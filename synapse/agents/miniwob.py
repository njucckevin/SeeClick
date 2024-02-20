import logging
from pathlib import Path
import os
import json
import copy
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By

from synapse.envs.miniwob.environment import MiniWoBEnv
from synapse.envs.miniwob.action import (
    MiniWoBType,
    MiniWoBElementClickXpath,
    MiniWoBElementClickOption,
    MiniWoBMoveXpath,
)
from synapse.memory.miniwob.build_memory import load_memory, retrieve_exemplar_name
from synapse.utils.llm import (
    generate_response,
    extract_from_response,
    num_tokens_from_messages,
    MAX_TOKENS,
)

from PIL import Image, ImageDraw

logger = logging.getLogger(__name__)

ENV_TO_FILTER = [
    "book-flight",
    "click-collapsible-2",
    "click-menu",
    "click-pie",
    "click-shape",
    "click-tab-2",
    "click-tab-2-hard",
    "count-shape",
    "email-inbox",
    "email-inbox-forward-nl",
    "email-inbox-forward-nl-turk",
    "email-inbox-nl-turk",
    "find-word",
    "grid-coordinate",
    "login-user-popup",
    "social-media",
    "social-media-some",
    "terminal",
    "tic-tac-toe",
    "use-autocomplete",
]


class Agent:
    def __init__(self, args):
        self.args = args
        self.env = MiniWoBEnv(subdomain=args.env_name, headless=args.headless)
        if self.args.env_name not in ENV_TO_FILTER:
            self.args.no_filter = True
        if not args.no_memory:
            self.memory = load_memory(args.memory_path)
        self.prompts = None
        self.prompt_type = None
        self.state = None
        self.task = None
        self.done = False
        self.reward = 0
        self.log_path = None
        self.trajectory = None
        self.conversation = None
        self.token_stats = None
        self.demo_traj = []

        self.record_traj = []   # 维护一个列表，保存一个traj中每个step对应的图片和action

    def reset(self, seed: int) -> None:
        self.state = self.env.reset(seed=seed, record_screenshots=True)
        self.task = self.env.get_task()
        self.done = False
        self.reward = 0
        self.trajectory = []
        self.conversation = []
        self.token_stats = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }
        """
        if self.args.no_memory:
            if self.args.env_name == "click-tab-2-hard":
                exemplar_name = "click-tab-2"
            elif self.args.env_name in [
                "email-inbox",
                "email-inbox-forward-nl",
                "email-inbox-forward-nl-turk",
            ]:
                exemplar_name = "email-inbox-nl-turk"
            else:
                exemplar_name = self.args.env_name
        else:
            query = "Task: " + self.task + "\nState:\n" + self.state
            exemplar_name = retrieve_exemplar_name(self.memory, query, 3)

        self.log_path = Path(
            os.path.join(
                self.args.log_dir,
                f"{self.args.model}/{self.args.env_name}/{f'no_filt_' if self.args.no_filter and self.args.env_name in ENV_TO_FILTER else ''}{f'no_mem_' if self.args.no_memory else ''}seed_{seed}{'' if exemplar_name == self.args.env_name else f'_{exemplar_name}'}.json",
            )
        )
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

        with open(os.path.join(self.args.memory_path, "exemplars.json"), "r") as rf:
            self.prompts = json.load(rf)[exemplar_name]
        demo = self.prompts["demo"]
        if self.args.no_filter:
            if "trajectory" not in demo[0]:
                self.prompt_type = "state_act"
                assert self.args.env_name != "click-pie"  # context limit
            else:
                self.prompt_type = "multi_state_act"
                assert self.args.env_name != "book-flight"  # context limit
        else:
            if "trajectory" not in demo[0]:
                if "obs" in demo[0]:
                    self.prompt_type = "obs_act"
                else:  # obs not available due to exemplar mismatch
                    self.prompt_type = "state_act"
            else:
                self.prompt_type = "multi_obs_act"

        self.demo_traj = []
        if self.prompt_type == "state_act" and "ablation_act_prompt" in self.prompts:
            self.demo_traj.append(
                {"role": "user", "content": self.prompts["ablation_act_prompt"]}
            )
        for d in demo:
            if self.prompt_type == "state_act":
                if "state" in d:  # fewer states due to context limit
                    self.demo_traj.append(
                        {
                            "role": "user",
                            "content": "Observation:\n" + d["state"] + "\nAction:",
                        }
                    )
                    self.demo_traj.append(
                        {"role": "assistant", "content": "```\n" + d["act"] + "\n```"}
                    )
            elif self.prompt_type == "multi_state_act":
                if exemplar_name in [
                    "login-user-popup",
                    "terminal",
                    "use-autocomplete",
                ]:  # context limit
                    if len(self.demo_traj) > 0:
                        break
                if all(
                    "state" in t for t in d["trajectory"]
                ):  # fewer states due to context limit
                    self.demo_traj.append(
                        {
                            "role": "user",
                            "content": "Task: " + d["task"] + "\nTrajectory:",
                        }
                    )
                    for t in d["trajectory"]:
                        self.demo_traj.append(
                            {
                                "role": "user",
                                "content": "Observation:\n" + t["state"] + "\nAction:",
                            }
                        )
                        self.demo_traj.append(
                            {
                                "role": "assistant",
                                "content": "```\n" + t["act"] + "\n```",
                            }
                        )
            elif self.prompt_type == "obs_act":
                self.demo_traj.append(
                    {
                        "role": "user",
                        "content": "Observation:\n" + d["obs"] + "\nAction:",
                    }
                )
                self.demo_traj.append(
                    {"role": "assistant", "content": "```\n" + d["act"] + "\n```"}
                )
            elif self.prompt_type == "multi_obs_act":
                self.demo_traj.append(
                    {"role": "user", "content": "Task: " + d["task"] + "\nTrajectory:"}
                )
                for t in d["trajectory"]:
                    self.demo_traj.append(
                        {
                            "role": "user",
                            "content": "Observation:\n" + t["obs"] + "\nAction:",
                        }
                    )
                    self.demo_traj.append(
                        {"role": "assistant", "content": "```\n" + t["act"] + "\n```"}
                    )
        """
    def filter(self) -> str:
        demo = self.prompts["demo"]
        if self.prompt_type in ["state_act", "multi_state_act"]:
            obs = self.state
        else:
            filter_with_code = False
            if self.prompt_type == "obs_act":
                if "code_filter_prompt" in self.prompts:
                    filter_with_code = True
                    filter_demo = ""  # create filter demo for possible LLM filtering in case code filtering fails
                    for d in demo:
                        if "state" in d:
                            filter_demo += "State:\n" + d["state"] + "\n"
                            filter_demo += "Observation:\n" + d["obs"] + "\n\n"
                    query = (
                        self.prompts["code_filter_prompt"]
                        .replace("<task>", self.task)
                        .replace("<state>", self.state)
                    )
                elif (
                    "filter_prompt" in self.prompts
                ):  # filter state into obs with specific prompts
                    query = self.prompts["filter_prompt"]
                    filter_demo = ""
                    for d in demo:
                        if "state" in d:
                            filter_demo += "State:\n" + d["state"] + "\n"
                            filter_demo += "Observation:\n" + d["obs"] + "\n\n"
                    query += filter_demo + "State:\n" + self.state + "\nObservation:"
                else:  # filter state into obs
                    filter_demo = ""
                    for d in demo:
                        if "state" in d:
                            filter_demo += "State:\n" + d["state"] + "\n"
                            filter_demo += "Observation:\n" + d["obs"] + "\n\n"
                    query = filter_demo + "State:\n" + self.state + "\nObservation:"
            else:
                cur_step = len(self.trajectory)
                if (
                    "code_filter_prompt" in self.prompts
                    and len(self.prompts["code_filter_prompt"][cur_step]) > 0
                ):
                    filter_with_code = True
                    filter_demo = ""  # create filter demo for possible LLM filtering in case code filtering fails
                    for d in demo:
                        if "state" in d["trajectory"][cur_step]:
                            filter_demo += (
                                "State:\n" + d["trajectory"][cur_step]["state"] + "\n"
                            )
                            filter_demo += (
                                "Observation:\n"
                                + d["trajectory"][cur_step]["obs"]
                                + "\n\n"
                            )
                    query = (
                        self.prompts["code_filter_prompt"][cur_step]
                        .replace("<task>", self.task)
                        .replace("<state>", self.state)
                    )
                else:
                    filter_demo = ""
                    for d in demo:
                        if "state" in d["trajectory"][cur_step]:
                            filter_demo += (
                                "State:\n" + d["trajectory"][cur_step]["state"] + "\n"
                            )
                            filter_demo += (
                                "Observation:\n"
                                + d["trajectory"][cur_step]["obs"]
                                + "\n\n"
                            )
                    query = filter_demo + "State:\n" + self.state + "\nObservation:"

            message = [{"role": "user", "content": query}]
            response, info = generate_response(
                messages=message,
                model=self.args.model,
                temperature=self.args.temperature,
                stop_tokens=["Action:", "Output:", "State:"],
            )
            self.conversation.append(
                {"input": message, "output": response, "token_stats": info}
            )
            for k, v in info.items():
                self.token_stats[k] += v

            if filter_with_code:
                obs_code = extract_from_response(response, "```")
                try:
                    logger.info(f"The code to extract observation:\n{obs_code}")
                    namespace = {"state": self.state}
                    exec(obs_code, namespace)
                    obs = namespace["obs"]
                except Exception as e:
                    logger.info(
                        f"{e}\nFailed to filter the raw state via code generation. Filter with LLM directly"
                    )
                    if self.prompt_type == "obs_act":
                        query = (
                            self.prompts["filter_prompt"]
                            + filter_demo
                            + "State:\n"
                            + self.state
                            + "\nObservation:"
                        )
                    else:
                        query = (
                            self.prompts["filter_prompt"][cur_step]
                            + filter_demo
                            + "State:\n"
                            + self.state
                            + "\nObservation:"
                        )
                    message = [{"role": "user", "content": query}]
                    response, info = generate_response(
                        messages=message,
                        model=self.args.model,
                        temperature=self.args.temperature,
                        stop_tokens=["Action:"],
                    )
                    self.conversation.append(
                        {"input": message, "output": response, "token_stats": info}
                    )
                    for k, v in info.items():
                        self.token_stats[k] += v
                    obs = response
            else:
                obs = response

            logger.info(f"filtered observation:\n{obs}")

        return obs

    def act(self, obs: str):
        sys_message = [
            {
                "role": "system",
                "content": "You are a large language model trained to navigate the web. To accomplish the task, use methods in the following Agent class to generate actions until you need the new state to proceed.\n```\nclass Agent:\n    def __init__(self, args):\n        ...\n\n    # Action: type a string via the keyboard\n    def type(self, characters: str) -> None:\n        ...\n\n    # Action: click an HTML element with a valid xpath\n    def click_xpath(self, xpath: str):\n        ...\n\n    # Actions: press a key on the keyboard, including:\n    # enter, space, arrowleft, arrowright, backspace, arrowup, arrowdown, command+a, command+c, command+v\n    def press(self, key_type: str) -> None:\n        ...\n\n    # Action: click an option HTML element in a list with a valid xpath\n    def click_option(self, xpath: str):\n        ...\n\n    # Action: move mouse cursor on an HTML element with a valid xpath\n    def movemouse(self, xpath: str):\n        ...\n```",
            }
        ]
        query_message = copy.deepcopy(self.demo_traj)
        if self.prompt_type in ["multi_state_act", "multi_obs_act"]:
            query_message.append(
                {"role": "user", "content": "Task: " + self.task + "\nTrajectory:"}
            )
            for t in self.trajectory:
                query_message.append(
                    {
                        "role": "user",
                        "content": "Observation:\n" + t["obs"] + "\nAction:",
                    }
                )
                query_message.append(
                    {"role": "assistant", "content": "```\n" + t["act"] + "\n```"}
                )
        query_message.append(
            {"role": "user", "content": "Observation:\n" + obs + "\nAction:"}
        )
        message = sys_message + query_message
        total_num_tokens = num_tokens_from_messages(message, self.args.model)
        if total_num_tokens > MAX_TOKENS[self.args.model]:
            self.conversation.append(
                {
                    "input": message,
                    "output": f"FAILED DUE TO THE CONTEXT LIMIT: {total_num_tokens}",
                }
            )
            return None
        response, info = generate_response(
            messages=message,
            model=self.args.model,
            temperature=self.args.temperature,
            stop_tokens=["Observation:"],
        )
        self.conversation.append(
            {"input": message, "output": response, "token_stats": info}
        )
        for k, v in info.items():
            self.token_stats[k] += v
        actions = extract_from_response(response, "```")

        self.trajectory.append(
            {
                "obs": obs,
                "act": actions,
            }
        )

        return actions
    """
    def step(self, action):
        self.state, reward, self.done, _ = self.env.step(action)
        if self.done:
            self.reward = reward
    """

    def step(self, action):
        # 重写一个step函数，用于在获取每个step执行前的screenshot，和执行要操作的元素或对应操作
        try:
            miniwob_state = self.env.instance.get_state()
            miniwob_sceenshot = miniwob_state.screenshot
            goal = miniwob_state.utterance

            # 获得操作对应的element
            if type(action) == MiniWoBType:
                action_record = {"action_type": "type", "typed_text": action._text}
            elif type(action) == MiniWoBElementClickOption:
                miniwob_element = self.env.instance.driver.find_element(By.XPATH, str(action.xpath))
                bbox = miniwob_element.rect
                bbox = [bbox['x'], bbox['y'], bbox['x']+bbox['width'], bbox['y']+bbox['height']]
                bbox = [item*2 for item in bbox]
                action_record = {"action_type": "click", "bbox": bbox}
            elif type(action) == MiniWoBElementClickXpath or type(action) == MiniWoBMoveXpath:
                miniwob_elements = self.env.instance.driver.find_elements(By.XPATH, str(action.xpath))
                if len(miniwob_elements) == 1:
                    miniwob_element = miniwob_elements[0]
                    bbox = miniwob_element.rect
                    bbox = [bbox['x'], bbox['y'], bbox['x'] + bbox['width'], bbox['y'] + bbox['height']]
                    bbox = [item * 2 for item in bbox]
                    action_record = {"action_type": "click", "bbox": bbox}
                elif len(miniwob_elements) > 1:
                    bboxes = []
                    for miniwob_element in miniwob_elements:
                        bbox = miniwob_element.rect
                        bbox = [bbox['x'], bbox['y'], bbox['x'] + bbox['width'], bbox['y'] + bbox['height']]
                        bbox = [item * 2 for item in bbox]
                        bboxes.append(bbox)
                    action_record = {"action_type": "click_seq", "bbox": bboxes}
                else:
                    return
            else:
                print("Action Type Error")
                input()
        except:     # 如果出现找不到元素的情况，则不执行任何操作，直接返回
            return

        miniwob_step = {"screenshot": miniwob_sceenshot, "action": action_record, "goal": goal}
        self.record_traj.append(miniwob_step)

        self.state, reward, self.done, _ = self.env.step(action)
        if self.done:
            self.reward = reward

    def log_results(self):
        filename = os.path.splitext(os.path.basename(self.log_path))[0]
        with open(self.log_path, "w") as f:
            json.dump(self.conversation, f, indent=2)
        if self.reward > 0:
            new_file_path = self.log_path.with_name(f"{filename}_success.json")
        else:
            new_file_path = self.log_path.with_name(f"{filename}_fail.json")
        os.rename(self.log_path, new_file_path)

    # Action: type a string via the keyboard
    def type(self, characters: str) -> None:
        action = MiniWoBType(characters)
        self.step(action)

    def click_xpath(self, xpath: str):
        action = MiniWoBElementClickXpath(xpath)
        self.step(action)

    def press(self, key_type: str) -> None:
        if key_type == "enter":
            action = MiniWoBType("\n")
        elif key_type == "space":
            action = MiniWoBType(" ")
        elif key_type == "arrowleft":
            action = MiniWoBType(Keys.LEFT)
        elif key_type == "arrowright":
            action = MiniWoBType(Keys.RIGHT)
        elif key_type == "backspace":
            action = MiniWoBType(Keys.BACKSPACE)
        elif key_type == "arrowup":
            action = MiniWoBType(Keys.UP)
        elif key_type == "arrowdown":
            action = MiniWoBType(Keys.DOWN)
        elif key_type in ["command+a", "command+c", "command+v"]:
            action = MiniWoBType(key_type)
        else:
            return
            raise ValueError("Invalid instruction")
        self.step(action)

    def click_option(self, xpath: str):
        action = MiniWoBElementClickOption(xpath)
        self.step(action)

    def movemouse(self, xpath: str):
        action = MiniWoBMoveXpath(xpath)
        self.step(action)

    def close(self):
        self.env.close()
