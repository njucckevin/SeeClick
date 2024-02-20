import logging
import numpy as np
import json
import os
import random
from pathlib import Path

from synapse.envs.mind2web.env_utils import (
    get_target_obs_and_act,
    get_top_k_obs,
    calculate_f1,
    parse_act_str,
    construct_act_str,
)
from synapse.utils.llm import (
    generate_response,
    num_tokens_from_messages,
    MAX_TOKENS,
    extract_from_response,
)
from synapse.memory.mind2web.build_memory import (
    load_memory,
    retrieve_exemplar_name,
    get_specifiers_from_sample,
)

logger = logging.getLogger(__name__)


def eval_sample(task_id, args, sample):
    # initialize metrics
    element_acc = []
    action_f1 = []
    step_success = []
    success = []
    token_stats = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    conversation = []
    episode_length = len(sample["action_reprs"])

    if args.no_trajectory:
        assert args.no_memory
        exemplars = [
            [
                {
                    "role": "user",
                    "content": "Observation:\n```\n<html> <div> <div> <a tock home page /> <button id=0 book a reservation. toggle open> <span> Book a reservation </span> </button> <button book a reservation. toggle open> </button> </div> <div> <select id=1 type> <option reservations true> Dine in </option> <option pickup> Pickup </option> <option delivery> Delivery </option> <option events> Events </option> <option wineries> Wineries </option> <option all> Everything </option> </select> <div id=2> <p> Celebrating and supporting leading women shaking up the industry. </p> <span> Explore now </span> </div> </div> </div> </html>\n```\nTask: Check for pickup restaurant available in Boston, NY on March 18, 5pm with just one guest\nPrevious actions:\nNone\nNext action:",
                },
                {"role": "assistant", "content": "`SELECT [1] [Pickup]`"},
            ],
            [
                {
                    "role": "user",
                    "content": "Observation:\n```\n<html> <div> <nav main menu> <ul> <li> <div button> Car Sales </div> <div id=0> <div> <div> <div> Buy A Car </div> <div> Plan Your Purchase </div> </div> <div> <h4> Its Tax Refund Time. Treat Yourself to an Upgrade. </h4> <p> With a variety of options, invest your refund in what you really want - a quality, used vehicle from Enterprise. </p> <a> View Inventory </a> </div> </div> </div> </li> <div id=1> Enterprise Fleet Management </div> </ul> </nav> <div region> <button id=2 selected pick-up date 03/19/2023> <span> <span> 19 </span> <div> <span> Mar </span> <span> 2023 </span> </div> </span> </button> </div> </div> </html>\n```\nTask: Find a mini van at Brooklyn City from April 5th to April 8th for a 22 year old renter.\nPrevious actions:\n[searchbox]  Pick-up & Return Location (ZIP, City or Airport) (... -> TYPE: Brooklyn\n[option]  Brooklyn, NY, US Select -> CLICK\nNext action:",
                },
                {"role": "assistant", "content": "`CLICK [2]`"},
            ],
            [
                {
                    "role": "user",
                    "content": "Observation:\n```\n<html> <form search> <input id=6385 search q blazer search by keyword /> <button submit search> </button> <button button close> </button> </form> </html>\n```\nTask: Find a black blazer for men with L size and add to wishlist.\nPrevious actions:\n[svg]   -> CLICK\nNext action:",
                },
                {"role": "assistant", "content": "`TYPE [6385] [blazer]`"},
            ],
        ]
    else:
        memory = load_memory(args.memory_path)
        with open(os.path.join(args.memory_path, "exemplars.json"), "r") as f:
            memory_mapping = json.load(f)
        if not args.no_memory:
            specifier = get_specifiers_from_sample(sample)
            retrieved_exemplar_names, scores = retrieve_exemplar_name(
                memory, specifier, args.retrieve_top_k
            )
            exemplars = [memory_mapping[name] for name in retrieved_exemplar_names]
        else:
            seed = 0
            random.seed(seed)
            exemplars = random.sample(memory_mapping, args.retrieve_top_k)

    sys_message = [
        {
            "role": "system",
            "content": "You are a large language model trained to navigate the web. Output the next action and wait for the next observation. Here is the action space:\n1. `CLICK [id]`: Click on an HTML element with its id.\n2. `TYPE [id] [value]`: Type a string into the element with the id.\n3. `SELECT [id] [value]`: Select a value for an HTML element by its id.",
        }
    ]

    prev_actions = []
    prev_obs = []
    previous_k = 5

    for s, act_repr in zip(sample["actions"], sample["action_reprs"]):
        _, target_act = get_target_obs_and_act(s)
        pos_candidates = [
            c for c in s["pos_candidates"] if c["rank"] < args.top_k_elements
        ]

        if args.no_trajectory:
            # Continue next loop if the ground truth element is not in the cleaned html
            if len(pos_candidates) == 0:
                element_acc.append(0)
                action_f1.append(0)
                step_success.append(0)
                prev_actions.append(act_repr)
                conversation.append("The ground truth element is not in cleaned html")
                continue

            obs, _ = get_top_k_obs(s, args.top_k_elements, use_raw=False)
            query = f"Observation:\n```\n{obs}\n```\nTask: {sample['confirmed_task']}\nPrevious actions:\n"
            if len(prev_actions) > 0:
                for a in prev_actions[-previous_k:]:
                    query += f"{a}\n"
            else:
                query += "None\n"
            query += "Next action:"
            query = [{"role": "user", "content": query}]
            prev_actions.append(act_repr)
        else:
            target_obs, _ = get_top_k_obs(s, args.previous_top_k_elements)
            # Continue next loop if the ground truth element is not in the cleaned html
            if len(pos_candidates) == 0:
                element_acc.append(0)
                action_f1.append(0)
                step_success.append(0)
                prev_obs.append("Observation: `" + target_obs + "`")
                prev_actions.append("Action: `" + target_act + "` (" + act_repr + ")")
                conversation.append("The ground truth element is not in cleaned html")
                continue

            query = []
            for o, a in zip(prev_obs, prev_actions):
                if len(query) == 0:
                    query.append(
                        {
                            "role": "user",
                            "content": f"Task: {sample['confirmed_task']}\nTrajectory:\n"
                            + o,
                        }
                    )
                else:
                    query.append({"role": "user", "content": o})
                query.append({"role": "assistant", "content": a})
            obs, _ = get_top_k_obs(s, args.top_k_elements, use_raw=False)
            if len(query) == 0:
                query.append(
                    {
                        "role": "user",
                        "content": f"Task: {sample['confirmed_task']}\nTrajectory:\n"
                        + "Observation: `"
                        + obs
                        + "`",
                    }
                )
            else:
                query.append({"role": "user", "content": "Observation: `" + obs + "`"})
            prev_obs.append("Observation: `" + target_obs + "`")
            prev_actions.append("Action: `" + target_act + "` (" + act_repr + ")")

        total_num_tokens = num_tokens_from_messages(sys_message + query, args.model)
        if total_num_tokens > MAX_TOKENS[args.model]:
            logger.info(
                f"Too many tokens in acting ({total_num_tokens} / {MAX_TOKENS[args.model]}), skipping..."
            )
            element_acc.append(0)
            action_f1.append(0)
            step_success.append(0)
            conversation.append(
                {
                    "input": sys_message + query,
                    "output": f"FAILED DUE TO THE CONTEXT LIMIT: {total_num_tokens}",
                }
            )
            continue

        demo_message = []
        for e_id, e in enumerate(exemplars):
            total_num_tokens = num_tokens_from_messages(
                sys_message + demo_message + e + query, args.model
            )
            if total_num_tokens > MAX_TOKENS[args.model]:
                logger.info(
                    f"Using {e_id} / {len(exemplars)} exemplars due to context limit"
                )
                break
            else:
                demo_message.extend(e)

        message = sys_message + demo_message + query
        response, info = generate_response(
            messages=message,
            model=args.model,
            temperature=args.temperature,
            stop_tokens=["Task:", "obs:"],
        )
        conversation.append({"input": message, "output": response, "token_stats": info})
        for k, v in info.items():
            token_stats[k] += v
        pred_act = extract_from_response(response, "`")
        pred_op, pred_id, pred_val = parse_act_str(pred_act)
        target_op, _, target_val = parse_act_str(target_act)

        # calculate metrics
        pos_ids = [c["backend_node_id"] for c in s["pos_candidates"]][:1]
        if pred_id in pos_ids:
            element_acc.append(1)
        else:
            element_acc.append(0)
        action_f1.append(
            calculate_f1(
                construct_act_str(pred_op, pred_val),
                construct_act_str(target_op, target_val),
            )
        )
        conversation.append({"pred_act": pred_act, "target_act": target_act})
        if pred_act == target_act:
            step_success.append(1)
        else:
            step_success.append(0)

    # check the last episode_length of step_success, if all 1, then success = 1
    if np.sum(step_success[-episode_length:]) == episode_length:
        success.append(1)
    else:
        success.append(0)

    conversation.append(
        {
            "element_acc": element_acc,
            "action_f1": action_f1,
            "step_success": step_success,
            "success": success,
        }
    )
    if args.no_trajectory:
        log_dir = Path(f"{args.log_dir}/{args.model}/{args.benchmark}/no_mem_no_traj")
    else:
        log_dir = Path(
            f"{args.log_dir}/{args.model}/{args.benchmark}{'/no_mem' if args.no_memory else ''}"
        )
    log_dir.mkdir(parents=True, exist_ok=True)
    with open(os.path.join(log_dir, f"{task_id}.json"), "w") as f:
        json.dump(conversation, f, indent=2)


def eval_sample_llama(
    task_id,
    args,
    sample,
    model,
    tokenizer,
):
    # initialize metrics
    element_acc = []
    action_f1 = []
    step_success = []
    success = []
    conversation = []
    episode_length = len(sample["action_reprs"])

    # prepare exemplars
    if args.no_trajectory:
        assert args.no_memory
        if args.retrieve_top_k != 3:
            assert (
                args.retrieve_top_k == 0
            ), "We only support zero-shot or three-shot evaluation"
            exemplars = []
        else:
            exemplars = [
                [
                    "Observation:\n```\n<html> <div> <div> <a tock home page /> <button id=0 book a reservation. toggle open> <span> Book a reservation </span> </button> <button book a reservation. toggle open> </button> </div> <div> <select id=1 type> <option reservations true> Dine in </option> <option pickup> Pickup </option> <option delivery> Delivery </option> <option events> Events </option> <option wineries> Wineries </option> <option all> Everything </option> </select> <div id=2> <p> Celebrating and supporting leading women shaking up the industry. </p> <span> Explore now </span> </div> </div> </div> </html>\n```\nTask: Check for pickup restaurant available in Boston, NY on March 18, 5pm with just one guest\nPrevious actions:\nNone\nNext action:",
                    "`SELECT [1] [Pickup]`",
                ],
                [
                    "Observation:\n```\n<html> <div> <nav main menu> <ul> <li> <div button> Car Sales </div> <div id=0> <div> <div> <div> Buy A Car </div> <div> Plan Your Purchase </div> </div> <div> <h4> Its Tax Refund Time. Treat Yourself to an Upgrade. </h4> <p> With a variety of options, invest your refund in what you really want - a quality, used vehicle from Enterprise. </p> <a> View Inventory </a> </div> </div> </div> </li> <div id=1> Enterprise Fleet Management </div> </ul> </nav> <div region> <button id=2 selected pick-up date 03/19/2023> <span> <span> 19 </span> <div> <span> Mar </span> <span> 2023 </span> </div> </span> </button> </div> </div> </html>\n```\nTask: Find a mini van at Brooklyn City from April 5th to April 8th for a 22 year old renter.\nPrevious actions:\n[searchbox]  Pick-up & Return Location (ZIP, City or Airport) (... -> TYPE: Brooklyn\n[option]  Brooklyn, NY, US Select -> CLICK\nNext action:",
                    "`CLICK [2]`",
                ],
                [
                    "Observation:\n```\n<html> <form search> <input id=6385 search q blazer search by keyword /> <button submit search> </button> <button button close> </button> </form> </html>\n```\nTask: Find a black blazer for men with L size and add to wishlist.\nPrevious actions:\n[svg]   -> CLICK\nNext action:",
                    "`TYPE [6385] [blazer]`",
                ],
            ]
    else:
        memory = load_memory(args.memory_path)
        with open(os.path.join(args.memory_path, "exemplars.json"), "r") as f:
            memory_mapping = json.load(f)
        if not args.no_memory:
            specifier = get_specifiers_from_sample(sample)
            retrieved_exemplar_names, scores = retrieve_exemplar_name(
                memory, specifier, args.retrieve_top_k
            )
            exemplars = [memory_mapping[name] for name in retrieved_exemplar_names]
        else:
            seed = 0
            random.seed(seed)
            exemplars = random.sample(memory_mapping, args.retrieve_top_k)

    # prepare prompt
    system_prompt = "You are a large language model trained to navigate the web. Output the next action and wait for the next observation. Here is the action space:\n1. `CLICK [id]`: Click on an HTML element with its id.\n2. `TYPE [id] [value]`: Type a string into the element with the id.\n3. `SELECT [id] [value]`: Select a value for an HTML element by its id."
    prev_actions = []
    prev_obs = []
    previous_k = 5

    # query the language model
    for s, act_repr in zip(sample["actions"], sample["action_reprs"]):
        _, target_act = get_target_obs_and_act(s)
        pos_candidates = [
            c for c in s["pos_candidates"] if c["rank"] < args.top_k_elements
        ]

        if args.no_trajectory:
            # Continue next loop if the ground truth element is not in the cleaned html
            if len(pos_candidates) == 0:
                element_acc.append(0)
                action_f1.append(0)
                step_success.append(0)
                prev_actions.append(act_repr)
                conversation.append("The ground truth element is not in cleaned html")
                continue

            # Add system prompt
            query = f"<<SYS>>\n{system_prompt}\n<</SYS>> </s>\n\n"
            # Add exemplars
            for e in exemplars:
                query += f"<s>[INST] {e[0]} [/INST] {e[1]} </s>"
            # Add current information
            obs, _ = get_top_k_obs(s, args.top_k_elements, use_raw=False)
            query += f"<s>[INST] Observation:\n```\n{obs}\n```\nTask: {sample['confirmed_task']}\nPrevious actions:\n"
            if len(prev_actions) > 0:
                for a in prev_actions[-previous_k:]:
                    query += f"{a}\n"
            else:
                query += "None\n"
            query += "Next action: [/INST]"
            prev_actions.append(act_repr)
        else:
            target_obs, _ = get_top_k_obs(s, args.previous_top_k_elements)
            # Continue next loop if the ground truth element is not in the cleaned html
            if len(pos_candidates) == 0:
                element_acc.append(0)
                action_f1.append(0)
                step_success.append(0)
                prev_obs.append("Observation: `" + target_obs + "`")
                prev_actions.append("Action: `" + target_act + "` (" + act_repr + ")")
                conversation.append("The ground truth element is not in cleaned html")
                continue

            # Add system prompt
            query = f"<<SYS>>\n{system_prompt}\n<</SYS>>\n\n"
            # Add exemplars
            for e in exemplars:
                query += ""
                for i in range(len(e)):
                    if i % 2 == 0:
                        query += f"[INST]\n{e[i]['content']}\n[/INST]\n"
                    else:
                        query += f"{e[i]['content']}\n"
            # Add current information
            for i in range(len(prev_obs)):
                o, a = prev_obs[i], prev_actions[i]
                if i == 0:
                    query += (
                        f"[INST]\nTask: {sample['confirmed_task']}\nTrajectory:\n"
                        + o
                        + "\n[/INST]\n"
                    )
                else:
                    query += "[INST]\n" + o + "\n[/INST]\n"
                query += a + "\n"
            obs, _ = get_top_k_obs(s, args.top_k_elements, use_raw=False)
            if len(prev_obs) == 0:
                query += (
                    f"[INST]\nTask: {sample['confirmed_task']}\nTrajectory:\nObservation: `"
                    + obs
                    + "`\n[/INST]\n"
                )
            else:
                query += "[INST]\nObservation: `" + obs + "`\n[/INST]\n"
            prev_obs.append("Observation: `" + target_obs + "`")
            prev_actions.append("Action: `" + target_act + "` (" + act_repr + ")")

        # generate action
        tok_enc = tokenizer.encode(query)
        print(f"INPUT TOKENS: {len(tok_enc)}")
        input = tokenizer.decode(tok_enc)
        model_input = tokenizer(query, return_tensors="pt").to("cuda")
        generation_output = model.generate(
            **model_input,
            pad_token_id=tokenizer.eos_token_id,
            max_new_tokens=100,
        )
        output = tokenizer.decode(generation_output[0])
        output = output[len(input) :]
        print("OUTPUTS:", output)
        conversation.append({"input": input, "output": output})
        if args.no_trajectory and "`" not in output:
            try:
                import re

                pattern = r"\[?(CLICK|SELECT|TYPE)\]?\s*\[(\d+)]\s*(?:\[(.+?)\])?"
                matches = re.findall(pattern, output.strip().replace("</s>", ""))
                result = [
                    "{} [{}] [{}]".format(m[0], m[1], m[2])
                    if len(m) > 2 and m[2]
                    else "{} [{}]".format(m[0], m[1])
                    for m in matches
                ]
                pred_act = result[0]
            except:
                pred_act = output.strip().replace("</s>", "")
        else:
            pred_act = extract_from_response(output, "`")
        pred_op, pred_id, pred_val = parse_act_str(pred_act)
        target_op, target_id, target_val = parse_act_str(target_act)

        # calculate metrics
        pos_ids = [c["backend_node_id"] for c in s["pos_candidates"]][:1]
        if pred_id in pos_ids:
            element_acc.append(1)
        else:
            element_acc.append(0)
        action_f1.append(
            calculate_f1(
                construct_act_str(pred_op, pred_val),
                construct_act_str(target_op, target_val),
            )
        )
        conversation.append({"pred_act": pred_act, "target_act": target_act})
        if pred_op == target_op and pred_id == target_id and pred_val == target_val:
            step_success.append(1)
        else:
            step_success.append(0)

    # check the last episode_length of step_success, if all 1, then success = 1
    if np.sum(step_success[-episode_length:]) == episode_length:
        success.append(1)
    else:
        success.append(0)

    conversation.append(
        {
            "element_acc": element_acc,
            "action_f1": action_f1,
            "step_success": step_success,
            "success": success,
        }
    )
    if args.lora_dir is None:
        if args.no_trajectory:
            log_dir = Path(
                f"{args.log_dir}/codellama/{args.benchmark}/{args.retrieve_top_k}shot/no_mem_no_traj_top{args.top_k_elements}"
            )
        else:
            if args.no_memory:
                log_dir = Path(
                    f"{args.log_dir}/codellama/{args.benchmark}/{args.retrieve_top_k}shot/no_mem_top{args.top_k_elements}"
                )
            else:
                log_dir = Path(
                    f"{args.log_dir}/codellama/{args.benchmark}/{args.retrieve_top_k}shot/top{args.top_k_elements}"
                )
    else:
        if args.no_trajectory:
            log_dir = Path(
                f"{args.log_dir}/codellama/{args.benchmark}/{args.retrieve_top_k}shot/naive_ft_top{args.top_k_elements}/no_mem_no_traj"
            )
        else:
            if args.no_memory:
                log_dir = Path(
                    f"{args.log_dir}/codellama/{args.benchmark}/{args.retrieve_top_k}shot/traj_ft_top{args.top_k_elements}/no_mem"
                )
            else:
                log_dir = Path(
                    f"{args.log_dir}/codellama/{args.benchmark}/{args.retrieve_top_k}shot/traj_ft_top{args.top_k_elements}"
                )

    log_dir.mkdir(parents=True, exist_ok=True)
    with open(os.path.join(log_dir, f"{task_id}.json"), "w") as f:
        json.dump(conversation, f, indent=2)
