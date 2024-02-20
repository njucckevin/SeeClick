# 在MiniWob环境中测试
# MiniWoBCoordClick中点击位置是参考body元素的偏移量
# 且当前selenium版本中是相对body元素中心的偏移量，因此点击位置需要校准
# 在服务器配置MiniWob环境的step：
# 1.下载chrome和chromedriver，遇到dkpg问题请在google上查找
# 2.无图形界面的服务器需要headless模式
# 3.可能需要禁掉各种http_proxy代理
# 注意MiniWoBCoordClick点击位置需要body元素的偏移量，body元素的偏移量和窗口大小有关，可以在get_screenshot中查看

import os
import random
import torch
import json
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import AutoPeftModelForCausalLM
from transformers.generation import GenerationConfig
import re
import logging
import ast
import argparse
import numpy as np

import sys
sys.path.append('..')
from synapse.envs.miniwob.environment import MiniWoBEnv
from synapse.envs.miniwob.action import (
    MiniWoBType,
    MiniWoBElementClickXpath,
    MiniWoBElementClickOption,
    MiniWoBMoveXpath,
    MiniWoBCoordClick,
    MiniWoBElementClickId,
)

logging.basicConfig(level=logging.INFO)


# 创建解析器
parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str, required=True)
parser.add_argument('--qwen_path', type=str, required=True)
parser.add_argument('--imgs_dir_temp', type=str, required=True)
parser.add_argument("--num_episodes", type=int, default=10)
parser.add_argument("--env_name", type=str, default="all")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--headless", action="store_true", default=True)
args = parser.parse_args()

seed = args.seed
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(seed)
random.seed(seed)

model_path = args.model_path
qwen_path = args.qwen_path
model = AutoModelForCausalLM.from_pretrained(model_path, device_map="cuda", trust_remote_code=True, bf16=True).eval() # load with model checkpoint
# model = AutoPeftModelForCausalLM.from_pretrained(model_path, device_map="cuda", trust_remote_code=True).eval()  # load with lora checkpoint
tokenizer = AutoTokenizer.from_pretrained(qwen_path, trust_remote_code=True)
model.generation_config = GenerationConfig.from_pretrained(qwen_path, trust_remote_code=True)

miniwob_imgs_dir_temp = args.imgs_dir_temp
miniwob_train = json.load(open('../data/miniwob_data_train.json', 'r'))     # load tasks from train set
miniwob_tasks = list(miniwob_train.keys())
if args.env_name != "all" and args.env_name not in miniwob_tasks:
    miniwob_tasks.append(args.env_name)
task_max_step = {k: (10 if (k != 'guess-number' and k != 'use-slider' and k != 'choose-date') else 30) for k in
                 miniwob_tasks}
prompt_origin = "Please generate the next move according to the ui screenshot, instruction and previous actions. Instruction: {}. Previous actions: {}"
result = {}
for env in tqdm(miniwob_tasks):
    if args.env_name != "all":
        if env != args.env_name:
            continue

    success = 0
    print("Task: " + env)
    for j in tqdm(range(args.num_episodes)):
        traj = []
        # 初始化MiniWob环境
        seed_task = random.randint(0, 1000000)
        miniwob_env = MiniWoBEnv(subdomain=env, headless=args.headless)
        miniwob_env.reset(seed=seed_task, record_screenshots=True)

        img_dir = miniwob_imgs_dir_temp

        reward = 0
        previous_actions = []
        for k in range(task_max_step[env]):

            # 获得MiniWob状态
            miniwob_state = miniwob_env.instance.get_state()
            state_screenshot = miniwob_state.screenshot
            img_path = os.path.join(img_dir, args.model_path + '-' + env + '-' + str(
                seed_task) + '-' + str(k) + '.jpg')
            state_screenshot.save(img_path)
            goal = miniwob_state.utterance

            # Agent生成下一步动作
            previous_step = ""
            for i, action in enumerate(previous_actions[-4:]):
                previous_step += 'Step' + str(i) + ': ' + action + ". "
            prompt = prompt_origin.format(goal, previous_step)

            query = tokenizer.from_list_format([{'image': img_path}, {'text': prompt}, ])
            with torch.no_grad():
                response, history = model.chat(tokenizer, query=query, history=None)

            action_step_record = {"img_path": img_path, "sentence": response, "success": False}
            traj.append(action_step_record)

            previous_actions.append(response)

            try:
                action_pred = ast.literal_eval(response)
            except:
                continue
            # 将动作转换为MiniWob Action
            try:
                if action_pred["action_type"] == 4:
                    click_x = action_pred['click_point'][0] * 160
                    click_y = action_pred['click_point'][1] * 210
                    miniwob_action = MiniWoBCoordClick(click_x - 150, click_y - 105)
                elif action_pred["action_type"] == 3:
                    typed_text = action_pred['typed_text']
                    miniwob_action = MiniWoBType(typed_text)
                else:
                    print("action undefined")
                    input()
                    continue
                # 执行动作，并判断是否结束
                _, reward, done, _ = miniwob_env.step(miniwob_action)
            except:
                continue
            # print(reward)
            if reward > 0.8:
                success += 1
                for item in traj:
                    item["success"] = True
                break

            if done:
                break

        miniwob_env.close()

    result[env] = success / args.num_episodes
    print("Task: " + env + "  Score: " + str(success / args.num_episodes))

print(result)
print("Average Score: " + str(np.mean(list(result.values()))))