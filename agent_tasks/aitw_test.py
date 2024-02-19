# evaluation on aitw
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
from PIL import Image
import numpy as np

import action_matching


# convert action to prediction format
def action2step(step_data):
    action_type = step_data["action_type_id"]

    if action_type == 4:
        if step_data["action_type_text"] == 'click':  # for click action, we calculate midpoint of touch and lift as the click point
            touch_point = step_data["touch"]
            lift_point = step_data["lift"]
            action_type_new = 4
            click_point = [(touch_point[0] + lift_point[0]) / 2, (touch_point[1] + lift_point[1]) / 2]
            click_point = [f"{item:.2f}" for item in click_point]
            click_point = "({},{})".format(click_point[0], click_point[1])
            action = "{{\"action_type\": {}, \"click_point\": {}}}".format(action_type_new, click_point)
        else:  # for scroll action, we assign an action_type_id for each scroll
            if step_data["action_type_text"] == 'scroll down':
                action_type_new = 0
            elif step_data["action_type_text"] == 'scroll up':
                action_type_new = 1
            elif step_data["action_type_text"] == 'scroll left':
                action_type_new = 8
            elif step_data["action_type_text"] == 'scroll right':
                action_type_new = 9
            action = "{{\"action_type\": {}}}".format(action_type_new)
    elif action_type == 3:
        typed_text = step_data["type_text"]
        action_type_new = action_type
        action = "{{\"action_type\": {}, \"typed_text\": \"{}\"}}".format(action_type_new, typed_text)
    else:
        action_type_new = action_type
        action = "{{\"action_type\": {}}}".format(action_type_new)

    return action


torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
random.seed(0)

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str, required=True)
parser.add_argument('--qwen_path', type=str, required=True)
parser.add_argument('--imgs_dir', type=str, required=True)
args = parser.parse_args()

model_path = args.model_path
qwen_path = args.qwen_path
# model = AutoModelForCausalLM.from_pretrained(model_path, device_map="cuda", trust_remote_code=True, bf16=True).eval() # load with model checkpoint
model = AutoPeftModelForCausalLM.from_pretrained(model_path, device_map="cuda", trust_remote_code=True).eval()  # load with lora checkpoint
tokenizer = AutoTokenizer.from_pretrained(qwen_path, trust_remote_code=True)
model.generation_config = GenerationConfig.from_pretrained(qwen_path, trust_remote_code=True)

aitw_imgs_dir = args.imgs_dir
aitw_test = json.load(open('../data/aitw_data_test.json', 'r'))
prompt_origin = "Please generate the next move according to the ui screenshot, instruction and previous actions. Instruction: {}. Previous actions: {}"
score_average = 0

for task, episodes in aitw_test.items():

    print("Task: " + task)

    corr_action = 0
    corr_type = 0
    num_text = 0
    corr_text = 0
    num_scroll = 0
    corr_scroll = 0
    num_click = 0
    corr_click = 0
    num_both_click = 0
    corr_both_click = 0
    num_wrong_format = 0
    num = 0
    for j, episode in tqdm(enumerate(episodes)):

        previous_actions = []

        for step in episode:
            img_filename = step["img_filename"] + '.png'
            img_path = os.path.join(aitw_imgs_dir, img_filename)
            if not os.path.exists(img_path):
                print("img not found")
                continue

            goal = step["goal"]

            previous_step = ""
            for i, action in enumerate(previous_actions[-4:]):
                previous_step += 'Step' + str(i) + ': ' + action + ". "

            action_step = action2step(step)
            previous_actions.append(action_step)

            action_ref = action_matching.action_2_format(step)

            prompt = prompt_origin.format(goal, previous_step)
            try:    # several sample's img dir lead to error, just jump it
                query = tokenizer.from_list_format([{'image': img_path}, {'text': prompt}, ])
                with torch.no_grad():
                    response, history = model.chat(tokenizer, query=query, history=None)
            except:
                continue

            num += 1

            try:
                action_pred = action_matching.pred_2_format(ast.literal_eval(response))
                annot_position = np.array(
                    [step["annot_position"][i:i + 4] for i in range(0, len(step["annot_position"]), 4)])
                check_match = action_matching.check_actions_match(action_pred["touch_point"], action_pred["lift_point"],
                                                                  action_pred["action_type"], action_ref["touch_point"],
                                                                  action_ref["lift_point"], action_ref["action_type"],
                                                                  annot_position)
                # step accuracy
                if check_match == True:
                    corr_action += 1
                    match_label = 1
                    logging.info("Step: " + str(j) + " right")
                else:
                    match_label = 0
                    logging.info("Step: " + str(j) + " wrong")

                # type accuracy
                if action_pred["action_type"] == action_ref["action_type"]:
                    corr_type += 1

                # text accuracy
                if action_ref["action_type"] == 3:
                    num_text += 1
                    if (action_pred["typed_text"] == action_ref["typed_text"]) or (
                            action_pred["typed_text"] in action_ref["typed_text"]) or (
                            action_ref["typed_text"] in action_pred["typed_text"]):
                        corr_text += 1

                if action_ref["action_type"] == 4:
                    # click accuracy
                    if action_matching.is_tap_action(action_ref["touch_point"], action_ref["lift_point"]):
                        num_click += 1
                        if match_label:
                            corr_click += 1
                    # scroll accuracy
                    else:
                        num_scroll += 1
                        if match_label:
                            corr_scroll += 1
                    if (action_pred["action_type"] == 4) and action_matching.is_tap_action(action_ref["touch_point"],
                                                                                           action_ref[
                                                                                               "lift_point"]) and action_matching.is_tap_action(
                            action_pred["touch_point"], action_pred["lift_point"]):
                        num_both_click += 1
                        if match_label:
                            corr_both_click += 1

            except:
                num_wrong_format += 1
                logging.info("Step: " + str(j) + " wrong format")

    score_average += corr_action / num

    logging.info("Action Acc: " + str(corr_action / num))
    logging.info("Type Acc: " + str(corr_type / num))
    logging.info("Text Acc: " + str(corr_text / num_text))
    logging.info("Click Acc: " + str(corr_click / num_click))
    logging.info("Scroll Acc: " + str(corr_scroll / num_scroll))
    logging.info("Both Click Acc: " + str(corr_both_click / num_both_click))
    logging.info("Num Both Click: " + str(num_both_click))
    logging.info("Num wrong format: " + str(num_wrong_format))

logging.info("Average score: " + str(score_average / 5))
