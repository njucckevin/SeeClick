# visualize&process mind2web data
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image, ImageDraw
import json
import os
from tqdm import tqdm
import random
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--imgs_dir', type=str, required=True)
args = parser.parse_args()


# show image with bbox
def show_image_with_bbox(image, bbox=None):

    img_width, img_height = image.size
    dpi = 40
    figsize = img_width / float(dpi), img_height / float(dpi)
    fig, ax = plt.subplots(1, figsize=figsize)
    ax.imshow(image)

    if bbox:
        x = int(bbox['x'])
        y = int(bbox['y'])
        width = int(bbox['width'])
        height = int(bbox['height'])
        rect = patches.Rectangle((x, y), width, height, linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
    plt.axis('off')
    plt.show()


# convert action to prediction format
def action2step(action, image_size):
    action_type = action["operation"]["original_op"]
    assert action_type in ['CLICK', 'TYPE', 'SELECT', 'HOVER', 'ENTER']  # five types of data

    point_x = action["bbox"]["x"] + (action["bbox"]["width"] / 2)
    point_y = action["bbox"]["y"] + (action["bbox"]["height"] / 2)
    click_point = [point_x / image_size[0], point_y / image_size[1]]
    click_point = [round(item, 3) for item in click_point]
    click_point = [f"{item:.2f}" for item in click_point]
    click_point = "({},{})".format(click_point[0], click_point[1])

    if action_type in ['CLICK', 'HOVER', 'ENTER']:  # following mind2web, these three actions are regarded as click
        action_step = "{{\"action_type\": {}, \"click_point\": {}}}".format(4, click_point)
    elif action_type == 'SELECT':
        select_value = action["operation"]["value"]
        action_step = "{{\"action_type\": {}, \"click_point\": {}, \"value\": \"{}\"}}".format(2, click_point, select_value)
    elif action_type == 'TYPE':
        typed_text = action["operation"]["value"]
        action_step = "{{\"action_type\": {}, \"click_point\": {}, \"value\": \"{}\"}}".format(3, click_point, typed_text)
    return action_step


mind2web_imgs_dir = args.imgs_dir
mind2web_train = json.load(open('../data/mind2web_data_train.json', 'r'))
train_step = []
prompt_origin = "Please generate the next move according to the ui screenshot, instruction and previous actions. Instruction: {}. Previous actions: {}"
step_i = 0
for episode in tqdm(mind2web_train):
    goal = episode["confirmed_task"]
    annot_id = episode["annotation_id"]
    previous_actions = []

    # print(episode["action_reprs"])

    for step in episode["actions"]:

        # Few actions can not find its corresponding bbox, jump these actions
        if "bbox" not in step:
            print("action not found")
            continue

        filename = annot_id + '-' + step["action_uid"] + '.jpg'
        img_path = os.path.join(mind2web_imgs_dir, filename)
        if not os.path.exists(img_path):
            print("img not found")
            input()
        image = Image.open(img_path)

        # visualize step data
        # show_image_with_bbox(image, step["bbox"])
        # print(step)
        # input()

        previous_step = ""
        for i, action in enumerate(previous_actions[-4:]):
            previous_step += 'Step' + str(i) + ': ' + action + ". "

        action_step = action2step(step, image.size)
        previous_actions.append(action_step)

        prompt = prompt_origin.format(goal, previous_step)

        conversations = []
        conv_user = {"from": "user", "value": "Picture 1: <img>{}</img>\n".format(img_path)}
        conv_user["value"] += prompt
        conv_ai = {"from": "assistant", "value": str(action_step)}
        conversations.append(conv_user)
        conversations.append(conv_ai)

        train_step.append({"id": "mind2web_{}".format(step_i), "conversations": conversations})
        step_i += 1

random.shuffle(train_step)
print("Num of total step: " + str(len(train_step)))
json.dump(train_step, open("../data/mind2web_train_sft.json", "w"))
