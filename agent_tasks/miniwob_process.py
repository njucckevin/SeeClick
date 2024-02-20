import json
from tqdm import tqdm
import random
import os
import re
import argparse
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image, ImageDraw

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
        x = int(bbox[0])
        y = int(bbox[1])
        width = int(bbox[2]-bbox[0])
        height = int(bbox[3]-bbox[1])
        rect = patches.Rectangle((x, y), width, height, linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
    plt.axis('off')
    plt.show()


miniwob_imgs_dir = args.imgs_dir
miniwob_train = json.load(open('../data/miniwob_data_train.json', 'r'))
train_step = []
prompt_origin = "Please generate the next move according to the ui screenshot, instruction and previous actions. Instruction: {}. Previous actions: {}"
step_i = 0
for task_name, data in tqdm(miniwob_train.items()):
    for episode in data:
        previous_actions = []
        for step in episode:
            img_filename = step["img_filename"]
            img_path = os.path.join(miniwob_imgs_dir, img_filename)

            goal = step["goal"]

            # visualize step data
            # image = Image.open(img_path)
            # bbox = step["bbox"] if "bbox" in step else None
            # show_image_with_bbox(image, bbox)
            # print(step)
            # input()

            previous_step = ""
            for i, action in enumerate(previous_actions[-4:]):
                previous_step += 'Step' + str(i) + ': ' + action + ". "

            if step["action_type"] == 'click':
                bbox = [step["bbox"][0] / 160, step["bbox"][1] / 210, step["bbox"][2] / 160, step["bbox"][3] / 210]
                click_point = [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]
                click_point = [round(item, 3) for item in click_point]
                click_point = [f"{item:.2f}" for item in click_point]
                click_point = "({},{})".format(click_point[0], click_point[1])
                action = "{{\"action_type\": {}, \"click_point\": {}}}".format(4, click_point)
            elif step["action_type"] == 'type':
                action = "{{\"action_type\": {}, \"typed_text\": \"{}\"}}".format(3, step["typed_text"])

            previous_actions.append(str(action))

            prompt = prompt_origin.format(goal, previous_step)

            conversations = []
            conv_user = {"from": "user", "value": "Picture 1: <img>{}</img>\n".format(img_path)}
            conv_user["value"] += prompt
            conv_ai = {"from": "assistant", "value": str(action)}
            conversations.append(conv_user)
            conversations.append(conv_ai)

            train_step.append({"id": "miniwob_{}".format(step_i), "conversations": conversations})
            step_i += 1

random.shuffle(train_step)
print("Num of total step: " + str(len(train_step)))
json.dump(train_step, open("../data/miniwob_train_sft.json", "w"))




