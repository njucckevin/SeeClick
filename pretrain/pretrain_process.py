# process data for pre-training
import json
from process_utils import is_english_simple, bbox_2_point, bbox_2_bbox
import task_prompts
from tqdm import tqdm
import os
import random

mobile_imgs = "/cpfs01/user/chengkanzhi/combined"
web_imgs = "/cpfs01/user/chengkanzhi/seeclick_web_imgs"
widgetcap_json = "/cpfs01/user/chengkanzhi/widget_captioning.json"
ricosca_json = "/cpfs01/user/chengkanzhi/ricosca.json"
screensum_json = "/cpfs01/user/chengkanzhi/screen_captioning.json"
web_json = "/cpfs01/user/chengkanzhi/seeclick_web.json"

# widget captioning & RICOSCA
widgetcap_train = json.load(open(widgetcap_json, "r"))
ricosca_train = json.load(open(ricosca_json, "r"))
mobile_text_2_point = []
mobile_text_2_bbox = []
mobile_data_loca = {"widgetcap": widgetcap_train, "ricosca": ricosca_train}
for data_name, data in mobile_data_loca.items():

    print("Processing " + str(data_name))
    for i, item in tqdm(enumerate(data)):
        img_filename = item["img_filename"]
        img_path = os.path.join(mobile_imgs, img_filename)

        goal = item["instruction"]
        click_point = bbox_2_point(item["bbox"])
        click_bbox = bbox_2_bbox(item["bbox"])

        # text_2_point
        conversations_point = []
        prompt = random.choice(task_prompts.loca_point_prompt).format(goal)
        conv_user = {"from": "user", "value": "Picture 1: <img>{}</img>\n".format(img_path)}
        conv_user["value"] += prompt
        conv_ai = {"from": "assistant", "value": click_point}
        conversations_point.append(conv_user)
        conversations_point.append(conv_ai)

        # text_2_bbox
        conversations_bbox = []
        prompt = random.choice(task_prompts.loca_bbox_prompt).format(goal)
        conv_user = {"from": "user", "value": "Picture 1: <img>{}</img>\n".format(img_path)}
        conv_user["value"] += prompt
        conv_ai = {"from": "assistant", "value": click_bbox}
        conversations_bbox.append(conv_user)
        conversations_bbox.append(conv_ai)

        mobile_text_2_point.append(
            {"id": "{}_loca_point_{}".format(data_name, i), "conversations": conversations_point})
        mobile_text_2_bbox.append({"id": "{}_loca_bbox_{}".format(data_name, i), "conversations": conversations_bbox})

print("Num of mobile_text_2_point: " + str(len(mobile_text_2_point)))
print("Num of mobile_text_2_bbox: " + str(len(mobile_text_2_bbox)))

# UI summarization
screensum_train = json.load(open(screensum_json, "r"))
mobile_screensum = []
print("Processing screensum")
i = 0
for i, item in tqdm(enumerate(screensum_train)):

    img_filename = item["img_filename"]
    img_path = os.path.join(mobile_imgs, img_filename)

    captions = item["captions"]
    random.shuffle(captions)
    for caption in captions[:3]:
        conversations = []
        prompt = random.choice(task_prompts.screen_caption_prompt)
        conv_user = {"from": "user", "value": "Picture 1: <img>{}</img>\n".format(img_path)}
        conv_user["value"] += prompt
        conv_ai = {"from": "assistant", "value": caption}
        conversations.append(conv_user)
        conversations.append(conv_ai)

        mobile_screensum.append(({"id": "screensum_{}".format(i), "conversations": conversations}))
        i += 1

print("Num of screensum: " + str(len(mobile_screensum)))

# widget captioning
widgetcap_train = json.load(open(widgetcap_json, "r"))
mobile_widgetcap = []
print("Processing widgetcap")
for i, item in tqdm(enumerate(widgetcap_train)):
    img_filename = item["img_filename"]
    img_path = os.path.join(mobile_imgs, img_filename)

    goal = item["instruction"]
    click_point = bbox_2_point(item["bbox"])

    conversations = []
    prompt = random.choice(task_prompts.widgetcap_prompt).format(click_point)
    conv_user = {"from": "user", "value": "Picture 1: <img>{}</img>\n".format(img_path)}
    conv_user["value"] += prompt
    conv_ai = {"from": "assistant", "value": goal}
    conversations.append(conv_user)
    conversations.append(conv_ai)

    mobile_widgetcap.append(({"id": "widgetcap_{}".format(i), "conversations": conversations}))

print("Num of widgetcap " + str(len(mobile_widgetcap)))

# web
web_train = json.load(open(web_json, "r"))
web_loca_point = []
web_loca_bbox = []
web_ocr_point = []
web_ocr_bbox = []
num_ele_valid = 0
print("Processing web")
for i, item in tqdm(enumerate(web_train)):

    img_filename = item["img_filename"]
    img_path = os.path.join(web_imgs, img_filename)

    eles_valid = []
    for ele in item["elements"]:
        if len([item for item in ele["bbox"] if item < 0]) != 0:
            continue
        if len(ele["instruction"]) > 60:
            continue
        if ('{' in ele["instruction"]) or ('}' in ele["instruction"]):
            continue
        if not is_english_simple(ele["instruction"]):
            continue
        eles_valid.append(ele)

    if len(eles_valid) == 0:
        continue
    num_ele_valid += len(eles_valid)

    # text_2_point
    random.shuffle(eles_valid)
    conversations = []
    prompt = random.choice(task_prompts.web_loca_all_point_prompt)
    prompt += ' '
    for j, item in enumerate(eles_valid):
        if j == 0:
            conv_user = {"from": "user", "value": "Picture 1: <img>{}</img>\n".format(img_path)}
            conv_user["value"] += prompt
            conv_user["value"] += item["instruction"]
        else:
            conv_user = {"from": "user", "value": ""}
            conv_user["value"] += item["instruction"]

        click_point = bbox_2_point(item["bbox"])
        conv_ai = {"from": "assistant", "value": click_point}
        conversations.append(conv_user)
        conversations.append(conv_ai)

    web_loca_point.append({"id": "loca_point_{}".format(i), "conversations": conversations})

    # text_2_bbox
    random.shuffle(eles_valid)
    conversations = []
    prompt = random.choice(task_prompts.web_loca_all_bbox_prompt)
    prompt += ' '
    for j, item in enumerate(eles_valid):
        if j == 0:
            conv_user = {"from": "user", "value": "Picture 1: <img>{}</img>\n".format(img_path)}
            conv_user["value"] += prompt
            conv_user["value"] += item["instruction"]
        else:
            conv_user = {"from": "user", "value": ""}
            conv_user["value"] += item["instruction"]

        click_point = bbox_2_bbox(item["bbox"])
        conv_ai = {"from": "assistant", "value": click_point}
        conversations.append(conv_user)
        conversations.append(conv_ai)
    web_loca_bbox.append({"id": "loca_bbox_{}".format(i), "conversations": conversations})

    # point_2_text
    random.shuffle(eles_valid)
    conversations = []
    prompt = random.choice(task_prompts.web_ocr_all_point_prompt)
    prompt += ' '
    for j, item in enumerate(eles_valid):
        click_point = bbox_2_point(item["bbox"])
        if j == 0:
            conv_user = {"from": "user", "value": "Picture 1: <img>{}</img>\n".format(img_path)}
            conv_user["value"] += prompt
            conv_user["value"] += click_point
        else:
            conv_user = {"from": "user", "value": ""}
            conv_user["value"] += click_point

        conv_ai = {"from": "assistant", "value": item["instruction"]}
        conversations.append(conv_user)
        conversations.append(conv_ai)

    web_ocr_point.append({"id": "ocr_point_{}".format(i), "conversations": conversations})

    # bbox_2_text
    random.shuffle(eles_valid)
    conversations = []
    prompt = random.choice(task_prompts.web_ocr_all_bbox_prompt)
    prompt += ' '
    for j, item in enumerate(eles_valid):
        click_point = bbox_2_bbox(item["bbox"])
        if j == 0:
            conv_user = {"from": "user", "value": "Picture 1: <img>{}</img>\n".format(img_path)}
            conv_user["value"] += prompt
            conv_user["value"] += click_point
        else:
            conv_user = {"from": "user", "value": ""}
            conv_user["value"] += click_point

        conv_ai = {"from": "assistant", "value": item["instruction"]}
        conversations.append(conv_user)
        conversations.append(conv_ai)

    web_ocr_bbox.append({"id": "ocr_bbox_{}".format(i), "conversations": conversations})

print("Num of valid elements: " + str(num_ele_valid))
print("Num of web_loca_point: " + str(len(web_loca_point)))
print("Num of web_loca_bbox: " + str(len(web_loca_bbox)))
print("Num of web_ocr_point: " + str(len(web_ocr_point)))
print("Num of web_ocr_bbox: " + str(len(web_ocr_bbox)))

# llava 150k
llava_data = []
with open("/cpfs01/user/chengkanzhi/mPLUG-Owl-main/data/llava_instruct_150k.jsonl", 'r') as f:
    for line in f:
        llava_data.append(json.loads(line))

llava_150k = []
for i, conversation in tqdm(enumerate(llava_data)):
    con_human = [item for item in conversation['conversations'] if item["from"] == 'human']
    con_gpt = [item for item in conversation['conversations'] if item["from"] == 'gpt']
    assert conversation['conversations'][0]['from'] == 'human'

    num_img = 0
    for item in conversation['conversations']:
        if '<image>' in item["value"]:
            num_img += 1
    assert num_img == 1

    img_filename = conversation['image']
    img_path = os.path.join('/cpfs01/user/chengkanzhi/coco/train2017', img_filename)

    conversations_new = []
    for j, item in enumerate(conversation['conversations']):
        if j == 0:
            assert '<image>\n' in item["value"] or '\n<image>' in item["value"]
        if item["from"] == "human":
            sentence = item["value"].replace("<image>\n", "").replace("\n<image>", "")
            if j == 0:
                conv_user = {"from": "user", "value": "Picture 1: <img>{}</img>\n".format(img_path)}
                conv_user["value"] += sentence
            else:
                conv_user = {"from": "user", "value": ""}
                conv_user["value"] += sentence
            conversations_new.append(conv_user)
        elif item["from"] == "gpt":
            sentence = item["value"].replace("<image>\n", "").replace("\n<image>", "")
            conv_ai = {"from": "assistant", "value": sentence}
            conversations_new.append(conv_ai)

    llava_150k.append({"id": "llava_{}".format(i), "conversations": conversations_new})

print("Num of llava: " + str(len(llava_150k)))

random.shuffle(mobile_text_2_point)
mobile_text_2_point = mobile_text_2_point[:]
random.shuffle(mobile_text_2_bbox)
mobile_text_2_bbox = mobile_text_2_bbox[:56000]
random.shuffle(mobile_screensum)
mobile_screensum = mobile_screensum[:]
random.shuffle(mobile_widgetcap)
mobile_widgetcap = mobile_widgetcap[:42000]
random.shuffle(web_loca_point)
web_loca_point = web_loca_point[:]
random.shuffle(web_loca_bbox)
web_loca_bbox = web_loca_bbox[:54000]
random.shuffle(web_ocr_point)
web_ocr_point = web_ocr_point[:54000]
random.shuffle(web_ocr_bbox)
web_ocr_bbox = web_ocr_bbox[:54000]
random.shuffle(llava_150k)
llava_150k = llava_150k[:]

sft_train = mobile_text_2_point + mobile_text_2_bbox + mobile_screensum + mobile_widgetcap + web_loca_point + web_loca_bbox + web_ocr_point + web_ocr_bbox + llava_150k
print("Num of sft: " + str(len(sft_train)))
json.dump(sft_train, open("../data/sft_train.json", "w"))

