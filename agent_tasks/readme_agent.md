# SeeClick for Downstream Agent Tasks

***
### Data Preparation
* Mind2Web: Download the [screenshots](https://box.nju.edu.cn/f/33e203d170ab48b0b922/) and [annotations](https://box.nju.edu.cn/f/e30b861fa7604668821b/) (train set and test set of Domain/Website/Task). 
Note that according to [mind2web](https://github.com/OSU-NLP-Group/Mind2Web), please **DO NOT** redistribute the unzipped data files online.

* AITW

* MiniWob

***
### Mind2Web
#### Prepare sft data for mind2web

Place the downloaded annotations in the data folder. Then process the mind2web training set to get the json file for sft LVLMs:
`python agent_tasks/mind2web_process.py --imgs_dir mind2web_imgs`, the `mind2web_imgs` should be replace by the actual dir of downloaded mind2web screenshots.

Uncomment `lines 84-87` to visualize the annotation episode of mind2web.

#### Evaluation on mind2web
After fine-tuning LVLM on the above sft data, the evaluation was performed on three subsets.
```
python agent_tasks/mind2web_test.py --model_path xxxx/checkpoint-1000 --qwen_path xxxx/Qwen-VL-Chat --imgs_dir mind2web_imgs --task website
```
* `model_path`: the trained checkpoint of LVLM/SeeClick model
* `qwen_path`: the origin checkpoint of Qwen-VL-Chat, for loading the tokenizer and config
* `imgs_dir`: the actual dir of downloaded mind2web screenshots
* `task`: evaluation subset, one of `domain`, `website` and `task`
