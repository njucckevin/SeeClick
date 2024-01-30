# SeeClick: Harnessing GUI Grounding for Advanced Visual GUI Agents

The model, data, and code for the paper: [SeeClick: Harnessing GUI Grounding for Advanced Visual GUI Agents](https://arxiv.org/abs/2401.10935)

Release Plans:

- [x] GUI grounding benchmark: *ScreenSpot*
- [x] Data for the GUI grounding Pre-training of SeeClick
- [x] Inference code & model checkpoint
- [ ] Other code and resources

***
### GUI Grounding Benchmark: *ScreenSpot*

*ScreenSpot* is an evaluation benchmark for GUI grounding, comprising over 1200 instructions from iOS, Android, macOS, Windows and Web environments, along with annotated element types (Text or Icon/Widget). See details and more examples in our paper.

Download the images and annotations of [*ScreenSpot*](https://box.nju.edu.cn/d/5b8892c1901c4dbeb715/). 

Each test sample contain: 
* `img_filename`: the interface screenshot file
* `instruction`: human instruction
* `bbox`: the bounding box of the target element corresponding to instruction
* `data_type`: "icon"/"text", indicates the type of the target element
* `data_souce`: interface platform, including iOS, Android, macOS, Windows and Web (Gitlab, Shop, Forum and Tool)

![Examples of *ScreenSpot*](assets/screenspot.png)

#### Evaluation Results

![Results on *ScreenSpot*](assets/screenspot_result.png)

***
### GUI Grounding Pre-training Data for SeeClick
Check [data](readme_data.md) for the GUI grounding pre-training datasets,
including the first open source large-scale web GUI grounding corpus collected from Common Crawl.

***
### Inference code & model checkpoint
SeeClick is built on [Qwen-VL](https://github.com/QwenLM/Qwen-VL) and is compatible with its Transformers 🤗 inference code.

All you need is to input a few lines of codes as the examples below.

Before running, set up the environment and install the required packages.
```angular2html
pip install -r requirements.txt
```
Then,
```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-VL-Chat", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("SeeClick-ckpt-dir", device_map="cuda", trust_remote_code=True, bf16=True).eval()
model.generation_config = GenerationConfig.from_pretrained("Qwen/Qwen-VL-Chat", trust_remote_code=True)

img_path = "assets/test_img.png"
prompt = "In this UI screenshot, what is the position of the element corresponding to the command \"{}\" (with point)?"
# prompt = "In this UI screenshot, what is the position of the element corresponding to the command \"{}\" (with bbox)?"  # Use this prompt for generating bounding box
ref = "add an event"   # response (0.17,0.06)
ref = "switch to Year"   # response (0.59,0.06)
ref = "search for events"   # response (0.82,0.06)
query = tokenizer.from_list_format([
    {'image': img_path}, # Either a local path or an url
    {'text': prompt.format(ref)},
])
response, history = model.chat(tokenizer, query=query, history=None)
print(response)
```
The SeeClick's checkpoint can be downloaded on [huggingface](https://huggingface.co/cckevinn/SeeClick/tree/main).
Please replace the `SeeClick-ckpt-dir` with the actual checkpoint dir. 

The prediction output represents the point of `(x, y)` or the bounding box of `(left, top, right, down)`,
each value is a [0, 1] decimal number indicating the ratio of the corresponding position to the width or height of the image.
We recommend using point for prediction because SeeClick is mainly trained for predicting click points on GUIs.

Thanks to [Qwen-VL](https://github.com/QwenLM/Qwen-VL) for their powerful model and wonderful open-sourced work.
***
### Citation
```
@misc{cheng2024seeclick,
      title={SeeClick: Harnessing GUI Grounding for Advanced Visual GUI Agents}, 
      author={Kanzhi Cheng and Qiushi Sun and Yougang Chu and Fangzhi Xu and Yantao Li and Jianbing Zhang and Zhiyong Wu},
      year={2024},
      eprint={2401.10935},
      archivePrefix={arXiv},
      primaryClass={cs.HC}
}
```
