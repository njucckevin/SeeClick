# SeeClick for Downstream Agent Tasks

***
### Preparation
#### Download Screenshots and Annotations
* Mind2Web: Download the [screenshots](https://box.nju.edu.cn/f/33e203d170ab48b0b922/) and [annotations](https://box.nju.edu.cn/f/e30b861fa7604668821b/) (train set and test set of Domain/Website/Task). 
Note that according to [mind2web](https://github.com/OSU-NLP-Group/Mind2Web), please **DO NOT** redistribute the unzipped data files online.

* AITW

* MiniWob

#### Setup Environment
```
conda create --name env_name python=3.8
source activate env_name
pip install -r requirements_agent.txt
```

***
### Mind2Web
#### Prepare sft data for mind2web

Place the downloaded annotations in the data folder. Then process the mind2web training set to get the json file for sft LVLMs:
```
cd agent_tasks
python mind2web_process.py --imgs_dir mind2web_imgs
```
The `mind2web_imgs` should be replace by the actual dir of downloaded mind2web screenshots.

Uncomment `lines 84-87` to visualize the annotation episode of mind2web.

#### Fine-tune SeeClick
```
bash finetune/finetune_lora_ds.sh --save-name SeeClick_test --max-length 704 --micro-batch-size 4 --save-interval 500 
    --train-epochs 10 --nproc-per-node 2 --data-path xxxx/mind2web_train_sft.json --learning-rate 3e-5 
    --gradient-accumulation-steps 8 --qwen-ckpt xxxx/Qwen-VL-Chat --pretrain-ckpt xxxx/SeeClick-pretrain
    --save-path xxxx/checkpoint_qwen
```
* `data-path`: sft data generated in the above step
* `qwen-ckpt`: origin Qwen-VL ckpt path for loading tokenizer
* `pretrain-ckpt`: base model for fine-tuning, e.g. SeeClick-pretrain or Qwen-VL
* `save-path`: directory to save training checkpoints

The fine-tuning scripts are similar to Qwen-VL, except for we use LoRA to fine-tune customized parameters, as in `finetune/finetune.py lines 315-327`.
This scripts fine-tune pre-train LVLM with LoRA and multi-GPU training; for more option like full-finetuning, Q-LoRA and single-GPU training, please refer to [Qwen-VL](https://github.com/QwenLM/Qwen-VL/tree/master?tab=readme-ov-file#finetuning).

#### Evaluation on mind2web
After fine-tuning LVLM on the above sft data, the evaluation was performed on three subsets.

Alternatively, we provide the fine-tuned [checkpoint](https://huggingface.co/cckevinn/SeeClick-mind2web) of SeeClick for evaluation.
```
cd agent_tasks
python mind2web_test.py --model_path xxxx/SeeClick-mind2web --qwen_path xxxx/Qwen-VL-Chat --imgs_dir mind2web_imgs --task website
```
* `model_path`: the trained checkpoint of LVLM/SeeClick model
* `qwen_path`: the origin checkpoint of Qwen-VL-Chat, for loading the tokenizer and config
* `imgs_dir`: the actual dir of downloaded mind2web screenshots
* `task`: evaluation subset, one of `domain`, `website` and `task`
