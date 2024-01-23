# SeeClick: Harnessing GUI Grounding for Advanced Visual GUI Agents

The model, data, and code for the paper: [SeeClick: Harnessing GUI Grounding for Advanced Visual GUI Agents](https://arxiv.org/abs/2401.10935)

Release Plans:

- [x] GUI grounding benchmark: *ScreenSpot*
- [x] Data for the GUI grounding Pre-training of SeeClick
- [ ] Inference code & model checkpoint
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
