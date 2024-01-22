# SeeClick: Harnessing GUI Grounding for Advanced Visual GUI Agents

The model, data, and code for the paper: "SeeClick: Harnessing GUI Grounding for Advanced Visual GUI Agents"

Release Plans:

- [x] GUI grounding benchmark: *ScreenSpot*
- [ ] Data for the GUI grounding Pre-training of SeeClick
- [ ] Inference code & model checkpoint


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
