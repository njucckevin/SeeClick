# GUI Grounding Pre-training Data for SeeClick

***
Tips: In GUI grounding data, the position of the target element is recorded in the `bbox` key, represented by `[left, top, right, bottom]`. 
Each value is a [0, 1] decimal number indicating the ratio of the corresponding position to the width or height of the image.
***
### Mobile data

The images for mobile data are part of the RICO dataset [1], 
which can be downloaded from [here](http://www.interactionmining.org/). 
Alternatively, we provide a packaged [zip file](https://box.nju.edu.cn/f/7ae5e9bd4bf840d4add3/).

#### Widget Captioning
Widget Captioning data are collected by [2]. 
The part used for SeeClick training can be downloaded in [here](https://box.nju.edu.cn/f/4019422e045b480f8945/).

Each sample contain:
* `img_filename`: the interface screenshot file
* `instruction`: human instruction
* `bbox`: the bounding box of the target element corresponding to instruction

#### RICOSCA
RICOSCA is a dataset automatically labeled using Android VH in [3].
The part used for SeeClick training can be downloaded in [here](https://box.nju.edu.cn/f/1b54f3b4bf864775b78c/).

Each sample contain:
* `img_filename`: the interface screenshot file
* `instruction`: automatically labeled instruction
* `bbox`: the bounding box of the target element corresponding to instruction

#### Screen Summarization
Screen Summarization data are collected by [4].
The part used for SeeClick training can be downloaded in [here](https://box.nju.edu.cn/f/6bcf4c17ec1b49d2806b/).

Each sample contain:
* `img_filename`: the interface screenshot file
* `captions`: a list of captions for the screenshot

***
### Web data
The web data used by SeeClick for training was crawled from websites provided by Common Crawl, containing more than 270k webpage screenshots and over 3 million webpage elements.
The crawled web screenshots is in [here](https://box.nju.edu.cn/f/6a804cf190dd490a808f/) (include 270k webpage screenshots, 130G), for convenience we also provide a [subset]((https://box.nju.edu.cn/f/813897fc4edc440a9e12/)) of 10,000 images. The annotation elements and text are available at [here](https://box.nju.edu.cn/f/3b0f6ccb8bed476c8e39/).

Each sample contain:
* `img_filename`: the interface screenshot file
* `url`: the url of the webpage
* `elements`: the target elements in the webpage
  * `instruction`: automatically crawled text/instruction for the element
  * `bbox`: the bounding box of the target element
  * `data_type`: "text"/"hover", the two types of element collected by SeeClick

***
### General data
We use [LLaVA-Instruct-150K](https://llava-vl.github.io/) as general data for training SeeClick.

***
[1] [Rico: A mobile app dataset for building data-driven design applications](https://dl.acm.org/doi/pdf/10.1145/3126594.3126651)

[2] [Widget Captioning: Generating Natural Language Description for Mobile User Interface Elements](https://arxiv.org/pdf/2010.04295.pdf)

[3] [Mapping Natural Language Instructions to Mobile UI Action Sequences](https://arxiv.org/pdf/2005.03776)

[4] [Screen2Words: Automatic Mobile UI Summarization with Multimodal Learning](https://dl.acm.org/doi/pdf/10.1145/3472749.3474765)