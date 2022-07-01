# Iterate on the model

My lack of understanding about network/tensorflow/keras internal prevented me to make further progress.

I tried changing the learning rate and other optimization parameters but I cannot get any improvement.


So I started to ask me some questions, as:
- Is this performance is a limitation of the network architecture?
- Since we measure performance in Average Precision, why don't we optimize this metric directly?

For the second question it became clear (after a google search) that we did not use AP in the optimization because it is not a differentiable function and neither convex.

In this search found the paper [Towards Accurate One-Stage Object Detection with AP-Loss](https://arxiv.org/abs/1904.06373) thats also gave me some insight about the first question.

In their related works section they present the SSD architecture as solution that trade of accuracy to processing efficiency.

At that moment, I realized that it was time to try other architectures.

# YOLOv5

After another google search I found the YOLOv5 model.

I followed the video tutorial and in less than a morning I was detecting qrcodes with better AP then any model that I have trained before.  The end!

![YOLOv5 github repository and video tutorial.](/report/imgs/iterateonmodel_001.png "YOLOv5 github repository and video tutorial.")

- [YOLOv5 + Roboflow Custom Training Tutorial](https://www.youtube.com/watch?v=x0ThXHbtqCQ)
- [YOLOv5 github repository](https://github.com/ultralytics/yolov5)

The tutorial make use of Roboflow app to store the dataset. I made the initial test with a different train/val/test split defined by roboflow. For this reason this is not directly comparable with previous result. 

Besides that the tutorial utilizes YOLOv5 small version and I was curious to evaluate other architecture sizes.

![YOLOv5 available sizes.](/report/imgs/iterateonmodel_002.png "YOLOv5 available sizes.")

Pause to organize the dataset in roboflow and the evaluation code of the different architecture sizes.

# Roboflow dataset management tool

The tutorial uses Roboflow to generate annotations in YOLOv5's own format. 

[Roboflow](https://roboflow.com) started being as being a dataset management tool. Today, the platform can also train and deploy ML models. For this task I used only the dataset annotation format conversion functionality.

- [Notebook with recipes for loading data to roboflow](https://github.com/diogoalves/qrcode-detection-experiments/tree/main/util/manipulate_annotations/util_generate_roboflow_annotations.ipynb)

The following dataset variations were created:
| Dataset                               	| Description                                                                            	|
|---------------------------------------	|----------------------------------------------------------------------------------------	|
| baseline-twoclasses-416               	| Dataset v2, with both qrcode and fips annotations. This size was used during tutorial. 	|
| baseline-twoclasses-640               	| Same as baseline-twoclasses-416, but with 640x640px images.                            	|
| baseline-twoclasses-640-more168images 	| Dataset v3, with both qrcode and fips annotations. With 640x640px images.              	|
| baseline-oneclass-416                 	| Same as baseline-twoclasses-416, but with only qrcode annotations.                     	|
| baseline-oneclass-640                 	| Same as baseline-twoclasses-640, but with only qrcode annotations.                     	|
| baseline-oneclass-640-more168images   	| Same as baseline-twoclasses-640-more168images, but with only qrcode annotations.       	|


Now all of them maintains the same train/val/test split from baseline project. 

When generating a dataset version, Roboflow can create more images using data augmentation. This was not used during the tutorial because seems that YOLOv5 model has already some buitin kind of data augumentation.

![Roboflow qrcode dataset panel.](/report/imgs/iterateonmodel_003.png "Roboflow qrcode dataset panel.")

# Results

Architecture size initial exploration.

Tried 100 epochs on each architecture size x dataset.

![YOLOv5 qrcode detection results.](/report/imgs/iterateonmodel_004.png "YOLOv5 qrcode detection results.")

### Some points
- Running the YOLOv5 experiments was considerably faster than the baseline Tensorflow experiments. One reason is that the YOLOv5 training code uses all available GPUs on the machine.
- Larger architectures tend to perform better, but not always. Overfitting?
- Datasets with fips and qrcodes seem to work better than ones with just qrcodes.
- The dataset with more images seems to improve the result on the simpler network.


After that, I tried using pretrained weights and fine-tuning with our dataset for more 300 epochs. At this point we achieved **AP@.5 95.3% in the test set**.

Ok, we got a meaningful improvement. First from 75.75% to 87.72%.
Now we achieved 95.3%

Probably it still possible to improve it a little, but for now I will stop here.