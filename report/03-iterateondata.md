# Iterate on data

After listening to Andrew Ng talk a bit about the importance of data in modern machine learning workflows during their machine learning and deep learning course.

I started to think that it would be possible to improve baseline model performance by maintaining the same network architecture and improving only the data.

Andrew´s presentation about the Data-centric AI Movement at the AI Developer Conference GTC March 2022 NVIDIA gave me more concrete ideas about what makes a good dataset.

> What makes a good dataset?
>-	Consistent and precises annotations.
>-	High quality and representative features.
>-	Dataset refers to the real problem that is being addressed.

Also, the presentation of *Finding millions of label errors with Cleanlab*, make me think that this kind of errors are very common and offered to me a good point to decide to try fixing the baseline dataset annotations.


# Consistent and precises annotations

Despite being a simple task to explain, the bounding box annotation task can be performed in different ways for the same image. Both can be correct, but for the model to work in the best way it is necessary to have consistency between them.

This section tries, through examples from the dataset itself, to promote consistency and precision in the dataset annotations.

1. Bounding Boxes are used to mark the location of QRCODES and FIPs in the images.
2. Bounding boxes are defined by 4 (four) integers: xmin, ymin, xmax, ymax.
3. A bounding box must be used for each object individually. The bounding box must not group multiple objects.

| ✓ OK 	| 🗙 NOT OK	|
|------	|------ 	|
| ![Example of individual bounding box.](/report/imgs/iterateondata_001.png "Example of individual bounding box.") 	| ![Example of bounding box grouping multiple objects.](/report/imgs/iterateondata_002.png "Example of bounding box grouping multiple objects.") 	|

4. Bounding boxes must be accurate. It shouldn't leave any unnecessary bords, it should be tight enough to contain the entire object without including unnecessary bords. The edges of the bounding box must touch the outermost pixels of the object we are annotating.

| ✓ OK 	| 🗙 NOT OK	|
|------	|------ 	|
| ![Example tight bounding box.](/report/imgs/iterateondata_003.png "Example tight bounding box.") 	| ![Example of loose bounding box.](/report/imgs/iterateondata_004.png "Example of loose bounding box.") 	|

> Unnecessary bords can lead to poor performance in the IoU metric. (https://pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/)

5. All objects, without exception, must be annoted. Small, low resolution objects should be annotated. Even knowing that the model will have difficulty detecting objects smaller than 10x10px (~1.5% of dimensions) they must be identified.

| ✓ OK 	| 🗙 NOT OK	|
|------	|------ 	|
| ![Example a small, low res annotated qrcode bounding box.](/report/imgs/iterateondata_005.png "Example a small, low res annotated qrcode bounding box.") 	| ![Example of a missing qrcode annotation.](/report/imgs/iterateondata_006.png "Example of a missing qrcode annotation.") 	|
| ![Example of multiple small qrcodes annotated.](/report/imgs/iterateondata_007.png "Example of multiple small qrcodes annotated.") 	| ![Example of a missing qrcode annotation 2.](/report/imgs/iterateondata_008.png "Example of a missing qrcode annotation 2.") 	|

6. Overlaps between bounding boxes of the same class should be avoided. As bounding box detectors are trained to consider the IoU, bouding box overlaps should be avoided.

7. Images must be at least 480 x 480 px which is the size of the network input.

# Dataset v2 - Fix annotations inconsistencies
Based on the definition of the previous section a new version of images annotations was produced.

## Changelog
- Adds missing bounding boxes on 9 images. 
    - train: 6, valid: 2, test: 1
    - Images: ['2211459923', '2236525717', '2236526151', '2237317748', '4441975297', '4443372421', '4884107014', '5030691999', '5266587566']. 
    - This added 20 more QR Codes.
    
- Adjusts some bounding box of 26 images.
    - train: 17, valid: 2, test: 7
    - Images: ['0193927093', '1134131119', '1213992780', '1476633170', '2124747258', '2126492335', '2332977452', '2423843117', '2425729926', '2468137356', '2880662710', '2943752525', '2961178381', '2985146218', '2986368381', '4059282918', '4352836095', '4353596623', '4353977850', '4354342598', '4413934670', '4413992286', '4567642653', '4597824773', '4957719411', '5121473172'].
    - This did not change the number of QR Codes.

# Dataset v3 - more images
Adds more images with the objective of improving model score.

All new images were shotted from 29/05/2022 to 04/06/2022 during walking tours in the city of São Paulo, Brazil.

All images were shooted using a Xiaomi Mi 9 SE phone with 3000x4000px (12MP) resolution.

Added 168 images of natural scenes. Many of them with small qrcodes.


# Results
The raw data is available in:

- Baseline: https://github.com/diogoalves/qrcode-detection-experiments/tree/main/training_metrics/baseline
- Baseline-dataset-v2: https://github.com/diogoalves/qrcode-detection-experiments/tree/main/training_metrics/baseline-dataset-v2
- Baseline-dataset-v3: https://github.com/diogoalves/qrcode-detection-experiments/tree/main/training_metrics/baseline-dataset-v3

Power BI desktop was used to plot optimization history.

The baseline run achieved qrcode was  AP@.5 75.75%.
![Baseline results.](/report/imgs/iterateondata_009.png "Baseline results.")



Running the same baseline code on dataset v2 got a small improvement. 
The qrcode AP@.5 came from 75.75% to 77.82%.

![Dataset v2 results.](/report/imgs/iterateondata_010.png "Dataset v2 results.")

But, running the same baseline code on dataset v3 apparently made no improvement. 😖

![Dataset v3 results.](/report/imgs/iterateondata_011.png "Dataset v3 results.")


# DVC toolset
DVC library was used to help track changes to the dataset.

> Data Version Control is a data and ML experiment management tool that takes advantage of the existing engineering toolset that you're already familiar with (Git, your IDE, CI/CD, etc).

In addition, the DVCLive library was used to help log optimization results.

> DVCLive is a Python library for logging machine learning metrics and other metadata in simple file formats, which is fully compatible with DVC. 

![DVC matches the right versions of data, code, and models for you 💘.](/report/imgs/iterateondata_012.png "DVC matches the right versions of data, code, and models for you 💘.")



# Label me

The Label me tool was used to help in the process of fixing Dataset v2 annotations and creating of Dataset v3 annotations.

From github, Label me is an ...
>  ImagePolygonal Annotation with Python (polygon, rectangle, circle, line, point and image-level flag annotation). 

![Qrcode and fips annotations with Label me.](/report/imgs/iterateondata_013.png "Qrcode and fips annotations with Label me.")

In addition, a notebook was created to help convert annotations from CSV model format to labelme json format. And also converting back to CSV model format.

![Annotation format conversion.](/report/imgs/iterateondata_014.png "Annotation format conversion.")



# References and tools
- [The Data-centric AI Movement #1 AI Developer Conference GTC March 2022 NVIDIA](https://www.youtube.com/watch?v=X7dcSQz0R24)
- [Finding millions of label errors with Cleanlab](https://datacentricai.org/blog/finding-millions-of-label-errors-with-cleanlab/)
- [Power BI dashboard with results](/report/assets/qrcodes.pbix)

- [DVC - Version Control System
for Machine Learning Projects](https://dvc.org/)
- [DVCLive - Log history helper](https://dvc.org/doc/dvclive)

- [Label me - data annotation tool](https://github.com/wkentaro/labelme)

- [Notebook with recipes for annotation format conversion](https://github.com/diogoalves/qrcode-detection-experiments/tree/main/util/manipulate_annotations/util_convert_csv2labelme2csv.ipynb)


