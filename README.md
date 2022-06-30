# qrcode-detection-experiments

This repository is part of my journey into machine learning and deep learning fields. So, a small warning is that will not be a surprise if you find some errors and misconceptions here.
The task idea came from Prof. Dra. Nina S. T. Hirata during a small talk after her Intro to Machine Learning class.

## Objective:
Try to improve the performance Qr Code detection task in natural scenes when comparing against a baseline project.

The baseline project (from 2019) achieved great improvements (mAP@0.5 77%) when compared with previous techniques. But as the deep learning field innovation pace is frenetic, we suspect that today we could get an even better result.


## Baseline:
The baseline is presented in the paper: An Evaluation of Deep Learning Techniques for Qr Code Detection (Leonardo Blanger and Nina S. T. Hirata). It evaluated variations of the popular Single Shot Detector deep learning architecture and proposed an architecture modification that allowed the detection aided by object subparts annotations. The solution achieved substantial improvements when compared with a non-deep learning technique that used Viola-Jones framework. Beyond that, it makes publicly a Dataset of 767 images of natural scenes with QR Codes and Find Patterns (FIPs) bounding box annotations.

**Result summary:**

![Table 1. Experimental results on the test set, in Average Precision.](/report/imgs/readme_001.png "Table 1. Experimental results on the test set, in Average Precision.")

Of evaluated architectures the ResNet5 Subparts achieved the best mAP@0.5 (77%).

Evaluating it against the FastQR work that used Viola-Jones framework.

![Table 2. Comparison against FastQR.](/report/imgs/readme_002.png "Table 2 Comparison against FastQR.")


> L. Blanger and N. S. T. Hirata, "An Evaluation of Deep Learning Techniques for Qr Code Detection," 2019 IEEE International Conference on Image Processing (ICIP), 2019, pp. 1625-1629, doi: 10.1109/ICIP.2019.8803075.

Code: https://github.com/leonardo-Blanger/subparts_ppn_keras

Dataset: https://github.com/ImageU/QR_codes_dataset


## Exploration

- [Idea 1 - Running baseline code.](/report/01-runningbaseline.md)
- [Idea 2 – Error analysis.](/report/02-erroranalysis.md)
- [Idea 3 – Iterate on the data.](/report/03-iterateondata.md)
- [Idea 4 – Train more.](/report/04-moretraining.md)
- [Idea 5 – Iterate on the model.](/report/05-iterateonmodel.md)


## Comparative results



