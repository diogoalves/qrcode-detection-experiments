# Error analysis

Following the recommendations learned from Structuring Machine Learning Projects course, decided to start trying to see current detection results to get some idea for next steps.

The idea is simple
1. Run model detection in validation dataset.
2. Select images that the model did not perform well.
3. Try to categorize it and add comments if necessary.

At the end, with luck, you will get a list of possible situations where the model is not performing well. And maybe get a clear indication of where to improve the model.

To generate the validation image results it was necessary replicate an already present code that was used to verify the training set.  This code generates images bounding box of qrcodes and FIPs. After that, the information about the Average Precision score was added.

![Validation images. Model detected on the left and the ground truth on the right.](/report/imgs/erroranalysis_001.png "Validation images. Model detected on the left and the ground truth on the right.")

At the end, the error analysis resulting table.

![Error analysis resulting table.](/report/imgs/erroranalysis_002.png "Error analysis resulting table.")

### Some points

1. The detector is having problems with small qrcodes. 
    - Is it a dataset low representation problem?
    - Is it a network architecture limitation? Current network input was 480x480px.


2. It seems that the ground truth annotations have some inconsistencies. Ex: Missing QRcode annotations, bounding box sizes.



# Small QRcodes

Returning to the error analysis table and adding an extra column that represented QRcode bounding box area after scaling it to network input dimensions (480x480px). With it, it was possible to characterize the Small QRcode as images with scaled area inferior to 450, about 22x22px of scaled qrcode bounding box.

![Scaled qrcoded area.](/report/imgs/erroranalysis_003.png "Scaled qrcoded area.")

## Does the training dataset have enough small images?
The training set has 962 qrcodes located in 567 images.
From them, 115 qrcodes are under this limit in only 38 images.


# Annotations inconsistencies

It seems that the ground truth annotations have some inconsistencies.

### Missing qrocode annotations

![Missing qrocode annotations 1.](/report/imgs/erroranalysis_004.png "Missing qrocode annotations 1.")
![Missing qrocode annotations 2.](/report/imgs/erroranalysis_005.png "Missing qrocode annotations 2.")

### Loose bounding boxes
![Loose bounding boxes.](/report/imgs/erroranalysis_006.png "Loose bounding boxes")


# Next steps

- Should I correct dataset inconsistencies?
- Should I add more small qrcodes in training set? How many?


# References and tools

- [Structuring Machine Learning Projects course](https://www.coursera.org/learn/machine-learning-projects/)


- [Specific video on Carrying Out Error Analysis](https://www.youtube.com/watch?v=JoAxZsdw_3w)