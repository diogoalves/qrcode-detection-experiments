# More training


After not making much progress when adding more training images (dataset v3) I decided to look into the code try to train more the model.

Another indicator for this direction came from Average Precision progress graph on the training set. The curve seemed to indicate an improvement tendency. Despite the Cost function is minimizing well. The AP metric was similarly low ~77% across the training, validation, and test sets.

So, it was more or less clear that we should try training the model for more than the 5000 iterations used in baseline experiment. 


# Fixing number of steps per epoch

The baseline project established 5000 gradient iterations as it optimization horizon. Looking for the code I realized that the paramater **steps_per_epoch** is hardcoded to 50, which makes a epoch not pass throught all training samples.  Batch-size of 8 samples * 50 steps per epoch corresponds to 400 training samples per Epoch.

I fixed it making the steps_per_epoch paramater equal to the floor division between train_size and batch_size.

![Fixed number of steps per epoch.](/report/imgs/moretraining_001.png "Fixed number of steps per epoch.")


# More powerful machines
It was clear that will need more computer power to explore this avenue. Prof. Nina Hirata managed to give me access to a series of deep learning capable shared machines 🙏.

So I started to use them to run all experiments. In short, all of shared machines are better than my local setup.

![Shared deeplearning machines.](/report/imgs/moretraining_002.png "Shared deeplearning machines.")

# Looking at the code that calculates the metrics

I took the opportunity to look at the file with the codes used to calculate the metrics IOU, AP, ...

https://github.com/Leonardo-Blanger/subparts_ppn_keras/blob/parts_based_detector/metrics.py

Here are some observations about these implementations.

##  IoU

The IoU calculation expects the dataset to be correctly annotated. That is, xmax value should be greater than xmin and ymax value should greater the ymin.

![IoU metric function.](/report/imgs/moretraining_003.png "IoU metric function.")

I am not sure that Labelme annotation tool follow this restriction. Anyway, I updated the annotations conversion code recipe to make sure that the right values arrive to this function.

## AP

I was getting an error when try to calculate AP of training set. The code was not prepared for images that did not have any FIPs annotated.

There was an image in the training set in this situation.

![Image without any annotated FIP in training set.](/report/imgs/moretraining_004.png "Image without any annotated FIP  in training set.")

So, we updated the code to be able to handle this situation.

![Updated AP metric function.](/report/imgs/moretraining_005.png "Updated AP metric function.")

## Precision

The precision method was defined twice in the metrics file.

The definition that was operating was a code that calculated recall metric.
I deleted the wrong definition.
Anyway, I couldn't find any place that used this code in the rest of the project.


# Save, stop and Resume
I tried to implement the save, stop and resume functionality of the optimization process.
The idea was to be able to change the learning rate value during training.

It took me a while but I think it's working.

# Results

I left training 2 or 3 days in one of the shared machines. 1442 epochs.

![More training results.](/report/imgs/moretraining_006.png "More training results.")

Test AP@.5 was 87.72%. (87% already in 473 epochs).

Seems to be meaningful progress when comparing to the 75.75% from baseline.

Looks like the model is overfitting.
I tried changing the learning rate and other optimization parameters but I cannot get any improvement.

Also, my lack of understanding about network/tensorflow/keras prevented me to make further progress.

