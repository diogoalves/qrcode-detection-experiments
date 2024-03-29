# Running baseline code

I had no previous experience with tensorflow/keras.  The availability jupyter notebooks in the baseline project were crucial to begin the exploration. 

I started running some of them in Google Colab.

- https://github.com/Leonardo-Blanger/subparts_ppn_keras/blob/parts_based_detector/QR_Code_experiments/Subparts%20QR%20Codes%20and%20FIPs/subparts_ppn_qr_codes_fips_resnet50.ipynb
- https://github.com/Leonardo-Blanger/subparts_ppn_keras/blob/parts_based_detector/QR_Code_experiments/comparisons/Recall_and_False_Positives.ipynb

But soon I realized that the free GPU offering would be not enough for this task. So, I decided to try to run them on my machine.



# ResNet50 layer name inconsistency between Tensorflow/Keras versions

I decided to try to use a more recent version of tools. 
Instead of replicating the same Tensorflow 1.0 environment that I used in Colab. Created a conda environment with tensorflow-2.6.0. my machine. 

I changed some library import line buts at end I had an issue where the network didn´t compile. The problem was that RestNet50 customization pointed for some layers names that were not found after Tensorflow update. It seems to related with the issue  https://github.com/tensorflow/tensorflow/issues/36237.

I couldn't get layer name equivalence between versions anywhere.
To solve the problem I had to run the model.summary command in Colab Tensorflow 1.x environment and in my machine's Tensorflow 2.x environment.  After that realized that outputs had the same amount of lines. 
 
 ![ResNet50 networkk. Left Tensorflow , right Tensorflow 2.](/report/imgs/runningbaseline_001.png "Model summary of the same ResNet50 network in Tensorflow 1 and 2.")
 
 I assumed that only the layer names have changed. With this information, updated network code and compiled to tensorflow 2.🤞

- [Resnet50 model.sumary Tensorflow 1.x .](/report/assets/resnet50_old.txt)
- [Resnet50 model.sumary Tensorflow 2.x .](/report/assets/resnet50_new.txt)

# batch-size didn´t fit in my gpu

My GPU is a modest NVIDIA GeForce 1650 4GB. Even installing a recent CUDA environment I realized that it is far from a good setup for deeplearning. To run the code I had to reduce the batch-size from 8 to 2 samples.

# Baseline results confirmed

After running the baseline code I could confirm that the results scores were very similar to the published paper.




## Tools
- google colab
- miniconda
- vscode
- tensorflow, keras and cuda libraries