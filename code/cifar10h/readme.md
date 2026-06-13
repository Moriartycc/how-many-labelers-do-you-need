This directory contains code and source files for reproducing numerical experiments in Appendix A for the paper:

**How many labelers do you have? A closer look at gold-standard labels**

Authors: Chen Cheng^1, Hilal Asi^2 and John Duchi^3

Affiliations:

1. University of Chicago, Department of Statistics

2. Apple Inc.

3. Stanford University, Departments of Statistics and Electrical Engineering

### Description

In this experiment, we consider Peterson et al.’s CIFAR-10H dataset (https://github.com/jcpeterson/cifar-10h), which consists of 10,000 images from CIFAR-10 test set with soft labeling in that for each image, we have approximately 50 labels from different annotators. Each 32 × 32 image in the dataset belongs to one of the ten classes airplane, automobile, bird, cat, dog, frog, horse, ship, or truck; labelers assign each image to one of the classes.

### Source files

* `./cifar10h-counts.npy` and `./cifar10h-probs.npy`: The original label counts for the CIFAR-10H dataset and the empirical probabilities from the counts, which were obtained from (https://github.com/jcpeterson/cifar-10h).
* `./outputs.mat`, `./pred.mat` and `./true.mat`: Intermediate weight files from the pretrained CIFAR10 outputs.
* `./resnet-calib.fig`, `./resnet-class.fig`, `./resnet-calib.pdf` and `./resnet-calib.pdf`: Outputs of the MATLAB code.
* `./resnet-calib-new.fig`, `./resnet-class-new.fig`, `./resnet-calib-new.pdf` and `./resnet-calib-new.pdf`: Modified displaying style files of the above outputs with the same data, which are the final versions appeared in the paper.
* `./main.py`: Python code applying the pretrained CIFAR10 models (https://github.com/huyvnphan/PyTorch\_CIFAR10) to generate intermediate last layer weights and features.
* `./main.m`: MATLAB source file reproducing the figures in the paper.



### Instructions and Configurations

* Run `./main.py` in Python 3.11 with the necessary packages installed in the code. The code requires external pretrained CIFAR10 models from (https://github.com/huyvnphan/PyTorch\_CIFAR10).
* Run `./main.m` in MATLAB 2022b or later.