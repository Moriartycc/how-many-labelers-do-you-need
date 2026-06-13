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

* `./original.tsv`: Individual 108\*39 labels.
* `./resnet-features.txt`: Last layer output of the pretrained ResNet50 model. The features were obtained by passing data in `./original.tsv` into the ResNet 50 model with weights obtained from (https://github.com/huyvnphan/PyTorch\_CIFAR10).
* `./main.m`: MATLAB source file to reproducing the figures in the paper.
* `./resnet-calib-bluebirds.pdf` and `./resnet-class-bluebirds.pdf`: outputs of the experiments in Fig 1(a) and Fig 1(b) of the paper.



### Instructions and Configurations