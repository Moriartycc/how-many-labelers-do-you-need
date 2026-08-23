This directory contains code and source files for reproducing the numerical experiments in Appendix A of the paper:

**How many labelers do you have? A closer look at gold-standard labels**

Authors: Chen Cheng^1, Hilal Asi^2 and John Duchi^3

Affiliations:

1. University of Chicago, Department of Statistics

2. Apple Inc.

3. Stanford University, Departments of Statistics and Electrical Engineering

### Description

In this experiment, we consider Peterson et al.'s [CIFAR-10H dataset](https://github.com/jcpeterson/cifar-10h), which consists of 10,000 images from the CIFAR-10 test set with approximately 50 labels from different annotators for each image. Each 32-by-32 image belongs to one of ten classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, or truck.

### Source files

* `./cifar10h-counts.npy` and `./cifar10h-probs.npy`: The original label counts for the CIFAR-10H dataset and the empirical probabilities from the counts, which were obtained from (https://github.com/jcpeterson/cifar-10h).
* `./outputs.mat`, `./pred.mat` and `./true.mat`: Intermediate weight files from the pretrained CIFAR10 outputs.
* `./resnet-calib.fig`, `./resnet-class.fig`, `./resnet-calib.pdf` and `./resnet-class.pdf`: Outputs of the MATLAB code.
* `./resnet-calib-new.fig`, `./resnet-class-new.fig`, `./resnet-calib-new.pdf` and `./resnet-class-new.pdf`: Restyled versions of the same outputs; these are the final versions appearing in the paper.
* `./main.py`: Python code applying the [pretrained CIFAR-10 models](https://github.com/huyvnphan/PyTorch_CIFAR10) to generate intermediate last-layer weights and features.
* `./main.m`: MATLAB source file reproducing the figures in the paper.



### Instructions and Configurations

* Run `./main.py` in Python 3.11 after installing the required packages. The code uses external [pretrained CIFAR-10 models](https://github.com/huyvnphan/PyTorch_CIFAR10).
* Run `./main.m` in MATLAB R2022b or later.
