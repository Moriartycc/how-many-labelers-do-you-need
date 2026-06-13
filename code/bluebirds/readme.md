This directory contains code and source files for reproducing numerical experiments in Sec 5.1 for the paper:



**How many labelers do you have? A closer look at gold-standard labels**



Authors: Chen Cheng^1, Hilal Asi^2 and John Duchi^3



Affiliations:



1. University of Chicago, Department of Statistics



2. Apple Inc.



3. Stanford University, Departments of Statistics and Electrical Engineering



### Description

The experiments use original data from the BlueBirds dataset (https://github.com/eaplatanios/noisy-labels), a small dataset of 108 images with ResNet features. The task is to classify each image as one of Indigo Bunting or Blue Grosbeak (two similarlooking blue bird species). For each image, we have 39 labels, obtained through Amazon

Mechanical Turk workers. We use an ImageNet pretrained ResNet50 model to generate image features, applying PCA to reduce the dimensionality from 2048 to 25.



### Source files

* `./original.tsv`: Individual 108\*39 labels.
* `./resnet-features.txt`: Last layer output of the pretrained ResNet50 model. The features were obtained by passing data in `./original.tsv` into the ResNet 50 model with weights obtained from (https://github.com/huyvnphan/PyTorch\_CIFAR10).
* `./main.m`: MATLAB source file to reproducing the figures in the paper.
* `./resnet-calib-bluebirds.pdf` and `./resnet-class-bluebirds.pdf`: outputs of the experiments in Fig 1(a) and Fig 1(b) of the paper.



### Instructions and Configurations

* Run `./main.m` in MATLAB 2022b or later. To run the code the user is required to install MOSEK with CVX. See further insturctions in (https://cvxr.com/cvx/doc/mosek.html).









