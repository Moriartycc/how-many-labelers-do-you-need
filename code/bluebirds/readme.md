This directory contains code and source files for reproducing the numerical experiments in Section 5.1 of the paper:



**How many labelers do you have? A closer look at gold-standard labels**



Authors: Chen Cheng^1, Hilal Asi^2 and John Duchi^3



Affiliations:



1. University of Chicago, Department of Statistics



2. Apple Inc.



3. Stanford University, Departments of Statistics and Electrical Engineering



### Description

The experiments use data from the [BlueBirds dataset](https://github.com/eaplatanios/noisy-labels), a small dataset of 108 images. The task is to classify each image as either an Indigo Bunting or a Blue Grosbeak (two similar-looking blue bird species). For each image, the dataset contains 39 labels obtained from Amazon Mechanical Turk workers. We use an ImageNet-pretrained ResNet50 model to generate image features and apply PCA to reduce their dimensionality from 2048 to 25.



### Source files

* `./original.tsv`: The 108-by-39 matrix of individual labels from the BlueBirds dataset.
* `./resnet-features.txt`: The 25-dimensional PCA-reduced image features derived from the last-layer output of an ImageNet-pretrained ResNet50 model.
* `./main.m`: MATLAB source file reproducing the figures in the paper.
* `./resnet-calib-bluebirds.pdf` and `./resnet-class-bluebirds.pdf`: outputs of the experiments in Fig 1(a) and Fig 1(b) of the paper.



### Instructions and Configurations

* Run `./main.m` in MATLAB R2022b or later. The code requires MOSEK with CVX; see the [CVX installation instructions for MOSEK](https://cvxr.com/cvx/doc/mosek.html).









