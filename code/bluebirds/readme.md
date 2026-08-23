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

* `./original.tsv`: Long-form data containing one row for each of 108 images and 39 labelers (4,212 rows total), with labeler ID, image ID, individual label, and reference label.
* `./resnet-features.txt`: Image ID followed by 2,048 last-layer ResNet50 features. The script applies PCA and retains 25 dimensions.
* `./main.m`: MATLAB source file reproducing the figures in the paper.
* `./resnet-class-bluebirds.pdf` and `./resnet-calib-bluebirds.pdf`: Figure 1(a) and Figure 1(b), respectively.



### Instructions and Configurations

* Run `./main.m` in MATLAB R2022b or later. The code requires MOSEK with CVX; see the [CVX installation instructions for MOSEK](https://cvxr.com/cvx/doc/mosek.html).

### Reproduction workflow

No external download is needed because both input files are included.

1. Start MATLAB R2022b and set the current folder to `code/bluebirds/`.
2. Confirm that CVX 2.2 can select MOSEK 9.3.22 and that a valid MOSEK license is available.
3. Run `main.m`.

The script reads `original.tsv` and `resnet-features.txt`, performs the repeated cross-validation experiment, and writes:

* `resnet-class-bluebirds.pdf` and `resnet-class-bluebirds.fig`;
* `resnet-calib-bluebirds.pdf` and `resnet-calib-bluebirds.fig`;
* `bluebirds.mat`, containing the plotted estimates and uncertainty bands.

### Runtime and randomness

The full script solves approximately 70,000 CVX problems: two fits for each of 35 labeler counts, 100 trials, and 10 folds. Runtime is many hours or longer on a CPU, depending on the MOSEK installation. The script uses MATLAB's `randperm` and `rand` without setting a seed, so a fresh full run will not be bit-for-bit identical. The two supplied PDF files are the paper outputs and can be inspected without rerunning the experiment.









