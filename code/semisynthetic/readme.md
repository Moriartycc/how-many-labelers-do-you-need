This directory contains code and source files for reproducing numerical experiments in Sec 5.2 for the paper:

**How many labelers do you have? A closer look at gold-standard labels**

Authors: Chen Cheng^1, Hilal Asi^2 and John Duchi^3

Affiliations:

1. University of Chicago, Department of Statistics

2. Apple Inc.

3. Stanford University, Departments of Statistics and Electrical Engineering

### Description

We adapt the original CIFAR-10 dataset, which consists of 6000 32*32 images from each of k=10 classes (60,000 total images). To mimic collecting messy data, rather than the single-label in the base CIFAR-10 data, we construct pseudo-labelers using a pretrained eighteen layer residual network (ResNet18). We fit majority vote and MLE based approaches investigated in our paper. We also investigate the standard Dawid-Skene (DS) and GLAD crowdsourcing methods.

### Source files

* `./raw_labels.py` and `./aggregation_labels.py`: Python code generating pseudo-labels and aggregating labels using majority vote and crowdsourcing approaches.
* `./main.py`: Python code generating test errors from the above pseudo-labelers in the file `./all_output.csv`.
* `./plot.py`: Python code reproducing meshed test error curves in Fig 2(a) and 2(b) from the output of `./main.py`.
* `./plot-repeats.py`: Python code reproducing test error curves under multiple trials in Fig 2(c) and 2(d) from the output of `.main.py`.
* `./all_output_varying_median.csv`: Output file from `./main.py` for reproducing Fig 2(a) and 2(b).
* `./all_output_105_20repeats.csv`: Output file from `./main.py` for reproducing Fig 2(c).
* `./all_output_42_20repeats.csv`: Output file from `./main.py` for reproducing Fig 2(d).
* `./hard-test-error.pdf`, `./soft-test-error.pdf`, `./105.pdf` and `./42.pdf`: Figure 2(a)-2(d) in the paper.



### Instructions and Configurations

* Run `./main.py` in Python 3.11 with the necessary packages installed in the code. The code requires external pretrained CIFAR10 models from (https://github.com/huyvnphan/PyTorch\_CIFAR10) and crowdsourcing toolkit from (https://github.com/Toloka/crowd-kit).
* Run `./plot.py` to reproduce Fig 2(a) and 2(b).
* Run `./plot-repeats.py` to reproduce Fig 2(c) and 2(d) choosing `./all_output_105_20repeats.csv` or `./all_output_42_20repeats.csv` as the input.