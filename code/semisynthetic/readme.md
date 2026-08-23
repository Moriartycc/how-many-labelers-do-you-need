This directory contains code and source files for reproducing the numerical experiments in Section 5.2 of the paper:

**How many labelers do you have? A closer look at gold-standard labels**

Authors: Chen Cheng^1, Hilal Asi^2 and John Duchi^3

Affiliations:

1. University of Chicago, Department of Statistics

2. Apple Inc.

3. Stanford University, Departments of Statistics and Electrical Engineering

### Description

We adapt the original CIFAR-10 dataset, which consists of 6,000 32-by-32 images from each of 10 classes (60,000 images in total). To mimic the collection of noisy labels, we replace the single label in the base CIFAR-10 data with pseudo-labels generated using a pretrained 18-layer residual network (ResNet18). We fit the majority-vote and MLE-based approaches investigated in the paper. We also investigate the standard Dawid-Skene (DS) and GLAD crowdsourcing methods.

### Source files

* `./raw_labels.py` and `./aggregation_labels.py`: Python code generating pseudo-labels and aggregating labels using majority vote and crowdsourcing approaches.
* `./main.py`: Python code generating test errors from the above pseudo-labelers in the file `./all_output.csv`.
* `./plot.py`: Python code reproducing meshed test error curves in Fig 2(a) and 2(b) from the output of `./main.py`.
* `./plot-repeats.py`: Python code reproducing test error curves under multiple trials in Fig 2(c) and 2(d) from the output of `./main.py`.
* `./all_output_varying_median.csv`: Output file from `./main.py` for reproducing Fig 2(a) and 2(b).
* `./all_output_105_20repeats.csv`: Output file from `./main.py` for reproducing Fig 2(c).
* `./all_output_42_20repeats.csv`: Output file from `./main.py` for reproducing Fig 2(d).
* `./hard-test-error.pdf`, `./soft-test-error.pdf`, `./105.pdf` and `./42.pdf`: Figure 2(a)-2(d) in the paper.



### Instructions and Configurations

* Run `./main.py` in Python 3.11 after installing the required packages. The code uses external [pretrained CIFAR-10 models](https://github.com/huyvnphan/PyTorch_CIFAR10) and the [Crowd-Kit](https://github.com/Toloka/crowd-kit) crowdsourcing toolkit.
* Run `./plot.py` to reproduce Fig 2(a) and 2(b).
* Run `./plot-repeats.py` to reproduce Fig 2(c) and 2(d) choosing `./all_output_105_20repeats.csv` or `./all_output_42_20repeats.csv` as the input.
