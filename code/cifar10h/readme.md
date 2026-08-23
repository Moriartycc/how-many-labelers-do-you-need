This directory contains code and source files for reproducing the numerical experiments in Appendix A of the paper:

**How many labelers do you have? A closer look at gold-standard labels**

Authors: Chen Cheng^1, Hilal Asi^2 and John Duchi^3

Affiliations:

1. University of Illinois Urbana-Champaign, Department of Statistics

2. Apple Inc.

3. Stanford University, Departments of Statistics and Electrical Engineering

### Description

In this experiment, we consider Peterson et al.'s [CIFAR-10H dataset](https://github.com/jcpeterson/cifar-10h), which consists of 10,000 images from the CIFAR-10 test set with approximately 50 labels from different annotators for each image. Each 32-by-32 image belongs to one of ten classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, or truck.

### Source files

* `./cifar10h-counts.npy` and `./cifar10h-probs.npy`: The original CIFAR-10H label counts and their empirical probabilities, obtained from the [CIFAR-10H repository](https://github.com/jcpeterson/cifar-10h).
* `./outputs.mat`, `./pred.mat` and `./true.mat`: Included intermediate features, model predictions, label counts, and reference labels consumed by `main.m`.
* `./resnet-calib.fig`, `./resnet-class.fig`, `./resnet-calib.pdf` and `./resnet-class.pdf`: Outputs of the MATLAB code.
* `./resnet-calib-new.fig`, `./resnet-class-new.fig`, `./resnet-calib-new.pdf` and `./resnet-class-new.pdf`: Restyled versions of the same outputs; these are the final versions appearing in the paper.
* `./main.py`: Python code applying the pretrained VGG19 model from [PyTorch_CIFAR10](https://github.com/huyvnphan/PyTorch_CIFAR10) to generate the three intermediate `.mat` files.
* `./main.m`: MATLAB source file generating the classification and calibration plots from the intermediate `.mat` files.



### Instructions and Configurations

* Run `./main.py` in Python 3.11 after installing the required packages. The code uses external [pretrained CIFAR-10 models](https://github.com/huyvnphan/PyTorch_CIFAR10).
* Run `./main.m` in MATLAB R2022b or later.

### Reproduction workflow

Run both stages from `code/cifar10h/`.

1. Configure the Python environment, `PYTHONPATH`, and pretrained weights as described in the root README.
2. Run `python main.py`. The script downloads the CIFAR-10 test images to `./data/` if necessary, reads `cifar10h-counts.npy` and `cifar10h-probs.npy`, applies the pretrained VGG19 model, and writes `outputs.mat`, `pred.mat`, and `true.mat`.
3. Start MATLAB R2022b with CVX 2.2, MOSEK 9.3.22, and a valid MOSEK license. Set the MATLAB current folder to `code/cifar10h/` and run `main.m`.
4. The MATLAB stage reads the three `.mat` files and writes `resnet-class.pdf` and `resnet-calib.pdf`. The included `resnet-class-new.pdf` and `resnet-calib-new.pdf` are restyled versions of those outputs and are the panels used in Supplement Figure A.1.

To rerun only the MATLAB analysis, begin at step 3 using the three included `.mat` files. To inspect the exact paper panels without computation, use the included `*-new.pdf` files.

### Runtime, hardware, and randomness

The Python feature-extraction stage and MATLAB optimization stage each typically require minutes to tens of minutes, depending on the CPU and MOSEK installation. The current Python script runs the VGG19 calculations on the CPU. The MATLAB script sets `rng('default')`, and the Python data loader does not shuffle, so runs from the same inputs and software environment are intended to be deterministic.
