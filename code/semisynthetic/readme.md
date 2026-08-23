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
* `./plot.py`: Python code plotting the meshed test-error curves underlying Figure 2(a) and Figure 2(b) from `./all_output_varying_median.csv`.
* `./plot-repeats.py`: Python code plotting repeated-trial test-error curves for Figure 2(c) or Figure 2(d), selected by the `file_name` variable near the top of the script.
* `./all_output_varying_median.csv`: Output file from `./main.py` for reproducing Fig 2(a) and 2(b).
* `./all_output_105_20repeats.csv`: Output file from `./main.py` for reproducing Fig 2(c).
* `./all_output_42_20repeats.csv`: Output file from `./main.py` for reproducing Fig 2(d).
* `./hard-test-error.pdf`, `./soft-test-error.pdf`, `./105.pdf` and `./42.pdf`: Figure 2(a)-2(d) in the paper.



### Instructions and Configurations

* Run `./main.py` in Python 3.11 after installing the required packages. The code uses external [pretrained CIFAR-10 models](https://github.com/huyvnphan/PyTorch_CIFAR10) and the [Crowd-Kit](https://github.com/Toloka/crowd-kit) crowdsourcing toolkit.
* Run `./plot.py` to plot the curves underlying Figure 2(a) and Figure 2(b).
* Set `file_name` in `./plot-repeats.py` to `./all_output_105_20repeats.csv` or `./all_output_42_20repeats.csv`, then run the script for Figure 2(c) or Figure 2(d), respectively.

### Quick reproduction from supplied outputs

Run all commands from `code/semisynthetic/`. With the root Python environment and `PYTHONPATH` configured as described in the repository README:

```text
python plot.py
```

This reads `all_output_varying_median.csv`, opens the three-dimensional comparison plot, and writes `output_plot.png`. The separately styled paper panels are supplied as `hard-test-error.pdf` and `soft-test-error.pdf`.

For the repeated-trial panels, set the `file_name` variable near the top of `plot-repeats.py`, then run:

```text
python plot-repeats.py
```

Use `all_output_105_20repeats.csv` for Figure 2(c) and `all_output_42_20repeats.csv` for Figure 2(d). The script displays the selected plot; the final paper versions are supplied as `105.pdf` and `42.pdf`.

### Full simulation workflow

Before running the experiment driver, create the directory `net_parameters/resnet18_finetune/`. The scripts download the CIFAR-10 training and test data to `../data/` if those data are absent and load the pretrained ResNet18 weights described in the root README.

`main.py` controls the experiment through `M_list` and `prob_l_list`. It overwrites `all_output.csv` at the start of every run, so preserve or rename that file before changing configurations.

1. For the varying-accuracy data underlying Figure 2(a)--(b), use `M_list = [2, 4, 8, 16, 32]` and `prob_l_list = [0.105, 0.11, 0.12, 0.14, 0.18, 0.26, 0.42, 0.74]`. Run `python main.py` and preserve the result as `all_output_varying_median.csv`.
2. For Figure 2(c), use the same `M_list` and `prob_l_list = [0.105] * 20`. Run `python main.py` and preserve the result as `all_output_105_20repeats.csv`.
3. For Figure 2(d), use the same `M_list` and `prob_l_list = [0.42] * 20`. Run `python main.py` and preserve the result as `all_output_42_20repeats.csv`.
4. Run the plotting scripts using the preserved CSV files as described above.

For every configuration, `main.py` calls the following stages in order:

1. `raw_labels.py` generates pseudo-labels and feature files;
2. `aggregation_labels.py` produces majority-vote, Dawid-Skene, and GLAD labels and probabilities;
3. `trainer.py` fits the final layer of the ResNet18 models and writes checkpoints under `net_parameters/resnet18_finetune/`;
4. `test.py` evaluates the fitted models and appends the results to `all_output.csv`.

Intermediate files include `raw_votes.pt`, `task_votes.csv`, `alpha_set.csv`, the method-specific label CSV files, train/test feature files, and fitted `.pt` checkpoints. They are regenerated and overwritten as the driver advances through configurations.

### Runtime, hardware, and randomness

Plotting the supplied CSV files normally takes less than a minute. A full simulation repeatedly processes all 60,000 CIFAR-10 images and performs 50 training epochs for each method and configuration. A 20-trial run takes many hours to several days. A CUDA-capable GPU substantially reduces runtime; CPU execution is considerably slower. Several gigabytes of free disk space are needed for the CIFAR-10 data, pretrained weights, intermediate files, and outputs.

The scripts use CUDA automatically when available. Their random-number seed calls are commented out, so newly generated pseudo-labels and trained models will not be bit-for-bit identical to the supplied results. The three supplied CSV files and four supplied PDFs are the archived paper outputs.
