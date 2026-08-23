JASA Reproducibility Materials for the paper:

**How many labelers do you have? A closer look at gold-standard labels**

Authors: Chen Cheng^1, Hilal Asi^2 and John Duchi^3

Affiliations:

1. University of Chicago, Department of Statistics

2. Apple Inc.

3. Stanford University, Departments of Statistics and Electrical Engineering

The directory [`manuscript/`](./manuscript/) contains the manuscript source files for the paper.

The directory [`code/`](./code/) contains the source files, data, and instructions for reproducing the experiments in the paper.

The code in this repository is released under the [MIT License](./LICENSE).

## Software requirements

The Python experiments use Python 3.11. Install the direct dependencies with:

```text
pip install -r requirements.txt
```

The experiments also use the `cifar10_models` module from the `PyTorch_CIFAR10` repository. Because that repository is not packaged for installation by `pip`, clone the pinned revision separately:

```text
git clone https://github.com/huyvnphan/PyTorch_CIFAR10.git
git -C PyTorch_CIFAR10 checkout 641cac24371b17052b9bb6e56af1c83b5e97cd7f
```

Add the absolute path of the cloned `PyTorch_CIFAR10` directory to `PYTHONPATH` before running the Python scripts.

The MATLAB experiments require MATLAB R2022b, CVX 2.2, and MOSEK 9.3.22. A valid MOSEK license is required.
