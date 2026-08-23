# Reproducibility materials

This repository accompanies the paper **How many labelers do you have? A closer look at gold-standard labels** by Chen Cheng (Department of Statistics, University of Chicago), Hilal Asi (Apple), and John Duchi (Departments of Statistics and Electrical Engineering, Stanford University).

## Repository contents

- [`manuscript/`](./manuscript/) contains the manuscript and supplementary-material source files.
- [`code/`](./code/) contains the code, data, archived outputs, and instructions for reproducing the computational figures.
- [`acc-form-2021.docx`](./acc-form-2021.docx) and [`acc-form-2021.pdf`](./acc-form-2021.pdf) provide the completed JASA Author Contributions Checklist.

The code is released under the [MIT License](./LICENSE).

## Software requirements

The Python experiments use Python 3.11. Install the direct dependencies with:

```text
pip install -r requirements.txt
```

The experiments also use the `cifar10_models` module from the `PyTorch_CIFAR10` repository. Because that repository is not packaged for installation by `pip`, clone the pinned revision separately:

```text
git clone https://github.com/huyvnphan/PyTorch_CIFAR10.git ../PyTorch_CIFAR10
git -C ../PyTorch_CIFAR10 checkout 641cac24371b17052b9bb6e56af1c83b5e97cd7f
```

Add the absolute path of the cloned `PyTorch_CIFAR10` directory to `PYTHONPATH` before running the Python scripts.

The upstream Git repository does not include the pretrained parameter files. Download its [official pretrained-weight archive](https://rutgers.box.com/shared/static/gkw08ecs797j2et1ksmbg1w5t3idf5r5.zip) and extract it so that the `.pt` files are in `../PyTorch_CIFAR10/cifar10_models/state_dicts/`.

The MATLAB experiments require MATLAB R2022b, CVX 2.2, and MOSEK 9.3.22. A valid MOSEK license is required.

## Reproduction map

Each experiment has its own master script because the paper combines MATLAB and Python workflows.

| Paper output | Working directory | Master script or plotting script | Supplied final output |
| --- | --- | --- | --- |
| Figure 1(a)--(b) | `code/bluebirds/` | `main.m` | `resnet-class-bluebirds.pdf`, `resnet-calib-bluebirds.pdf` |
| Figure 2(a)--(b) | `code/semisynthetic/` | `main.py`, then `plot.py` | `hard-test-error.pdf`, `soft-test-error.pdf` |
| Figure 2(c) | `code/semisynthetic/` | `main.py`, then `plot-repeats.py` using `all_output_105_20repeats.csv` | `105.pdf` |
| Figure 2(d) | `code/semisynthetic/` | `main.py`, then `plot-repeats.py` using `all_output_42_20repeats.csv` | `42.pdf` |
| Supplement Figure A.1(a)--(b) | `code/cifar10h/` | `main.py`, then `main.m` | `resnet-class-new.pdf`, `resnet-calib-new.pdf` |

The experiment-specific README files give the required inputs, exact execution order, generated intermediates, runtime scale, and randomness information. The repository contains no computational tables; all computational outputs reported in the paper are figures.
