This directory contains code and source files for reproducing the numerical experiments in the paper:



**How many labelers do you have? A closer look at gold-standard labels**



The experiments and instructions for Section 5.1 of the paper are in the directory [`bluebirds/`](./bluebirds/).



The experiments and instructions for Section 5.2 of the paper are in the directory [`semisynthetic/`](./semisynthetic/).



The experiments and instructions for Appendix A of the paper are in the directory [`cifar10h/`](./cifar10h/).

Each directory has its own master script because the workflows use different software and datasets:

| Paper section | Directory | Master workflow |
| --- | --- | --- |
| Section 5.1, Figure 1 | [`bluebirds/`](./bluebirds/) | MATLAB: `main.m` |
| Section 5.2, Figure 2 | [`semisynthetic/`](./semisynthetic/) | Python: `main.py`, `plot.py`, and `plot-repeats.py` |
| Appendix A, Figure A.1 | [`cifar10h/`](./cifar10h/) | Python: `main.py`; MATLAB: `main.m` |

Run every script from its experiment directory because the scripts use relative paths. Precomputed numerical outputs and the final paper figures are included for inspection without rerunning the long simulations.

