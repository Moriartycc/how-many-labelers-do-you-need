# Computational experiments

This directory contains the code, data, archived outputs, and execution instructions for the numerical experiments in **How many labelers do you have? A closer look at gold-standard labels**.

- [`bluebirds/`](./bluebirds/) covers Section 5.1 and Figure 1.
- [`semisynthetic/`](./semisynthetic/) covers Section 5.2 and Figure 2.
- [`cifar10h/`](./cifar10h/) covers Appendix A and Supplement Figure A.1.

Each directory has its own master script because the workflows use different software and datasets:

| Paper section | Directory | Master workflow |
| --- | --- | --- |
| Section 5.1, Figure 1 | [`bluebirds/`](./bluebirds/) | MATLAB: `main.m` |
| Section 5.2, Figure 2 | [`semisynthetic/`](./semisynthetic/) | Python: `main.py`, `plot.py`, and `plot-repeats.py` |
| Appendix A, Figure A.1 | [`cifar10h/`](./cifar10h/) | Python: `main.py`; MATLAB: `main.m` |

Run each script from its experiment directory because the scripts use relative paths. The repository includes precomputed numerical outputs and the final paper figures for inspection without rerunning the long simulations.

