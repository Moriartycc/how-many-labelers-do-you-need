import math
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.ticker as mticker
import numpy as np

file_name = 'all_output_42_20repeats.csv'
M_set = [2, 4, 8, 16, 32]

def plot_method(method_name, fig, color):
    outputs = pd.read_csv(file_name, index_col=0)
    outputs = outputs[outputs["2"] == method_name]
    y_med, y_l, y_u = [], [], []
    for M in M_set:
        z = (100 - outputs[outputs["0"] == M]["3"]) / 100
        y_med.append(np.percentile(z, 50))
        y_l.append(np.percentile(z, 20))
        y_u.append(np.percentile(z, 80))

    if method_name == "MV_probas":
        method_name = "MLE"
    markers, caps, bars = plt.errorbar(x=M_set, y=y_med, yerr=[np.subtract(y_med, y_l), np.subtract(y_u, y_med)], marker='o', color=color, capsize=7, label=method_name.replace("_probas", " prob"))
    [bar.set_alpha(0.3) for bar in bars]
    [cap.set_alpha(0.3) for cap in caps]

fig = plt.figure()
plt.yscale("log")
plot_method("MV", fig, "b")
plot_method("MV_probas", fig, "c")
plot_method("DS", fig, "k")
plot_method("DS_probas", fig, "g")
plot_method("GLAD", fig, "m")
plot_method("GLAD_probas", fig, "r")
plt.legend(loc="upper right")
plt.ylabel('test error')
plt.xlabel('number of labelers')
plt.show()


