import math
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.ticker as mticker
import numpy as np

def log_tick_formatter(val, pos=None):
    return f"$10^{{{val:g}}}$"

def plot_method(method_name, ax, color):
    outputs = pd.read_csv('all_output_varying_median.csv', index_col=0)
    # x = outputs[outputs["2"] == "MV"]["0"]
    # y = outputs[outputs["2"] == "MV"]["1"]
    z = np.log10((100 - outputs[outputs["2"] == method_name]["3"]) / 100)
    x, y = np.meshgrid([2, 4, 8, 16, 32], [0.105, 0.11, 0.12, 0.14, 0.18, 0.26, 0.42, 0.74])
    z = z.values.reshape(8, 5)
    surf = ax.plot_surface(x, y, z, color=color, antialiased=True, alpha=.2)
    surf._edgecolors2d = surf._edgecolor3d
    surf._facecolors2d = surf._facecolor3d
    if method_name == "MV_probas":
        method_name = "MLE"
    surf = ax.plot_wireframe(x, y, z, color=color, linewidth=1.5, antialiased=True, alpha=.75, label=method_name.replace("_probas", " prob"))

def plot_u_distance_method(method_name, ax, color):
    outputs = pd.read_csv('all_output_adversarial_median.csv', index_col=0)
    # x = outputs[outputs["2"] == "MV"]["0"]
    # y = outputs[outputs["2"] == "MV"]["1"]
    z = outputs[outputs["2"] == method_name]
    z = z[z.columns[4:]]
    z_star = outputs[outputs["2"] == "true_train_prob"]
    z_star = z_star[z_star.columns[4:]]
    z = z.div(np.sqrt(np.square(z).sum(axis=1)), axis=0)
    z_star = z_star.div(np.sqrt(np.square(z_star).sum(axis=1)), axis=0)
    z = np.minimum(np.sqrt(np.square(np.subtract(z, np.asarray(z_star))).sum(axis=1)), np.sqrt(np.square(np.add(z, np.asarray(z_star))).sum(axis=1)))
    x, y = np.meshgrid([2, 4, 8, 16, 32], [0.105, 0.11, 0.12, 0.14, 0.18, 0.26, 0.42, 0.74])
    z = z.values.reshape(8, 5)
    # print(z)
    surf = ax.plot_surface(x, y, z, color=color, antialiased=True, alpha=.2)
    surf._edgecolors2d = surf._edgecolor3d
    surf._facecolors2d = surf._facecolor3d
    if method_name == "MV_probas":
        method_name = "MLE"
    surf = ax.plot_wireframe(x, y, z, color=color, linewidth=1.5, antialiased=True, alpha=.75, label=method_name.replace("_probas", " prob"))


# outputs = pd.read_csv('outputs_processed.csv', index_col=0)
# x = outputs[outputs["2"] == "MV"]["0"]
# y = outputs[outputs["2"] == "MV"]["1"]
# z = outputs[outputs["2"] == "MV"]["3"]
# x, y = np.meshgrid([2, 4, 8, 16, 32], [0.105, 0.11, 0.12, 0.14, 0.18, 0.26])
# z = z.values.reshape(6, 6)
# z = z[:, 1:6]
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.zaxis.set_major_formatter(mticker.FuncFormatter(log_tick_formatter))
ax.zaxis.set_major_locator(mticker.MaxNLocator(integer=False))
plot_method("MV", ax, "b")
# plot_u_distance_method("MV", ax, "b")
plot_method("MV_probas", ax, "c")
# plot_u_distance_method("MV_probas", ax, "c")
plot_method("DS", ax, "k")
# plot_u_distance_method("DS", ax, "k")
# plot_method("DS_probas", ax, "g")
# plot_u_distance_method("DS_probas", ax, "g")
plot_method("GLAD", ax, "m")
# plot_u_distance_method("GLAD", ax, "m")
# plot_method("GLAD_probas", ax, "r")
# plot_u_distance_method("GLAD_probas", ax, "r")

ax.legend(loc=9, fontsize='small', fancybox=True, shadow=True, ncol=4)
ax.set_xlabel('number of labelers')
ax.set_ylabel('median labeler accuracy')
ax.set_zlabel('test error')
# ax.set_zlabel('$\| u - u^\star \|_2$')
plt.show()
plt.savefig('output_plot.png')
