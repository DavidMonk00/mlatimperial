from matplotlib import pyplot as plt
import numpy as np


def plotCorrelation(data):
    corr_data = data.copy()
    names = list(corr_data.columns)
    correlations = corr_data.corr().abs()
    fig = plt.figure(figsize=(50, 50))
    ax = fig.add_subplot(111)
    ax.matshow(correlations, vmin=-1, vmax=1)
    # fig.colorbar(cax)
    ticks = np.arange(0, len(names), 1)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(names)
    ax.set_yticklabels(names)
    plt.show()
