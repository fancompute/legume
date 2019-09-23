import matplotlib.pyplot as plt
import numpy as np


# TODO: Make this more general
def bands(gme, lightcone=True, ax=None, figsize=(4,5)):

    X = np.arange(len(gme.kpoints[0, :]))

    if ax is None:
        fig, ax = plt.subplots(1, 1, constrained_layout=True, figsize=figsize)

    plt.plot(X, gme.freqs, 'o', c="#1f77b4", label="", ms=4, mew=1)

    if lightcone:
        vec_LL = np.sqrt(np.square(gme.kpoints[0, :]) + np.square(gme.kpoints[1, :])) / np.pi /2
        ax.fill_between(X, vec_LL, gme.freqs[:].max(), facecolor="#cccccc", zorder=4, alpha=0.5)

    ax.set_xlim(left=0, right=len(gme.kpoints[0, :])-1)
    ax.set_ylim(bottom=0.0, top=gme.freqs[:].max())
    ax.set_xticks([])
    ax.set_xlabel('Wave vector')
    ax.set_ylabel('Frequency')

    plt.show()

    return ax
