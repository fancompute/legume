import matplotlib.pyplot as plt
import numpy as np


# TODO: Make this more general
def bands(gme, lightcone=True, ax=None):
    if ax is None:
        fig, ax = plt.subplots(1, 1, constrained_layout=True, figsize=(4, 5))

    plt.plot(gme.kpoints[0, :] / np.pi, gme.freqs, 'o', c="#1f77b4", label="")

    if lightcone:
        ax.fill_between(gme.kpoints[0, :], gme.kpoints[0, :], gme.freqs[:].max(), facecolor="#cccccc", zorder=4,
                        alpha=0.5)

    ax.set_xlim(left=0.0, right=1.0)
    ax.set_ylim(bottom=0.0, top=gme.freqs[:].max())
    ax.set_xlabel('Wave vector')
    ax.set_ylabel('Frequency')

    plt.show(block=False)

    return ax
