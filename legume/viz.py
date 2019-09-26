import matplotlib.pyplot as plt
import numpy as np


# TODO: Make this more general
def bands(gme, lightcone=True, ax=None, figsize=(4,5), ls='o'):

    if np.all(gme.kpoints[0,:]==0) and not np.all(gme.kpoints[1,:]==0) \
        or np.all(gme.kpoints[1,:]==0) and not np.all(gme.kpoints[0,:]==0):
        X = np.sqrt(np.square(gme.kpoints[0,:]) + np.square(gme.kpoints[1,:])) / np.pi
    else:
        X = np.arange(len(gme.kpoints[0, :]))

    if ax is None:
        fig, ax = plt.subplots(1, 1, constrained_layout=True, figsize=figsize)

    ax.plot(X, gme.freqs, ls, c="#1f77b4", label="", ms=4, mew=1)

    if lightcone:
        eps_clad = [gme.phc.claddings[0].eps_avg, gme.phc.claddings[-1].eps_avg]
        vec_LL = np.sqrt(np.square(gme.kpoints[0, :]) + np.square(gme.kpoints[1, :])) \
            / 2 / np.pi / np.sqrt(max(eps_clad))
        ax.fill_between(X, vec_LL, gme.freqs[:].max(), facecolor="#cccccc", zorder=4, alpha=0.5)

    ax.set_xlim(left=0, right=max(X))
    ax.set_ylim(bottom=0.0, top=gme.freqs[:].max())
    # ax.set_xticks([])
    ax.set_xlabel('Wave vector')
    ax.set_ylabel('Frequency')

    plt.show()

    return ax
