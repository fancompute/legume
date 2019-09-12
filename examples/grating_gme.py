import argparse

import matplotlib.pyplot as plt
import numpy as np

import pygme


def plot_bands(gme, cone=True, csv_file=None):
    fig, ax = plt.subplots(1, constrained_layout=True, figsize=(4, 5))
    plt.plot(gme.kpoints[0, :] / np.pi, gme.freqs, '-', c="#1f77b4", label="")
    if cone:
        ax.fill_between(gme.kpoints[0, :], gme.kpoints[0, :], gme.freqs[:].max(), facecolor="#cccccc", zorder=4,
                        alpha=0.5)
    if csv_file is not None:
        data = np.loadtxt(csv_file, comments='%', delimiter=',')
        comsol_K = data[:, 0]
        comsol_bands = data[:, 1:]
        plt.plot(comsol_K, comsol_bands, 'o', markeredgecolor='k', color='none', label="COMSOL")
        # plt.legend()

    ax.set_xlim(left=0.0, right=1.0)
    ax.set_ylim(bottom=0.0, top=gme.freqs[:].max())
    ax.set_xlabel('Wave vector')
    ax.set_ylabel('Frequency')
    plt.show()


parser = argparse.ArgumentParser()
parser.add_argument('--overview', action='store_true')
parser.add_argument('-gmax', default=5)
parser.add_argument('-neig', default=5)
parser.add_argument('-ymax', default=0.1)
parser.add_argument('-W', default=0.50)
parser.add_argument('-H', default=0.33)
parser.add_argument('-D', default=0.20)
parser.add_argument('-epsrb', default=12)
parser.add_argument('-epsrt', default=12)
args = parser.parse_args()

lattice = pygme.Lattice([1, 0], [0, args.ymax])
phc = pygme.PhotCryst(lattice)

# Substrate
phc.add_layer(d=args.D, eps_b=args.epsrb)

# Grating
phc.add_layer(d=args.H, eps_b=1)

grating = pygme.Poly(eps=args.epsrt, x_edges=[-args.W / 2, -args.W / 2, +args.W / 2, +args.W / 2],
                     y_edges=np.array([0.5, -0.5, -0.5, 0.5]) * args.ymax)
phc.layers[-1].add_shape(grating)

gme = pygme.GuidedModeExp(phc, gmax=args.gmax)

if args.overview:
    phc.plot_overview()
    # gme.plot_overview_ft()

path = phc.lattice.bz_path(['G', np.array([np.pi, 0])], [25])

gme.run(kpoints=path.kpoints, gmode_inds=np.arange(0, 8), N_g_array=500, verbose=False, numeig=args.neig)

plot_bands(gme, csv_file='./grating_bands_filtered.csv')
plot_bands(gme, csv_file='./grating_bands.csv')
