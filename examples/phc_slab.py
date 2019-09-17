import argparse

import numpy as np

import pygme

parser = argparse.ArgumentParser()
parser.add_argument('--overview', action='store_true')
parser.add_argument('-gmax', default=5, type=int)
parser.add_argument('-neig', default=6, type=int)
parser.add_argument('-ymax', default=0.1, type=float)
parser.add_argument('-r', default=0.40, type=float)
parser.add_argument('-d', default=0.55, type=float)
parser.add_argument('-epsr', default=12, type=float)
args = parser.parse_args()

lattice = pygme.Lattice('hexagonal')
phc = pygme.PhotCryst(lattice)

phc.add_layer(d=args.d, eps_b=args.epsr)
hole = pygme.Circle(x_cent=0, y_cent=0, r=args.r)
phc.layers[-1].add_shape(hole)

gme = pygme.GuidedModeExp(phc, gmax=args.gmax)

if args.overview:
    phc.plot_overview()
    # gme.plot_overview_ft()

path = phc.lattice.bz_path(['G', 'M', 'K', 'G'], [50, 50, 50])
options = {'gmode_inds': np.arange(0, 6), 'gmode_npts': 100, 'numeig': args.neig, 'verbose': False}

gme.run(kpoints=path.kpoints, options=options)

import matplotlib.pyplot as plt

fig, ax = plt.subplots(1, constrained_layout=True)
plt.plot(gme.freqs, 'o')
ax.set_ylim([0, 0.8])
ax.set_xlim([0, gme.freqs.shape[0]-1])
plt.xticks(path.pt_inds, path.pt_labels)
ax.xaxis.grid('True')
plt.show()

