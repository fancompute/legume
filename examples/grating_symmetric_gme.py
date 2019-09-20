import argparse

import numpy as np

import legume

parser = argparse.ArgumentParser()
parser.add_argument('--overview', action='store_true')
parser.add_argument('-gmax', default=5, type=int)
parser.add_argument('-neig', default=5, type=int)
parser.add_argument('-ymax', default=0.1, type=float)
parser.add_argument('-W', default=0.50, type=float)
parser.add_argument('-H', default=0.25, type=float)
parser.add_argument('-epsr', default=12, type=float)
args = parser.parse_args()

lattice = legume.Lattice([1, 0], [0, args.ymax])
phc = legume.PhotCryst(lattice)

phc.add_layer(d=args.H, eps_b=1)

grating = legume.Poly(eps=args.epsr, x_edges=[-args.W / 2, -args.W / 2, +args.W / 2, +args.W / 2],
                     y_edges=np.array([0.5, -0.5, -0.5, 0.5]) * args.ymax)
phc.layers[-1].add_shape(grating)

gme = legume.GuidedModeExp(phc, gmax=args.gmax)

if args.overview:
    # phc.plot_overview()
    gme.plot_overview_ft()

path = phc.lattice.bz_path(['G', np.array([np.pi, 0])], [10])
options = {'gmode_inds': np.arange(0, 8), 'gmode_npts': 500, 'numeig': args.neig, 'verbose': False}

gme.run(kpoints=path.kpoints, **options)

ax = legume.viz.bands(gme)
data = np.loadtxt('./examples/filtered_symmetric_H0.25_W0.50.csv', comments='%', delimiter=',')
# data = np.loadtxt('./examples/unfiltered_symmetric_H0.25_W0.50.csv', comments='%', delimiter=',')
ax.plot(data[:, 0], data[:, 1:], 'o', markeredgecolor='k', color='none', label="COMSOL")
