import argparse

import numpy as np

import pygme

parser = argparse.ArgumentParser()
parser.add_argument('--overview', action='store_true')
parser.add_argument('-gmax', default=5, type=int)
parser.add_argument('-neig', default=10, type=int)
parser.add_argument('-ymax', default=0.1, type=float)
parser.add_argument('-W', default=0.50, type=float)
parser.add_argument('-H', default=0.33, type=float)
parser.add_argument('-D', default=0.20, type=float)
parser.add_argument('-epsrb', default=12, type=float)
parser.add_argument('-epsrt', default=12, type=float)
args = parser.parse_args()

lattice = pygme.Lattice([1, 0], [0, args.ymax])
phc = pygme.PhotCryst(lattice)

# Substrate
phc.add_layer(d=args.D, eps_b=args.epsrb)

# Grating
phc.add_layer(d=args.H, eps_b=1)

grating = pygme.Poly(eps=args.epsrt, x_edges=np.array([-args.W / 2, -args.W / 2, +args.W / 2, +args.W / 2]),
                     y_edges=np.array([0.5, -0.5, -0.5, 0.5]) * args.ymax)
phc.layers[-1].add_shape(grating)

gme = pygme.GuidedModeExp(phc, gmax=float(args.gmax))

if args.overview:
    phc.plot_overview()
    # gme.plot_overview_ft()

path = phc.lattice.bz_path(['G', np.array([np.pi, 0])], [35])
options = {'gmode_inds': np.arange(0, 8), 'gmode_npts': 500, 'numeig': args.neig, 'verbose': False}

gme.run(kpoints=path.kpoints, **options)

ax = pygme.viz.bands(gme)
# data = np.loadtxt('./examples/filtered_symmetric_H0.25_W0.50.csv', comments='%', delimiter=',')
# data = np.loadtxt('./examples/grating_bands.csv', comments='%', delimiter=',')
# ax.plot(data[:, 0], data[:, 1:], 'o', markeredgecolor='k', color='none', label="COMSOL")