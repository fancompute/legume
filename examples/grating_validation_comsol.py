import argparse

import numpy as np
import matplotlib.pyplot as plt
import legume

parser = argparse.ArgumentParser()
parser.add_argument('--overview', action='store_true')
parser.add_argument('--verbose', action='store_true')
parser.add_argument('-gmax', default=10, type=int)
parser.add_argument('-gmode_npts', default=1000, type=int)
parser.add_argument('-neig', default=5, type=int)
parser.add_argument('-ymax', default=0.1, type=float)
parser.add_argument('-W', default=0.50, type=float)
parser.add_argument('-H', default=0.25, type=float)
parser.add_argument('-D', default=0.25, type=float)
parser.add_argument('-epsr', default=12, type=float)
args = parser.parse_args()

options = {'gmode_inds': np.arange(0, 10), 'gmode_npts': args.gmode_npts, 'numeig': args.neig, 'verbose': args.verbose}

lattice = legume.Lattice([1, 0], [0, args.ymax])
path = lattice.bz_path(['G', np.array([np.pi, 0])], [25])
box = legume.Poly(eps=args.epsr, x_edges=[-args.W / 2, -args.W / 2, +args.W / 2, +args.W / 2],
                     y_edges=np.array([0.5, -0.5, -0.5, 0.5]) * args.ymax)

data_ce = np.loadtxt('./examples/data/filtered_complete_etch_H0.25_W0.50.csv', comments='%', delimiter=',')
data_pe = np.loadtxt('./examples/data/filtered_partial_etch_H0.25_W0.50_D0.25.csv', comments='%', delimiter=',')
data_spe = np.loadtxt('./examples/data/filtered_symmetric_partial_etch_H0.25_W0.50_D0.25.csv', comments='%', delimiter=',')

# Complete etch
print("Running complete etch...")

phc_ce = legume.PhotCryst(lattice)
phc_ce.add_layer(d=args.H, eps_b=1)
phc_ce.layers[-1].add_shape(box)
# phc_ce.plot_overview()
gme_ce = legume.GuidedModeExp(phc_ce, gmax=args.gmax)
gme_ce.run(kpoints=path.kpoints, **options)

# Partial etch
print("Running partial etch...")

phc_pe = legume.PhotCryst(lattice)
phc_pe.add_layer(d=args.D, eps_b=args.epsr)
phc_pe.add_layer(d=args.H, eps_b=1)
phc_pe.layers[-1].add_shape(box)
# phc_pe.plot_overview()
gme_pe = legume.GuidedModeExp(phc_pe, gmax=args.gmax)
gme_pe.run(kpoints=path.kpoints, **options)

# Symmetric partial etch
print("Running symmetric partial etch...")

phc_spe = legume.PhotCryst(lattice)
phc_spe.add_layer(d=args.H, eps_b=1)
phc_spe.layers[-1].add_shape(box)
phc_spe.add_layer(d=args.D, eps_b=args.epsr)
phc_spe.add_layer(d=args.H, eps_b=1)
phc_spe.layers[-1].add_shape(box)
# phc_spe.plot_overview()
gme_spe = legume.GuidedModeExp(phc_spe, gmax=args.gmax)
gme_spe.run(kpoints=path.kpoints, **options)

# Plot
fig, axs = plt.subplots(1,3,constrained_layout=True)
legume.viz.bands(gme_ce, ls='-', ax=axs[0])
axs[0].plot(data_ce[:,0], data_ce[:, 1:], 'o', markeredgecolor='k', color='none', label="COMSOL")
axs[0].set_title("Complete etch")

legume.viz.bands(gme_pe, ls='-', ax=axs[1])
axs[1].plot(data_pe[:,0], data_pe[:, 1:], 'o', markeredgecolor='k', color='none', label="COMSOL")
axs[1].set_title("Partial etch")

legume.viz.bands(gme_spe, ls='-', ax=axs[2])
axs[2].plot(data_spe[:,0], data_spe[:, 1:], 'o', markeredgecolor='k', color='none', label="COMSOL")
axs[2].set_title("Symmetric partial etch")