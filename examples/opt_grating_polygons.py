""" This attempts to perform topology optimization of a grating through a density defined across N polygons
"""

import argparse

import autograd
import matplotlib.pyplot as plt
from matplotlib import gridspec
import numpy as np
from autograd import grad

import legume
from legume.backend import backend as bd
from legume.optimizers import adam_optimize

parser = argparse.ArgumentParser()
parser.add_argument('--verbose', action='store_true')
parser.add_argument('--optimize', action='store_true')
parser.add_argument('-N_epochs', default=20, type=int)
parser.add_argument('-learning_rate', default=0.1, type=float)
parser.add_argument('-gmax', default=5, type=int)
parser.add_argument('-gmode_npts', default=2000, type=int)
parser.add_argument('-neig', default=10, type=int)
parser.add_argument('-ymax', default=0.1, type=float)
parser.add_argument('-N_polygons', default=20, type=int)
parser.add_argument('-H', default=0.33, type=float)
parser.add_argument('-D', default=0.20, type=float)
parser.add_argument('-eps', default=12, type=float)
args = parser.parse_args()

options = {'gmode_inds': np.arange(0, 8, 2),
		   'gmode_npts': args.gmode_npts,
		   'numeig': args.neig,
		   'verbose': args.verbose}

lattice = legume.Lattice([1, 0], [0, args.ymax])
path = lattice.bz_path(['G', np.array([np.pi / 2, 0])], [35])


def projection(rho, eta=0.5, beta=100):
	return bd.divide(bd.tanh(beta * eta) + bd.tanh(beta * (rho - eta)), bd.tanh(beta * eta) + bd.tanh(beta * (1 - eta)))


def make_grating():
	phc = legume.PhotCryst(lattice)
	phc.add_layer(d=args.D, eps_b=args.eps)
	phc.add_layer(d=args.H, eps_b=1)
	return legume.GuidedModeExp(phc, gmax=float(args.gmax))


def parameterize_density_layer(layer, rho, eta=0.5, beta=100):
	N = len(rho)
	X = np.linspace(-0.5, 0.5, N + 1)
	rho_proj = projection(rho, eta, beta)
	for i in range(len(X) - 1):
		x0 = X[i]
		x1 = X[i + 1]
		eps = 1 + rho_proj[i] * (args.eps - 1)
		grating = legume.Poly(eps=eps, x_edges=np.array([x0, x0, x1, x1]),
							  y_edges=np.array([0.5, -0.5, -0.5, 0.5]) * args.ymax)
		layer.add_shape(grating)


def objective(rho):
	parameterize_density_layer(gme.phc.layers[-1], rho, eta=0.5, beta=10)
	gme.run(kpoints=path.kpoints, **options)
	tgt_freqs = gme.freqs[0:10, 1]
	return bd.sqrt(bd.var(tgt_freqs))


# Initialize structure

# rho_0 = np.zeros((args.N_polygons,)) + 0.1
# rho_0[int(0.25 * len(rho_0)):int(0.75 * len(rho_0))] = 1.0

x0 = np.linspace(-np.pi, +np.pi, args.N_polygons)
rho_0 = 0.5 + np.sin(4*x0) * 0.5

# Compute results for initial structure
legume.set_backend('numpy')
objective_grad = grad(objective)
gme = make_grating()
parameterize_density_layer(gme.phc.layers[-1], rho_0, eta=0.5, beta=10)
gme.run(kpoints=path.kpoints, **options)

fig = plt.figure(constrained_layout=True)
gs = gridspec.GridSpec(3, 2, figure=fig, height_ratios=[1.0, 0.33, 0.15])

gme.phc.plot_overview(cmap='Reds', fig=fig, subplot_spec=gs[0,0], cbar=False)
ax1 = fig.add_subplot(gs[1, 0])
legume.viz.bands(gme, ax=ax1)

if args.optimize:
	## Optimize
	legume.set_backend('autograd')
	(rho_opt, ofs) = adam_optimize(objective, rho_0, objective_grad, step_size=args.learning_rate, Nsteps=args.N_epochs,
								   options={'direction': 'min', 'disp': ['of', 'params']})
	legume.set_backend('numpy')

	ax3 = fig.add_subplot(gs[2, :])
	ax3.plot(ofs, "o-")
	ax3.set_xlabel("Epoch")
	ax3.set_ylabel("Cost function")

	## Display results (recompute at optimal structure)
	gme = make_grating()
	parameterize_density_layer(gme.phc.layers[-1], rho_opt, eta=0.5, beta=10)
	gme.run(kpoints=path.kpoints, **options)

	gme.phc.plot_overview(cmap='Blues', fig=fig, subplot_spec=gs[0, 1], cbar=False)
	ax2 = fig.add_subplot(gs[1, 1])
	legume.viz.bands(gme, ax=ax2)

fig.tight_layout(pad=0)
