""" This attempts to perform topology optimization of a grating through a density defined across N polygons
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np
from autograd import grad

import legume
from legume.backend import backend as bd
from legume.optimizers import adam_optimize

parser = argparse.ArgumentParser()
parser.add_argument('--verbose', action='store_true')
parser.add_argument('-N_epochs', default=9, type=int)
parser.add_argument('-learning_rate', default=0.1, type=float)
parser.add_argument('-gmax', default=5, type=float)
parser.add_argument('-gmode_npts', default=2000, type=int)
parser.add_argument('-lr', default=0.1, type=float)
parser.add_argument('-epochs', default=10, type=int)
parser.add_argument('-eta', default=0.5, type=float)
parser.add_argument('-beta', default=10, type=float)
parser.add_argument('-neig', default=10, type=int)
parser.add_argument('-ymax', default=0.1, type=float)
parser.add_argument('-N_polygons', default=20, type=int)
parser.add_argument('-H', default=0.33, type=float)
parser.add_argument('-D', default=0.20, type=float)
parser.add_argument('-eps', default=12, type=float)
args = parser.parse_args()

options = {'gmode_inds': np.arange(0, 2, 8),  # [0, 3, 5, 7],
		   'gmode_npts': args.gmode_npts,
		   'numeig': args.neig,
		   'verbose': args.verbose,
		   'compute_im': False}

lattice = legume.Lattice([1, 0], [0, args.ymax])
path = lattice.bz_path(['G', np.array([np.pi, 0])], [25])

def summarize_results(gme):
	"""Summarize the results of a gme run.

	Plots the bands and the real space structure
	"""
	fig = plt.figure(constrained_layout=True)
	gs = fig.add_gridspec(ncols=2, nrows=1)
	ax1 = fig.add_subplot(gs[0])
	legume.viz.bands(gme, ax=ax1, Q=True)
	gme.phc.plot_overview(fig=fig, gridspec=gs[1], cbar=False)


def projection(rho, eta=0.5, beta=100):
	return bd.divide(bd.tanh(beta * eta) + bd.tanh(beta * (rho - eta)), bd.tanh(beta * eta) + bd.tanh(beta * (1 - eta)))


def make_simulation(polygons):
	phc = legume.PhotCryst(lattice)
	phc.add_layer(d=args.D, eps_b=args.eps)
	phc.add_layer(d=args.H, eps_b=1)
	for polygon in polygons: phc.layers[-1].add_shape(polygon)
	gme = legume.GuidedModeExp(phc, gmax=float(args.gmax))
	return gme


def generate_polygons(rho, eta=0.5, beta=100):
	N = len(rho)
	X = np.linspace(-0.5, 0.5, N + 1)

	rho_proj = projection(rho, eta, beta)

	polygons = []
	for i in range(len(X) - 1):
		x0 = X[i]
		x1 = X[i + 1]
		eps = 1 + rho_proj[i] * (args.eps - 1)
		polygon = legume.Poly(eps=eps, x_edges=np.array([x0, x0, x1, x1]),
							  y_edges=np.array([0.5, -0.5, -0.5, 0.5]) * args.ymax)
		polygons.append(polygon)

	return polygons


def objective(rho):
	polygons = generate_polygons(rho, eta=args.eta, beta=args.beta)
	gme = make_simulation(polygons)
	gme.run(kpoints=path.kpoints, **options)
	tgt_freqs = gme.freqs[0:5, 1]
	return bd.max(bd.abs(tgt_freqs[0] - tgt_freqs))

# Initialize structure
x0 = np.linspace(-np.pi, +np.pi, args.N_polygons)
rho_0 = 0.5 + np.sin(4 * x0) * 0.5

# Compute results for initial structure
legume.set_backend('numpy')
polygons = generate_polygons(rho_0, eta=args.eta, beta=args.beta)
gme = make_simulation(polygons)
gme.run(kpoints=path.kpoints, **options)
summarize_results(gme)

## Optimize
legume.set_backend('autograd')
objective_grad = grad(objective)
(rho_opt, ofs) = adam_optimize(objective, rho_0, objective_grad, step_size=args.lr, Nsteps=args.epochs,
							   options={'direction': 'min', 'disp': ['of']})

legume.set_backend('numpy')

## Display results (recompute at optimal structure)
polygons = generate_polygons(rho_opt, eta=args.eta, beta=args.beta)
gme = make_simulation(polygons)
gme.run(kpoints=path.kpoints, **options)
summarize_results(gme)
