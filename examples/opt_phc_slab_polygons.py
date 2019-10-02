""" This attempts to perform topology optimization of a photonic crystal slab band gap through a density
defined across a grid of polygons
"""

import argparse

import autograd
import matplotlib.pyplot as plt
import numpy as np
from autograd import grad

import legume
from legume.backend import backend as bd
from legume.optimizers import adam_optimize

parser = argparse.ArgumentParser()
parser.add_argument('--verbose', action='store_true')
parser.add_argument('--optimize', action='store_true')
parser.add_argument('-N_epochs', default=10, type=int)
parser.add_argument('-learning_rate', default=0.5, type=float)
parser.add_argument('-gmax', default=5, type=float)
parser.add_argument('-gmode_npts', default=2000, type=int)
parser.add_argument('-neig', default=10, type=int)
parser.add_argument('-N_polygons', default=20, type=int)
parser.add_argument('-H', default=0.33, type=float)
parser.add_argument('-D', default=0.20, type=float)
parser.add_argument('-eps_b', default=12, type=float)
args = parser.parse_args()

options = {'gmode_inds': np.array([0, 2]),
		   'gmode_npts': args.gmode_npts,
		   'numeig': args.neig,
		   'verbose': args.verbose}

lattice = legume.Lattice('hexagonal')
path = lattice.bz_path(['G', 'M', 'K', 'G'], [10, 10, 10])

def projection(rho, eta=0.5, beta=100):
	return bd.divide(bd.tanh(beta * eta) + bd.tanh(beta * (rho - eta)), bd.tanh(beta * eta) + bd.tanh(beta * (1 - eta)))


def make_simulation(polygons):
	phc = legume.PhotCryst(lattice)
	phc.add_layer(d=args.D, eps_b=args.eps_b)
	for polygon in polygons: phc.layers[-1].add_shape(polygon)
	gme = legume.GuidedModeExp(phc, gmax=args.gmax)
	return gme


def generate_polygons(rho, eta=0.5, beta=10):
	assert rho.shape[0] == rho.shape[1], "For now only square grids"
	N = rho.shape[0]

	X = np.linspace(-0.5, +0.5, N+1)
	Y = np.linspace(-0.5, +0.5, N+1)

	rho_proj = projection(rho, eta, beta)

	polygons = []
	for i in range(len(X) - 1):
		x0 = X[i]
		x1 = X[i + 1]
		for j in range(len(Y) - 1):
			y0 = Y[j]
			y1 = Y[j + 1]

			eps = 12 + (1 - 12) * rho_proj[i,j]
			polygon = legume.Poly(eps=eps,
								  x_edges=np.array([x0, x1, x1, x0]),
								  y_edges=np.array([y0, y0, y1, y1]))
			polygons.append(polygon)

	return polygons


def objective(rho):
	polygons = generate_polygons(rho, eta=0.5, beta=10)
	gme = make_simulation(polygons)
	gme.run(kpoints=path.kpoints, **options)
	gap_size = bd.min(gme.freqs[:,1])-bd.max(gme.freqs[:,0])
	return gap_size

legume.set_backend('numpy')

r0 = 0.2

# # Can verify starting structure
# phc2=legume.PhotCryst(lattice)
# phc2.add_layer(d=args.D, eps_b=args.eps_b)
# phc2.layers[-1].add_shape(legume.Circle(eps=1, x_cent=0.0, y_cent=0.0, r=r0))
# gme2=legume.GuidedModeExp(phc2, gmax=args.gmax)
# gme2.run(kpoints=path.kpoints, **options)
# legume.viz.bands(gme2)

# Initialize the polygon slab
N = args.N_polygons
X = np.linspace(-0.5, +0.5, N)
Y = np.linspace(-0.5, +0.5, N)
(Y, X) = np.meshgrid(Y, X)
R = np.sqrt(np.square(X) + np.square(Y))
rho_0 = np.zeros((N,N))
rho_0[R<r0] = 1.0

polygons = generate_polygons(rho_0, eta=0.5, beta=10)
gme = make_simulation(polygons)
gme.run(kpoints=path.kpoints, **options)
gap_size = bd.min(gme.freqs[:,1])-bd.max(gme.freqs[:,0])

# Optimize
legume.set_backend('autograd')
objective_grad = grad(objective)
(rho_opt, ofs) = adam_optimize(objective, rho_0, objective_grad, step_size=args.learning_rate, Nsteps=args.N_epochs,
							   options={'direction': 'max', 'disp': ['of']})

legume.set_backend('numpy')
polygons = generate_polygons(rho_opt, eta=0.5, beta=10)
gme = make_simulation(polygons)
gme.run(kpoints=path.kpoints, **options)
legume.viz.bands(gme)

gme.plot_overview_ft()
gme.phc.plot_overview()
