
import argparse

import autograd
import matplotlib.pyplot as plt
from matplotlib import gridspec
import numpy as np
from autograd import grad

import legume
from legume.backend import backend as bd
from legume.optimizers import adam_optimize
import skimage

parser = argparse.ArgumentParser()
parser.add_argument('--verbose', action='store_true')
parser.add_argument('--optimize', action='store_true')
parser.add_argument('-N_epochs', default=20, type=int)
parser.add_argument('-learning_rate', default=0.1, type=float)
parser.add_argument('-gmax', default=5, type=float)
parser.add_argument('-gmode_npts', default=2000, type=int)
parser.add_argument('-neig', default=10, type=int)
parser.add_argument('-N_polygons', default=20, type=int)
parser.add_argument('-H', default=0.33, type=float)
parser.add_argument('-D', default=0.20, type=float)
parser.add_argument('-eps', default=12, type=float)
args = parser.parse_args()

options = {'gmode_inds': np.arange(0, 8, 2),
		   'gmode_npts': args.gmode_npts,
		   'numeig': args.neig,
		   'verbose': args.verbose}

lattice = legume.Lattice('square')
path = lattice.bz_path(['G', 'M', 'X', 'G'], [10, 10, 10])

def projection(rho, eta=0.5, beta=100):
	return bd.divide(bd.tanh(beta * eta) + bd.tanh(beta * (rho - eta)), bd.tanh(beta * eta) + bd.tanh(beta * (1 - eta)))


def make_grating():
	phc = legume.PhotCryst(lattice)
	phc.add_layer(d=args.D, eps_b=args.eps)
	return legume.GuidedModeExp(phc, gmax=args.gmax)


def parameterize_density_layer(layer, rho, eta=0.5, beta=100):
	assert rho.shape[0] == rho.shape[1], "For now only square grids"
	N = rho.shape[0]

	X = np.linspace(-1.0, +1.0, N+1)
	Y = np.linspace(-1.0, +1.0, N+1)

	rho_proj = projection(rho, eta, beta)
	plt.figure()
	plt.imshow(rho_proj)
	plt.show()

	for i in range(len(X) - 1):
		x0 = X[i]
		x1 = X[i + 1]
		for j in range(len(Y) - 1):
			y0 = Y[i]
			y1 = Y[i + 1]

			eps = 1 + rho_proj[i,j] * (args.eps - 1)
			grating = legume.Poly(eps=eps,
								  x_edges=np.array([x0, x1, x1, x0]),
								  y_edges=np.array([y0, y0, y1, y1]))
			layer.add_shape(grating)


def objective(rho):
	parameterize_density_layer(gme.phc.layers[-1], rho, eta=0.5, beta=10)
	gme.run(kpoints=path.kpoints, **options)
	tgt_freqs = gme.freqs[0:10, 1]
	return bd.sqrt(bd.var(tgt_freqs))


N = args.N_polygons
# Initialize structure
(rr, cc) = skimage.draw.circle(N/2, N/2, N/5)
rho_0 = np.zeros((N,N))
rho_0[rr,cc] = 1
# plt.figure()
# plt.imshow(rho_0)

# Compute results for initial structure
legume.set_backend('numpy')
gme = make_grating()
parameterize_density_layer(gme.phc.layers[-1], rho_0, eta=0.5, beta=10)
# gme.run(kpoints=path.kpoints, **options)

gme.phc.plot_overview()
