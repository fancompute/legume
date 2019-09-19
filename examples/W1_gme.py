import argparse
from memory_profiler import profile

import numpy as np
import autograd.numpy as npa
from autograd import grad

import pygme
from pygme import GuidedModeExp, Circle, PhotCryst, Lattice

@profile
def W1(ra):
	Ny = 10
	# Initialize a lattice
	lattice = Lattice([1., 0], [0, Ny*np.sqrt(3)/2])
	# Initialize a PhC (by default with eps = 1 in upper and lower cladding, we set upper one to 5)
	phc = PhotCryst(lattice)
	# Add a layer to the PhC with thickness 1 and background permittivity 10
	phc.add_layer(d=0.5, eps_b=12.)
	# Add a shape to this layer 
	for ih in range(Ny):
		if ih != Ny//2:
			circ = Circle(x_cent=(ih%2)*0.5, y_cent = (-Ny//2 + ih)*np.sqrt(3)/2, r=ra)
			phc.layers[-1].add_shape(circ)

	if args.overview:
		phc.plot_overview(cladding='True', Ny=500)

	gme = GuidedModeExp(phc, gmax=3)
	path = phc.lattice.bz_path(['G', np.array([np.pi, 0])], [50])
	gme.run(kpoints=path.kpoints, gmode_inds=[0], verbose=False, numeig=20)
	return gme.freqs

@profile
def grad_freq():
	of = lambda r: W1(r)[0, 1]
	grad_test = grad(of)
	grad_test(0.3)

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--overview', action='store_true')
	parser.add_argument('-backend', default='numpy', type=str)
	parser.add_argument('-ra', default=0.3, type=float)
	args = parser.parse_args()

	pygme.set_backend(args.backend)

	W1(args.ra)

	if args.backend=='autograd':
		grad_freq()

