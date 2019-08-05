import numpy as np
import matplotlib.pyplot as plt
import pygme.utils as utils
from .guided_modes import guided_modes
from .backend import backend as bd

class GuidedModeExp(object):
	'''
	Main simulation class of the guided-mode expansion
	'''
	def __init__(self, phc, gmax=3):
		# Object of class Phc which will be simulated
		self.phc = phc
		# Maximum reciprocal lattice wave-vector length in units of 2pi/a
		self.gmax = gmax

		# Initialize the reciprocal lattice vectors and compute the FT of all
		# the layers of the PhC
		self._init_reciprocal()
		self.compute_ft()

	def _init_reciprocal(self):
		'''
		Initialize reciprocal lattice vectors based on self.phc and self.gmax
		'''
		n1max = np.int_((2*np.pi*self.gmax)/np.linalg.norm(self.phc.lattice.b1))
		n2max = np.int_((2*np.pi*self.gmax)/np.linalg.norm(self.phc.lattice.b2))

		# This constructs the reciprocal lattice in a way that is suitable
		# for Toeplitz-Block-Toeplitz inversion of the permittivity in the main
		# code. However, one caveat is that the hexagonal lattice symmetry is 
		# not preserved. For that, the option to construct a hexagonal mesh in 
		# reciprocal space could is needed.
		inds1 = np.tile(np.arange(-n1max, n1max + 1), (2*n2max + 1, 1))  \
						 .reshape((2*n2max + 1)*(2*n1max + 1), order='F')
		inds2 = np.tile(np.arange(-n2max, n2max + 1), 2*n1max + 1)

		gvec = self.phc.lattice.b1[:, np.newaxis].dot(inds1[np.newaxis, :]) + \
				self.phc.lattice.b2[:, np.newaxis].dot(inds2[np.newaxis, :])

		# Save the reciprocal lattice vectors
		self.gvec = gvec

		# Save the number of vectors along the b1 and the b2 directions 
		# Note: gvec.shape[1] = n1g*n2g
		self.n1g = 2*n1max + 1
		self.n2g = 2*n2max + 1

	def compute_ft(self):
		'''
		Compute the unique FT coefficients of the permittivity, eps(g-g') for
		every layer in the PhC.
		'''
		(n1max, n2max) = (self.n1g, self.n2g)
		G1 = self.gvec - self.gvec[:, [0]]
		G2 = np.zeros((2, n1max*n2max))

		for ind1 in range(n1max):
			G2[:, ind1*n2max:(ind1+1)*n2max] = self.gvec[:, [ind1*n2max]] - \
							self.gvec[:, range(n2max)]

		for layer in self.phc.layers + self.phc.claddings:
			T1 = bd.zeros(self.gvec.shape[1])
			T2 = bd.zeros(self.gvec.shape[1])
			for shape in layer.shapes:
				# Note: compute_ft() returns the FT of a function that is one 
				# inside the shape and zero outside
				T1 = T1 + (shape.eps - layer.eps_b)*shape.compute_ft(G1)
				T2 = T2 + (shape.eps - layer.eps_b)*shape.compute_ft(G2)

			# Apply some final coefficients
			# Note the hacky way to set the zero element so as to work with
			# 'autograd' backend
			ind0 = bd.arange(T1.size) < 1  
			T1 = T1 / layer.lattice.ec_area
			T1 = T1*(1-ind0) + layer.eps_avg*ind0
			T2 = T2 / layer.lattice.ec_area
			T2 = T2*(1-ind0) + layer.eps_avg*ind0

			# Store T1 and T2
			layer.T1 = T1
			layer.T2 = T2

		# Store the g-vectors to which T1 and T2 correspond
		self.G1 = G1
		self.G2 = G2

	def plot_overview_ft(self, Nx=100, Ny=100, cladding=False):
		'''
		Plot the permittivity of the PhC cross-sections as computed from an 
		inverse Fourier transform with the GME reciprocal lattice vectors.
		'''
		(xgrid, ygrid) = self.phc.lattice.xy_grid(Nx=Nx, Ny=Ny)

		if cladding:
			all_layers = [self.phc.claddings[0]] + self.phc.layers + \
							[self.phc.claddings[1]]
		else:
			all_layers = self.phc.layers
		N_layers = len(all_layers)

		fig, ax = plt.subplots(1, N_layers, constrained_layout=True)
		if N_layers==1: ax=[ax]

		(eps_min, eps_max) = (all_layers[0].eps_b, all_layers[0].eps_b)
		ims = []
		for (indl, layer) in enumerate(all_layers):
			ft_coeffs = np.hstack((layer.T1, layer.T2, 
								np.conj(layer.T1), np.conj(layer.T2)))
			gvec = np.hstack((self.G1, self.G2, 
								-self.G1, -self.G2))

			eps_r = utils.ftinv(ft_coeffs, gvec, xgrid, ygrid)
			eps_min = min([eps_min, np.amin(np.real(eps_r))])
			eps_max = max([eps_max, np.amax(np.real(eps_r))])
			extent = [xgrid[0], xgrid[-1], ygrid[0], ygrid[-1]]

			im = utils.plot_eps(np.real(eps_r), ax=ax[indl], extent=extent, 
							cbar=False)
			ims.append(im)
			if cladding:
				if indl > 0 and indl < N_layers-1:
					ax[indl].set_title("xy in layer %d" % indl)
				elif indl==N_layers-1:
					ax[0].set_title("xy in lower cladding")
					ax[-1].set_title("xy in upper cladding")
			else:
				ax[indl].set_title("xy in layer %d" % indl)
		
		for il in range(N_layers):
			ims[il].set_clim(vmin=eps_min, vmax=eps_max)
		plt.colorbar(ims[-1])
		plt.show()

	def run(self, kpoints=np.array([[0], [0]]), gmode_inds=[0], N_g_array=100, 
				gmode_step=1e-1):
		''' 
		Run the simulation. Basically:

		- Compute the guided modes over a grid of g-points
		- Compute the inverse matrix of the FT of the permittivity eps(G - G')
			in every phc layer, with G, G' reciprocal lattice vectors
		- Iterate over the k points:
			- interpolate the guided mode frequency and coefficient for every 
				(G + k) 
			- compute the matrix for diagonalization and eigenvalues
			- compute imaginary part of eigenvalues perturbatively
		'''

		# Bloch momenta over which band structure is imulated 
		self.kpoints = kpoints
		# Indexes of modes to be included in the expansion
		gmode_inds = np.array(gmode_inds)
		self.gmode_inds = gmode_inds
		# Change this if switching to a solver that allows for variable numeig
		self.numeig = self.gvec.shape[1]
		# Number of points over which guided modes are computed
		self.N_g_array = N_g_array
		# Step in frequency in the search for guided mode solutions
		self.gmode_step = gmode_step

		kmax = np.amax(np.sqrt(np.square(kpoints[0, :]) +
							np.square(kpoints[1, :])))
		Gmax = np.amax(np.sqrt(np.square(self.gvec[0, :]) +
							np.square(self.gvec[1, :])))
		# Array of g-points over which the guided modes will be computed
		g_array = np.linspace(1e-4, Gmax + kmax, N_g_array)
		# Array of average permittivity of every layer (including claddings)
		eps_array = np.array(list(layer.eps_avg for layer in \
			[self.phc.claddings[0]] + self.phc.layers + 
			[self.phc.claddings[1]]), dtype=np.float64)
		# Array of thickness of every layer (not including claddings)
		d_array = np.array(list(layer.d for layer in self.phc.layers), 
							dtype=np.float64)

		# Compute guided modes
		self.gmode_te = gmode_inds[np.remainder(gmode_inds, 2) == 0]
		self.gmode_tm = gmode_inds[np.remainder(gmode_inds, 2) != 0]

		if self.gmode_te.size > 0:
			(omegas_te, coeffs_te) = guided_modes(g_array, eps_array, d_array, 
						step=gmode_step, n_modes=1 + np.amax(self.gmode_te)//2, 
						tol=1e-6, mode='TE')
		if self.gmode_tm.size > 0:
			(omegas_tm, coeffs_tm) = guided_modes(g_array, eps_array, d_array, 
						step=gmode_step, n_modes=1 + np.amax(self.gmode_tm)//2, 
						tol=1e-6, mode='TM')
		# print(Gmax, self.gvec.shape)
		print(omegas_te, coeffs_te, coeffs_te[0][0].shape) 
		# print(np.array(omegas_tm).shape, np.array(coeffs_tm).shape) 

		self.compute_ft()	# Just in case something changed after __init__()
		self.compute_eps_inv()


	def compute_eps_inv(self):
		'''
		Construct the inverse FT matrices for the permittivity in each layer
		'''
		for layer in self.phc.layers:
			# For now we just use the numpy inversion. Later on we could 
			# implement the Toeplitz-Block-Toeplitz inversion (faster)
			eps_mat = bd.toeplitz_block(self.n2g, layer.T1, layer.T2)
			layer.eps_inv_mat = bd.inv(eps_mat)
		