import numpy as np
import matplotlib.pyplot as plt
from pygme import utils as utils
from .guided_modes import guided_modes
from .backend import backend as bd
import time
from itertools import zip_longest
from functools import reduce

class GuidedModeExp(object):
	'''
	Main simulation class of the guided-mode expansion
	'''
	def __init__(self, phc, gmax=3):
		# Object of class Phc which will be simulated
		self.phc = phc
		# Number of layers
		self.N_layers = len(phc.layers)
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

	def run(self, gmode_inds, kpoints=np.array([[0], [0]]), N_g_array=100, 
				gmode_step=1e-1, verbose=True):
		t_start = time.time()
		def print_vb(*args):
			if verbose: print(*args)
		''' 
		Run the simulation. Basically:

		- Compute the guided modes over a grid of g-points
		- Compute the inverse matrix of the FT of the permittivity eps(G - G')
			in every phc layer, with G, G' reciprocal lattice vectors
		- Iterate over the k points:
			- compute the Hermitian matrix for diagonalization 
			- compute the real eigenvalues and corresponding eigenvectors
		'''

		# Bloch momenta over which band structure is imulated 
		self.kpoints = kpoints
		# Indexes of modes to be included in the expansion
		gmode_inds = np.array(gmode_inds)
		self.gmode_inds = gmode_inds
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
		(self.g_array, self.eps_array, self.d_array) = \
								(g_array, eps_array, d_array)

		# Compute guided modes
		t = time.time()
		self.gmode_te = gmode_inds[np.remainder(gmode_inds, 2) == 0]
		self.gmode_tm = gmode_inds[np.remainder(gmode_inds, 2) != 0]
		reshape_list = lambda x: [list(filter(lambda y: y is not None, i)) \
						for i in zip_longest(*x)]

		if self.gmode_te.size > 0:
			(omegas_te, coeffs_te) = guided_modes(g_array, eps_array, d_array, 
						step=gmode_step, n_modes=1 + np.amax(self.gmode_te)//2, 
						tol=1e-6, mode='TE')
			self.omegas_te = reshape_list(omegas_te)
			self.coeffs_te = reshape_list(coeffs_te)
		else:
			self.omegas_te = [[]]
			self.coeffs_te = [[]]

		if self.gmode_tm.size > 0:
			(omegas_tm, coeffs_tm) = guided_modes(g_array, eps_array, d_array, 
						step=gmode_step, n_modes=1 + np.amax(self.gmode_tm)//2, 
						tol=1e-6, mode='TM')
			self.omegas_tm = reshape_list(omegas_tm)
			self.coeffs_tm = reshape_list(coeffs_tm)
		else:
			self.omegas_tm = [[]]
			self.coeffs_tm = [[]]

		print_vb("%1.4f seconds for guided mode computation"% (time.time()-t)) 

		# Compute inverse matrix of FT of permittivity
		t = time.time()
		self.compute_ft()	# Just in case something changed after __init__()
		self.compute_eps_inv()
		print_vb("%1.4f seconds for inverse matrix of Fourier-space "\
			"permittivity"% (time.time()-t))

		# Loop over all k-points, construct the matrix and diagonalize
		freqs = []
		for ik, k in enumerate(kpoints.T):
			print_vb("Running k-point %d of %d" % (ik+1, kpoints.shape[1]))
			mat = self.construct_mat(k=k)

			# Diagonalize using numpy.linalg.eigh() for now; should maybe switch 
			# to scipy.sparse.linalg.eigsh() in the future
			# NB: we shift the matrix by np.eye to avoid problems at the zero-
			# frequency mode at Gamma
			(freq2, vec) = bd.eigh(mat + bd.eye(mat.shape[0]))
			freqs.append(bd.sort(bd.sqrt(
					bd.abs(freq2 - bd.ones(self.numeig)))))

		# Store the eigenfrequencies taking the standard reduced frequency 
		# convention for the units (2pi a/c)
		self.mat = mat	
		self.freqs = bd.array(freqs)/2/np.pi

		print_vb("%1.4f seconds total time to run"% (time.time()-t_start))

	def compute_eps_inv(self):
		'''
		Construct the inverse FT matrices for the permittivity in each layer
		'''
		for layer in self.phc.layers + self.phc.claddings:
			# For now we just use the numpy inversion. Later on we could 
			# implement the Toeplitz-Block-Toeplitz inversion (faster)
			eps_mat = bd.toeplitz_block(self.n2g, layer.T1, layer.T2)
			layer.eps_inv_mat = bd.inv(eps_mat)

	def construct_mat(self, k):
		'''
		Construct the Hermitian matrix for diagonalization for a given k
		'''

		# We only know how large the basis set will be at most
		# We will truncate the matrix later on but for now initialize like this
		max_basis = self.gvec.shape[1]*self.gmode_inds.size
		mat = bd.zeros((max_basis, max_basis), dtype=np.complex128)

		# Number of G points included for every mode
		self.modes_numg = []

		# G + k vectors
		gkx = self.gvec[0, :] + k[0]
		gky = self.gvec[1, :] + k[1]
		gk = np.sqrt(np.square(gkx + 1e-20) + np.square(gky))

		# Unit vectors in the propagation direction; we add a tiny component 
		# in the x-direction to avoid problems at gk = 0
		pkx = (gkx + 1e-20) / gk
		pky = gky / gk

		# Unit vectors in-plane orthogonal to the propagation direction
		qkx = -gky / gk
		qky = (gkx + 1e-20) / gk

		pp = np.outer(qkx, qkx) + np.outer(qky, qky)
		pq = np.outer(pkx, qkx) + np.outer(pky, qky)
		qq = np.outer(qkx, qkx) + np.outer(qky, qky)

		def interp_coeff(coeffs, il, ic, indmode, gs):
			'''
			Interpolate the A/B coefficient (ic = 0/1) in layer number il
			'''
			param_list = [coeffs[i][il, ic, 0] for i in range(len(coeffs))]
			return np.interp(gk[indmode], gs, np.array(param_list)).ravel()

		def interp_guided(im, omegas, coeffs):
			'''
			Interpolate all the relevant guided mode parameters over gk
			'''
			gs = self.g_array[-len(omegas[im]):]
			indmode = np.argwhere(gk > gs[0])
			oms = np.interp(gk[indmode], gs, omegas[im])

			As, Bs, chis = (np.zeros((self.N_layers + 2, 
					indmode.size), dtype=np.complex128) for i in range(3))

			for il in range(self.N_layers + 2):
				As[il, :] = interp_coeff(coeffs[im], il, 0, indmode, gs)
				Bs[il, :] = interp_coeff(coeffs[im], il, 1, indmode, gs)
				chis[il, :] = np.sqrt(self.eps_array[il]*oms**2 - 
								gk[indmode]**2, dtype=np.complex128).ravel()
			return (indmode, oms, As, Bs, chis)

		def get_guided(mode):
			'''
			Get all the guided mode parameters over 'gk' for mode number 'mode'
			Variable 'indmode' stores the indexes of 'gk' over which a guided
			mode solution was found
			'''
			if mode%2 == 0:
				im_te = np.argwhere(mode1==self.gmode_te)[0][0]
				(indmode, oms, As, Bs, chis) = interp_guided(
							im_te, self.omegas_te, self.coeffs_te)
			else:
				im_tm = np.argwhere(mode1==self.gmode_tm)[0][0]
				(indmode, oms, As, Bs, chis) = interp_guided(
							im_tm, self.omegas_tem, self.coeffs_tm)
			return (indmode, oms, As, Bs, chis)

		# Loop over modes and build the matrix block-by-block
		count1 = 0
		modes_numg = []
		for im1 in range(self.gmode_inds.size):
			mode1 = self.gmode_inds[im1]
			(indmode1, oms1, As1, Bs1, chis1) = get_guided(mode1)
			modes_numg.append(indmode1.size)
			count2 = 0

			for im2 in range(im1, self.gmode_inds.size):
				mode2 = self.gmode_inds[im2]
				(indmode2, oms2, As2, Bs2, chis2) = get_guided(mode2)
				# TE-TE
				if mode1%2 + mode2%2 == 0:
					mat_block = self.mat_te_te(k, gkx, gky, gk, indmode1, oms1,
									As1, Bs1, chis1, indmode2, oms2, As2, Bs2, 
									chis2, qq)
				mat[count1:count1+indmode1.size, 
					count2:count2+indmode2.size] = mat_block
				count2 += indmode2.size
			count1 += indmode1.size

		N_basis = count1 # this should also be equal to count 2 at this point
		# Change below if switching to a solver that allows for variable numeig
		self.numeig = N_basis
		# Store a list of how many g-points were used for each mode index
		self.modes_numg.append(modes_numg) 

		# Take only the filled part of the matrix
		mat = mat[0:N_basis, 0:N_basis]

		'''
		If the matrix is within numerical precision to real symmetric, 
		make it explicitly so. This will speed up the diagonalization and will
		often be the case, specifically when there is in-plane inversion
		symmetry in the PhC elementary cell
		'''
		if bd.amax(bd.abs(bd.imag(mat))) < 1e-10*bd.amax(bd.abs(bd.real(mat))):
			mat = bd.real(mat)

		'''
		Make the matrix Hermitian (note that only upper part of the blocks, i.e.
		(im2 >= im1) was computed
		'''
		mat = bd.triu(mat) + bd.transpose(bd.conj(bd.triu(mat, 1)))  

		return mat


	'''===========MATRIX ELEMENTS BETWEEN GUIDED MODES BELOW============'''

	def mat_te_te(self, k, gkx, gky, gk, indmode1, oms1,
						As1, Bs1, chis1, indmode2, oms2, As2, Bs2, 
						chis2, qq):
		'''
		Matrix block for TE-TE mode coupling
		Notation is following Vasily Zabelin's thesis
		not Andreani and Gerace PRB 2006
		'''
		
		# Contribution from lower cladding
		# NB: therer might be a better way than this (ind1, ind2) below
		(ind1, ind2) = np.meshgrid(indmode2, indmode1)
		mat = self.phc.claddings[0].eps_inv_mat[ind2, ind1]* \
				self.phc.claddings[0].eps_avg**2 / \
				np.outer(np.conj(chis1[0, :]), chis2[0, :]) * \
				np.outer(np.conj(Bs1[0, :]), Bs2[0, :]) * \
				1j / (chis2[0, :] - np.conj(chis1[0, :][:, np.newaxis]))
		# NB: Check sign of the J function in the thesis
		# raise Exception 

		# Contribution from upper cladding
		mat = mat + self.phc.claddings[1].eps_inv_mat[ind2, ind1]* \
				self.phc.claddings[1].eps_avg**2 / \
				np.outer(np.conj(chis1[-1, :]), chis2[-1, :]) * \
				np.outer(np.conj(As1[-1, :]), As2[-1, :]) * \
				1j / (chis2[-1, :] - np.conj(chis1[-1, :][:, np.newaxis]))

		# raise Exception

		# Contributions from layers
		def I(il, alpha): 
			return -1j/(alpha + 1e-20)*(np.exp(1j*alpha*self.d_array[il-1]) - 1)

		# note: self.N_layers = self.phc.layers.shape so without claddings
		for il in range(1, self.N_layers+1):
			mat = mat + self.phc.layers[il-1].eps_inv_mat[ind2, ind1] *\
			self.phc.layers[il-1].eps_avg**2 / \
			np.outer(np.conj(chis1[il, :]), chis2[il, :]) * \
			(np.outer(np.conj(As1[il, :]), As2[il, :]) * \
			I(il, (chis2[il, :] - np.conj(chis1[il, :][:, np.newaxis]))) + \
			np.outer(np.conj(Bs1[il, :]), Bs2[il, :]) * \
			I(il, (-chis2[il, :] + np.conj(chis1[il, :][:, np.newaxis]))) -
			np.outer(np.conj(As1[il, :]), Bs2[il, :]) * \
			I(il, (-chis2[il, :] - np.conj(chis1[il, :][:, np.newaxis]))) -
			np.outer(np.conj(Bs1[il, :]), As2[il, :]) * \
			I(il, (chis2[il, :] + np.conj(chis1[il, :][:, np.newaxis])))  )

		# Final pre-factor		
		mat = mat * np.outer(oms1**2, oms2**2) * (qq[ind2, ind1])

		raise Exception
		return mat