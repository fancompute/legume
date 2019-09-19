import numpy as np
import matplotlib.pyplot as plt
import pygme.utils as utils
from .slab_modes import guided_modes, rad_modes
from .backend import backend as bd
import time
from itertools import zip_longest
from functools import reduce
from pygme.utils import I_alpha, J_alpha, get_value

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

		# Number of G points included for every mode, will be defined after run
		self.modes_numg = []
		# Total number of basis vectors (equal to np.sum(self.modes_numg))
		self.N_basis = []

		# Eigenfrequencies and eigenvectors
		self.freqs = []
		self.eigvecs = []

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

	def _run_options(self, options):
		default_options = {
			# Indexes of modes to be included in the expansion
			'gmode_inds'   : [0],

			# Number of points over which guided modes are computed
			'gmode_npts'   : 1000,

			# Step in frequency in the search for guided mode solutions
			'gmode_step'   : 1e-2,

			# Tolerance in the minimum and maximum omega value when looking for 
			# the guided-mode solutions
			'gmode_tol'    : 1e-10,

			# Number of eigen-frequencies to be stored (starting from lowest)
			'numeig'	   : 10,

			# Print information at intermmediate steps
			'verbose'	   : True
			}

		for key in default_options.keys():
			if key not in options.keys():
				options[key] = default_options[key]

		for key in options.keys():
			if key not in default_options.keys():
				raise ValueError("Unknown run() argument '%s'" % key)

		for (option, value) in options.items():
			# Make sure 'gmode_inds' is a numpy array
			if option.lower() == 'gmode_inds':
				value = np.array(value)
			# Set all the options as class attributes
			setattr(self, option, value)

	def _get_guided(self, gk, mode):
		'''
		Get all the guided mode parameters over 'gk' for mode number 'mode'
		Variable 'indmode' stores the indexes of 'gk' over which a guided
		mode solution was found
		'''

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
			indmode = np.argwhere(gk > gs[0]).ravel()
			oms = np.interp(gk[indmode], gs, omegas[im])
			# prit('k=' )

			As, Bs, chis = (np.zeros((self.N_layers + 2, 
					indmode.size), dtype=np.complex128) for i in range(3))

			for il in range(self.N_layers + 2):
				As[il, :] = interp_coeff(coeffs[im], il, 0, indmode, gs)
				Bs[il, :] = interp_coeff(coeffs[im], il, 1, indmode, gs)
				chis[il, :] = np.sqrt(self.eps_array[il]*oms**2 - 
								gk[indmode]**2, dtype=np.complex128).ravel()
			return (indmode, oms, As, Bs, chis)

		if mode%2 == 0:
			(indmode, oms, As, Bs, chis) = interp_guided(
						mode//2, self.omegas_te, self.coeffs_te)
		else:
			(indmode, oms, As, Bs, chis) = interp_guided(
						mode//2, self.omegas_tm, self.coeffs_tm)
		return (indmode, oms, As, Bs, chis)

	def _get_rad(self, gkr, omr, pol, clad):
		'''
		Get all the radiative mode parameters over 'gkr' at frequency 'omr' with
		polarization 'pol' and out-going in cladding 'clad'
		'''
		chis = np.zeros((self.N_layers + 2, gkr.size), dtype=np.complex128)
		for il in range(self.N_layers + 2):
			chis[il, :] = np.sqrt(self.eps_array[il]*omr**2 - 
							gkr**2, dtype=np.complex128).ravel()
		(Xs, Ys) = rad_modes(omr, gkr, self.eps_array, self.d_array, pol, clad)

		# raise Exception
		
		return (Xs, Ys, chis)

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
			T1 = layer.compute_ft(G1)
			T2 = layer.compute_ft(G2)

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

	def run(self, kpoints=np.array([[0], [0]]), **kwargs):
		''' 
		Run the simulation. Basically:

		- Compute the guided modes over a grid of g-points
		- Compute the inverse matrix of the FT of the permittivity eps(G - G')
			in every phc layer, with G, G' reciprocal lattice vectors
		- Iterate over the k points:
			- compute the Hermitian matrix for diagonalization 
			- compute the real eigenvalues and corresponding eigenvectors
		'''
		t_start = time.time()
		
		def print_vb(*args):
			if self.verbose: print(*args)

		# Parse the input arguments
		self._run_options(kwargs)
		# Bloch momenta over which band structure is simulated 
		self.kpoints = kpoints

		kmax = np.amax(np.sqrt(np.square(kpoints[0, :]) +
							np.square(kpoints[1, :])))
		Gmax = np.amax(np.sqrt(np.square(self.gvec[0, :]) +
							np.square(self.gvec[1, :])))
		# Array of g-points over which the guided modes will be computed
		g_array = np.linspace(1e-3, Gmax + kmax, self.gmode_npts)
		# Array of average permittivity of every layer (including claddings)
		eps_array = np.array(list(get_value(layer.eps_avg) for layer in \
			[self.phc.claddings[0]] + self.phc.layers + 
			[self.phc.claddings[1]]), dtype=np.float64)
		# Array of thickness of every layer (not including claddings)
		d_array = np.array(list(get_value(layer.d) for layer in \
			self.phc.layers), dtype=np.float64)
		(self.g_array, self.eps_array, self.d_array) = \
								(g_array, eps_array, d_array)

		# Compute guided modes
		t = time.time()
		self.gmode_te = self.gmode_inds[np.remainder(self.gmode_inds, 2) == 0]
		self.gmode_tm = self.gmode_inds[np.remainder(self.gmode_inds, 2) != 0]
		reshape_list = lambda x: [list(filter(lambda y: y is not None, i)) \
						for i in zip_longest(*x)]

		if self.gmode_te.size > 0:
			(omegas_te, coeffs_te) = guided_modes(g_array, eps_array, d_array, 
					step=self.gmode_step, n_modes=1 + np.amax(self.gmode_te)//2, 
					tol=self.gmode_tol, pol='TE')
			self.omegas_te = reshape_list(omegas_te)
			self.coeffs_te = reshape_list(coeffs_te)
		else:
			self.omegas_te = [[]]
			self.coeffs_te = [[]]

		if self.gmode_tm.size > 0:
			(omegas_tm, coeffs_tm) = guided_modes(g_array, eps_array, d_array, 
					step=self.gmode_step, n_modes=1 + np.amax(self.gmode_tm)//2, 
					tol=self.gmode_tol, pol='TM')
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
		print_vb("%1.4f seconds for inverse matrix of Fourier-space"
			"permittivity"% (time.time()-t))

		# Loop over all k-points, construct the matrix, diagonalize, and compute
		# radiative losses for the modes requested by kinds_rad and minds_rad
		t_rad = 0
		freqs = []
		freqs_im = []
		eigvecs = []
		for ik, k in enumerate(kpoints.T):
			print_vb("Running k-point %d of %d" % (ik+1, kpoints.shape[1]))
			mat = self.construct_mat(k=k)
			if self.numeig > mat.shape[0]:
				raise ValueError("Requested number of eigenvalues 'numeig' "
					"larger than total size of basis set. Reduce 'numeig' or"
					"increase 'gmax'. ")

			# Diagonalize using numpy.linalg.eigh() for now; should maybe switch 
			# to scipy.sparse.linalg.eigsh() in the future
			# NB: we shift the matrix by np.eye to avoid problems at the zero-
			# frequency mode at Gamma
			(freq2, evec) = bd.eigh(mat + bd.eye(mat.shape[0]))
			freq = bd.sort(bd.sqrt(bd.abs(freq2[:self.numeig]
						- bd.ones(self.numeig))))
			freqs.append(freq)
			eigvecs.append(evec[:, :self.numeig])
			# raise Exception

		# Store the eigenfrequencies taking the standard reduced frequency 
		# convention for the units (2pi a/c)
		self.freqs = bd.array(freqs)/2/np.pi
		self.eigvecs = eigvecs

		print_vb("%1.4f seconds total time to run"% (time.time()-t_start))

	def compute_eps_inv(self):
		'''
		Construct the inverse FT matrices for the permittivity in each layer
		'''
		for layer in self.phc.layers + self.phc.claddings:
			# For now we just use the numpy inversion. Later on we could 
			# implement the Toeplitz-Block-Toeplitz inversion (faster)
			eps_mat = bd.toeplitz_block(self.n1g, layer.T1, layer.T2)
			layer.eps_inv_mat = bd.inv(eps_mat)

	def construct_mat(self, k):
		'''
		Construct the Hermitian matrix for diagonalization for a given k
		'''

		# We will construct the matrix block by block
		mat_blocks = [[] for i in range(self.gmode_inds.size)]

		# G + k vectors
		gkx = self.gvec[0, :] + k[0]
		gky = self.gvec[1, :] + k[1]
		gk = np.sqrt(np.square(gkx + 1e-20) + np.square(gky))

		# Unit vectors in the propagation direction; we add a tiny component 
		# in the x-direction to avoid problems at gk = 0
		pkx = (gkx + 1e-20) / gk
		pky = gky / gk

		# Unit vectors in-plane orthogonal to the propagation direction
		qkx = gky / gk
		qky = -(gkx + 1e-20) / gk

		pp = np.outer(pkx, pkx) + np.outer(pky, pky)
		pq = np.outer(pkx, qkx) + np.outer(pky, qky)
		qq = np.outer(qkx, qkx) + np.outer(qky, qky)

		# Loop over modes and build the matrix block-by-block
		modes_numg = []
		for im1 in range(self.gmode_inds.size):
			mode1 = self.gmode_inds[im1]
			(indmode1, oms1, As1, Bs1, chis1) = self._get_guided(gk, mode1)
			modes_numg.append(indmode1.size)

			if len(modes_numg) > 1:
				mat_blocks[im1].append(bd.zeros((modes_numg[-1], 
					bd.sum(modes_numg[:-1]))))

			for im2 in range(im1, self.gmode_inds.size):
				mode2 = self.gmode_inds[im2]
				(indmode2, oms2, As2, Bs2, chis2) = self._get_guided(gk, mode2)

				if mode1%2 + mode2%2 == 0:
					mat_block = self.mat_te_te(indmode1, oms1,
									As1, Bs1, chis1, indmode2, oms2, As2, Bs2, 
									chis2, qq)
				elif mode1%2 + mode2%2 == 2:
					mat_block = self.mat_tm_tm(gk, indmode1, oms1,
									As1, Bs1, chis1, indmode2, oms2, As2, Bs2, 
									chis2, pp)
				elif mode1%2==0 and mode2%2==1:
					mat_block = self.mat_te_tm(indmode1, oms1,
									As1, Bs1, chis1, indmode2, oms2, As2, Bs2, 
									chis2, pq.transpose(), 1j)
				elif mode1%2==1 and mode2%2==0:
					# Note: TM-TE is just hermitian conjugate of TE-TM
					# with switched indexes 1 <-> 2
					mat_block = self.mat_te_tm(indmode2, oms2,
									As2, Bs2, chis2, indmode1, oms1, As1, Bs1, 
									chis1, pq, -1j) 
					mat_block = np.conj(np.transpose(mat_block))

				mat_blocks[im1].append(mat_block)

		# Store how many modes total were included in the matrix
		self.N_basis.append(np.sum(modes_numg))
		# Store a list of how many g-points were used for each mode index
		self.modes_numg.append(modes_numg) 

		# Stack all the blocks together
		mat_rows = [bd.hstack(mb) for mb in mat_blocks]
		mat = bd.vstack(mat_rows)

		# raise Exception

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
		self.mat = mat  

		return mat

	def compute_rad(self, kind, minds=[0]):
		'''
		Compute the radiation losses of the eigenmodes after the dispersion
		has been computed.
		Input
			kind 		    : index of the k-point for the computation
			minds           : indexes of which modes to be computed 
							  (max value must be smaller than self.numeig)
		Output
			freqs_im		: imaginary part of the frequencies 
		'''
		if self.freqs == []:
			raise RuntimeError("Run the GME computation first!")
		if np.max(np.array(minds)) > self.numeig - 1:
			raise ValueError("Requested mode index out of range for the %d "
				"stored eigenmodes" % self.numeig)
		
		# G + k vectors
		gkx = self.gvec[0, :] + self.kpoints[0, kind]
		gky = self.gvec[1, :] + self.kpoints[1, kind]
		gk = np.sqrt(np.square(gkx + 1e-20) + np.square(gky))

		# Iterate over all the modes to be computed
		rad_tot = []
		(coup_l, coup_u) = ([], [])
		for im in minds:
			omr = 2*np.pi*self.freqs[kind, im]
			evec = self.eigvecs[kind][:, im]

			# Reciprocal vedctors within the radiative cone for the claddings
			indmoder = [np.argwhere(gk**2 <= \
					self.phc.claddings[0].eps_avg*omr**2).ravel(), 
						np.argwhere(gk**2 <= \
					self.phc.claddings[1].eps_avg*omr**2).ravel()
						]
			gkr = [gk[indmode] for indmode in indmoder]
			rad_coup = {'te': [np.zeros((indmode.size, ), dtype=np.complex128) 
							for indmode in indmoder],
						'tm': [np.zeros((indmode.size, ), dtype=np.complex128) 
							for indmode in indmoder]}

			[Xs, Ys, chis] = [{'te': [], 'tm': []} for i in range(3)]
			for clad_ind in [0, 1]:
				for pol in ['te', 'tm']:
					(X, Y, chi) = self._get_rad(gkr[clad_ind], omr, 
							pol=pol, clad=clad_ind)
					Xs[pol].append(X)
					Ys[pol].append(Y)
					chis[pol].append(chi)
			# Iterate over the 'gmode_inds' basis of the PhC mode
			count = 0
			for im1 in range(self.gmode_inds.size):
				mode1 = self.gmode_inds[im1]
				(indmode1, oms1, As1, Bs1, chis1) = self._get_guided(gk, mode1)
				# Iterate over lower cladding (0) and upper cladding (1)
				for clad_ind in [0, 1]:			
					# Radiation to TE-polarized states
					if mode1%2 == 0:
						qq = (np.outer(gkx[indmode1], gkx[indmoder[clad_ind]])
							+ np.outer(gky[indmode1], gky[indmoder[clad_ind]]))\
							/ np.outer(gk[indmode1], gk[indmoder[clad_ind]])
						rad = self.rad_te_te(indmode1, oms1, As1, Bs1, chis1, 
							indmoder[clad_ind], omr, Xs['te'][clad_ind], 
							Ys['te'][clad_ind], chis['te'][clad_ind], qq)
					else:
						rad = self.rad_tm_te(indmode1, oms1, As1, Bs1, chis1, 
							indmoder[clad_ind], omr, Xs['te'][clad_ind], 
							Ys['te'][clad_ind], chis['te'][clad_ind])
					# raise Exception
					rad = rad*bd.conj(evec[count:
						count+self.modes_numg[kind][im1]][:, np.newaxis])
					rad_coup['te'][clad_ind] += bd.sum(rad, axis=0)

					# Radiation to TM-polarized states
					if mode1%2 == 0:
						qp = (np.outer(gky[indmode1], gkx[indmoder[clad_ind]])
							- np.outer(gkx[indmode1], gky[indmoder[clad_ind]]))\
							/ np.outer(gk[indmode1], gk[indmoder[clad_ind]])
						rad = self.rad_te_tm(indmode1, oms1, As1, Bs1, chis1, 
							indmoder[clad_ind], omr, Xs['tm'][clad_ind], 
							Ys['tm'][clad_ind], chis['tm'][clad_ind], qp)
					else:
						rad = self.rad_tm_tm(indmode1, oms1, As1, Bs1, chis1, 
							indmoder[clad_ind], omr, Xs['te'][clad_ind], 
							Ys['te'][clad_ind], chis['te'][clad_ind])
					
					rad = rad*bd.conj(evec[count:
						count+self.modes_numg[kind][im1]][:, np.newaxis])
					rad_coup['tm'][clad_ind] += bd.sum(rad, axis=0)
				count += self.modes_numg[kind][im1]
				
				rad_dos = [self.phc.claddings[i].eps_avg/bd.sqrt(
						self.phc.claddings[i].eps_avg*omr**2 - gkr[i]**2) / \
						4 / np.pi for i in [0, 1]]
				# raise Exception
				rad_t = 0
				(c_l, c_u) = ({}, {})
				for pol in ['te', 'tm']:
					c_l[pol] = rad_coup[pol][0]
					c_u[pol] = rad_coup[pol][1]
					rad_t = rad_t + \
						np.pi*bd.sum(bd.square(bd.abs(c_l[pol]))*rad_dos[0]) + \
						np.pi*bd.sum(bd.square(bd.abs(c_u[pol]))*rad_dos[1])
				rad_tot.append(bd.imag(bd.sqrt(omr**2 + 1j*rad_t)))
				# raise Exception
				# NB: think about the normalization of the couplings!
				coup_l.append(c_l)
				coup_u.append(c_u)
				
		# Finally compute radiation rate in units of frequency	
		freqs_im = bd.array(rad_tot)/2/np.pi
		return (freqs_im, coup_l, coup_u)


	'''===========MATRIX ELEMENTS BETWEEN SLAB MODES BELOW============'''
	# Notation is following Andreani and Gerace PRB 2006

	def mat_te_te(self, indmode1, oms1,
						As1, Bs1, chis1, indmode2, oms2, As2, Bs2, 
						chis2, qq):
		'''
		Matrix block for TE-TE mode coupling
		'''
		
		# Contribution from lower cladding
		indmat = np.ix_(indmode1, indmode2)
		mat = self.phc.claddings[0].eps_inv_mat[indmat]* \
				self.phc.claddings[0].eps_avg**2 * \
				np.outer(np.conj(Bs1[0, :]), Bs2[0, :]) * \
				J_alpha(chis2[0, :] - np.conj(chis1[0, :][:, np.newaxis]))

		# Contribution from upper cladding
		mat = mat + self.phc.claddings[1].eps_inv_mat[indmat]* \
				self.phc.claddings[1].eps_avg**2 * \
				np.outer(np.conj(As1[-1, :]), As2[-1, :]) * \
				J_alpha(chis2[-1, :] - np.conj(chis1[-1, :][:, np.newaxis]))

		# Contributions from layers
		# note: self.N_layers = self.phc.layers.shape so without claddings
		for il in range(1, self.N_layers+1):
			mat = mat + self.phc.layers[il-1].eps_inv_mat[indmat] *\
			self.phc.layers[il-1].eps_avg**2 * ( \
			np.outer(np.conj(As1[il, :]), As2[il, :])*I_alpha(chis2[il, :] -\
				np.conj(chis1[il, :][:, np.newaxis]), self.d_array[il-1]) + \
			np.outer(np.conj(Bs1[il, :]), Bs2[il, :])*I_alpha(-chis2[il, :] +\
				np.conj(chis1[il, :][:, np.newaxis]), self.d_array[il-1]) + \
			np.outer(np.conj(As1[il, :]), Bs2[il, :])*I_alpha(-chis2[il, :] -\
				np.conj(chis1[il, :][:, np.newaxis]), self.d_array[il-1]) +
			np.outer(np.conj(Bs1[il, :]), As2[il, :])*I_alpha(chis2[il, :] +\
				np.conj(chis1[il, :][:, np.newaxis]), self.d_array[il-1])  )

		# Final pre-factor		
		mat = mat * np.outer(oms1**2, oms2**2) * (qq[indmat])

		return mat

	def mat_tm_tm(self, gk, indmode1, oms1,
						As1, Bs1, chis1, indmode2, oms2, As2, Bs2, 
						chis2, pp):
		'''
		Matrix block for TM-TM mode coupling
		'''
		
		# Contribution from lower cladding
		indmat = np.ix_(indmode1, indmode2)
		mat = self.phc.claddings[0].eps_inv_mat[indmat]*(pp[indmat] * \
				np.outer(np.conj(chis1[0, :]), chis2[0, :]) + \
				np.outer(gk[indmode1], gk[indmode2])) * \
				np.outer(np.conj(Bs1[0, :]), Bs2[0, :]) * \
				J_alpha(chis2[0, :] - np.conj(chis1[0, :][:, np.newaxis]))

		# Contribution from upper cladding
		mat = mat + self.phc.claddings[1].eps_inv_mat[indmat]*(pp[indmat] * \
				np.outer(np.conj(chis1[-1, :]), chis2[-1, :]) + \
				np.outer(gk[indmode1], gk[indmode2])) * \
				np.outer(np.conj(As1[-1, :]), As2[-1, :]) * \
				J_alpha(chis2[-1, :] - np.conj(chis1[-1, :][:, np.newaxis]))

		# Contributions from layers
		# note: self.N_layers = self.phc.layers.shape so without claddings
		for il in range(1, self.N_layers+1):
			mat = mat + self.phc.layers[il-1].eps_inv_mat[indmat]*( \
			(pp[indmat] * np.outer(np.conj(chis1[il, :]), chis2[il, :]) + \
				np.outer(gk[indmode1], gk[indmode2])) * ( \
			np.outer(np.conj(As1[il, :]), As2[il, :])*I_alpha(chis2[il, :] -\
				np.conj(chis1[il, :][:, np.newaxis]), self.d_array[il-1]) + \
			np.outer(np.conj(Bs1[il, :]), Bs2[il, :])*I_alpha(-chis2[il, :] +\
				np.conj(chis1[il, :][:, np.newaxis]), self.d_array[il-1]) ) - \
			(pp[indmat] * np.outer(np.conj(chis1[il, :]), chis2[il, :]) - \
				np.outer(gk[indmode1], gk[indmode2])) * ( \
			np.outer(np.conj(As1[il, :]), Bs2[il, :])*I_alpha(-chis2[il, :] -\
				np.conj(chis1[il, :][:, np.newaxis]), self.d_array[il-1]) +
			np.outer(np.conj(Bs1[il, :]), As2[il, :])*I_alpha(chis2[il, :] +\
				np.conj(chis1[il, :][:, np.newaxis]), self.d_array[il-1]))  )
		# Note: in Vitaly's thesis, there's a typo on line 3 of eq. (3.41), 
		# the term in brackets should be A*B*I + B*A*I instead of minus

		return mat

	def mat_te_tm(self, indmode1, oms1,
						As1, Bs1, chis1, indmode2, oms2, As2, Bs2, 
						chis2, qp, signed_1j):
		'''
		Matrix block for TM-TE mode coupling
		'''
		
		# Contribution from lower cladding
		indmat = np.ix_(indmode1, indmode2)
		mat = - self.phc.claddings[0].eps_inv_mat[indmat] * \
				self.phc.claddings[0].eps_avg * chis2[0, :][np.newaxis, :]* \
				np.outer(np.conj(Bs1[0, :]), Bs2[0, :]) * \
				J_alpha(chis2[0, :] - np.conj(chis1[0, :][:, np.newaxis]))

		# Contribution from upper cladding
		mat = mat + self.phc.claddings[1].eps_inv_mat[indmat] * \
				self.phc.claddings[1].eps_avg * chis2[-1, :][np.newaxis, :] * \
				np.outer(np.conj(As1[-1, :]), As2[-1, :]) * \
				J_alpha(chis2[-1, :] - np.conj(chis1[-1, :][:, np.newaxis]))

		# Contributions from layers
		# note: self.N_layers = self.phc.layers.shape so without claddings
		for il in range(1, self.N_layers+1):
			mat = mat + signed_1j * self.phc.layers[il-1].eps_inv_mat[indmat] *\
			self.phc.layers[il-1].eps_avg * chis2[il, :][np.newaxis, :] * ( \
			np.outer(np.conj(As1[il, :]), As2[il, :])*I_alpha(chis2[il, :] -\
				np.conj(chis1[il, :][:, np.newaxis]), self.d_array[il-1]) - \
			np.outer(np.conj(Bs1[il, :]), Bs2[il, :])*I_alpha(-chis2[il, :] +\
				np.conj(chis1[il, :][:, np.newaxis]), self.d_array[il-1]) - \
			np.outer(np.conj(As1[il, :]), Bs2[il, :])*I_alpha(-chis2[il, :] -\
				np.conj(chis1[il, :][:, np.newaxis]), self.d_array[il-1]) +
			np.outer(np.conj(Bs1[il, :]), As2[il, :])*I_alpha(chis2[il, :] +\
				np.conj(chis1[il, :][:, np.newaxis]), self.d_array[il-1])  )

		# Final pre-factor
		mat = mat * (oms1**2)[:, np.newaxis] * qp[indmat]

		return mat

	def rad_te_te(self, indmode1, oms1,
				As1, Bs1, chis1, indmoder, omr, Xsr, Ysr, 
				chisr, qq):
		'''
		Coupling of TE guided modes to TE radiative modes
		'''

		# Contribution from lower cladding
		indmat = np.ix_(indmode1, indmoder)
		mat = self.phc.claddings[0].eps_inv_mat[indmat]* \
				self.phc.claddings[0].eps_avg**2 * (\
				np.outer(np.conj(Bs1[0, :]), Ysr[0, :]) * J_alpha(
				-np.conj(chis1[0, :][:, np.newaxis])-chisr[np.newaxis, 0])+
				np.outer(np.conj(Bs1[0, :]), Xsr[0, :]) * J_alpha(
				-np.conj(chis1[0, :][:, np.newaxis])+chisr[np.newaxis, 0]))

		# Contribution from upper cladding
		mat = mat + self.phc.claddings[-1].eps_inv_mat[indmat]* \
				self.phc.claddings[-1].eps_avg**2 * (\
				np.outer(np.conj(As1[-1, :]), Ysr[-1, :]) * J_alpha(
				-np.conj(chis1[-1, :][:, np.newaxis])+chisr[np.newaxis, -1])+
				np.outer(np.conj(As1[-1, :]), Xsr[-1, :]) * J_alpha(
				-np.conj(chis1[-1, :][:, np.newaxis])-chisr[np.newaxis, -1]))

		# Contributions from layers
		# note: self.N_layers = self.phc.layers.shape so without claddings
		for il in range(1, self.N_layers+1):
			mat = mat + self.phc.layers[il-1].eps_inv_mat[indmat] *\
			self.phc.layers[il-1].eps_avg**2 * ( \
			np.outer(np.conj(As1[il, :]), Ysr[il, :])*I_alpha(chisr[il, :] -\
				np.conj(chis1[il, :][:, np.newaxis]), self.d_array[il-1]) + \
			np.outer(np.conj(Bs1[il, :]), Xsr[il, :])*I_alpha(-chisr[il, :] +\
				np.conj(chis1[il, :][:, np.newaxis]), self.d_array[il-1]) + \
			np.outer(np.conj(As1[il, :]), Xsr[il, :])*I_alpha(-chisr[il, :] -\
				np.conj(chis1[il, :][:, np.newaxis]), self.d_array[il-1]) +
			np.outer(np.conj(Bs1[il, :]), Ysr[il, :])*I_alpha(chisr[il, :] +\
				np.conj(chis1[il, :][:, np.newaxis]), self.d_array[il-1])  )

		# Final pre-factor
		mat = mat * np.outer(oms1**2, omr*np.ones(indmoder.size)) * qq

		return mat

	def rad_te_tm(self, indmode1, oms1,
				As1, Bs1, chis1, indmoder, omr, Xsr, Ysr, 
				chisr, qp):
		'''
		Coupling of TE guided modes to TM radiative modes
		'''

		# Contribution from lower cladding
		indmat = np.ix_(indmode1, indmoder)
		mat = self.phc.claddings[0].eps_inv_mat[indmat]* \
				self.phc.claddings[0].eps_avg * (\
				np.outer(np.conj(Bs1[0, :]), -chisr[0, :]*Ysr[0, :]) * J_alpha(
				-np.conj(chis1[0, :][:, np.newaxis])-chisr[np.newaxis, 0])+
				np.outer(np.conj(Bs1[0, :]), chisr[0, :]*Xsr[0, :]) * J_alpha(
				-np.conj(chis1[0, :][:, np.newaxis])+chisr[np.newaxis, 0]))

		# Contribution from upper cladding
		mat = mat + self.phc.claddings[-1].eps_inv_mat[indmat]* \
			self.phc.claddings[-1].eps_avg * (\
			np.outer(np.conj(As1[-1, :]), -chisr[-1, :]*Ysr[-1, :]) * J_alpha(
				-np.conj(chis1[-1, :][:, np.newaxis])-chisr[np.newaxis, -1])+
			np.outer(np.conj(As1[-1, :]), chisr[-1, :]*Xsr[-1, :]) * J_alpha(
				-np.conj(chis1[-1, :][:, np.newaxis])+chisr[np.newaxis, -1]))

		# Contributions from layers
		# note: self.N_layers = self.phc.layers.shape so without claddings
		for il in range(1, self.N_layers+1):
			mat = mat + self.phc.layers[il-1].eps_inv_mat[indmat] *\
			self.phc.layers[il-1].eps_avg * chisr[il, :][np.newaxis, :] * ( -\
			np.outer(np.conj(As1[il, :]), Ysr[il, :])*I_alpha(chisr[il, :] -\
				np.conj(chis1[il, :][:, np.newaxis]), self.d_array[il-1]) + \
			np.outer(np.conj(Bs1[il, :]), Xsr[il, :])*I_alpha(-chisr[il, :] +\
				np.conj(chis1[il, :][:, np.newaxis]), self.d_array[il-1]) + \
			np.outer(np.conj(As1[il, :]), Xsr[il, :])*I_alpha(-chisr[il, :] -\
				np.conj(chis1[il, :][:, np.newaxis]), self.d_array[il-1]) -
			np.outer(np.conj(Bs1[il, :]), Ysr[il, :])*I_alpha(chisr[il, :] +\
				np.conj(chis1[il, :][:, np.newaxis]), self.d_array[il-1])  )

		# Final pre-factor
		mat = mat * 1j * (oms1**2)[:, np.newaxis] * qp

		return mat