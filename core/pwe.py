import numpy as np
import matplotlib.pyplot as plt
import utils.utils as utils
from .backend import backend as bd

class PlaneWaveExp(object):
	'''
	Main simulation class of the guided-mode expansion
	'''
	def __init__(self, layer, gmax=3, eps_eff=None):
		# Object of class Layer which will be simulated
		self.layer = layer
		# Maximum reciprocal lattice wave-vector length in units of 2pi/a
		self.gmax = gmax

		if not eps_eff:
			eps_eff = layer.eps_b

		self.eps_eff = eps_eff

		# Initialize the reciprocal lattice vectors and compute the eps FT
		self._init_reciprocal()
		self.compute_ft()

	def _init_reciprocal(self):
		'''
		Initialize reciprocal lattice vectors based on self.layer and self.gmax
		'''
		n1max = np.int_((2*np.pi*self.gmax)/np.linalg.norm(self.layer.lattice.b1))
		n2max = np.int_((2*np.pi*self.gmax)/np.linalg.norm(self.layer.lattice.b2))

		# This constructs the reciprocal lattice in a way that is suitable
		# for Toeplitz-Block-Toeplitz inversion of the permittivity in the main
		# code. However, one caveat is that the hexagonal lattice symmetry is 
		# not preserved. For that, the option to construct a hexagonal mesh in 
		# reciprocal space could is needed.
		inds1 = np.tile(np.arange(-n1max, n1max + 1), (2*n2max + 1, 1))  \
						 .reshape((2*n2max + 1)*(2*n1max + 1), order='F')
		inds2 = np.tile(np.arange(-n2max, n2max + 1), 2*n1max + 1)

		gvec = self.layer.lattice.b1[:, np.newaxis].dot(inds1[np.newaxis, :]) + \
				self.layer.lattice.b2[:, np.newaxis].dot(inds2[np.newaxis, :])

		# Save the reciprocal lattice vectors
		self.gvec = gvec

		# Save the number of vectors along the b1 and the b2 directions 
		# Note: gvec.shape[1] = n1g*n2g
		self.n1g = 2*n1max + 1
		self.n2g = 2*n2max + 1

	def compute_ft(self):
		'''
		Compute the unique FT coefficients of the permittivity, eps(g-g')
		'''
		(n1max, n2max) = (self.n1g, self.n2g)
		G1 = self.gvec - self.gvec[:, [0]]
		G2 = np.zeros((2, n1max*n2max))

		for ind1 in range(n1max):
			G2[:, ind1*n2max:(ind1+1)*n2max] = self.gvec[:, [ind1*n2max]] - \
							self.gvec[:, range(n2max)]

		# print(self.gvec,'\n', G1/2/np.pi,'\n', G2/2/np.pi,'\n')

		T1 = bd.zeros(self.gvec.shape[1])
		T2 = bd.zeros(self.gvec.shape[1])
		eps_avg = self.eps_eff
		
		for shape in self.layer.shapes:
			# Note: compute_ft() returns the FT of a function that is one 
			# inside the shape and zero outside
			T1 = T1 + (shape.eps - self.eps_eff)*shape.compute_ft(G1)
			T2 = T2 + (shape.eps - self.eps_eff)*shape.compute_ft(G2)
			eps_avg = (eps_avg*(self.layer.lattice.ec_area - shape.area) + 
						shape.eps*shape.area)/self.layer.lattice.ec_area

		# Apply some final coefficients
		# Note the hacky way to set the zero element so as to work with
		# 'autograd' backend
		ind0 = bd.arange(T1.size) < 1  
		T1 = T1 / self.layer.lattice.ec_area
		T1 = T1*(1-ind0) + eps_avg*ind0
		T2 = T2 / self.layer.lattice.ec_area
		T2 = T2*(1-ind0) + eps_avg*ind0

		# Store T1 and T2
		self.T1 = T1
		self.T2 = T2

		# Store the g-vectors to which T1 and T2 correspond
		self.G1 = G1
		self.G2 = G2

	def plot_overview_ft(self, Nx=100, Ny=100):
		'''
		Plot the permittivity of the layer as computed from an 
		inverse Fourier transform with the GME reciprocal lattice vectors.
		'''
		(xgrid, ygrid) = self.layer.lattice.xy_grid(Nx=Nx, Ny=Ny)

		fig, ax = plt.subplots(1, 1, constrained_layout=True)

		ft_coeffs = np.hstack((self.T1, self.T2, 
							np.conj(self.T1), np.conj(self.T2)))
		gvec = np.hstack((self.G1, self.G2, -self.G1, -self.G2))

		eps_r = utils.ftinv(ft_coeffs, gvec, xgrid, ygrid)
		extent = [xgrid[0], xgrid[-1], ygrid[0], ygrid[-1]]

		utils.plot_eps(np.real(eps_r), ax=ax, extent=extent, cbar=True)
		ax.set_title("xy in PWE layer")

		plt.show()

	def run(self, kpoints=np.array([[1e-5], [0]]), pol='te'):
		''' 
		Run the simulation. Input:
			- kpoints, [2xNk] numpy array over which band structure is simulated
			- pol, polarization of the computation (TE/TM)
		'''
		 
		self.kpoints = kpoints
		self.pol = pol.lower()

		self.compute_ft()
		self.compute_eps_inv()

		# Change this if switching to a solver that allows for variable numeig
		self.numeig = self.gvec.shape[1]

		freqs = bd.zeros((kpoints.shape[1], self.numeig))

		for ik, k in enumerate(kpoints.T):
			# Construct the matrix for diagonalization
			if self.pol == 'te':
				mat = bd.dot(bd.transpose(k[:, bd.newaxis] + self.gvec), 
								(k[:, bd.newaxis] + self.gvec))
				mat = mat * self.eps_inv_mat
			elif self.pol == 'tm':
				Gk = bd.sqrt(bd.square(k[0] + self.gvec[0, :]) + \
						bd.square(k[1] + self.gvec[1, :]))
				mat = bd.outer(Gk, Gk)
				mat = mat * self.eps_inv_mat
			else:
				raise ValueError("Polarization should be 'TE' or 'TM'")

			# Diagonalize using numpy.linalg.eigh() for now; should maybe switch 
			# to scipy.sparse.linalg.eish() in the future
			# NB: we shift the matrix by np.eye to avoid problems at the zero-
			# frequency mode at Gamma
			(freq2, vec) = bd.eigh(mat + bd.eye(mat.shape[0]))
			freqs = bd.sqrt(bd.abs(freq2 - bd.ones(self.numeig)))

		# Store the eigenfrequencies taking the standard reduced frequency 
		# convention for the units (2pi a/c)	
		self.freqs = freqs/2/np.pi

	def compute_eps_inv(self):
		'''
		Construct the inverse FT matrix of the permittivity
		'''

		# For now we just use the numpy inversion. Later on we could 
		# implement the Toeplitz-Block-Toeplitz inversion (faster)
		eps_mat = bd.toeplitz_block(self.n2g, self.T1, self.T2)
		self.eps_inv_mat = bd.inv(eps_mat)
		