import numpy as np
from utils import *

class GME(object):
	'''
	Main simulation class of the guided-mode expansion
	'''
	def __init__(self, phc, kpoints=np.array([0, 0]), gmax=3, 
						gmode_inds=1, numeig=10, om_target=0):
		# Object of class Phc which will be simulated
		self.phc = phc
		# Bloch momenta over which band structure is imulated 
		self.kpoints = kpoints
		# Maximum reciprocal lattice wave-vector length in units of 2pi/a
		self.gmax = gmax
		# Indexes of modes to be included in the expansion
		self.gmode_inds = gmode_inds
		# Number of eigenvalues to be computed
		self.numeig = numeig
		# Eigenvalues closes in magnitude to the target will be computed
		self.om_target = om_target

		# Initialize the reciprocal lattice vectors
		self._init_reciprocal()

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
		inds1 = np.tile(np.arange(-n1max, n1max + 1), 2*n2max + 1)
		inds2 = np.tile(np.arange(-n2max, n2max + 1), (2*n1max + 1, 1))  \
						 .reshape((2*n1max + 1)*(2*n2max + 1), order='F')

		gvec = self.phc.lattice.b1[:, np.newaxis].dot(inds1[np.newaxis, :]) + \
				self.phc.lattice.b2[:, np.newaxis].dot(inds2[np.newaxis, :])

		self.gvec = gvec
		self.n1g = 2*n1max + 1
		self.n2g = 2*n2max + 1

	def get_ft(self):
		'''
		Compute the unique FT coefficients of the permittivity, eps(g-g') for
		every layer in the PhC.
		'''
		(n1max, n2max) = (self.n1g, self.n2g)
		G1 = self.gvec - self.gvec[:, [0]]
		G2 = np.zeros((2, n1max*n2max))
		for ind1 in range(n1max):
			G2[:, ind1*n1max:(ind1+1)*n2max] = self.gvec[:, [ind1*n2max]] - \
							self.gvec[:, range(n2max)]

		# print(self.gvec,'\n', G1/2/np.pi,'\n', G2/2/np.pi,'\n')

		for layer in self.phc.layers:
			T1 = np.zeros(self.gvec.shape[1])
			T2 = np.zeros(self.gvec.shape[1])
			for shape in layer.shapes:
				T1 = T1 + shape.compute_ft(G1)
				T2 = T2 + shape.compute_ft(G2)
			# Store T1 and T2
			layer.T1 = T1
			layer.T2 = T2

		# Store the g-vectors to which T1 and T2 correspond
		self.G1 = G1
		self.G2 = G2

	def plot_eps(self, dx=1e-2, dy=1e-2):
		'''
		Plot the permittivity of the PhC cross-sections as computed from an 
		inverse Fourier transform with the GME reciprocal lattice vectors
		Implemented for 'square' or 'hexagonal' lattice only
		'''
		if self.phc.lattice.type not in ['hexagonal', 'square']:
			raise(NotImplementedError, "gme.plot_eps() is only implemented \
				for a lattice initialized as 'square' or 'hexagonal'")

		if not hasattr(self, 'T1'):
			self.get_ft()

		(xgrid, ygrid) = self.phc.xy_grid(dx=dx, dy=dy)

		dgx = np.abs(self.phc.lattice.b1[0])
		dgy = np.abs(self.phc.lattice.b2[1])
		nx = np.int_(2*np.max(self.gvec[0, :]/dgx))
		ny = np.int_(2*np.max(self.gvec[1, :]/dgy))
		nxtot = 2*nx + 1
		nytot = 2*ny + 1
		eps_ft = np.zeros((nxtot, nytot), dtype=np.complex128)

		N_layers = len(self.phc.layers)
		fig, ax = plt.subplots(N_layers)
		# Hacky way to make sure that the loop below works for N_layers = 1
		if N_layers == 1:
			ax = [ax]

		for (indl, layer) in enumerate(self.phc.layers):
			for jG in range(self.gvec.shape[1]):
				nG1 = np.int_(self.G1[:, jG]/dgx)
				nG2 = np.int_(self.G2[:, jG]/dgy)
	
				eps_ft[nx + nG1[0], ny + nG1[1]] = layer.T1[jG];
				eps_ft[nx + nG2[0], ny + nG2[1]] = layer.T1[jG];
				eps_ft[nx - nG1[0], ny - nG1[1]] = np.conj(layer.T1[jG]);
				eps_ft[nx - nG2[0], ny - nG2[1]] = np.conj(layer.T1[jG]);

			im = ax[indl].imshow(np.abs(np.fft.fft2(eps_ft)))
			ax[indl].set_title("xy in layer %d" % indl)
			plt.colorbar(im)
		plt.show()

	def run(self):
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

		