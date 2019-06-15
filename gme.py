import numpy as np
from utils import plot_reciprocal

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
		plot_reciprocal(self)

	def _init_reciprocal(self):
		'''
		Initialize reciprocal lattice vectors based on self.phc and self.gmax
		'''
		n1max = int((2*np.pi*self.gmax)/np.linalg.norm(self.phc.lattice.b1))
		n2max = int((2*np.pi*self.gmax)/np.linalg.norm(self.phc.lattice.b2))

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

		