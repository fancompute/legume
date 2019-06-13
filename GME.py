import numpy as np

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
