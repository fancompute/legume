import unittest

import numpy as np

from pygme import GuidedModeExp, Circle, PhotCryst, Lattice
import matplotlib.pyplot as plt
from itertools import zip_longest
import scipy.io

class TestGME(unittest.TestCase):
	def test_freqs_matlab(self):
		# Initialize a lattice
		lattice = Lattice('square')
		# Initialize a PhC with asymmetric cladding
		phc = PhotCryst(lattice, eps_u=5.)
		phc.add_layer(d=0.5, eps_b=12.)
		phc.add_shape(Circle(r=0.2, x_cent=0.1, y_cent=0.2))

		# Find the modes of the structure and compare to the saved .mat file
		gme = GuidedModeExp(phc, gmax=4)
		options = {'gmode_inds': [0, 1, 2, 3], 'numeig': 10, 'verbose': False}
		gme.run(kpoints=np.array([[0.1], [0]]), options=options)

		dat = scipy.io.loadmat('./tests/data/gme_freqs.mat')
		diff = np.sum(np.abs(gme.freqs - dat['Ek']/2/np.pi))

		self.assertLessEqual(diff, 1e-4)

if __name__ == '__main__':
    unittest.main()