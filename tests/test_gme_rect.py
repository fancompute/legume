import unittest

import numpy as np

from legume import GuidedModeExp, Circle, PhotCryst, Lattice
import matplotlib.pyplot as plt
from itertools import zip_longest

class TestGME(unittest.TestCase):
	pass

if __name__ == '__main__':
    # unittest.main()
	# Initialize a rectangular-lattice PhC with asymmetric cladding
	lattice = Lattice([1., 0], [0., .5])
	phc = PhotCryst(lattice, eps_l=5.)
	phc.add_layer(d=0.5, eps_b=12.)
	phc.add_shape(Circle(r=0.2, x_cent=0.1))

	gme = GuidedModeExp(phc, gmax=3)
	# gme.plot_overview_ft(cladding=True)
	options = {'gmode_inds': [0, 2], 'numeig': 10, 'verbose': True}
	gme.run(kpoints=np.array([[0.1], [0.2]]), **options)
	print(gme.freqs)
	(freqs_im, coup_l, coup_u) = gme.compute_rad(kind=0, minds=[1, 2])
	print(freqs_im)