import unittest

import numpy as np

from pygme import GuidedModeExp, Circle, PhotCryst, Lattice
import matplotlib.pyplot as plt
from itertools import zip_longest

class TestGME(unittest.TestCase):
	pass

if __name__ == '__main__':
    # unittest.main()
	# Initialize a lattice
	lattice = Lattice([1., 0], [0., 2.])
	# Initialize a PhC (by default with eps = 1 in upper and lower cladding, we set upper one to 5)
	phc = PhotCryst(lattice, eps_u=1.)
	# Add a layer to the PhC with thickness 1 and background permittivity 10
	phc.add_layer(d=0.5, eps_b=12.)
	# Add a shape to this layer 
	phc.add_shape(Circle(r=0.2))
	# phc.claddings[0].add_shape(Circle(r=0.1, eps=5))
	# Plot an overview picture
	# phc.plot_overview(cladding='True')

	gme = GuidedModeExp(phc, gmax=1)
	# gme.plot_overview_ft(cladding=True)
	gme.run(kpoints=np.array([[0], [0]]), gmode_inds=[0], N_g_array=500, numeig=10)
	print(gme.freqs)