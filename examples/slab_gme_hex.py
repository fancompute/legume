import unittest

import numpy as np

from legume import GuidedModeExp, Circle, PhotCryst, Lattice
import legume
import matplotlib.pyplot as plt
from itertools import zip_longest

if __name__ == '__main__':
	# Initialize a rectangular-lattice PhC
	lattice = Lattice('hexagonal')
	phc = PhotCryst(lattice, eps_l=1., eps_u=1.)
	# Make a path in the BZ 
	npts = 10 # npts = 1 just will just take the three high-symmetry points
	path = lattice.bz_path(['G', 'M', 'K', 'G'], [npts])

	# Optional: add an un-patterned layer
	# phc.add_layer(d=0.5, eps_b=12.)

	# Add a patterned layer
	phc.add_layer(d=0.5, eps_b=12.)
	phc.add_shape(Circle(r=0.3, x_cent=0, y_cent=0))
	# phc.plot_overview() # show that the PhC looks like

	# Initialize the guided mode expansion
	gme = GuidedModeExp(phc, gmax=6)
	options = {'gmode_inds': [0, 1, 2, 3], 'numeig': 10, 'verbose': False, 
		'gmode_npts': 2000}
	# And run it
	gme.run(kpoints=path.kpoints, **options)

	freqs_im = []
	for kind in range(path.kpoints[0, :].size):
		(freq_im, _, _) = gme.compute_rad(kind=kind, minds=range(10))
		freqs_im.append(freq_im)
	freqs_im = np.array(freqs_im)

	# Either print or visualize the results
	if npts==1:
		print("Real part of frequencies:      \n", gme.freqs)	
		print("Imaginary part of frequencies: \n", gme.freqs)
	else:
		legume.viz.bands(gme)

	# np.savetxt("hex_ra0_rb0.3_da0.5_db0.5_epsl5_epsa12_epsb12_epsu1_mode1+2+3+4_Ng" + 
	# 	str(gme.gvec.shape[1]) + "_kx0.1_ky0.1.txt", np.vstack((gme.freqs.ravel(), freqs_im.ravel())).T, "%1.6f")

	# gme.plot_field_xy('d', kind=0, mind=0, z=0.25,
 #                component='xyz', val='re', Nx=200, Ny=200)