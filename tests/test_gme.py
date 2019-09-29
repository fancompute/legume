import unittest

import numpy as np

from legume import GuidedModeExp, Circle, PhotCryst, Lattice
import matplotlib.pyplot as plt
from itertools import zip_longest
import scipy.io

class TestGME(unittest.TestCase):
    def test_square(self):
        # Initialize a lattice
        lattice = Lattice('square')
        # Initialize a PhC with asymmetric cladding
        phc = PhotCryst(lattice, eps_u=5.)
        phc.add_layer(d=0.5, eps_b=12.)
        phc.add_shape(Circle(r=0.2, x_cent=0.1, y_cent=0.2))

        # Find the modes of the structure and compare to the saved .mat file
        gme = GuidedModeExp(phc, gmax=4)
        options = {'gmode_inds': [0, 1, 2, 3], 'numeig': 10, 'verbose': False, 
                    'gmode_npts': 2000}
        gme.run(kpoints=np.array([[0.1], [0.2]]), **options)
        (freqs_im, _, _) = gme.compute_rad(kind=0, 
                                    minds=range(10))

        dat = scipy.io.loadmat('./tests/data/gme_square.mat')
        diff_real = np.sum(np.abs(gme.freqs - dat['Eks']/2/np.pi) / \
                                    (gme.freqs+1e-10))
        diff_imag = np.sum(np.abs(freqs_im - dat['Pks']/2/np.pi) / \
                                    (np.abs(freqs_im)+1e-10))

        self.assertLessEqual(diff_real, 1e-4)
        self.assertLessEqual(diff_imag, 1e-3)

    def test_rect(self):
        lattice = Lattice([1., 0], [0., .5])

        phc = PhotCryst(lattice, eps_u=5.)
        phc.add_layer(d=0.5, eps_b=12.)
        phc.add_shape(Circle(r=0.2, x_cent=0.1))

        gme = GuidedModeExp(phc, gmax=3)
        options = {'gmode_inds': [0, 1, 2, 3], 'numeig': 10, 'verbose': False}
        gme.run(kpoints=np.array([[0., 0.1], [0., 0.2]]), **options)
        freqs_im = []
        for kind in [0, 1]:
            (f_im, _, _) = gme.compute_rad(kind=kind, minds=range(10))
            freqs_im.append(f_im)
        freqs_im = np.array(freqs_im)

        dat = scipy.io.loadmat('./tests/data/gme_rect.mat')
        diff_real = np.sum(np.abs(gme.freqs - dat['Eks']/2/np.pi) / \
                                    (gme.freqs+1e-10))
        diff_imag = np.sum(np.abs(freqs_im - dat['Pks']/2/np.pi) / \
                                    (np.abs(freqs_im)+1e-10))

        self.assertLessEqual(diff_real, 1e-4)
        self.assertLessEqual(diff_imag, 1e-3)

    def test_hex(self):
        lattice = Lattice('hexagonal')
        phc = PhotCryst(lattice, eps_l=5.)
        phc.add_layer(d=0.5, eps_b=12.)
        phc.add_shape(Circle(r=0.2, x_cent=0.1, y_cent=0.2))

        gme = GuidedModeExp(phc, gmax=6)
        # gme.plot_overview_ft(cladding=True)
        options = {'gmode_inds': [1, 2, 3], 'numeig': 10, 'verbose': False}
        gme.run(kpoints=np.array([[0.1], [0.1]]), **options)
        (freqs_im, _, _) = gme.compute_rad(kind=0, minds=range(10))

        dat = scipy.io.loadmat('./tests/data/gme_hex.mat')
        diff_real = np.sum(np.abs(gme.freqs - dat['Eks']/2/np.pi) / \
                                    (gme.freqs+1e-10))
        diff_imag = np.sum(np.abs(freqs_im - dat['Pks']/2/np.pi) / \
                                    (np.abs(freqs_im)+1e-10))

        # Lower tolerance allowed because the MATLAB reciprocal space is 
        # hexagonal while the legume one is a parallelogram
        self.assertLessEqual(diff_real, 1e-3)
        self.assertLessEqual(diff_imag, 1e-1)

if __name__ == '__main__':
    unittest.main()
