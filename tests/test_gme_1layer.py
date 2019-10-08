import unittest

import numpy as np

from legume import GuidedModeExp, Circle, Poly, PhotCryst, Lattice
import matplotlib.pyplot as plt
from itertools import zip_longest
import scipy.io

class TestGME(unittest.TestCase):
    '''
    Tests of single-layer legume vs. stored data from my MATLAB GME
    Asymmetric claddings everywhere.
    ''' 
    def test_square(self):
        '''
        Test a square-lattice PhC with a triangular hole
        '''

        lattice = Lattice('square')
        phc = PhotCryst(lattice, eps_u=5.)
        phc.add_layer(d=0.5, eps_b=12.)
        poly = Poly(eps=3, x_edges=[-0.1, 0.2, 0], y_edges = [-0.1, -0.1, 0.3])
        phc.add_shape(poly)

        # Find the modes of the structure and compare to the saved .mat file
        gme = GuidedModeExp(phc, gmax=4)
        options = {'gmode_inds': [0, 1, 2, 3], 'numeig': 10, 'verbose': False, 
                    'gmode_npts': 2000}
        gme.run(kpoints=np.array([[0.1], [0.2]]), **options)

        dat = scipy.io.loadmat('./tests/data/gme_square.mat')
        diff_real = np.sum(np.abs(gme.freqs - dat['Eks']/2/np.pi) / \
                                    (gme.freqs+1e-10))
        diff_imag = np.sum(np.abs(gme.freqs_im - dat['Pks']/2/np.pi) / \
                                    (np.abs(gme.freqs_im)+1e-10))

        self.assertLessEqual(diff_real, 1e-4)
        self.assertLessEqual(diff_imag, 1e-3)

    def test_rect(self):
        '''
        Test a rectangular-lattice PhC with a circular hole
        '''
        lattice = Lattice([1., 0], [0., .5])

        phc = PhotCryst(lattice, eps_u=5.)
        phc.add_layer(d=0.5, eps_b=12.)
        phc.add_shape(Circle(eps=1, r=0.2, x_cent=0, y_cent=0))

        gme = GuidedModeExp(phc, gmax=3)
        options = {'gmode_inds': [0, 1, 2, 3], 'numeig': 10, 'verbose': False}
        gme.run(kpoints=np.array([[0., 0.1], [0., 0.2]]), **options)

        dat = scipy.io.loadmat('./tests/data/gme_rect.mat')
        diff_real = np.sum(np.abs(gme.freqs - dat['Eks']/2/np.pi) / \
                                    (gme.freqs+1e-10))
        diff_imag = np.sum(np.abs(gme.freqs_im - dat['Pks']/2/np.pi) / \
                                    (np.abs(gme.freqs_im)+1e-10))

        self.assertLessEqual(diff_real, 1e-4)
        self.assertLessEqual(diff_imag, 1e-3)

    def test_hex(self):
        '''
        Test a hexagonal-lattice PhC with a circular hole and
        gmode_compute = 'interp'
        '''
        lattice = Lattice('hexagonal')
        phc = PhotCryst(lattice, eps_l=5.)
        phc.add_layer(d=0.5, eps_b=12.)
        phc.add_shape(Circle(r=0.2, x_cent=0.1, y_cent=0.2))

        gme = GuidedModeExp(phc, gmax=6)
        # gme.plot_overview_ft(cladding=True)
        options = {'gmode_inds': [1, 2, 3], 'numeig': 10, 'verbose': False, 
                    'gmode_compute': 'interp'}
        gme.run(kpoints=np.array([[0.1], [0.1]]), **options)

        dat = scipy.io.loadmat('./tests/data/gme_hex.mat')
        diff_real = np.sum(np.abs(gme.freqs - dat['Eks']/2/np.pi) / \
                                    (gme.freqs+1e-10))
        diff_imag = np.sum(np.abs(gme.freqs_im - dat['Pks']/2/np.pi) / \
                                    (np.abs(gme.freqs_im)+1e-10))

        # Lower tolerance allowed because the MATLAB reciprocal space is 
        # hexagonal while the legume one is a parallelogram
        self.assertLessEqual(diff_real, 1e-3)
        self.assertLessEqual(diff_imag, 1e-1)

if __name__ == '__main__':
    unittest.main()
