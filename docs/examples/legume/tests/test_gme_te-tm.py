import unittest

import numpy as np

from legume import GuidedModeExp, Circle, Poly, PhotCryst, Lattice
import matplotlib.pyplot as plt
from itertools import zip_longest
import scipy.io


class TestGME_TE_TM(unittest.TestCase):
    '''
    Test of single-layer legume vs. stored data from my MATLAB GME
    Lowest-order TE and TM guided bands
    '''
    def test_square(self):
        '''
        Test a square-lattice PhC with a circular hole
        '''

        lattice = Lattice('square')
        phc = PhotCryst(lattice, eps_l=5.)
        phc.add_layer(d=0.5, eps_b=12.)
        phc.add_shape(Circle(r=0.2))

        # Find the modes of the structure and compare to the saved .mat file
        gme = GuidedModeExp(phc, gmax=5)
        options = {'gmode_inds': [0, 1], 'numeig': 10, 'verbose': False}
        gme.run(kpoints=np.array([[0., 0.1], [0., 0.2]]), **options)

        dat = scipy.io.loadmat('./tests/data/gme_square_te-tm.mat')

        # Check real and imaginary parts for k = [0, 0]
        diff_real_k0 = np.sum(
            np.abs(gme.freqs[0, 1:] - dat['Eks'][0, 1:] / 2 / np.pi) /
            (gme.freqs[0, 1:] + 1e-10))
        diff_imag_k0 = np.sum(
            np.abs(gme.freqs_im[0, 1:] - dat['Pks'][0, 1:] / 2 / np.pi) /
            (gme.freqs_im[0, 1:] + 1e-10))

        # Check real and imaginary parts for k = [0.1, 0.2]
        diff_real_k1 = np.sum(
            np.abs(gme.freqs[1, 1:] - dat['Eks'][1, 1:] / 2 / np.pi) /
            (gme.freqs[1, 1:] + 1e-10))
        diff_imag_k1 = np.sum(
            np.abs(gme.freqs_im[1, 1:] - dat['Pks'][1, 1:] / 2 / np.pi) /
            (gme.freqs_im[1, 1:] + 1e-10))

        self.assertLessEqual(diff_real_k0, 1e-4)
        self.assertLessEqual(diff_imag_k0, 1e-3)
        self.assertLessEqual(diff_real_k1, 1e-4)
        self.assertLessEqual(diff_imag_k1, 1e-3)


if __name__ == '__main__':
    unittest.main()
