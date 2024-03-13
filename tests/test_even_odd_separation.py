import unittest

import numpy as np

from legume import GuidedModeExp, Circle, Poly, PhotCryst, Lattice
import matplotlib.pyplot as plt
from itertools import zip_longest
import scipy.io


class TestGME_symmetry(unittest.TestCase):
    '''
    Test of legume eigenergies separated by the 
    vertical symmetry vs data from GME Fortran code
    of Univ. di Pavia.
    '''
    def test_symmetry(self):
        '''
        Test symmetry separation of photonic modes
        '''
        def load_GME_Fort(file):

            k, e, im = [], [], []
            with open(file, 'r') as f:
                for line in f:
                    values = [s for s in line.split()]
                    if len(values) > 0:
                        if values[0] != '#':  #
                            k.append(float(values[0]))
                            e.append(float(values[1]))
                            im.append(float(values[2]))
            k = np.asarray(k)
            e = np.asarray(e)
            im = np.asarray(im)

            return (k, e, im)

        # Load fortran data
        Gx_both_k, Gx_both_E, Gx_both_im = load_GME_Fort(
            './tests/data/gme_gx_both.out')

        cut_point = 4000 # How many modes are compared
        lattice = Lattice('square')
        phc_asymm = PhotCryst(lattice, eps_l=1.45**2)
        phc_asymm.add_layer(d=0.5, eps_b=3.54**2)
        phc_asymm.layers[-1].add_shape(Circle(eps=1, r=0.2))
        path_gx = lattice.bz_path(["G", "X"], [50])
        gme_asymm = GuidedModeExp(phc_asymm, 4.01, truncate_g='abs')

        # Dense matrix test
        gme_options = {
            'gmode_inds': [0, 1, 2, 3],
            'numeig': 96,
            'verbose': False,
            'kz_symmetry': 'both',
            'symm_thr': 4e-8,
            'angles': path_gx['angles'],
            "compute_im": True,
            'use_sparse' :False
        }
        gme_asymm.run(kpoints=path_gx['kpoints'], **gme_options)

        diff_gx = np.sum(
            np.abs(gme_asymm.freqs.T.flatten()[:cut_point] -
                   Gx_both_E[:cut_point]))
        diff_gx_im = np.sum(
            np.abs(gme_asymm.freqs_im.T.flatten()[:cut_point] -
                   Gx_both_im[:cut_point]))

        self.assertLessEqual(diff_gx, 5e-8)
        self.assertLessEqual(diff_gx_im, 5e-7)

        # Sparse matrix test
        gme_options['use_sparse'] = True
        gme_asymm.run(kpoints=path_gx['kpoints'], **gme_options)

        diff_gx = np.sum(
            np.abs(gme_asymm.freqs.T.flatten()[:cut_point] -
                   Gx_both_E[:cut_point]))
        diff_gx_im = np.sum(
            np.abs(gme_asymm.freqs_im.T.flatten()[:cut_point] -
                   Gx_both_im[:cut_point]))

        self.assertLessEqual(diff_gx, 5e-8)
        self.assertLessEqual(diff_gx_im, 5e-7)


if __name__ == '__main__':
    unittest.main()
