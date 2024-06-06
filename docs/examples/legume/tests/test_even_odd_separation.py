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
            for line in open(file, 'r'):
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

        lattice = legume.Lattice('square')
        phc_asymm = legume.PhotCryst(lattice, eps_l=1.45**2)
        phc_asymm.add_layer(d=d, eps_b=3.54**2)
        phc_asymm.layers[-1].add_shape(Circle(eps=1, r=0.2))
        path_gx = lattice.bz_path(["G", "X"], [num_points])
        gme_asymm = GuidedModeExp(phc_asymm, g_max, truncate_g='abs')

        gme_options = {
            'gmode_inds': [0, 1, 2, 3],
            'numeig': 96,
            'verbose': True,
            'symmetry': 'both',
            'symm_thr': 4e-8,
            'angles': path_gx['angles'],
            "compute_im": True
        }
        gme_asymm.run(kpoints=path_gx['kpoints'], **gme_options)

        Gx_both_k, Gx_both_E, Gx_both_im = load_GME_Fort(
            './tests/data/gme_gx_both.out')

        diff_gx = np.sum(
            np.abs(gme_asymm.freqs.T.flatten()[:cut_points] -
                   Gx_both_E[:cut_points]))
        diff_gx_im = np.sum(
            np.abs(gme_asymm.freqs_im.T.flatten()[:cut_points] -
                   Gx_both_im[:cut_points]))

        self.assertLessEqual(diff_gx, 1e-7)
        self.assertLessEqual(diff_gx_im, 1e-7)


if __name__ == '__main__':
    unittest.main()
