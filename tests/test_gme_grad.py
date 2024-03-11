import unittest

import numpy as np

from legume import GuidedModeExp, Circle, Poly, PhotCryst, Lattice
import legume
import matplotlib.pyplot as plt
from itertools import zip_longest
import scipy.io


class TestGMEgrad(unittest.TestCase):
    '''
    Tests of the gradient of a multi-layer legume computation vs. numerical
    finite-difference gradient
    '''
    def test_rect(self):
        '''
        Test the gradient for a rectangular-lattice PhC with a circular hole
        '''

        try:
            import autograd.numpy as npa
            from autograd import grad
        except:
            return 0

        lattice = Lattice([1., 0], [0., .5])

        def of(params):
            '''Define and solve a phc'''

            d = params[0]
            r = params[1]
            x = params[2]
            # Asymmetric claddings
            phc = PhotCryst(lattice, eps_u=5.)

            # First layer with a circular hole
            phc.add_layer(d=d, eps_b=12.)
            phc.add_shape(Circle(eps=1, r=r, x_cent=0, y_cent=0))

            # Second layer with a triangular hole
            phc.add_layer(d=d, eps_b=10.)
            poly = Poly(eps=3, x_edges=[x, 0.2, 0], y_edges=[-0.1, -0.1, 0.3])
            phc.add_shape(poly)

            # Define and run the GME
            gme = GuidedModeExp(phc, gmax=3, truncate_g="tbt")
            options = {'gmode_inds': [0, 1, 2], 'numeig': 5, 'verbose': False}
            gme.run(kpoints=np.array([[0., 0.1], [0., 0.2]]), **options)

            return npa.sum(gme.freqs / 2 / gme.freqs_im)

        # Define d, r, x
        params = npa.array([0.5, 0.2, -0.1])

        # Autograd gradients
        legume.set_backend('autograd')
        gr_ag = grad(of)(params)

        # Numerical gradients
        gr_num = legume.utils.grad_num(of, params)

        diff_d = np.abs(gr_num[0] - gr_ag[0]) / gr_num[0]
        diff_r = np.abs(gr_num[1] - gr_ag[1]) / gr_num[1]
        diff_x = np.abs(gr_num[2] - gr_ag[2]) / gr_num[2]

        # Check grad w.r.t. layer thickness
        self.assertLessEqual(diff_d, 1e-1)
        # Check grad w.r.t. circle radius
        self.assertLessEqual(diff_r, 1e-1)
        # Check grad w.r.t. triangle vertex position
        self.assertLessEqual(diff_x, 1e-1)


if __name__ == '__main__':
    unittest.main()
