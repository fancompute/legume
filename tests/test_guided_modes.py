import unittest

import numpy as np

import legume
import matplotlib.pyplot as plt
import scipy.io

class Test_Guided(unittest.TestCase):
    """
    Test of the guided mode computation vs. stored data for a three-layer
    grating as in Zhou et al. Optica 3 (2016)
    """ 
    def test_gm(self):
        """
        Test a square-lattice PhC with a circular hole
        """

        # Grating parameters
        ymax = 0.1      # ficticious supercell length in y-direction
        W = 0.45        # width of dielectric rods
        H = 1.5         # total height of grating
        D = 0.1         # thickness of added parts
        Wa = (1-W)/2    # width of added parts
        epss = 1.45**2  # permittivity of the rods
        epsa = 1.1**2   # permittivity of the added parts

        # Make grating
        # Initialize the lattice and the PhC
        lattice = legume.Lattice([1, 0], [0, ymax])
        phc = legume.PhotCryst(lattice)

        # First layer
        phc.add_layer(d=D, eps_b=epss)
        rect_add = legume.Poly(eps=epsa, x_edges=np.array([-0.5, -0.5, -W / 2, -W / 2]),
                             y_edges=np.array([0.5, -0.5, -0.5, 0.5]) * ymax)
        rect_air = legume.Poly(eps=1, x_edges=np.array([W / 2, W / 2, 0.5, 0.5]),
                             y_edges=np.array([0.5, -0.5, -0.5, 0.5]) * ymax)
        phc.add_shape([rect_add, rect_air])

        # Second layer
        phc.add_layer(d=H-2*D, eps_b=epss)
        rect_air1 = legume.Poly(eps=1, x_edges=np.array([-0.5, -0.5, -W / 2, -W / 2]),
                             y_edges=np.array([0.5, -0.5, -0.5, 0.5]) * ymax)
        rect_air2 = legume.Poly(eps=1, x_edges=np.array([W / 2, W / 2, 0.5, 0.5]),
                             y_edges=np.array([0.5, -0.5, -0.5, 0.5]) * ymax)
        phc.add_shape([rect_air1, rect_air2])

        # Third layer
        phc.add_layer(d=D, eps_b=epss)
        rect_air = legume.Poly(eps=1, x_edges=np.array([-0.5, -0.5, -W / 2, -W / 2]),
                             y_edges=np.array([0.5, -0.5, -0.5, 0.5]) * ymax)
        rect_add = legume.Poly(eps=epsa, x_edges=np.array([W / 2, W / 2, 0.5, 0.5]),
                             y_edges=np.array([0.5, -0.5, -0.5, 0.5]) * ymax)
        phc.add_shape([rect_add, rect_air])

        # Run a small GME simulation
        gme = legume.GuidedModeExp(phc, gmax=5)
        options = {'gmode_inds': np.arange(4), 'verbose': False, 
                    'gmode_step': 1e-3, 'gmode_tol': 1e-12,
                    'gmode_compute': 'interp',
                    'compute_im': False}

        gme.run(kpoints=np.array([[0.], [0.]]), **options)

        gms = np.zeros((4, gme.g_array[0].size))
        for im in range(2):
            gms[2*im, -len(gme.omegas_te[0][im]):] = gme.omegas_te[0][im]
            gms[2*im+1, -len(gme.omegas_tm[0][im]):] = gme.omegas_tm[0][im]

        gms_store = np.load("./tests/data/guided_modes_grating.npy")

        self.assertLessEqual(np.sum(np.abs(gms_store-gms)), 1e-5)

if __name__ == '__main__':
    unittest.main()
