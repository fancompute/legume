import unittest

import numpy as np

import legume


class TestShapes(unittest.TestCase):
    def test_poly(self):
        eps = 1
        x_edges = np.array([-1.0, +1.0, +1.0, -1.0])
        y_edges = np.array([-1.0, -1.0, +1.0, +1.0])
        poly = legume.Poly(eps, x_edges, y_edges)

        x_edges = np.array([-1.0, -1.0, +1.0, +1.0])
        y_edges = np.array([-1.0, +1.0, +1.0, -1.0])
        self.assertRaises(ValueError, legume.Poly, *(eps, x_edges, y_edges))

        eps = 5
        y_edges = np.array([0.05, -0.05, -0.05, 0.05])
        x_edges = np.array([-0.25, -0.25, 0.25, 0.25])
        poly = legume.Poly(eps, x_edges, y_edges)

        y_edges = np.array([0.05, 0.05, -0.05, -0.05])
        x_edges = np.array([-0.25, 0.25, 0.25, -0.25])
        self.assertRaises(ValueError, legume.Poly, *(eps, x_edges, y_edges))

        list_xc = np.linspace(-1, +1, 9)
        list_yc = np.linspace(-1, +1, 9)
        list_a = np.linspace(0.1, 1.1, 9)
        for a in list_a:
            for xc in list_xc:
                for yc in list_yc:
                    x_edges = xc + np.array([-a / 2, +a / 2, +a / 2, -a / 2])
                    y_edges = yc + np.array([-a / 2, -a / 2, +a / 2, -a / 2])
                    poly = legume.Poly(eps, x_edges, y_edges)


if __name__ == '__main__':
    unittest.main()
