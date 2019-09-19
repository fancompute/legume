import unittest

import numpy as np

import pygme


class TestShapes(unittest.TestCase):
	def test_poly(self):
		x_edges = [-1.0, +1.0, +1.0, -1.0]
		y_edges = [-1.0, -1.0, +1.0, +1.0]
		self.assertTrue(pygme.shapes.Poly._check_counterclockwise(x_edges, y_edges, verbose=True))

		x_edges = [-1.0, -1.0, +1.0, +1.0]
		y_edges = [-1.0, +1.0, +1.0, -1.0]
		self.assertFalse(pygme.shapes.Poly._check_counterclockwise(x_edges, y_edges, verbose=True))

		y_edges = np.array([0.05, -0.05, -0.05, 0.05])
		x_edges = np.array([-0.25, -0.25, 0.25, 0.25])
		self.assertTrue(pygme.shapes.Poly._check_counterclockwise(x_edges, y_edges, verbose=True))

		y_edges = np.array([0.05, 0.05, -0.05, -0.05])
		x_edges = np.array([-0.25, 0.25, 0.25, -0.25])
		self.assertFalse(pygme.shapes.Poly._check_counterclockwise(x_edges, y_edges, verbose=True))


		list_xc = np.linspace(-1, +1, 9)
		list_yc = np.linspace(-1, +1, 9)
		list_a =  np.linspace(0.1, 1.1, 9)
		for a in list_a:
			for xc in list_xc:
				for yc in list_yc:
					x_edges = xc + np.array([-a/2, +a/2, +a/2, -a/2])
					y_edges = yc + np.array([-a/2, -a/2, +a/2, -a/2])
					self.assertTrue(pygme.shapes.Poly._check_counterclockwise(x_edges, y_edges, verbose=True))


if __name__ == '__main__':
	unittest.main()
