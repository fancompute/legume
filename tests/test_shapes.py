import unittest
import pygme
import numpy as np

class TestShapes(unittest.TestCase):
	def test_poly(self):

		x_edges = [-1.0, +1.0, +1.0, -1.0]
		y_edges = [-1.0, -1.0, +1.0, +1.0]
		self.assertTrue(pygme.shapes.Poly._check_counterclockwise(x_edges, y_edges, verbose=True))

		x_edges = [-1.0, -1.0, +1.0, +1.0]
		y_edges = [-1.0, +1.0, +1.0, -1.0]
		self.assertFalse(pygme.shapes.Poly._check_counterclockwise(x_edges, y_edges, verbose=True))

		y_edges = np.array([0.05, -0.05, -0.05, 0.05])
		x_edges = np.array([-0.25, -0.25,  0.25,  0.25])
		self.assertTrue(pygme.shapes.Poly._check_counterclockwise(x_edges, y_edges, verbose=True))

if __name__ == '__main__':
    unittest.main()