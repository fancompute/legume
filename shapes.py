import numpy as np
import matplotlib.path as mpltPath

class Shape(object):
	''' 
	Parent class for shapes. Each child class should have the following methods:
		- compute_ft: discrete FT assuming 1 inside the shape and 0 outside
		- is_inside: checks if points (x, y) are inside the shape
	and 
	'''
	def __init__(self, eps=1):
		self.eps = eps

class Circle(Shape):
	'''
	Define class for a circular shape
	'''
	def __init__(self, eps=1, x=0, y=0, r=0):
		super().__init__(eps=eps)
		self.x = x
		self.y = y
		self.r = r
		self.area=np.pi*r**2

	def compute_ft(self, gvec):
		pass

	def is_inside(self, x, y):
		return (np.square(x - self.x) + np.square(y - self.y)
							<= np.square(self.r))

class Poly(Shape):
	'''
	Define class for a polygonal shape
	'''
	def __init__(self, eps=1, x_edges=0, y_edges=0):
		super().__init__(eps)
		# Make extra sure that the last point of the polygon is the same as the 
		# first point
		self.x_edges = x_edges
		self.y_edges = y_edges
		self.x_edges.append(x_edges[0])
		self.y_edges.append(y_edges[0])

		self.area = 1 # FIX THIS

	def compute_ft(self, gvec):
		pass

	def is_inside(self, x, y):
		vert = np.vstack((np.array(self.x_edges), np.array(self.y_edges)))
		path = mpltPath.Path(vert.T)
		points = np.vstack((np.array(x).ravel(), np.array(y).ravel()))
		test = path.contains_points(points.T)
		return test.reshape((x.shape))

class Square(Poly):
	'''
	Define class for a square shape
	'''
	def __init__(self, eps=1, x_cent=0, y_cent=0, a=0):
		self.x_cent = x_cent
		self.y_cent = y_cent
		x_edges = [x_cent - a/2, x_cent - a/2, x_cent + a/2, x_cent + a/2]
		y_edges = [y_cent - a/2, y_cent + a/2, y_cent + a/2, y_cent - a/2]
		super().__init__(eps, x_edges, y_edges)