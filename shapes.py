import numpy as np
from scipy.special import jv as besselj 
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
		'''
		FT of a 2D function equal to 1 inside the circle and 0 outside
		Input: 
			- gvec: [2 x Ng] numpy array
		'''
		print(gvec, gvec.shape)
		gx = np.array(gvec[0, :])
		gy = np.array(gvec[1, :])
		gabs = np.sqrt(np.abs(np.square(gx)) + np.abs(np.square(gy)))
		gind = gabs > 1e-10
		ft = np.pi*self.r**2*np.ones(gabs.shape, dtype=np.complex128)

		ft[gind] = np.exp(1j*gx[gind]*self.x + 1j*gy[gind]*self.y)*2* \
						np.pi/gabs[gind]*besselj(1, gabs[gind]*self.r)

		return ft

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
		'''
		Computing polygonal shape FT as per Lee, IEEE TAP (1984)
		NB: the vertices of the polygonal should be defined in a 
		counter-clockwise manner!
		Input: 
			- gvec: [Ng x 2] numpy array
		'''
		gx = gvec[:, 0]
		gy = gvec[:, 1]
		# To be done...


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