import numpy as np
from scipy.special import jv as besselj 
import matplotlib.path as mpltPath

from .backend import backend as bd

class Shape(object):
	''' 
	Parent class for shapes. Each child class should have the following methods:
		- compute_ft: discrete FT assuming 1 inside the shape and 0 outside
		- is_inside: checks if points (x, y) are inside the shape
	and 
	'''
	def __init__(self, eps=1):
		self.eps = eps
		self.area = bd.real(self.compute_ft(bd.array([[0], [0]])))

	def parse_ft_gvec(self, gvec):
		if type(gvec) == list:
			gvec = np.array(gvec)
		elif type(gvec) != np.ndarray:
			raise(TypeError, "Input gvec must be either of type list or \
									np.ndarray")

		if len(gvec.shape) != 2:
			raise(ValueError, "Input gvec must be a 2D array")

		elif gvec.shape[0] != 2:
			if gvec.shape[1] == 2:
				gvec = gvec.T
			else:
				raise(ValueError, "Input gvec must have length 2 \
				 				along one axis")

		return (gvec[0, :], gvec[1, :])

class Circle(Shape):
	'''
	Define class for a circular shape
	'''
	def __init__(self, eps=1, x=0, y=0, r=0):
		self.x = x
		self.y = y
		self.r = r
		super().__init__(eps=eps)

	def compute_ft(self, gvec):
		'''
		FT of a 2D function equal to 1 inside the circle and 0 outside
		Input: 
			- gvec: [2 x Ng] numpy array
		'''
		(gx, gy) = self.parse_ft_gvec(gvec)

		gabs = np.sqrt(np.abs(np.square(gx)) + np.abs(np.square(gy)))
		gabs += 1e-10 # To avoid numerical instability at zero

		ft = bd.exp(1j*gx*self.x + 1j*gy*self.y)*self.r* \
							2*np.pi/gabs*bd.bessel1(gabs*self.r)

		return ft

	def is_inside(self, x, y):
		return (np.square(x - self.x) + np.square(y - self.y)
							<= np.square(self.r))

class Poly(Shape):
	'''
	Define class for a polygonal shape
	'''
	def __init__(self, eps=1, x_edges=0, y_edges=0):
		# Make extra sure that the last point of the polygon is the same as the 
		# first point
		self.x_edges = x_edges
		self.y_edges = y_edges
		self.x_edges.append(x_edges[0])
		self.y_edges.append(y_edges[0])
		super().__init__(eps)

	def compute_ft(self, gvec):
		'''
		Computing polygonal shape FT as per Lee, IEEE TAP (1984)
		NB: the vertices of the polygonal should be defined in a 
		counter-clockwise manner!
		Input: 
			- gvec: [2 x Ng] numpy array
		'''
		(gx, gy) = self.parse_ft_gvec(gvec)

		xj = np.array(self.x_edges)
		yj = np.array(self.y_edges)
		npts = xj.shape[0]
		ng = gx.shape[0]
		gx = gx[:, np.newaxis]
		gy = gy[:, np.newaxis]
		xj = xj[np.newaxis, :]
		yj = yj[np.newaxis, :]

		ft = np.zeros((ng), dtype=bd.complex);

		aj = (np.roll(xj, -1, axis=1) - xj + 1e-10) / \
				(np.roll(yj, -1, axis=1) - yj + 1e-20)
		bj = xj - aj * yj

		# We first handle the Gx = 0 case
		ind_gx0 = np.abs(gx[:, 0]) < 1e-10
		ind_gx = ~ind_gx0
		if np.sum(ind_gx0) > 0:
			# And first the Gy = 0 case
			ind_gy0 = np.abs(gy[:, 0]) < 1e-10
			if np.sum(ind_gy0*ind_gx0) > 0:
				ft[ind_gx0*ind_gy0] = np.sum(xj * np.roll(yj, -1, axis=1) - \
								yj * np.roll(xj, -1, axis=1))/2
				# Remove the Gx = 0, Gy = 0 component
				ind_gx0[ind_gy0] = False

			# Compute the remaining Gx = 0 components
			a2j = 1 / aj
			b2j = yj - a2j * xj
			bgtemp = gy[ind_gx0, :] * b2j
			agtemp1 = gx[ind_gx0, :].dot(xj) + gy[ind_gx0, :].dot(a2j * xj)
			agtemp2 = gx[ind_gx0, :].dot(np.roll(xj, -1, axis=1)) + \
					gy[ind_gx0, :].dot(a2j * np.roll(xj, -1, axis=1));
			ft[ind_gx0] = np.sum(np.exp(1j*bgtemp) * (np.exp(1j*agtemp2) - \
					np.exp(1j*agtemp1)) / (gy[ind_gx0, :] * (gx[ind_gx0, :] + \
					gy[ind_gx0, :].dot(a2j))), axis=1)

		# Finally compute the general case for Gx != 0
		if np.sum(ind_gx) > 0:
			bgtemp = gx[ind_gx, :].dot(bj)
			agtemp1 = gy[ind_gx, :].dot(yj) + gx[ind_gx, :].dot(aj * yj)
			agtemp2 = gy[ind_gx, :].dot(np.roll(yj, -1, axis=1)) + \
						gx[ind_gx, :].dot(aj * np.roll(yj, -1, axis=1))
			ft[ind_gx] = -np.sum(np.exp(1j*bgtemp) * (np.exp(1j * agtemp2) - \
						np.exp(1j * agtemp1)) / (gx[ind_gx, :] * \
						(gy[ind_gx, :] + gx[ind_gx, :].dot(aj))), axis=1)

		return ft

	def is_inside(self, x, y):
		vert = np.vstack((np.array(self.x_edges), np.array(self.y_edges)))
		path = mpltPath.Path(vert.T)
		points = np.vstack((np.array(x).ravel(), np.array(y).ravel()))
		test = path.contains_points(points.T)
		return test.reshape((x.shape))

	def rotate(self, angle):
		'''
		Rotate a polygon around its center of mass by angle radians
		'''

		rotmat = np.array([[np.cos(angle), -np.sin(angle)], \
							[np.sin(angle), np.cos(angle)]])
		(xj, yj) = (np.array(self.x_edges), np.array(self.y_edges))
		com_x = np.sum((xj + np.roll(xj, -1)) * (xj * np.roll(yj, -1) - \
					np.roll(xj, -1) * yj))/6/self.area
		com_y = np.sum((yj + np.roll(yj, -1)) * (xj * np.roll(yj, -1) - \
					np.roll(xj, -1) * yj))/6/self.area
		new_coords = rotmat.dot(np.vstack((xj-com_x, yj-com_y)))

		self.x_edges = new_coords[0, :] + com_x
		self.y_edges = new_coords[1, :] + com_y

class Square(Poly):
	'''
	Define class for a square shape
	'''
	def __init__(self, eps=1, x_cent=0, y_cent=0, a=0):
		self.x_cent = x_cent
		self.y_cent = y_cent
		self.a = a
		x_edges = [x_cent - a/2, x_cent + a/2, x_cent + a/2, x_cent - a/2]
		y_edges = [y_cent - a/2, y_cent - a/2, y_cent + a/2, y_cent + a/2]
		super().__init__(eps, x_edges, y_edges)