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
		gx = np.array(gvec[0, :])
		gy = np.array(gvec[1, :])
		gabs = np.sqrt(np.abs(np.square(gx)) + np.abs(np.square(gy)))
		gind = gabs > 1e-10
		ft = np.pi*self.r**2*np.ones(gabs.shape, dtype=np.complex128)

		ft[gind] = np.exp(1j*gx[gind]*self.x + 1j*gy[gind]*self.y)*self.r* \
							2*np.pi/gabs[gind]*besselj(1, gabs[gind]*self.r)

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

		self.area = np.real(self.compute_ft(np.array([[0], [0]])))

	def compute_ft(self, gvec):
		'''
		Computing polygonal shape FT as per Lee, IEEE TAP (1984)
		NB: the vertices of the polygonal should be defined in a 
		counter-clockwise manner!
		Input: 
			- gvec: [2 x Ng] numpy array
		'''
		gx = np.array(gvec[0, :])
		gy = np.array(gvec[1, :])
		xj = np.array(self.x_edges)
		yj = np.array(self.y_edges)
		npts = xj.shape[0]
		ng = gx.shape[0]
		gx = gx[:, np.newaxis]
		gy = gy[:, np.newaxis]
		xj = xj[np.newaxis, :]
		yj = yj[np.newaxis, :]

		ft = np.zeros((ng));

		aj = (np.roll(xj, -1, axis=1) - xj + 1e-10) / \
				(np.roll(yj, -1, axis=1) - yj + 1e-10)
		bj = xj - aj * yj
		bgtemp = gx.dot(bj)
		agtemp1 = gy.dot(yj) + gx.dot(aj * yj)
		agtemp2 = gy.dot(np.roll(yj, -1, axis=1)) + \
					gx.dot(aj * np.roll(yj, -1, axis=1))
		ft = -np.sum(np.exp(1j*bgtemp) * (np.exp(1j * agtemp2) - \
					np.exp(1j * agtemp1)) / (gx * (gy + gx.dot(aj)) + 1e-10), 
					axis=1)
		a2j = 1 / aj
		b2j = yj - a2j * xj
		ind_gx0 = np.where(np.abs(gx[:, 0]) < 1e-10)[0]
		if ind_gx0.size > 0:
			bgtemp = gy[ind_gx0, :] * b2j
			agtemp1 = gx[ind_gx0, :].dot(xj) + gy[ind_gx0, :].dot(a2j * xj)
			agtemp2 = gx[ind_gx0, :].dot(np.roll(xj, -1, axis=1)) + \
					gy[ind_gx0, :].dot(a2j * np.roll(xj, -1, axis=1));
			ft[ind_gx0] = np.sum(np.exp(1j*bgtemp) * (np.exp(1j*agtemp2) - \
					np.exp(1j*agtemp1)) / (gy[ind_gx0, :] * (gx[ind_gx0, :] + \
					gy[ind_gx0, :].dot(a2j)) + 1e-10), axis=1);

		ind_gxy0 = np.where((np.abs(gx[:, 0]) < 1e-10) * 
							(np.abs(gx[:, 0]) < 1e-10))[0]
		if ind_gxy0.size > 0:
			ft[ind_gxy0] = np.sum(xj * np.roll(yj, -1, axis=1) - \
						yj * np.roll(xj, -1, axis=1))/2;
		
		return ft

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
		x_edges = [x_cent - a/2, x_cent + a/2, x_cent + a/2, x_cent - a/2]
		y_edges = [y_cent - a/2, y_cent - a/2, y_cent + a/2, y_cent + a/2]
		super().__init__(eps, x_edges, y_edges)