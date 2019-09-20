# This file defines the PhotCryst, Layer, and Lattice classes

import numpy as np
import matplotlib.pyplot as plt

import legume.utils as utils
import legume.backend as bd
from .shapes import Shape, Circle, Poly, Square
from .backend import backend as bd

class PhotCryst(object):
	'''
	Class for a photonic crystal which can contain a number of layers
	'''
	def __init__(self, lattice, eps_l=1, eps_u=1):
		# Define permittivity of lower and upper cladding
		# For now claddings are just ShapesLayer, maybe extend to freeform
		self.claddings = []
		self.claddings.append(ShapesLayer(lattice, eps_b=eps_l, z_min=-1e50))
		self.claddings.append(ShapesLayer(lattice, eps_b=eps_u, z_max=1e50))
		
		self.lattice = lattice

		# Initialize an empty list of layers
		self.layers = []

	def z_grid(self, Nz=100):
		''' 
		Define a z-grid for visualization purposes once some layers have been 
		added
		'''
		zmin = self.layers[0].z_min - 1
		zmax = self.layers[-1].z_max + 1
    
		return np.linspace(zmin, zmax, Nz)

	def add_layer(self, d, eps_b=1, layer_type='shapes'):
		'''
		Add a layer with thickness d and background permittivity eps_b
		'''
		if self.layers == []:
			z_min = 0
		else:
			z_min = self.layers[-1].z_max

		if layer_type.lower() == 'shapes':
			layer = ShapesLayer(self.lattice, z_min, z_min + d, eps_b=eps_b)
		elif layer_type.lower() == 'freeform':
			layer = FreeformLayer(self.lattice, z_min, z_min + d, eps_b=eps_b)
		else:
			raise ValueError("'layer_type' must be 'shapes' or 'freeform'")

		self.claddings[1].z_min = z_min + d
		self.layers.append(layer)

	def add_shape(self, *args, **kwargs):
		'''
		Add a shape to layer number layer_ind
		'''
		cladding = kwargs.get('cladding', None)
		print(cladding)
		layer = kwargs.get('layer', -1)
		if cladding is not None:
			if cladding==0 or cladding.lower()=='l':
				lay = self.claddings[0]
			elif cladding==1 or cladding.lower()=='u':
				lay = self.claddings[1]
			else:
				raise ValueError("'cladding' must be 0 or 'l' for lower" \
					"cladding and 1 or 'u' for upper cladding")
		else:
			if layer >= len(self.layers):
				raise ValueError("Layer index larger than total number of "\
					"layers")
			else:
				lay = self.layers[layer]
		if lay.layer_type == 'shapes':
			lay.add_shape(*args)
		else:
			raise TypeError("Shapes can only be added to a ShapesLayer")

	def get_eps(self, points):
		'''
		Compute the permittivity of the PhC at a set of points defined by
		a tuple of x, y, z positions which are same-size arrays
		'''
		(xmesh, ymesh, zmesh) = points
		a_shape = xmesh.shape
		if (ymesh.shape != a_shape) or (ymesh.shape != a_shape):
			raise ValueError(
					"xmesh, ymesh and zmesh must have the same shape")

		eps_r = np.zeros(a_shape)

		# eps_r[zmesh < self.layers[0].z_min] = self.eps_l
		# eps_r[zmesh >= self.layers[-1].z_max] = self.eps_u
		a1 = self.lattice.a1
		a2 = self.lattice.a2

		for layer in self.layers + self.claddings:
			zlayer = (zmesh >= layer.z_min) * (zmesh < layer.z_max)
			eps_r[zlayer] = layer.eps_b

			# Slightly hacky way to include the periodicity
			a_p = min([np.linalg.norm(a1), 
					   np.linalg.norm(a2)])
			nmax = np.int_(np.sqrt(np.square(np.max(abs(xmesh))) + 
							np.square(np.max(abs(ymesh))))/a_p) + 1

			for shape in layer.shapes:
				for n1 in range(-nmax, nmax):
					for n2 in range(-nmax, nmax):
						in_shape = shape.is_inside(xmesh + 
							n1*a1[0] + n2*a2[0], ymesh + 
							n1*a1[1] + n2*a2[1])
						eps_r[in_shape*zlayer] = shape.eps

		return eps_r

	def get_eps_bounds(self):
		# Returns the minimum and maximum permittivity of the structure
		eps_min = self.claddings[0].eps_b
		eps_max = self.claddings[0].eps_b

		for layer in self.layers + self.claddings:
			eps_min = min([eps_min, layer.eps_b])
			eps_max = max([eps_max, layer.eps_b])
			for shape in layer.shapes:
				eps_min = min([eps_min, shape.eps])
				eps_max = max([eps_max, shape.eps])

		return (eps_min, eps_max)

	def plot_cross(self, cross_section='xy', pos=0, Npts=[100, 100]):
		'''
		Plot a cross-section of the PhC at position pos along the third axis
		'''
		if cross_section == 'xy':
			utils.plot_xy(self, z=pos, Nx=Npts[0], Ny=Npts[1])
		elif cross_section == 'xz':
			utils.plot_xz(self, y=pos, Nx=Npts[0], Nz=Npts[1])
		elif cross_section == 'yz':
			utils.plot_yz(self, x=pos, Ny=Npts[0], Nz=Npts[1])
		else:
			raise ValueError("Cross-section must be in {'xy', 'yz', 'xz'}")

	def plot_overview(self, Nx=100, Ny=100, Nz=50, cladding=False):
		'''
		Plot an overview of PhC cross-sections
		'''

		(eps_min, eps_max) = self.get_eps_bounds()
		fig, ax = plt.subplots(1, 2, constrained_layout=True)
		utils.plot_xz(self, ax=ax[0], Nx=Nx, Nz=Nz,
					clim=[eps_min, eps_max], cbar=False)
		ax[0].set_title("xz at y = 0")
		utils.plot_yz(self, ax=ax[1], Ny=Ny, Nz=Nz,
					clim=[eps_min, eps_max], cbar=True)
		ax[1].set_title("yz at x = 0")

		if cladding:
			all_layers = [self.claddings[0]] + self.layers + [self.claddings[1]]
		else:
			all_layers = self.layers
		N_layers = len(all_layers)

		fig, ax = plt.subplots(1, N_layers, constrained_layout=True)
		if N_layers==1: ax=[ax]

		for indl in range(N_layers):
			zpos = (all_layers[indl].z_max + all_layers[indl].z_min)/2
			utils.plot_xy(self, z=zpos, ax=ax[indl], Nx=Nx, Ny=Ny,
					clim=[eps_min, eps_max], cbar=indl==N_layers-1)
			if cladding:
				if indl > 0 and indl < N_layers-1:
					ax[indl].set_title("xy in layer %d" % indl)
				elif indl==N_layers-1:
					ax[0].set_title("xy in lower cladding")
					ax[-1].set_title("xy in upper cladding")
			else:
				ax[indl].set_title("xy in layer %d" % indl)
		plt.show()

''' ======================== Layers ======================== '''

class Layer(object):
	'''
	Class for a single layer in the potentially multi-layer PhC
	'''
	def __init__(self, lattice, z_min=0, z_max=0):
		# Define beginning and end in z-direction
		self.z_min = z_min
		self.z_max = z_max
		self.d = z_max - z_min

		self.lattice = lattice

	def compute_ft(self, gvec):
		'''
		Compute fourier transform over gvec: [2 x Ng] numpy array
		'''
		raise NotImplementedError("compute_ft() needs to be implemented by"
			"Layer subclasses")

class ShapesLayer(Layer):
	'''
	Layer with permittivity defined by Shape objects
	'''
	def __init__(self, lattice, z_min=0, z_max=0, eps_b=1):
		super().__init__(lattice, z_min, z_max)

		# Define background permittivity
		self.eps_b = eps_b

		# Initialize average permittivity - needed for guided-mode computation
		self.eps_avg = np.array(eps_b)

		# Initialize an empty list of shapes
		self.layer_type = 'shapes'
		self.shapes = []

	def add_shape(self, *args):
		'''
		Add a shape to the layer
		'''

		for shape in args:
			if isinstance(shape, Shape):
				self.shapes.append(shape)
				self.eps_avg = self.eps_avg + (shape.eps - self.eps_b) * \
								shape.area/self.lattice.ec_area
			else:
				raise ValueError("Arguments to add_shape must be an instance"
					"of legume.Shape (e.g legume.Circle or legume.Poly)")

	def compute_ft(self, gvec):
		'''
		Compute fourier transform over gvec: [2 x Ng] numpy array
		'''
		FT = bd.zeros(gvec.shape[1])
		for shape in self.shapes:
			# Note: compute_ft() returns the FT of a function that is one 
			# inside the shape and zero outside
			FT = FT + (shape.eps - self.eps_b)*shape.compute_ft(gvec)

		# Apply some final coefficients
		# Note the hacky way to set the zero element so as to work with
		# 'autograd' backend
		ind0 = bd.abs(gvec[0, :]) + bd.abs(gvec[1, :]) < 1e-10  
		FT = FT / self.lattice.ec_area
		FT = FT*(1-ind0) + self.eps_avg*ind0

		return FT

class FreeformLayer(Layer):
	'''
	Layer with permittivity defined by a freeform distribution on a grid
	'''
	def __init__(self, lattice, z_min=0, z_max=0, eps_init=1, res=10):
		super().__init__(lattice, z_min, z_max, eps_b)

		# Initialize average permittivity - needed for guided-mode computation
		self.eps_avg = np.array(eps_init)

		# Initialize an empty list of shapes
		self.layer_type = 'freeform'
		self._init_grid(res)

	def _init_grid(res):
		'''
		Initialize a grid with resolution res, with res[0] pixels along the 
		lattice.a1 direction and res[1] pixels along the lattice.a2 direction
		'''
		res = np.array(res)
		if res.size == 1: 
			res = res * np.ones((2,))

	def compute_ft(self, gvec):
		'''
		Compute fourier transform over gvec: [2 x Ng] numpy array
		'''
		raise NotImplementedError("compute_ft() is not yet imlemented for"
			"the free form layer")

''' ======================== Lattice ======================== '''
		
class Lattice(object):
	'''
	Class for constructing a Bravais lattice
	'''
	def __init__(self, *args):
		'''
		Initialize a Bravais lattice.
		If a single argument is passed, then
			- 'square': initializes a square lattice
			- 'hexagonal': initializes a hexagonal lattice
			(lattice constant 1 in both cases)
		If two arguments are passed, they should each be 2-element arrays
		defining the elementary vectors of the lattice
		'''

		(a1, a2) = self._parse_input(*args)
		self.a1 = a1[0:2]
		self.a2 = a2[0:2]

		ec_area = bd.norm(bd.cross(a1, a2))
		a3 = bd.array([0, 0, 1])

		b1 = 2*np.pi*bd.cross(a2, a3)/bd.dot(a1, bd.cross(a2, a3)) 
		b2 = 2*np.pi*bd.cross(a3, a1)/bd.dot(a2, bd.cross(a3, a1))

		bz_area = bd.norm(bd.cross(b1, b2))

		# Reciprocal lattice vectors
		self.b1 = b1[0:2]
		self.b2 = b2[0:2]

		self.ec_area = ec_area	# Elementary cell area
		self.bz_area = bz_area	# Brillouin zone area

	def __repr__(self):
		return "Lattice(a1 = [%.4f, %.4f], a2 = [%.4f, %.4f])" % \
			   (self.a1[0], self.a1[1], self.a2[0], self.a2[1])

	def _parse_input(self, *args):
		if len(args) == 1:
			if args[0] == 'square':
				self.type = 'square'
				a1 = bd.array([1, 0, 0])
				a2 = bd.array([0, 1, 0])
			elif args[0] == 'hexagonal':
				self.type = 'hexagonal'
				a1 = bd.array([0.5, bd.sqrt(3)/2, 0])
				a2 = bd.array([0.5, -bd.sqrt(3)/2, 0])
			else:
				raise ValueError("Lattice can be 'square' or 'hexagonal," \
					"or defined through two primitive vectors.")

		elif len(args) == 2:
			self.type = 'custom'
			a1 = bd.hstack((bd.array(args[0]), 0))
			a2 = bd.hstack((bd.array(args[1]), 0))

		return (a1, a2)

	def xy_grid(self, Nx=100, Ny=100):
		''' 
		Define an xy-grid for visualization purposes based on the lattice
		vectors of the PhC (not sure if it works for very weird lattices)
		'''
		ymax = np.abs(max([self.a1[1], self.a2[1]]))
		ymin = -ymax

		xmax = np.abs(max([self.a1[0], self.a2[0]]))
		xmin = -xmax

		return (np.linspace(xmin, xmax, Nx), np.linspace(ymin, ymax, Ny))

	def bz_path(self, pts, ns):
		'''
		Make a path in the Brillouin zone 
			- pts is a list of points 
			- ns is a list of length either 1 or len(pts) - 1, specifying 
				how many points are to be added between each two pts
		'''

		npts = len(pts)
		if npts < 2:
			raise ValueError("At least two points must be given")

		if len(ns) == 1:
			ns = ns[0]*np.ones(npts-1, dtype=np.int_)
		elif len(ns) == npts - 1:
			ns = np.array(ns)
		else:
			raise ValueError("Length of ns must be either 1 or len(pts) - 1")

		kpoints = np.zeros((2, np.sum(ns) + 1))
		inds = [0]
		count = 0

		for ip in range(npts - 1):
			p1 = self._parse_point(pts[ip])
			p2 = self._parse_point(pts[ip + 1])
			kpoints[:, count:count+ns[ip]] = p1[:, np.newaxis] + np.outer(\
						(p2 - p1), np.linspace(0, 1, ns[ip], endpoint=False))
			count = count+ns[ip]
			inds.append(count)
		kpoints[:, -1] = p2

		path = type('', (), {})() # Create an "empty" object
		path.kpoints = kpoints
		path.pt_labels = [str(pt) for pt in pts]
		path.pt_inds = inds

		return path

	def _parse_point(self, pt):
		'''
		Returns a numpy array corresponding to a BZ point pt
		'''
		if type(pt) == np.ndarray:
			return pt
		elif type(pt) == str:
			if pt.lower() == 'g' or pt.lower() == 'gamma':
				return np.array([0, 0])

			if pt.lower() == 'x':
				if self.type == 'square':
					return np.array([np.pi, 0])
				else:
					raise ValueError("'X'-point is only defined for lattice"\
						"initialized as 'square'.")

			if pt.lower() == 'm':
				if self.type == 'square':
					return np.array([np.pi, np.pi])
				elif self.type == 'hexagonal':
					return np.array([np.pi, np.pi/np.sqrt(3)])
				else:
					raise ValueError("'М'-point is only defined for lattice" \
						"initialized as 'square' or 'hexagonal'.")

			if pt.lower() == 'k':
				if self.type == 'hexagonal':
					return np.array([4/3*np.pi, 0])
				else:
					raise ValueError("'K'-point is only defined for lattice" \
						"initialized as 'hexagonal'.")
					
		raise ValueError("Something was wrong with BZ point definition")



