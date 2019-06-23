import numpy as np
import utils.utils as utils
import matplotlib.pyplot as plt
from .shapes import Circle, Poly, Square

# Class for a photonic crystal which can contain a number of layers
class PhotCryst(object):
	def __init__(self, lattice, eps_l=1, eps_u=1):
		# Define permittivity of lower and upper cladding
		self.eps_l = eps_l
		self.eps_u = eps_u

		'''
		Define lattice parameters; variable lattice is initialized throug the 
		init_lattice() function in utils.py
		Note: for twisted layers, a separate lattice should be initialized 
		for every object in the Layer class
		'''
		self.lattice = lattice

		# Initialize an empty list of layers
		self.layers = []

	def xy_grid(self, dx=2e-2, dy=2e-2):
		''' 
		Define an xy-grid for visualization purposes based on the lattice
		vectors of the PhC (not sure if it works for very weird lattices)
		'''
		ymax = np.abs(max([self.lattice.a1[1], self.lattice.a2[1]]))
		ymin = -ymax

		xmax = np.abs(max([self.lattice.a1[0], self.lattice.a2[0]]))
		xmin = -xmax

		return (np.arange(xmin, xmax, dx),np.arange(ymin, ymax, dy))

	def z_grid(self, dz=2e-2):
		''' 
		Define a z-grid for visualization purposes once some layers have been 
		added
		'''
		zmin = self.layers[0].z_min - 1
		zmax = self.layers[-1].z_max + 1
		return np.arange(zmin, zmax, dz)

	def add_layer(self, d, eps_b=1):
		'''
		Add a layer with thickness d and background permittivity eps_b
		'''
		if self.layers == []:
			z_min = 0
		else:
			z_min = self.layers[-1].z_max

		layer = Layer(self.lattice, z_min, z_min + d, eps_b=eps_b)
		self.layers.append(layer)

	def add_shape(self, *args, layer_ind=-1):
		'''
		Add a shape to layer number layer_ind
		'''
		if layer_ind >= len(self.layers):
			raise(ValueError("Layer index larger than total number of layers"))
		else:
			self.layers[layer_ind].add_shape(*args)

	def get_eps(self, points):
		'''
		Compute the permittivity of the PhC at a set of points defined by
		a tuple of x, y, z positions which are same-size arrays
		'''
		(xmesh, ymesh, zmesh) = points
		a_shape = xmesh.shape
		if (ymesh.shape != a_shape) or (ymesh.shape != a_shape):
			raise (ValueError(
					"xmesh, ymesh and zmesh must have the same shape"))

		eps_r = np.zeros(a_shape)

		eps_r[zmesh < self.layers[0].z_min] = self.eps_l
		eps_r[zmesh >= self.layers[-1].z_max] = self.eps_u
		a1 = self.lattice.a1
		a2 = self.lattice.a2

		for layer in self.layers:
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
		eps_min = min([self.eps_l, self.eps_u])
		eps_max = max([self.eps_l, self.eps_u])

		for layer in self.layers:
			eps_min = min([eps_min, layer.eps_b])
			eps_max = max([eps_max, layer.eps_b])
			for shape in layer.shapes:
				eps_min = min([eps_min, shape.eps])
				eps_max = max([eps_max, shape.eps])

		return (eps_min, eps_max)

	def plot_cross(self, cross_section='xy', pos=0, res=[1e-2, 1e-2]):
		'''
		Plot a cross-section of the PhC at position pos along the third axis
		'''
		if cross_section == 'xy':
			utils.plot_xy(self, z=pos, dx=res[0], dy=res[1])
		elif cross_section == 'xz':
			utils.plot_xz(self, y=pos, dx=res[0], dz=res[1])
		elif cross_section == 'yz':
			utils.plot_yz(self, x=pos, dy=res[0], dz=res[1])
		else:
			raise(ValueError("Cross-section must be in {'xy', 'yz', 'xz'}"))

	def plot_overview(self, res=[1e-2, 1e-2, 2e-2]):
		'''
		Plot an overview of PhC cross-sections
		'''

		(eps_min, eps_max) = self.get_eps_bounds()
		fig, ax = plt.subplots(1, 2, constrained_layout=True)
		utils.plot_xz(self, ax=ax[0], dx=res[0], dz=res[2],
					clim=[eps_min, eps_max], cbar=False)
		ax[0].set_title("xz at y = 0")
		utils.plot_yz(self, ax=ax[1], dy=res[1], dz=res[2],
					clim=[eps_min, eps_max], cbar=True)
		ax[1].set_title("yz at x = 0")

		N_layers = len(self.layers)
		fig, ax = plt.subplots(1, N_layers, constrained_layout=True)

		# Hacky way to make sure that the loop below works for N_layers = 1
		if N_layers == 1:
			ax = [ax]

		for indl in range(N_layers):
			zpos = (self.layers[indl].z_max + self.layers[indl].z_min)/2
			utils.plot_xy(self, z=zpos, ax=ax[indl], dx=res[0], dy=res[1],
					clim=[eps_min, eps_max], cbar=indl==N_layers-1)
			ax[indl].set_title("xy in layer %d" % indl)
		plt.show()


class Layer(object):
	'''
	Class for a single layer in the potentially multi-layer PhC
	'''
	def __init__(self, lattice, z_min, z_max, eps_b=1):
		# Define beginning and end in z-direction
		self.z_min = z_min
		self.z_max = z_max
		self.d = z_max - z_min

		# Define background permittivity
		self.eps_b = eps_b
		# Define lattice parameters (right now inherited from Phc)
		self.lattice = lattice

		# Initialize average permittivity - needed for guided-mode computation
		self.eps_avg = eps_b

		# Initialize an empty list of shapes
		self.shapes = []

	def add_shape(self, *args):
		'''
		Add a shape to the layer
		'''

		if len(args) == 1:
			shape = args[0]
		elif len(args) == 2:
			(sh_type, params) = args

			if sh_type == 'circle':
				shape = Circle(params['eps'], params['x'], 
								params['y'], params['r'])
			elif sh_type == 'square':
				shape = Square(params['eps'], params['x_cent'], 
								params['y_cent'], params['a'])
			elif sh_type == 'poly':
				shape = Poly(params['eps'], params['x_edges'], 
								params['y_edges'])
			else:
				raise(NotImplementedError("Shape must be one of \
							{'circle', 'square', 'poly'}"))
		else:
			raise(ValueError, "Arguments to add_shape() must be either a Shape \
				instance or a tuple of (shape_type, parameters)")

		self.shapes.append(shape)
		self.eps_avg = (self.eps_avg*(self.lattice.ec_area - shape.area) + 
						shape.eps*shape.area)/self.lattice.ec_area


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
		self.a1 = a1
		self.a2 = a2

		ec_area = np.linalg.norm(np.cross(a1, a2))
		a3 = np.array([0, 0, 1])

		b1_3d = 2*np.pi*np.cross(a2, a3)[0:2]/np.dot(a1, np.cross(a2, a3)[0:2]) 
		b2_3d = 2*np.pi*np.cross(a3, a1)[0:2]/np.dot(a2, np.cross(a3, a1)[0:2])

		bz_area = np.linalg.norm(np.cross(b1_3d, b2_3d))

		# Reciprocal lattice vectors
		self.b1 = b1_3d[0:2]
		self.b2 = b2_3d[0:2]

		self.ec_area = ec_area	# Elementary cell area
		self.bz_area = bz_area	# Brillouin zone area

	def _parse_input(self, *args):
		if len(args) == 1:
			if args[0] == 'square':
				self.type = 'square'
				a1 = np.array([1, 0])
				a2 = np.array([0, 1])
			elif args[0] == 'hexagonal':
				self.type = 'hexagonal'
				a1 = np.array([0.5, np.sqrt(3)/2])
				a2 = np.array([0.5, -np.sqrt(3)/2])
			else:
				raise(ValueError, "Lattice can be 'square' or 'hexagonal, \
					or defined through two primitive vectors.")

		elif len(args) == 2:
			self.type = 'custom'
			a1 = args[0]
			a2 = args[1]

		return (a1, a2)
