import numpy as np
import matplotlib.pyplot as plt

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

	def add_hole(self, shape, params, layer_ind=0):
		'''
		Add a hole to layer number layer_ind
		'''
		if layer_ind >= len(self.layers):
			raise(ValueError("Layer index larger than total number of layers"))
		else:
			self.layers[layer_ind].add_hole(shape, params)

	def plot_xz(self, dx=2e-2, dz=2e-2):
		'''
		Plot an xz-cross section showing all the layers and holes
		'''
		zmin = self.layers[0].z_min - 1
		zmax = self.layers[-1].z_max + 1

		xmax = np.abs(max([self.lattice['a1'][0], self.lattice['a2'][0]]))
		xmin = -xmax

		nx = int((xmax - xmin)//dx)
		nz = int((zmax - zmin)//dz)

		xgrid = xmin + np.arange(nx)*dx
		zgrid = zmin + np.arange(nz)*dz

		[xmesh, zmesh] = np.meshgrid(xgrid, zgrid)
 
		eps_r = np.zeros((nz, nx))

		eps_r[zmesh < self.layers[0].z_min] = self.eps_l
		eps_r[zmesh > self.layers[-1].z_max] = self.eps_u

		for layer in self.layers:
			zlayer = (zmesh >= layer.z_min) * (zmesh < layer.z_max)
			eps_r[zlayer] = layer.eps_b
			for hole in layer.holes:
				xhole = hole.is_inside(xmesh, np.zeros(np.shape(xmesh)))
				eps_r[xhole*zlayer] = hole.eps_h

		plt.imshow(eps_r, cmap = "Greys", origin='lower',
						 extent=[xmin, xmax, zmin, zmax])
		plt.colorbar()
		plt.show()


# Class for a single layer in the PhC
class Layer(object):
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

		# Initialize an empty list of holes
		self.holes = []

	def add_hole(self, shape, params):
		# Add a hole with permittivity eps_h to the layer
		if shape == 'circle':
			hole = Circle(params['eps_h'], params['x'], 
							params['y'], params['r'])
		else:
			raise(NotImplementedError("Hole must be one of {'circle',}"))

		self.holes.append(hole)
		self.eps_avg = (self.eps_avg*(self.lattice['ec_area']- hole.area) + 
						hole.eps_h*hole.area)/self.lattice['ec_area']

	# NB: to implement method to compute FT of permittivity 


class Circle(object):
	'''
	Define class for a circular hole
	Other types of holes classes can also be added; they need to compute and 
	store the area of the hole and to have a method to compute the in-plane FT 
	'''
	def __init__(self, eps_h=1, x=0, y=0, r=0):
		self.eps_h=eps_h
		self.x = x
		self.y = y
		self.r = r
		self.area = np.pi*r**2

	def compute_ft(self, gvec):
		pass

	def is_inside(self, x, y):
		return np.square(x - self.x) + np.square(y - self.y) < np.square(self.r)