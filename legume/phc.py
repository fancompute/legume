import matplotlib.pyplot as plt
import numpy as np

import legume.utils as utils
from .layer import ShapesLayer


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
			if cladding == 0 or cladding.lower() == 'l':
				lay = self.claddings[0]
			elif cladding == 1 or cladding.lower() == 'u':
				lay = self.claddings[1]
			else:
				raise ValueError("'cladding' must be 0 or 'l' for lower" \
								 "cladding and 1 or 'u' for upper cladding")
		else:
			if layer >= len(self.layers):
				raise ValueError("Layer index larger than total number of " \
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
								   np.square(np.max(abs(ymesh)))) / a_p) + 1

			for shape in layer.shapes:
				for n1 in range(-nmax, nmax):
					for n2 in range(-nmax, nmax):
						in_shape = shape.is_inside(xmesh +
												   n1 * a1[0] + n2 * a2[0], ymesh +
												   n1 * a1[1] + n2 * a2[1])
						eps_r[in_shape * zlayer] = shape.eps

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
		if N_layers == 1: ax = [ax]

		for indl in range(N_layers):
			zpos = (all_layers[indl].z_max + all_layers[indl].z_min) / 2
			utils.plot_xy(self, z=zpos, ax=ax[indl], Nx=Nx, Ny=Ny,
						  clim=[eps_min, eps_max], cbar=indl == N_layers - 1)
			if cladding:
				if indl > 0 and indl < N_layers - 1:
					ax[indl].set_title("xy in layer %d" % indl)
				elif indl == N_layers - 1:
					ax[0].set_title("xy in lower cladding")
					ax[-1].set_title("xy in upper cladding")
			else:
				ax[indl].set_title("xy in layer %d" % indl)
		plt.show()
