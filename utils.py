import numpy as np
import matplotlib.pyplot as plt

def init_lattice(a1=np.array([1, 0]), a2=np.array([0, 1])):
	'''
	Initialize a Bravais lattice; default is square
	'''
	ec_area = np.linalg.norm(np.cross(a1, a2))
	a3 = np.array([0, 0, 1])

	b1_3d = 2*np.pi*np.cross(a2, a3)[0:2]/np.dot(a1, np.cross(a2, a3)[0:2]) 
	b2_3d = 2*np.pi*np.cross(a3, a1)[0:2]/np.dot(a2, np.cross(a3, a1)[0:2])

	bz_area = np.linalg.norm(np.cross(b1_3d, b2_3d))

	lattice = {	'a1': a1,
				'a2': a2,
				'b1': b1_3d[0:2],
				'b2': b2_3d[0:2],
				'ec_area': ec_area,
				'bz_area': bz_area}

	return lattice

def plot_eps(eps_r, clim=None, ax=None, extent=None, cmap="Greys", cbar=False):

	if ax is None:
		fig, ax = plt.subplots(1, constrained_layout=True)

	im = ax.imshow(eps_r, cmap=cmap, origin='lower', extent=extent)
	if clim:
		im.set_clim(vmin=clim[0], vmax=clim[1])

	if cbar:
		plt.colorbar(im)

def plot_xz(phc, y=0, dx=2e-2, dz=2e-2, ax=None, clim=None, cbar=False):
	'''
	Plot an xz-cross section showing all the layers and shapes
	'''
	zmin = phc.layers[0].z_min - 1
	zmax = phc.layers[-1].z_max + 1

	xmax = np.abs(max([phc.lattice['a1'][0], phc.lattice['a2'][0]]))
	xmin = -xmax

	nx = int((xmax - xmin)//dx)
	nz = int((zmax - zmin)//dz)

	xgrid = xmin + np.arange(nx)*dx
	zgrid = zmin + np.arange(nz)*dz

	[xmesh, zmesh] = np.meshgrid(xgrid, zgrid)
	ymesh = y*np.ones(xmesh.shape)

	eps_r = phc.get_eps((xmesh, ymesh, zmesh))
	extent = [xmin, xmax, zmin, zmax]

	plot_eps(eps_r, clim=clim, ax=ax, extent=extent, cbar=cbar)


def plot_xy(phc, z=0, dx=2e-2, dy=2e-2, ax=None, clim=None, cbar=False):
	'''
	Plot an xy-cross section showing all the layers and shapes
	'''
	ymax = np.abs(max([phc.lattice['a1'][1], phc.lattice['a2'][1]]))
	ymin = -ymax

	xmax = np.abs(max([phc.lattice['a1'][0], phc.lattice['a2'][0]]))
	xmin = -xmax

	nx = int((xmax - xmin)//dx)
	ny = int((ymax - ymin)//dy)

	xgrid = xmin + np.arange(nx)*dx
	ygrid = ymin + np.arange(ny)*dy

	[xmesh, ymesh] = np.meshgrid(xgrid, ygrid)
	zmesh = z*np.ones(xmesh.shape)

	eps_r = phc.get_eps((xmesh, ymesh, zmesh))
	extent = [xmin, xmax, ymin, ymax]

	plot_eps(eps_r, clim=clim, ax=ax, extent=extent, cbar=cbar)


def plot_yz(phc, x=0, dy=2e-2, dz=2e-2, ax=None, clim=None, cbar=False):
	'''
	Plot a yz-cross section showing all the layers and shapes
	'''
	zmin = phc.layers[0].z_min - 1
	zmax = phc.layers[-1].z_max + 1

	ymax = np.abs(max([phc.lattice['a1'][1], phc.lattice['a2'][1]]))
	ymin = -ymax

	ny = int((ymax - ymin)//dy)
	nz = int((zmax - zmin)//dz)

	ygrid = ymin + np.arange(ny)*dy
	zgrid = zmin + np.arange(nz)*dz

	[ymesh, zmesh] = np.meshgrid(ygrid, zgrid)
	xmesh = x*np.ones(ymesh.shape)

	eps_r = phc.get_eps((xmesh, ymesh, zmesh))
	extent = [ymin, ymax, zmin, zmax]

	plot_eps(eps_r, clim=clim, ax=ax, extent=extent, cbar=cbar)