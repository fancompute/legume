import numpy as np
import matplotlib.pyplot as plt

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

	xmax = np.abs(max([phc.lattice.a1[0], phc.lattice.a2[0]]))
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
	ymax = np.abs(max([phc.lattice.a1[1], phc.lattice.a2[1]]))
	ymin = -ymax

	xmax = np.abs(max([phc.lattice.a1[0], phc.lattice.a2[0]]))
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

	ymax = np.abs(max([phc.lattice.a1[1], phc.lattice.a2[1]]))
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


def plot_reciprocal(gme):
	'''
	Plot the reciprocal lattice of a GME object
	'''
	plt.plot(gme.gvec[0, :], gme.gvec[1, :], 'bx')
	plt.show()