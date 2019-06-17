import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import toeplitz

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
	(xgrid, zgrid) = (phc.xy_grid(dx=dx)[0], phc.z_grid(dz=dz))

	[xmesh, zmesh] = np.meshgrid(xgrid, zgrid)
	ymesh = y*np.ones(xmesh.shape)

	eps_r = phc.get_eps((xmesh, ymesh, zmesh))
	extent = [xgrid[0], xgrid[-1], zgrid[0], zgrid[-1]]

	plot_eps(eps_r, clim=clim, ax=ax, extent=extent, cbar=cbar)


def plot_xy(phc, z=0, dx=2e-2, dy=2e-2, ax=None, clim=None, cbar=False):
	'''
	Plot an xy-cross section showing all the layers and shapes
	'''
	(xgrid, ygrid) = phc.xy_grid(dx=dx, dy=dy)
	[xmesh, ymesh] = np.meshgrid(xgrid, ygrid)
	zmesh = z*np.ones(xmesh.shape)

	eps_r = phc.get_eps((xmesh, ymesh, zmesh))
	extent = [xgrid[0], xgrid[-1], ygrid[0], ygrid[-1]]

	plot_eps(eps_r, clim=clim, ax=ax, extent=extent, cbar=cbar)


def plot_yz(phc, x=0, dy=2e-2, dz=2e-2, ax=None, clim=None, cbar=False):
	'''
	Plot a yz-cross section showing all the layers and shapes
	'''
	(ygrid, zgrid) = (phc.xy_grid(dy=dy)[1], phc.z_grid(dz=dz))
	[ymesh, zmesh] = np.meshgrid(ygrid, zgrid)
	xmesh = x*np.ones(ymesh.shape)

	eps_r = phc.get_eps((xmesh, ymesh, zmesh))
	extent = [ygrid[0], ygrid[-1], zgrid[0], zgrid[-1]]

	plot_eps(eps_r, clim=clim, ax=ax, extent=extent, cbar=cbar)


def plot_reciprocal(gme):
	'''
	Plot the reciprocal lattice of a GME object
	'''
	fig, ax = plt.subplots(1, constrained_layout=True)
	plt.plot(gme.gvec[0, :], gme.gvec[1, :], 'bx')
	ax.set_title("Reciprocal lattice")
	plt.show()


def toeplitz_block(n, T1, T2):
	'''
	Constructs a Hermitian Toeplitz-block-Toeplitz matrix with n blocks and 
	T1 in the first row and T2 in the first column of every block in the first
	row of blocks 
	'''
	ntot = T1.shape[0]
	p = int(ntot/n) # Linear size of each block
	print(p)
	Tmat = np.zeros((ntot, ntot), dtype=np.complex128)
	for ind1 in range(n):
	    for ind2 in range(ind1, n):
	        toep1=T1[(ind2-ind1)*p:(ind2-ind1+1)*p]
	        toep2=T2[(ind2-ind1)*p:(ind2-ind1+1)*p]
	        Tmat[ind1*p:(ind1+1)*p, ind2*p:(ind2+1)*p] = \
	        		toeplitz(toep2, toep1)

	return np.triu(Tmat) + np.conj(np.transpose(np.triu(Tmat,1)))