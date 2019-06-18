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
		plt.colorbar(im, ax=ax)

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

def ftinv(ft_coeff, gvec, xgrid, ygrid):
	''' 
	Returns the discrete inverse Fourier transform over a real-space mesh 
	defined by 'xgrid', 'ygrid', computed given a number of FT coefficients 
	'ft_coeff' defined over a set of reciprocal vectors 'gvec'.
	This could be sped up through an fft function but written like this it is 
	more general as we don't have to deal with grid and lattice issues.
	'''
	(xmesh, ymesh) = np.meshgrid(xgrid, ygrid)
	ftinv = np.zeros(xmesh.shape, dtype=np.complex128)

	# Take only the unique components
	(_, ind_unique) = np.unique(gvec, return_index=True, axis=1)

	for indg in ind_unique:
		ftinv += ft_coeff[indg]*np.exp(-1j*gvec[0, indg]*xmesh - \
							1j*gvec[1, indg]*ymesh)

	return ftinv

def ft2square(lattice, ft_coeff, gvec):
	'''
	Make a square array of Fourier components given a number of them defined 
	over a set of reciprocal vectors gvec.
	NB: function hasn't really been tested, just storing some code.
	'''
	if lattice.type not in ['hexagonal', 'square']:
		raise(NotImplementedError, "ft2square probably only works for \
				 a lattice initialized as 'square' or 'hexagonal'")

	dgx = np.abs(lattice.b1[0])
	dgy = np.abs(lattice.b2[1])
	nx = np.int_(np.abs(np.max(gvec[0, :])/dgx))
	ny = np.int_(np.abs(np.max(gvec[1, :])/dgy))
	nxtot = 2*nx + 1
	nytot = 2*ny + 1
	eps_ft = np.zeros((nxtot, nytot), dtype=np.complex128)
	gx_grid = np.arange(-nx, nx)*dgx
	gy_grid = np.arange(-ny, ny)*dgy

	for jG in range(gvec.shape[1]):
		nG = np.int_(gvec[:, jG]/[dgx, dgy])
		eps_ft[nx + nG1[0], ny + nG1[1]] = ft_coeff[jG]

	return (eps_ft, gx_grid, gy_grid)

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