'''
Various utilities used in the main code.
NOTE: there should be no autograd functions here, only plain numpy.
'''

import numpy as np
from scipy.linalg import toeplitz
import matplotlib.pyplot as plt

def plot_eps(eps_r, clim=None, ax=None, extent=None, cmap='Greys', cbar=False):

	if ax is None:
		fig, ax = plt.subplots(1, constrained_layout=True)

	im = ax.imshow(eps_r, cmap=cmap, origin='lower', extent=extent)
	if clim:
		im.set_clim(vmin=clim[0], vmax=clim[1])

	if cbar:
		plt.colorbar(im, ax=ax)
		
	return im

def plot_xz(phc, y=0, Nx=100, Nz=50, ax=None, clim=None, cbar=False, cmap='Greys'):
	'''
	Plot an xz-cross section showing all the layers and shapes
	'''
	(xgrid, zgrid) = (phc.lattice.xy_grid(Nx=Nx)[0], phc.z_grid(Nz=Nz))

	[xmesh, zmesh] = np.meshgrid(xgrid, zgrid)
	ymesh = y*np.ones(xmesh.shape)

	eps_r = phc.get_eps((xmesh, ymesh, zmesh))
	extent = [xgrid[0], xgrid[-1], zgrid[0], zgrid[-1]]

	plot_eps(eps_r, clim=clim, ax=ax, extent=extent, cbar=cbar, cmap=cmap)


def plot_xy(phc, z=0, Nx=100, Ny=100, ax=None, clim=None, cbar=False, cmap='Greys'):
	'''
	Plot an xy-cross section showing all the layers and shapes
	'''
	(xgrid, ygrid) = phc.lattice.xy_grid(Nx=Nx, Ny=Ny)
	[xmesh, ymesh] = np.meshgrid(xgrid, ygrid)
	zmesh = z*np.ones(xmesh.shape)

	eps_r = phc.get_eps((xmesh, ymesh, zmesh))
	extent = [xgrid[0], xgrid[-1], ygrid[0], ygrid[-1]]

	plot_eps(eps_r, clim=clim, ax=ax, extent=extent, cbar=cbar, cmap=cmap)


def plot_yz(phc, x=0, Ny=100, Nz=50, ax=None, clim=None, cbar=False, cmap='Greys'):
	'''
	Plot a yz-cross section showing all the layers and shapes
	'''
	(ygrid, zgrid) = (phc.lattice.xy_grid(Ny=Ny)[1], phc.z_grid(Nz=Nz))
	[ymesh, zmesh] = np.meshgrid(ygrid, zgrid)
	xmesh = x*np.ones(ymesh.shape)

	eps_r = phc.get_eps((xmesh, ymesh, zmesh))
	extent = [ygrid[0], ygrid[-1], zgrid[0], zgrid[-1]]

	plot_eps(eps_r, clim=clim, ax=ax, extent=extent, cbar=cbar, cmap=cmap)


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
	(g_unique, ind_unique) = np.unique(gvec, return_index=True, axis=1)

	for indg in ind_unique:
		ftinv += ft_coeff[indg]*np.exp(-1j*gvec[0, indg]*xmesh - \
							1j*gvec[1, indg]*ymesh)

	# # Do the x- and y-transforms separately 
	# # I wrote this but then realized it doesn't improve anything
	# (gx_u, indx) = np.unique(gvec[0, :], return_inverse=True)
	# for ix, gx in enumerate(gx_u):
	# 	ind_match = np.where(indx==ix)[0]
	# 	(gy_u, indy) = np.unique(gvec[1, ind_match], return_index=True)
	# 	term = np.zeros(xmesh.shape, dtype=np.complex128)
	# 	for iy, gy in enumerate(gy_u):
	# 		# print(ft_coeff[indx[indy[iy]]])
	# 		term += ft_coeff[ind_match[indy[iy]]]*np.exp(-1j*gy*ymesh)
	# 	ftinv += term*np.exp(-1j*gx*xmesh)

	# # Can also be defined through a DFT matrix but it doesn't seem faster and 
	# # it's *very* memory intensive.
	# exp_matrix = xmesh.reshape((-1, 1)).dot(g_unique[[0], :]) + \
	# 				ymesh.reshape((-1, 1)).dot(g_unique[[1], :])

	# dft_matrix = np.exp(1j*exp_matrix)
	# ftinv = dft_matrix.dot(ft_coeff[ind_unique]).reshape(xmesh.shape)
	# print(ftinv)
	return ftinv

def ft2square(lattice, ft_coeff, gvec):
	'''
	Make a square array of Fourier components given a number of them defined 
	over a set of reciprocal vectors gvec.
	NB: function hasn't really been tested, just storing some code.
	'''
	if lattice.type not in ['hexagonal', 'square']:
		raise NotImplementedError("ft2square probably only works for" \
				 "a lattice initialized as 'square' or 'hexagonal'")

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

def grad_num(fn, arg, step_size=1e-7):
	''' Numerically differentiate `fn` w.r.t. its argument `arg` 
	`arg` can be a numpy array of arbitrary shape
	`step_size` can be a number or an array of the same shape as `arg` '''

	N = arg.size
	shape = arg.shape
	gradient = np.zeros((N,))
	f_old = fn(arg)

	if type(step_size) == float:
		step = step_size*np.ones((N))
	else:
		step = step_size.ravel()

	for i in range(N):
		arg_new = arg.flatten()
		arg_new[i] += step[i]
		f_new_i = fn(arg_new.reshape(shape))
		gradient[i] = (f_new_i - f_old) / step[i]

	return gradient.reshape(shape)

def toeplitz_block(n, T1, T2):
	'''
	Constructs a Hermitian Toeplitz-block-Toeplitz matrix with n blocks and 
	T1 in the first row and T2 in the first column of every block in the first
	row of blocks 
	'''
	ntot = T1.shape[0]
	p = int(ntot/n) # Linear size of each block
	Tmat = np.zeros((ntot, ntot), dtype=T1.dtype)
	for ind1 in range(n):
		for ind2 in range(ind1, n):
			toep1 = T1[(ind2-ind1)*p:(ind2-ind1+1)*p]
			toep2 = T2[(ind2-ind1)*p:(ind2-ind1+1)*p]
			Tmat[ind1*p:(ind1+1)*p, ind2*p:(ind2+1)*p] = \
					toeplitz(toep2, toep1)

	return np.triu(Tmat) + np.conj(np.transpose(np.triu(Tmat,1)))

	return np.triu(Tmat) + np.conj(np.transpose(np.triu(Tmat,1)))

def get_value(x):
	'''
	This is for when using the 'autograd' backend and you want to detach an 
	ArrayBox and just convert it to a numpy array.
	'''
	if str(type(x)) == "<class 'autograd.numpy.numpy_boxes.ArrayBox'>":
		return x._value
	else:
		return x

def RedhefferStar(SA,SB): #SA and SB are both 2x2 matrices;
	assert type(SA) == np.ndarray, 'not np.matrix'
	assert type(SB) == np.ndarray, 'not np.matrix'

	I = 1;
	# once we break every thing like this, we should still have matrices
	SA_11 = SA[0, 0]; SA_12 = SA[0, 1]; SA_21 = SA[1, 0]; SA_22 = SA[1, 1];
	SB_11 = SB[0, 0]; SB_12 = SB[0, 1]; SB_21 = SB[1, 0]; SB_22 = SB[1, 1];

	D = 1.0/(I-SB_11*SA_22);
	F = 1.0/(I-SA_22*SB_11);

	SAB_11 = SA_11 + SA_12*D*SB_11*SA_21;
	SAB_12 = SA_12*D*SB_12;
	SAB_21 = SB_21*F*SA_21;
	SAB_22 = SB_22 + SB_21*F*SA_22*SB_12;

	SAB = np.array([[SAB_11, SAB_12],[SAB_21, SAB_22]])
	return SAB;

def I_alpha(a, d): # integrate exp(iaz)dz from -d/2 to d/2
	a = a + 1e-20
	return 2 / a * np.sin(a*d/2)

def J_alpha(a): # integrate exp(iaz)dz from 0 to inf
	a = a + 1e-20
	return 1j / a