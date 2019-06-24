import numpy as np
from scipy.optimize import fsolve, bisect
''' 
Function to compute the guided modes of a multi-layer structure
Input
	g_array  		: numpy array of wave vector amplitudes 
	eps_array		: numpy array of slab permittivities, starting with lower 
					  cladding and ending with upper cladding
	d_array			: thicknesses of each layer
	n_modes			: maximum number of solutions to look for, starting from 
					  the lowest-frequency one
	omega_lb		: lower bound of omega
	omega_ub		: upper bound of omega
	step			: step size for root finding (i.e. the expected separation btw modes)
Output
	om_guided   	: array of size n_modes x length(g_array) with the guided 
					  mode frequencies
	(Will need further outputs in the future)  
'''
def guided_modes(g_array, eps_array, d_array, n_modes=1, omega_lb=0, omega_ub=20, step=1e-3, tol=1e-4, mode='TE'):
 	### One can consider optimizing the optimization routine.
 	### Currently, we do 'bisection' in all the regions of interest until n_modes target is met
 	### Alternative strategy is to call optimize fsolve from omega=0
 	### Both are implemented below for reference.
 	if n_modes==1: ## using fsolve
 		pass ## can implement when it is needed..
 	# 	cur = step ## initial guess
 	# 	prev = 0 ## previous guess
 	# 	n_more_times = 2 ## try n_more_times after finding a solution to find a smaller solution
 	# 	not_found = True
 	# 	while not_found:
 	# 		omega = fsolve(D22, [cur], args=(gs,epses,ds))
 	# 		if omega>0:
 	# 			not_found=False
 	# 			lb = prev
 	# 			ub = omega
 	# 			for i in range(n_more_times):
 	# 				new_omega = fsolve(D22, [guess], args=(gs,epses,ds))
 	# 				if new_omega>0:
 	# 					omega = new_omega
 	# 					ub = (prev + omega)/2
 	# 				else:
 	# 					lb = (prev + omega)/2
 	# 					omega = new_omega
 	# 				if omega_trial=
 	#	return

	omega_bounds = np.arange(omega_lb, omega_ub, step)
	for i in range(omega_bounds[:-1]):
		lb=omega_bounds[i]
		ub=omega_boudns[i+1]
		try:


 	return om_guided 

def chi(omega, g, eps):
	'''
	Function to compute k_z
	Input
		omega			: frequency * 2π , in units of light speed / unit length
		eps_array		: slab permittivity
		g 				: wave vector along propagation direction (ß_x)
	Output
		k_z
	'''
	return np.sqrt(eps*omega**2 - g**2, dtype=np.complex)

def D22_TM(omega, g_array, eps_array, d_array, n_modes=1):
	'''
	Function to get eigen modes by solving D22=0
	Input
		omega 			: frequency * 2π , in units of light speed / unit length
		g_array			: shape[M+1,1], wave vector along propagation direction (ß_x)
		eps_array		: shape[M+1,1], slab permittivities
		d_array			: thicknesses of each layer
	Output
		abs(D_22) 
	(S matrices describe layers 0...M-1, T matrices describe layers 1...M-1)
	(num_layers = M-1)
	'''
	assert len(g_array)==len(eps_array), 'g_array and eps_array should both have length = num_layers+2'
	assert len(d_array)==len(eps_array)-2, 'd_array should have length = num_layers'
	chi_array = chi(omega, g_array, eps_array)
	S11 = eps_array[1:]*chi_array[:-1] + eps_array[:-1]*chi_array[1:]
	S12 = eps_array[1:]*chi_array[:-1] - eps_array[:-1]*chi_array[1:]
	S22 = S11
	S21 = S12
	S_matrices = 0.5/eps_array[1:]/chi_array[:-1] * np.array([[S11,S12],[S21,S22]]).transpose([2,0,1]) ## this is how TE differs from TM
	T11 = np.exp(1j*chi_array[1:-1]*d_array)
	T22 = np.exp(-1j*chi_array[1:-1]*d_array)
	T_matrices = np.array([[T11,np.zeros_like(T11)],[np.zeros_like(T11),T22]]).transpose([2,0,1])
	D = S_matrices[0,:,:]
	for i,S in enumerate(S_matrices[1:]):
		T = T_matrices[i]
		D = S.dot(T.dot(D))
	return (D[1,1])

def D22_TE(omega, g_array, eps_array, d_array, n_modes=1):
	'''
	Function to get eigen modes by solving D22=0
	Input
		omega 			: frequency * 2π , in units of light speed / unit length
		g_array			: shape[M+1,1], wave vector along propagation direction (ß_x)
		eps_array		: shape[M+1,1], slab permittivities
		d_array			: thicknesses of each layer
	Output
		abs(D_22) 
	(S matrices describe layers 0...M-1, T matrices describe layers 1...M-1)
	(num_layers = M-1)
	'''
	assert len(g_array)==len(eps_array), 'g_array and eps_array should both have length = num_layers+2'
	assert len(d_array)==len(eps_array)-2, 'd_array should have length = num_layers'
	chi_array = chi(omega, g_array, eps_array)
	S11 = eps_array[1:]*chi_array[:-1] + eps_array[:-1]*chi_array[1:]
	S12 = eps_array[1:]*chi_array[:-1] - eps_array[:-1]*chi_array[1:]
	S22 = S11
	S21 = S12
	S_matrices = 0.5/chi_array[:-1] * np.array([[S11,S12],[S21,S22]]).transpose([2,0,1]) ## this is how TE differs from TM
	T11 = np.exp(1j*chi_array[1:-1]*d_array)
	T22 = np.exp(-1j*chi_array[1:-1]*d_array)
	T_matrices = np.array([[T11,np.zeros_like(T11)],[np.zeros_like(T11),T22]]).transpose([2,0,1])
	D = S_matrices[0,:,:]
	for i,S in enumerate(S_matrices[1:]):
		T = T_matrices[i]
		D = S.dot(T.dot(D))
	return (D[1,1])

