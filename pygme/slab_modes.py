import numpy as np
from scipy.optimize import brentq
from pygme.utils import RedhefferStar, I_alpha, J_alpha

def guided_modes(g_array, eps_array, d_array, n_modes=1, 
			step=1e-3, tol=1e-4, pol='TE'):
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
	step			: step size for root finding (should be smaller than the 
						minimum expected separation between modes)
	tol 			: tolerance in the omega boundaries for the root finding 
	pol 			: polarization, 'te' or 'tm'
	Output
	om_guided   	: array of size n_modes x length(g_array) with the guided 
					  mode frequencies
	coeffs_guided	: A, B coefficients of the modes in every layer
	'''
	om_guided = []
	coeffs_guided = []
	for ig, g in enumerate(g_array):

		om_lb = g/np.sqrt(eps_array[1:-1].max())
		om_ub = g/np.sqrt(max(eps_array[0],eps_array[-1]))
		if ig > 0:
			if len(omegas) == n_modes:
				# Dispersion cannot be faster than speed of light;
				# step is added just to be on the safe side
				om_ub = min(om_ub, 
					omegas[-1] + step + (g_array[ig] - g_array[ig-1]))

		(omegas, coeffs) = guided_mode_given_g(g=g, eps_array=eps_array, 
			d_array=d_array, n_modes=n_modes, omega_lb=om_lb, omega_ub=om_ub,
			step=step, tol=tol, pol=pol)
		om_guided.append(omegas)
		coeffs_guided.append(coeffs)
	return (om_guided, coeffs_guided)

def guided_mode_given_g(g, eps_array, d_array, n_modes=1, 
				omega_lb=None, omega_ub=None,
				step=1e-2, tol=1e-2, pol='TE'):
	'''
	Currently, we do 'bisection' in all the regions of interest until n_modes 
	target is met. For alternative variations, see guided_modes_draft.py
	'''
	if omega_lb is None:
		omega_lb = g/np.sqrt(eps_array[1:-1].max())
	if omega_ub is None:
		omega_ub = g/np.sqrt(max(eps_array[0],eps_array[-1]))

	if pol.lower()=='te':
		D22real = lambda x,*args: D22_TE(x,*args).real
		D22test = lambda x,*args: D22s_vec(x,*args,pol='TE').real
	elif pol.lower()=='tm':
		D22real = lambda x,*args: D22_TM(x,*args).real
		D22test = lambda x,*args: D22s_vec(x,*args,pol='TM').real
	else:
		raise ValueError("Polarization should be 'TE' or 'TM'.")

	# Making sure the bounds go all the way to om_ub - tol
	omega_bounds = np.append(np.arange(omega_lb + tol, omega_ub - tol, step), 
					omega_ub-tol) 

	omega_solutions = [] ## solving for D22_real
	coeffs = []

	D22s = D22test(omega_bounds, g, eps_array, d_array).real
	sign_change = np.where(D22s[0:-1]*D22s[1:] < 0)[0]
	lb = omega_bounds[0]

	for i in sign_change:
		if len(omega_solutions) >= n_modes:
			break
		lb = omega_bounds[i]
		ub = omega_bounds[i+1]
		omega = brentq(D22real,lb,ub,args=(g,eps_array,d_array))
		omega_solutions.append(omega)
		chi_array = chi(omega, g, eps_array)
		if pol.lower()=='te' or pol.lower()=='tm':				
			AB = AB_matrices(omega, g, eps_array, d_array, 
								chi_array, pol)
			norm = normalization_coeff(omega, g, eps_array, d_array, 
								AB, pol)
			# print(norm)
 
			coeffs.append(AB / np.sqrt(norm))
		else:
			raise ValueError("Polarization should be 'TE' or 'TM'")

	return (omega_solutions, coeffs)

def chi(omega, g, epses):
	'''
	Function to compute k_z
	Input
		omega			: frequency * 2π , in units of light speed / unit length
		epses 			: slab permittivity array
		g 				: wave vector along propagation direction (ß_x)
	Output
		k_z
	'''
	return np.sqrt(epses*omega**2 - g**2, dtype=np.complex)

def chi_vec(omegas, g, eps):
	'''
	Here omegas is an array and eps is a single number
	'''
	return np.sqrt(eps*omegas**2 - g**2, dtype=np.complex)

def S_T_matrices_TM(omega, g, eps_array, d_array):
	'''
	Function to get a list of S and T matrices for D22 calculation
	'''
	assert len(d_array)==len(eps_array)-2, \
			'd_array should have length = num_layers'
	chi_array = chi(omega, g, eps_array)
	# print(chi_array)
	S11 = (chi_array[:-1]/eps_array[:-1] + chi_array[1:]/eps_array[1:])
	S12 = -chi_array[:-1]/eps_array[:-1] + chi_array[1:]/eps_array[1:]
	S22 = S11
	S21 = S12
	S_matrices = 0.5 / (chi_array[1:]/eps_array[1:]).reshape(-1,1,1) * \
		np.array([[S11,S12],[S21,S22]]).transpose([2,0,1])
	T11 = np.exp(1j*chi_array[1:-1]*d_array/2)
	T22 = np.exp(-1j*chi_array[1:-1]*d_array/2)
	T_matrices = np.array([[T11,np.zeros_like(T11)],
		[np.zeros_like(T11),T22]]).transpose([2,0,1])
	return S_matrices, T_matrices

def D22_TM(omega, g, eps_array, d_array):
	'''
	Function to get eigen modes by solving D22=0
	Input
		omega 			: frequency * 2π , in units of light speed/unit length
		g   			: wave vector along propagation direction (ß_x)
		eps_array		: shape[M+1,1], slab permittivities
		d_array			: thicknesses of each layer
	Output
		abs(D_22) 
	(S matrices describe layers 0...M-1, T matrices describe layers 1...M-1)
	(num_layers = M-1)
	'''
	S_matrices, T_matrices = S_T_matrices_TM(omega, g, eps_array, d_array)
	D = S_matrices[0,:,:]
	for i,S in enumerate(S_matrices[1:]):
		T = T_matrices[i]
		D = S.dot(T.dot(T.dot(D)))
	return D[1,1]

def S_T_matrices_TE(omega, g, eps_array, d_array):
	'''
	Function to get a list of S and T matrices for D22 calculation
	'''
	assert len(d_array)==len(eps_array)-2, \
		'd_array should have length = num_layers'
	chi_array = chi(omega, g, eps_array)

	S11 = (chi_array[:-1] + chi_array[1:])
	S12 = -chi_array[:-1] + chi_array[1:]
	S22 = S11
	S21 = S12
	S_matrices = 0.5 / chi_array[1:].reshape(-1,1,1) * \
		np.array([[S11,S12],[S21,S22]]).transpose([2,0,1])
	T11 = np.exp(1j*chi_array[1:-1]*d_array/2)
	T22 = np.exp(-1j*chi_array[1:-1]*d_array/2)
	T_matrices = np.array([[T11,np.zeros_like(T11)],
		[np.zeros_like(T11),T22]]).transpose([2,0,1])
	return S_matrices, T_matrices

def D22_TE(omega, g, eps_array, d_array):
	'''
	Function to get eigen modes by solving D22=0
	Input
		omega 			: frequency * 2π , in units of light speed/unit length
		g   			: wave vector along propagation direction (ß_x)
		eps_array		: shape[M+1,1], slab permittivities
		d_array			: thicknesses of each layer
	Output
		abs(D_22) 
	(S matrices describe layers 0...M-1, T matrices describe layers 1...M-1)
	(num_layers = M-1)
	'''
	S_matrices, T_matrices = S_T_matrices_TE(omega, g, eps_array, d_array)
	D = S_matrices[0,:,:]
	for i,S in enumerate(S_matrices[1:]):
		T = T_matrices[i]
		D = S.dot(T.dot(T.dot(D)))
	return D[1,1]

def D22s_vec(omegas, g, eps_array, d_array, pol='TM'):
	'''
	Vectorized function to compute the matrix element D22 that needs to be zero
	Input
		omegas			: list of frequencies
		g   			: wave vector along propagation direction (ß_x)
		eps_array		: shape[M+1,1], slab permittivities
		d_array			: thicknesses of each layer
		pol 	 		: 'TE'/'TM'
	Output
		D_22 	  		: list of the D22 matrix elements corresponding to each 
							omega

	Note: This function is used to find intervals at which D22 switches sign.
	It is currently not used in the root finding, but it could be useful if 
	there is a routine that can take advantage of the vectorization. 
	'''

	N_oms = omegas.size # mats below will be of shape [2*N_oms, 2]

	def S_TE(eps1, eps2, chis1, chis2):
		# print((np.real(chis1) + np.imag(chis1)) / chis1)
		S11 = 0.5 / chis2 * (chis1 + chis2)
		S12 = 0.5 / chis2 * (-chis1 + chis2)
		return (S11, S12, S12, S11)

	def S_TM(eps1, eps2, chis1, chis2):
		S11 = 0.5 / (chis2/eps2) * (chis1/eps1 + chis2/eps2)
		S12 = 0.5 / (chis2/eps2) * (-chis1/eps1 + chis2/eps2)
		return (S11, S12, S12, S11)

	def S_T_prod(mats, omegas, g, eps1, eps2, d):
		'''
		Get the i-th S and T matrices for an array of omegas given the i-th slab 
		thickness d and permittivity of the slab eps1 and the next layer eps2
		'''

		chis1 = chi_vec(omegas, g, eps1)
		chis2 = chi_vec(omegas, g, eps2)

		if pol.lower() == 'te':
			(S11, S12, S21, S22) = S_TE(eps1, eps2, chis1, chis2)
		elif pol.lower() == 'tm':
			(S11, S12, S21, S22) = S_TM(eps1, eps2, chis1, chis2)
		
		T11 = np.exp(1j*chis1*d)
		T22 = np.exp(-1j*chis1*d)

		T_dot_mats = np.zeros(mats.shape, dtype=np.complex)
		T_dot_mats[0::2, :] = mats[0::2, :]*T11[:, np.newaxis]
		T_dot_mats[1::2, :] = mats[1::2, :]*T22[:, np.newaxis]

		S_dot_T = np.zeros(mats.shape, dtype=np.complex)
		S_dot_T[0::2, 0] = S11*T_dot_mats[0::2, 0] + S12*T_dot_mats[1::2, 0]
		S_dot_T[0::2, 1] = S11*T_dot_mats[0::2, 1] + S12*T_dot_mats[1::2, 1]
		S_dot_T[1::2, 0] = S21*T_dot_mats[0::2, 0] + S22*T_dot_mats[1::2, 0]
		S_dot_T[1::2, 1] = S21*T_dot_mats[0::2, 1] + S22*T_dot_mats[1::2, 1]

		return S_dot_T

	# Starting matrix array is constructed from S0
	(eps1, eps2) = (eps_array[0], eps_array[1])
	chis1 = chi_vec(omegas, g, eps1)
	chis2 = chi_vec(omegas, g, eps2)

	if pol.lower() == 'te':
		(S11, S12, S21, S22) = S_TE(eps1, eps2, chis1, chis2)
	elif pol.lower() == 'tm':
		(S11, S12, S21, S22) = S_TM(eps1, eps2, chis1, chis2)

	mats = np.zeros((2*N_oms, 2), dtype=np.complex)
	mats[0::2, 0] = S11
	mats[1::2, 0] = S21
	mats[0::2, 1] = S12
	mats[1::2, 1] = S22

	for il in range(1, eps_array.size - 1):
		mats = S_T_prod(mats, omegas, g, eps_array[il], 
							eps_array[il+1], d_array[il-1])

	D22s = mats[1::2, 1]
	return D22s

def AB_matrices(omega, g, eps_array, d_array, chi_array=None, pol='TE'):
	'''
	Function to calculate A,B coeff
	Output: array of shape [M+1,2]
	'''
	assert len(d_array)==len(eps_array)-2, \
		'd_array should have length = num_layers'
	if chi_array is None:
		chi_array = chi(omega, g, eps_array)

	if pol.lower()=='te':
		S_matrices, T_matrices = \
				S_T_matrices_TE(omega, g, eps_array, d_array)
	elif pol.lower()=='tm':
		S_matrices, T_matrices = \
				S_T_matrices_TM(omega, g, eps_array, d_array)
	else:
		raise Exception("Polarization should be 'TE' or 'TM'.")
	A0 = 0
	B0 = 1 
	AB0 = np.array([A0, B0]).reshape(-1,1)

	ABs = [AB0, T_matrices[0].dot(S_matrices[0].dot(AB0))] ### A, B coeff for each layer
	for i,S in enumerate(S_matrices[1:]):
		term = S_matrices[i+1].dot(T_matrices[i].dot(ABs[-1]))
		if i < len(S_matrices)-2:
			term = T_matrices[i+1].dot(term)
		ABs.append(term)
	return np.array(ABs)

def normalization_coeff(omega, g, eps_array, d_array, ABref, pol='TE'):
	'''
	Normalization of the guided modes (i.e. the A and B coeffs)
	'''
	assert len(d_array)==len(eps_array)-2, \
		'd_array should have length = num_layers'
	chi_array = chi(omega, g, eps_array)
	As = ABref[:, 0].ravel()
	Bs = ABref[:, 1].ravel()
	if pol == 'TM': 
		term1 = (np.abs(Bs[0])**2) * J_alpha(chi_array[0]-chi_array[0].conj())
		term2 = (np.abs(As[-1])**2) * J_alpha(chi_array[-1]-chi_array[-1].conj())
		term3 = (
				(As[1:-1] * As[1:-1].conj()) * \
				I_alpha(chi_array[1:-1]-chi_array[1:-1].conj(),d_array) + \
				(Bs[1:-1] * Bs[1:-1].conj()) * \
				I_alpha(chi_array[1:-1].conj()-chi_array[1:-1],d_array) + \
				(As[1:-1].conj() * Bs[1:-1]) * \
				I_alpha(-chi_array[1:-1]-chi_array[1:-1].conj(),d_array) + \
				(As[1:-1] * Bs[1:-1].conj()) * \
				I_alpha(chi_array[1:-1]+chi_array[1:-1].conj(),d_array)
				)
		return term1 + term2 + np.sum(term3)
	elif pol == 'TE':
		term1 = (np.abs(chi_array[0])**2 + g**2) * \
			(np.abs(Bs[0])**2) * J_alpha(chi_array[0]-chi_array[0].conj())
		term2 = (np.abs(chi_array[-1])**2 + g**2) * \
			(np.abs(As[-1])**2) * J_alpha(chi_array[-1]-chi_array[-1].conj())
		term3 = (np.abs(chi_array[1:-1])**2 + g**2) * (
				(As[1:-1] * As[1:-1].conj()) * \
				I_alpha(chi_array[1:-1]-chi_array[1:-1].conj(), d_array) + \
				(Bs[1:-1] * Bs[1:-1].conj()) * \
				I_alpha(chi_array[1:-1].conj()-chi_array[1:-1], d_array)
				) - \
				(np.abs(chi_array[1:-1])**2 - g**2) * (
				(As[1:-1].conj() * Bs[1:-1]) * \
				I_alpha(-chi_array[1:-1]-chi_array[1:-1].conj(), d_array) + \
				(As[1:-1] * Bs[1:-1].conj()) * \
				I_alpha(chi_array[1:-1]+chi_array[1:-1].conj(), d_array)
				)
		return term1 + term2 + np.sum(term3)
	else:
		raise Exception('Polarization should be TE or TM.')


def rad_modes(omega, g_array, eps_array, d_array, pol='TE', clad=0):
	''' 
	Function to compute the leaky modes of a multi-layer structure
	Input
	g_array  		: numpy array of wave vector amplitudes 
	eps_array		: numpy array of slab permittivities, starting with lower 
					  cladding and ending with upper cladding

	d_array			: thicknesses of each layer
	omega			: frequency of the radiative mode
	pol 			: polarization, 'te' or 'tm'
	clad 			: cladding index, 0 (lower) or 1 (upper)
	Output
	Xs, Ys 			: X, Y coefficients of the modes in every layer
	'''

	Xs = np.zeros((eps_array.size, g_array.size), dtype=np.complex128)
	Ys = np.zeros((eps_array.size, g_array.size), dtype=np.complex128)
	for ig, g in enumerate(g_array):
		# Get the scattering and transfer matrices
		if pol.lower()=='te' and clad==0:
			S_mat, T_mat = S_T_matrices_TE(omega, g, np.flip(eps_array), 
							np.flip(d_array))
		elif pol.lower()=='te' and clad==1:
			S_mat, T_mat = S_T_matrices_TE(omega, g, eps_array, d_array)
		elif pol.lower()=='tm' and clad==0:
			S_mat, T_mat = S_T_matrices_TM(omega, g, np.flip(eps_array), 
							np.flip(d_array))
		elif pol.lower()=='tm' and clad==1:
			S_mat, T_mat = S_T_matrices_TM(omega, g, eps_array, d_array)
		
		# Compute the transfer matrix coefficients
		coeffs = np.zeros((2, eps_array.size), dtype=np.complex128)
		coeffs[:, 0] = [1, 0]
		coeffs[:, 1] = T_mat[0].dot(S_mat[0].dot(coeffs[:, 0]))
		for i, S in enumerate(S_mat[1:-1]):
			T2 = T_mat[i+1]
			T1 = T_mat[i]
			coeffs[:, i+2] = T2.dot(S.dot(T1.dot(coeffs[:, i+1])))
		coeffs[:, -1] = S_mat[-1].dot(T_mat[-1].dot(coeffs[:, -2]))

		coeffs = coeffs / coeffs[0, -1] 
		if pol=='te':
			c_ind = [0, -1]
			coeffs = coeffs/np.sqrt(eps_array[c_ind[clad]])

		# Assign correctly based on which cladding the modes radiate to
		if clad == 0:
			Xs[:, ig] = np.flip(coeffs[0, :])
			Ys[:, ig] = np.flip(coeffs[1, :])
		elif clad == 1:
			Xs[:, ig] = coeffs[1, :]
			Ys[:, ig] = coeffs[0, :]

	'''
	(Xs, Ys) corresponds to the X, W coefficients for TE radiative modes in 
	Andreani and Gerace PRB (2006), and to the Z, Y coefficients for TM modes
	'''
	return (Xs, Ys)



