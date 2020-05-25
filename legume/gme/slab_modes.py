import numpy as np
from legume.backend import backend as bd
from legume.utils import get_value

"""===========HELPER FUNCTIONS FOR Z-INTEGRATION============"""
def I_alpha(a, d): # integrate exp(iaz)dz from -d/2 to d/2
    a = a + 1e-20
    return 2 / a * bd.sin(a*d/2)

def J_alpha(a): # integrate exp(iaz)dz from 0 to inf
    a = a + 1e-20
    return 1j / a

"""============GUIDED MODE COMPUTATION============"""
def guided_modes(g_array: np.ndarray, eps_array: np.ndarray,
                d_array: np.ndarray, n_modes: int=1, 
                step: float=1e-3, tol: float=1e-4, pol: str='TE'):
    """ 
    Function to compute the guided modes of a multi-layer structure
    Input
    g_array         : array of wave vector amplitudes 
    eps_array       : array of slab permittivities, starting with lower 
                      cladding and ending with upper cladding
    d_array         : array of thickness of each layer
    n_modes         : maximum number of solutions to look for, starting from 
                      the lowest-frequency one
    omega_lb        : lower bound of omega
    omega_ub        : upper bound of omega
    step            : step size for root finding (should be smaller than the 
                        minimum expected separation between modes)
    tol             : tolerance in the omega boundaries for the root finding 
    pol             : polarization, 'te' or 'tm'

    Output
    om_guided       : array of size n_modes x length(g_array) with the guided 
                      mode frequencies
    coeffs_guided   : A, B coefficients of the modes in every layer
    """

    om_guided = []
    coeffs_guided = []
    for ig, g in enumerate(g_array):

        g_val = max([g, 1e-4])
        compute_g = True

        eps_val = get_value(eps_array)
        om_lb = g_val/np.sqrt(eps_val[1:-1]).max()
        om_ub = g_val/np.sqrt(max(eps_val[0], eps_val[-1]))
        if ig > 0:
            if len(omegas) == n_modes:
                # Dispersion cannot be faster than speed of light;
                # step is added just to be on the safe side
                om_ub = min(om_ub, get_value(omegas[-1]) + step + 
                                        (g_array[ig] - g_array[ig-1]))
            """
            Check if the guided mode needs to be computed at all; when using the
            gmode_compute = 'exact' option, there might be identical g-points
            in the g_array, so we don't want to compute those multiple times
            """
            compute_g = np.abs(g - g_array[ig-1]) > 1e-8

        if compute_g:
            (omegas, coeffs) = guided_mode_given_g(g=g_val, eps_array=eps_array, 
            d_array=d_array, n_modes=n_modes,
            omega_lb=om_lb, omega_ub=om_ub, step=step, tol=tol, pol=pol)

        om_guided.append(omegas)
        coeffs_guided.append(coeffs)
    return (om_guided, coeffs_guided)

def guided_mode_given_g(g, eps_array, d_array, n_modes=1, 
                omega_lb=None, omega_ub=None,
                step=1e-2, tol=1e-2, pol='TE'):
    """
    Finds the first 'n_modes' guided modes of polarization 'pol' for a given 'g'
    """

    # Set lower and upper bound on the possible solutions
    eps_val = get_value(eps_array)
    d_val = get_value(d_array)
    if omega_lb is None:
        omega_lb = g/np.sqrt(eps_val[1:-1].max())
    if omega_ub is None:
        omega_ub = g/np.sqrt(max(eps_val[0],eps_val[-1]))

    omega_lb = omega_lb*(1+tol)
    omega_ub = omega_ub*(1-tol)

    # D22real is used in the fsolve; D22test is vectorized and used on a test 
    # array of omega-s to find sign flips
    D22real = lambda x,*args: bd.real(D22(x, *args, pol=pol))
    D22test = lambda x,*args: bd.real(D22s_vec(x, *args, pol=pol))

    # Making sure the bounds go all the way to omega_ub
    omega_bounds = np.append(np.arange(omega_lb, omega_ub, step), omega_ub) 

    # Variables storing the solutions
    omega_solutions = [] 
    coeffs = []

    # Find omegas between which D22 changes sign
    D22s = D22test(omega_bounds, g, eps_val, d_val).real
    sign_change = np.where(D22s[0:-1]*D22s[1:] < 0)[0]
    lb = omega_bounds[0]

    # Use fsolve to find the first 'n_modes' guided modes
    for i in sign_change:
        if len(omega_solutions) >= n_modes:
            break
        lb = omega_bounds[i]
        ub = omega_bounds[i+1]

        # Compute guided mode frequency 
        omega = bd.fsolve_D22(D22real, lb, ub, g, eps_array, d_array)
        omega_solutions.append(omega)
        chi_array = chi(omega, g, eps_array)
        if pol.lower()=='te' or pol.lower()=='tm':
            # Compute A-B coefficients              
            AB = AB_matrices(omega, g, eps_array, d_array, 
                                chi_array, pol)
            # Normalize
            norm = normalization_coeff(omega, g, eps_array, d_array, 
                                chi_array, AB, pol)
            coeffs.append(AB / bd.sqrt(norm))
        else:
            raise ValueError("Polarization should be 'TE' or 'TM'")

    return (omega_solutions, coeffs)

def chi(omega, g, eps):
    """
    Function to compute chi_j, the z-direction wave-vector in each layer j
    Either omega is an array and eps is a float, or vice versa.
    Input
        omega           : frequency * 2π , in units of light speed/unit length
        eps             : slab permittivity array
        g               : wave vector along propagation direction 
    Output
        chi             : array of chi_j for all layers j including claddings
    """
    sqarg = bd.array(eps*omega**2 - g**2, dtype=bd.complex)
    return bd.where(bd.real(sqarg) >=0, bd.sqrt(sqarg),
                         1j*bd.sqrt(-sqarg))

def chis_3layer(omega, g, eps_array):
    """
    """
    (eps1, eps2, eps3) = [e for e in eps_array]
    chis1 = 1j*bd.sqrt(g**2 - eps1*omega**2)
    chis2 = bd.array(bd.sqrt(-g**2 + eps2*omega**2), dtype=bd.complex)
    chis3 = 1j*bd.sqrt(g**2 - eps3*omega**2)
    
    return (chis1, chis2, chis3)

# def chis_nlayer(omega, g, eps_array):
#     """
#     """
#     chis = []
#     for e in eps_array:
#         if g**2 - e*omega**2 > 0:
#             chis.append(1j*bd.sqrt(g**2 - e*omega**2))
#         else:
#             chis.append(bd.sqrt(-g**2 + e*omega**2).astype(bd.complex))
    
#     return bd.array(chis)

def S_T_matrices_TM(omega, g, eps_array, d_array):
    """
    Function to get a list of S and T matrices for D22 calculation
    """
    assert len(d_array)==len(eps_array)-2, \
            'd_array should have length = num_layers'
    chi_array = chi(omega, g, eps_array)
    # print(chi_array)
    S11 = (chi_array[:-1]/eps_array[:-1] + chi_array[1:]/eps_array[1:])
    S12 = -chi_array[:-1]/eps_array[:-1] + chi_array[1:]/eps_array[1:]
    S22 = S11
    S21 = S12
    S_matrices = 0.5 / (chi_array[1:]/eps_array[1:]).reshape(-1,1,1) * \
        bd.array([[S11,S12],[S21,S22]]).transpose([2,0,1])
    T11 = bd.exp(1j*chi_array[1:-1]*d_array/2)
    T22 = bd.exp(-1j*chi_array[1:-1]*d_array/2)
    T_matrices = bd.array([[T11,bd.zeros(T11.shape)],
        [bd.zeros(T11.shape),T22]]).transpose([2,0,1])
    return S_matrices, T_matrices

def S_T_matrices_TE(omega, g, eps_array, d_array):
    """
    Function to get a list of S and T matrices for D22 calculation
    """
    assert len(d_array)==len(eps_array)-2, \
        'd_array should have length = num_layers'
    chi_array = chi(omega, g, eps_array)

    S11 = (chi_array[:-1] + chi_array[1:])
    S12 = -chi_array[:-1] + chi_array[1:]
    S22 = S11
    S21 = S12
    S_matrices = 0.5 / chi_array[1:].reshape(-1,1,1) * \
        bd.array([[S11,S12],[S21,S22]]).transpose([2,0,1])
    T11 = bd.exp(1j*chi_array[1:-1]*d_array/2)
    T22 = bd.exp(-1j*chi_array[1:-1]*d_array/2)
    T_matrices = bd.array([[T11,bd.zeros(T11.shape)],
        [bd.zeros(T11.shape),T22]]).transpose([2,0,1])
    return S_matrices, T_matrices

def D22(omega, g, eps_array, d_array, pol='TM'):
    """
    Function to get TE guided modes by solving D22=0
    Input
        omega           : frequency * 2π , in units of light speed/unit length
        g               : wave vector along propagation direction
        eps_array       : shape[M+1,1], slab permittivities
        d_array         : thicknesses of each layer
    Output
        D_22
    """
    if eps_array.size == 3:
        (eps1, eps2, eps3) = [e for e in eps_array]
        # (chis1, chis2, chis3) = [chi(omega, g, e) for e in eps_array]
        (chis1, chis2, chis3) = chis_3layer(omega, g, eps_array)

        tcos = -1j*bd.cos(chis2*d_array)
        tsin = -bd.sin(chis2*d_array)

        if pol.lower() == 'te':
            D22 = chis2*(chis1 + chis3)*tcos + \
                    (chis1*chis3 + bd.square(chis2))*tsin
        elif pol.lower() == 'tm':    
            D22 = chis2/eps2*(chis1/eps1 + chis3/eps3)*tcos + \
                    (chis1/eps1*chis3/eps3 + bd.square(chis2/eps2))*tsin
        return D22
    else:
        if pol.lower() == 'te':
            S_mat, T_mat = S_T_matrices_TE(omega, g, eps_array, d_array)
        elif pol.lower() == 'tm':
            S_mat, T_mat = S_T_matrices_TM(omega, g, eps_array, d_array)
        else:
            raise ValueError("Polarization should be 'TE' or 'TM'.")

        D = S_mat[0,:,:]
        for i,S in enumerate(S_mat[1:]):
            T = T_mat[i]
            D = bd.dot(S, bd.dot(T, bd.dot(T, D)))
        return D[1,1]

def D22s_vec(omegas, g, eps_array, d_array, pol='TM'):
    """
    Vectorized function to compute the matrix element D22 that needs to be zero
    Input
        omegas          : list of frequencies
        g               : wave vector along propagation direction (ß_x)
        eps_array       : shape[M+1,1], slab permittivities
        d_array         : thicknesses of each layer
        pol             : 'TE'/'TM'
    Output
        D_22            : list of the D22 matrix elements corresponding to each 
                            omega

    Note: This function is used to find intervals at which D22 switches sign.
    It is currently not used in the root finding, but it could be useful if 
    there is a routine that can take advantage of the vectorization. 
    """
    if isinstance(omegas, float):
        omegas = np.array([omegas])

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
        """
        Get the i-th S and T matrices for an array of omegas given the i-th slab 
        thickness d and permittivity of the slab eps1 and the next layer eps2
        """

        chis1 = chi(omegas, g, eps1)
        chis2 = chi(omegas, g, eps2)

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

    if eps_array.size == 3:
        (eps1, eps2, eps3) = [e for e in eps_array]
        # (chis1, chis2, chis3) = [chi(omegas, g, e) for e in eps_array]
        (chis1, chis2, chis3) = chis_3layer(omegas, g, eps_array)

        tcos = -1j*bd.cos(chis2*d_array)
        tsin = -bd.sin(chis2*d_array)

        if pol.lower() == 'te':
            D22s = chis2*(chis1 + chis3)*tcos + \
                    (chis1*chis3 + bd.square(chis2))*tsin
        elif pol.lower() == 'tm':    
            D22s = chis2/eps2*(chis1/eps1 + chis3/eps3)*tcos + \
                    (chis1/eps1*chis3/eps3 + bd.square(chis2/eps2))*tsin

    else:
        # Starting matrix array is constructed from S0
        (eps1, eps2) = (eps_array[0], eps_array[1])
        chis1 = chi(omegas, g, eps1)
        chis2 = chi(omegas, g, eps2)

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
    """
    Function to calculate A,B coeff
    Output: array of shape [M+1,2]
    """
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
    AB0 = bd.array([A0, B0]).reshape(-1,1)

    # A, B coeff for each layer
    ABs = [AB0, bd.dot(T_matrices[0], bd.dot(S_matrices[0], AB0))]
    for i,S in enumerate(S_matrices[1:]):
        term = bd.dot(S_matrices[i+1], bd.dot(T_matrices[i], ABs[-1]))
        if i < len(S_matrices)-2:
            term = bd.dot(T_matrices[i+1], term)
        ABs.append(term)
    return bd.array(ABs)

def normalization_coeff(omega, g, eps_array, d_array, chi_array, ABref, 
                            pol='TE'):
    """
    Normalization of the guided modes (i.e. the A and B coeffs)
    """
    assert len(d_array)==len(eps_array)-2, \
        'd_array should have length = num_layers'
    if chi_array is None:
        chi_array = chi(omega, g, eps_array)
    As = ABref[:, 0].ravel()
    Bs = ABref[:, 1].ravel()
    if pol == 'TM': 
        term1 = (bd.abs(Bs[0])**2) * J_alpha(chi_array[0]-bd.conj(chi_array[0]))
        term2 = (bd.abs(As[-1])**2) * \
                    J_alpha(chi_array[-1]-bd.conj(chi_array[-1]))
        term3 = (
                (bd.abs(As[1:-1])**2 + bd.abs(Bs[1:-1])**2) * \
                I_alpha(chi_array[1:-1]-bd.conj(chi_array[1:-1]),d_array) + 
                (bd.conj(As[1:-1]) * Bs[1:-1] + As[1:-1] * bd.conj(Bs[1:-1])) *
                I_alpha(-chi_array[1:-1]-bd.conj(chi_array[1:-1]),d_array)  )
        return term1 + term2 + bd.sum(term3)
    elif pol == 'TE':
        term1 = (bd.abs(chi_array[0])**2 + g**2) * \
            (bd.abs(Bs[0])**2) * J_alpha(chi_array[0]-bd.conj(chi_array[0]))
        term2 = (bd.abs(chi_array[-1])**2 + g**2) * \
            (bd.abs(As[-1])**2) * J_alpha(chi_array[-1]-bd.conj(chi_array[-1]))
        term3 = (bd.abs(chi_array[1:-1])**2 + g**2) * (
                (bd.abs(As[1:-1])**2 + bd.abs(Bs[1:-1])**2) * \
                I_alpha(chi_array[1:-1]-bd.conj(chi_array[1:-1]), d_array)) + \
                (g**2 - bd.abs(chi_array[1:-1])**2) * (
                (bd.conj(As[1:-1]) * Bs[1:-1] + As[1:-1] * bd.conj(Bs[1:-1])) *
                I_alpha(-chi_array[1:-1]-bd.conj(chi_array[1:-1]), d_array)  )
        return term1 + term2 + bd.sum(term3)
    else:
        raise Exception('Polarization should be TE or TM.')


def rad_modes(omega: float, g_array: np.ndarray, eps_array: np.ndarray,
            d_array: np.ndarray, pol: str='TE', clad: int=0):
    """ 
    Function to compute the radiative modes of a multi-layer structure
    Input
    g_array         : numpy array of wave vector amplitudes 
    eps_array       : numpy array of slab permittivities, starting with lower 
                      cladding and ending with upper cladding

    d_array         : thicknesses of each layer
    omega           : frequency of the radiative mode
    pol             : polarization, 'te' or 'tm'
    clad            : radiating into cladding index, 0 (lower) or 1 (upper)
    Output
    Xs, Ys          : X, Y coefficients of the modes in every layer
    """

    Xs, Ys = [], []
    for ig, g in enumerate(g_array):
        g_val = max([g, 1e-10])
        # Get the scattering and transfer matrices
        if pol.lower()=='te' and clad==0:
            S_mat, T_mat = S_T_matrices_TE(omega, g_val, eps_array[::-1], 
                            d_array[::-1])
        elif pol.lower()=='te' and clad==1:
            S_mat, T_mat = S_T_matrices_TE(omega, g_val, eps_array, d_array)
        elif pol.lower()=='tm' and clad==0:
            S_mat, T_mat = S_T_matrices_TM(omega, g_val, eps_array[::-1], 
                            d_array[::-1])
        elif pol.lower()=='tm' and clad==1:
            S_mat, T_mat = S_T_matrices_TM(omega, g_val, eps_array, d_array)
        
        # Compute the transfer matrix coefficients
        coeffs = [bd.array([0, 1])]
        coeffs.append(bd.dot(T_mat[0], bd.dot(S_mat[0], coeffs[0])))
        for i, S in enumerate(S_mat[1:-1]):
            T2 = T_mat[i+1]
            T1 = T_mat[i]
            coeffs.append(bd.dot(T2, bd.dot(S, bd.dot(T1, coeffs[-1]))))
        coeffs.append(bd.dot(S_mat[-1], bd.dot(T_mat[-1], coeffs[-1])))
        coeffs = bd.array(coeffs, dtype=bd.complex).transpose()

        # Normalize
        coeffs = coeffs / coeffs[1, -1] 
        if pol=='te':
            c_ind = [0, -1]
            coeffs = coeffs/bd.sqrt(eps_array[c_ind[clad]])/omega
        # Assign correctly based on which cladding the modes radiate to
        if clad == 0:
            Xs.append(coeffs[0, ::-1].ravel())
            Ys.append(coeffs[1, ::-1].ravel())
        elif clad == 1:
            Xs.append(coeffs[1, :].ravel())
            Ys.append(coeffs[0, :].ravel())

    Xs = bd.array(Xs, dtype=bd.complex).transpose()
    Ys = bd.array(Ys, dtype=bd.complex).transpose()

    # Fix the dimension if g_array is an empty list
    if len(g_array)==0:
        Xs = bd.ones((eps_array.size, 1))*Xs
        Ys = bd.ones((eps_array.size, 1))*Ys

    """
    (Xs, Ys) corresponds to the X, W coefficients for TE radiative modes in 
    Andreani and Gerace PRB (2006), and to the Z, Y coefficients for TM modes

    Note that there's an error in the manuscript; within our definitions, the 
    correct statement should be: X3 = 0 for states out-going in the lower 
    cladding; normalize through W1; and W1 = 0 for states out-going in the upper
    cladding; normalize through X3.
    """
    return (Xs, Ys)



