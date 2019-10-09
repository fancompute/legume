import numpy as np

from legume.backend import backend as bd

'''
Notation is following Andreani and Gerace PRB 2006 with the exception that all
wave-vectors are always defined as (eps*omega^2/c^2 - g^2) regardless of 
whether that makes them real or imaginary
'''

'''===========HELPER FUNCTIONS FOR Z-INTEGRATION============'''
def I_alpha(a, d): # integrate exp(iaz)dz from -d/2 to d/2
    a = a + 1e-20
    return 2 / a * bd.sin(a*d/2)

def J_alpha(a): # integrate exp(iaz)dz from 0 to inf
    a = a + 1e-20
    return 1j / a

'''===========MATRIX ELEMENTS BETWEEN GUIDED SLAB MODES============'''
def mat_te_te(eps_array, d_array, eps_inv_mat, indmode1, oms1,
                    As1, Bs1, chis1, indmode2, oms2, As2, Bs2, 
                    chis2, qq):
    '''
    Matrix block for TE-TE mode coupling
    '''
    
    # Contribution from lower cladding
    indmat = np.ix_(indmode1, indmode2)
    mat = eps_inv_mat[0][indmat]* \
            eps_array[0]**2 * \
            np.outer(np.conj(Bs1[0, :]), Bs2[0, :]) * \
            J_alpha(chis2[0, :] - np.conj(chis1[0, :][:, np.newaxis]))

    # Contribution from upper cladding
    mat = mat + eps_inv_mat[-1][indmat]* \
            eps_array[-1]**2 * \
            np.outer(np.conj(As1[-1, :]), As2[-1, :]) * \
            J_alpha(chis2[-1, :] - np.conj(chis1[-1, :][:, np.newaxis]))

    # Contributions from layers
    for il in range(1, d_array.size+1):
        mat = mat + eps_inv_mat[il][indmat] *\
        eps_array[il]**2 * ( \
        np.outer(np.conj(As1[il, :]), As2[il, :])*I_alpha(chis2[il, :] -\
            np.conj(chis1[il, :][:, np.newaxis]), d_array[il-1]) + \
        np.outer(np.conj(Bs1[il, :]), Bs2[il, :])*I_alpha(-chis2[il, :] +\
            np.conj(chis1[il, :][:, np.newaxis]), d_array[il-1]) + \
        np.outer(np.conj(As1[il, :]), Bs2[il, :])*I_alpha(-chis2[il, :] -\
            np.conj(chis1[il, :][:, np.newaxis]), d_array[il-1]) +
        np.outer(np.conj(Bs1[il, :]), As2[il, :])*I_alpha(chis2[il, :] +\
            np.conj(chis1[il, :][:, np.newaxis]), d_array[il-1])  )

    # Final pre-factor      
    mat = mat * np.outer(oms1**2, oms2**2) * (qq[indmat])

    return mat

def mat_tm_tm(eps_array, d_array, eps_inv_mat, gk, indmode1, oms1,
                    As1, Bs1, chis1, indmode2, oms2, As2, Bs2, 
                    chis2, pp):
    '''
    Matrix block for TM-TM mode coupling
    '''
    
    # Contribution from lower cladding
    indmat = np.ix_(indmode1, indmode2)
    mat = eps_inv_mat[0][indmat]*(pp[indmat] * \
            np.outer(np.conj(chis1[0, :]), chis2[0, :]) + \
            np.outer(gk[indmode1], gk[indmode2])) * \
            np.outer(np.conj(Bs1[0, :]), Bs2[0, :]) * \
            J_alpha(chis2[0, :] - np.conj(chis1[0, :][:, np.newaxis]))

    # Contribution from upper cladding
    mat = mat + eps_inv_mat[-1][indmat]*(pp[indmat] * \
            np.outer(np.conj(chis1[-1, :]), chis2[-1, :]) + \
            np.outer(gk[indmode1], gk[indmode2])) * \
            np.outer(np.conj(As1[-1, :]), As2[-1, :]) * \
            J_alpha(chis2[-1, :] - np.conj(chis1[-1, :][:, np.newaxis]))

    # Contributions from layers
    for il in range(1, d_array.size+1):
        mat = mat + eps_inv_mat[il][indmat]*( \
        (pp[indmat] * np.outer(np.conj(chis1[il, :]), chis2[il, :]) + \
            np.outer(gk[indmode1], gk[indmode2])) * ( \
        np.outer(np.conj(As1[il, :]), As2[il, :])*I_alpha(chis2[il, :] -\
            np.conj(chis1[il, :][:, np.newaxis]), d_array[il-1]) + \
        np.outer(np.conj(Bs1[il, :]), Bs2[il, :])*I_alpha(-chis2[il, :] +\
            np.conj(chis1[il, :][:, np.newaxis]), d_array[il-1]) ) - \
        (pp[indmat] * np.outer(np.conj(chis1[il, :]), chis2[il, :]) - \
            np.outer(gk[indmode1], gk[indmode2])) * ( \
        np.outer(np.conj(As1[il, :]), Bs2[il, :])*I_alpha(-chis2[il, :] -\
            np.conj(chis1[il, :][:, np.newaxis]), d_array[il-1]) +
        np.outer(np.conj(Bs1[il, :]), As2[il, :])*I_alpha(chis2[il, :] +\
            np.conj(chis1[il, :][:, np.newaxis]), d_array[il-1]))  )
    # Note: in Vitaly's thesis, there's a typo on line 3 of eq. (3.41), 
    # the term in brackets should be A*B*I + B*A*I instead of minus

    return mat

def mat_te_tm(eps_array, d_array, eps_inv_mat, indmode1, oms1,
                    As1, Bs1, chis1, indmode2, oms2, As2, Bs2, 
                    chis2, qp, signed_1j):
    '''
    Matrix block for TM-TE mode coupling
    '''
    
    # Contribution from lower cladding
    indmat = np.ix_(indmode1, indmode2)
    mat = eps_inv_mat[0][indmat] * \
            eps_array[0] * 1j*chis2[0, :][np.newaxis, :]* \
            np.outer(np.conj(Bs1[0, :]), Bs2[0, :]) * \
            J_alpha(chis2[0, :] - np.conj(chis1[0, :][:, np.newaxis]))

    # Contribution from upper cladding
    mat = mat - eps_inv_mat[-1][indmat] * \
            eps_array[-1] * 1j*chis2[-1, :][np.newaxis, :] * \
            np.outer(np.conj(As1[-1, :]), As2[-1, :]) * \
            J_alpha(chis2[-1, :] - np.conj(chis1[-1, :][:, np.newaxis]))

    # Contributions from layers
    for il in range(1, d_array.size+1):
        mat = mat + signed_1j * eps_inv_mat[il][indmat] *\
        eps_array[il] * chis2[il, :][np.newaxis, :] * (-\
        np.outer(np.conj(As1[il, :]), As2[il, :])*I_alpha(chis2[il, :] -\
            np.conj(chis1[il, :][:, np.newaxis]), d_array[il-1]) + \
        np.outer(np.conj(Bs1[il, :]), Bs2[il, :])*I_alpha(-chis2[il, :] +\
            np.conj(chis1[il, :][:, np.newaxis]), d_array[il-1]) + \
        np.outer(np.conj(As1[il, :]), Bs2[il, :])*I_alpha(-chis2[il, :] -\
            np.conj(chis1[il, :][:, np.newaxis]), d_array[il-1]) -
        np.outer(np.conj(Bs1[il, :]), As2[il, :])*I_alpha(chis2[il, :] +\
            np.conj(chis1[il, :][:, np.newaxis]), d_array[il-1])  )

    # Final pre-factor
    mat = mat * (oms1**2)[:, np.newaxis] * qp[indmat]

    return mat

'''===========MATRIX ELEMENTS BETWEEN GUIDED AND RADIATIVE MODES============'''
def rad_te_te(eps_array, d_array, eps_inv_mat, indmode1, oms1,
            As1, Bs1, chis1, indmoder, omr, Xsr, Ysr, 
            chisr, qq):
    '''
    Coupling of TE guided modes to TE radiative modes
    '''

    # Contribution from lower cladding
    indmat = np.ix_(indmode1, indmoder)
    mat = eps_inv_mat[0][indmat]* \
            eps_array[0]**2 * (\
            np.outer(np.conj(Bs1[0, :]), Ysr[0, :]) * J_alpha(
            -np.conj(chis1[0, :][:, np.newaxis])-chisr[np.newaxis, 0])+
            np.outer(np.conj(Bs1[0, :]), Xsr[0, :]) * J_alpha(
            -np.conj(chis1[0, :][:, np.newaxis])+chisr[np.newaxis, 0]))

    # Contribution from upper cladding
    mat = mat + eps_inv_mat[-1][indmat]* \
            eps_array[-1]**2 * (\
            np.outer(np.conj(As1[-1, :]), Ysr[-1, :]) * J_alpha(
            -np.conj(chis1[-1, :][:, np.newaxis])+chisr[np.newaxis, -1])+
            np.outer(np.conj(As1[-1, :]), Xsr[-1, :]) * J_alpha(
            -np.conj(chis1[-1, :][:, np.newaxis])-chisr[np.newaxis, -1]))

    # Contributions from layers
    for il in range(1, d_array.size+1):
        mat = mat + eps_inv_mat[il][indmat] *\
        eps_array[il]**2 * ( \
        np.outer(np.conj(As1[il, :]), Ysr[il, :])*I_alpha(chisr[il, :] -\
            np.conj(chis1[il, :][:, np.newaxis]), d_array[il-1]) + \
        np.outer(np.conj(Bs1[il, :]), Xsr[il, :])*I_alpha(-chisr[il, :] +\
            np.conj(chis1[il, :][:, np.newaxis]), d_array[il-1]) + \
        np.outer(np.conj(As1[il, :]), Xsr[il, :])*I_alpha(-chisr[il, :] -\
            np.conj(chis1[il, :][:, np.newaxis]), d_array[il-1]) +\
        np.outer(np.conj(Bs1[il, :]), Ysr[il, :])*I_alpha(chisr[il, :] +\
            np.conj(chis1[il, :][:, np.newaxis]), d_array[il-1])  )

    # Final pre-factor
    mat = mat * bd.outer(oms1**2, omr*bd.ones(indmoder.size)) * qq

    return mat

def rad_te_tm(eps_array, d_array, eps_inv_mat, indmode1, oms1,
            As1, Bs1, chis1, indmoder, omr, Xsr, Ysr, 
            chisr, qp):
    '''
    Coupling of TE guided modes to TM radiative modes
    '''

    # Contribution from lower cladding
    indmat = np.ix_(indmode1, indmoder)
    mat = eps_inv_mat[0][indmat]* \
            eps_array[0] * (\
            np.outer(np.conj(Bs1[0, :]), -chisr[0, :]*Ysr[0, :]) * J_alpha(
            -np.conj(chis1[0, :][:, np.newaxis])-chisr[np.newaxis, 0])+
            np.outer(np.conj(Bs1[0, :]), chisr[0, :]*Xsr[0, :]) * J_alpha(
            -np.conj(chis1[0, :][:, np.newaxis])+chisr[np.newaxis, 0]))

    # Contribution from upper cladding
    mat = mat + eps_inv_mat[-1][indmat]* \
        eps_array[-1] * (\
        np.outer(np.conj(As1[-1, :]), -chisr[-1, :]*Ysr[-1, :]) * J_alpha(
            -np.conj(chis1[-1, :][:, np.newaxis])+chisr[np.newaxis, -1])+
        np.outer(np.conj(As1[-1, :]), chisr[-1, :]*Xsr[-1, :]) * J_alpha(
            -np.conj(chis1[-1, :][:, np.newaxis])-chisr[np.newaxis, -1]))

    # Contributions from layers
    for il in range(1, d_array.size+1):
        mat = mat + eps_inv_mat[il][indmat] *\
        eps_array[il] * chisr[il, :][np.newaxis, :] * ( -\
        np.outer(np.conj(As1[il, :]), Ysr[il, :])*I_alpha(chisr[il, :] -\
            np.conj(chis1[il, :][:, np.newaxis]), d_array[il-1]) + \
        np.outer(np.conj(Bs1[il, :]), Xsr[il, :])*I_alpha(-chisr[il, :] +\
            np.conj(chis1[il, :][:, np.newaxis]), d_array[il-1]) + \
        np.outer(np.conj(As1[il, :]), Xsr[il, :])*I_alpha(-chisr[il, :] -\
            np.conj(chis1[il, :][:, np.newaxis]), d_array[il-1]) -\
        np.outer(np.conj(Bs1[il, :]), Ysr[il, :])*I_alpha(chisr[il, :] +\
            np.conj(chis1[il, :][:, np.newaxis]), d_array[il-1])  )

    # Final pre-factor
    mat = mat * 1j * (oms1**2)[:, np.newaxis] * qp

    return mat

def rad_tm_te(eps_array, d_array, eps_inv_mat, indmode1, oms1,
            As1, Bs1, chis1, indmoder, omr, Xsr, Ysr, 
            chisr, pq):
    '''
    Coupling of TM guided modes to TE radiative modes
    '''

    # Contribution from lower cladding
    indmat = np.ix_(indmode1, indmoder)
    mat = eps_inv_mat[0][indmat]* eps_array[0] * \
        (1j*chis1[0, :]*np.conj(Bs1[0, :]))[:, np.newaxis] * (\
            Ysr[0, :][np.newaxis, :] * J_alpha(
            -np.conj(chis1[0, :][:, np.newaxis])-chisr[np.newaxis, 0]) +
            Xsr[0, :][np.newaxis, :] * J_alpha(
            -np.conj(chis1[0, :][:, np.newaxis])+chisr[np.newaxis, 0]))

    # Contribution from upper cladding
    mat = mat - eps_inv_mat[-1][indmat]*eps_array[-1]*\
        (1j*chis1[-1, :]*np.conj(As1[-1, :]))[:, np.newaxis] * (\
            Ysr[-1, :][np.newaxis, :] * J_alpha(
            -np.conj(chis1[-1, :][:, np.newaxis])+chisr[np.newaxis, -1]) +
            Xsr[-1, :][np.newaxis, :] * J_alpha(
            -np.conj(chis1[-1, :][:, np.newaxis])-chisr[np.newaxis, -1]))

    # Contributions from layers
    for il in range(1, d_array.size+1):
        mat = mat - 1j * eps_inv_mat[il][indmat] *\
        eps_array[il] * chis1[il, :][:, np.newaxis] * ( -\
        np.outer(np.conj(As1[il, :]), Ysr[il, :])*I_alpha(chisr[il, :] -\
            np.conj(chis1[il, :][:, np.newaxis]), d_array[il-1]) + \
        np.outer(np.conj(Bs1[il, :]), Xsr[il, :])*I_alpha(-chisr[il, :] +\
            np.conj(chis1[il, :][:, np.newaxis]), d_array[il-1]) - \
        np.outer(np.conj(As1[il, :]), Xsr[il, :])*I_alpha(-chisr[il, :] -\
            np.conj(chis1[il, :][:, np.newaxis]), d_array[il-1]) +\
        np.outer(np.conj(Bs1[il, :]), Ysr[il, :])*I_alpha(chisr[il, :] +\
            np.conj(chis1[il, :][:, np.newaxis]), d_array[il-1])  )

    # Final pre-factor
    mat = mat * omr * pq

    return mat

def rad_tm_tm(eps_array, d_array, eps_inv_mat, gk, indmode1, oms1,
            As1, Bs1, chis1, indmoder, omr, Xsr, Ysr, 
            chisr, pp):
    '''
    Coupling of TM guided modes to TM radiative modes
    '''

    # Contribution from lower cladding
    indmat = np.ix_(indmode1, indmoder)
    mat = eps_inv_mat[0][indmat] * \
            np.conj(Bs1[0, :])[:, np.newaxis] *(\
            (np.outer(gk[indmode1], gk[indmoder]) + 
            np.outer(chis1[0, :], chisr[0, :]) * pp) * 
            Ysr[0, :][np.newaxis, :] * J_alpha(
            -np.conj(chis1[0, :][:, np.newaxis])-chisr[np.newaxis, 0]) +
            (np.outer(gk[indmode1], gk[indmoder]) - 
            np.outer(chis1[0, :], chisr[0, :]) * pp) *
            Xsr[0, :][np.newaxis, :] * J_alpha(
            -np.conj(chis1[0, :][:, np.newaxis])+chisr[np.newaxis, 0]))

    # Contribution from upper cladding
    mat = mat + eps_inv_mat[-1][indmat] * \
            np.conj(As1[-1, :])[:, np.newaxis] * (\
            (np.outer(gk[indmode1], gk[indmoder]) - 
            np.outer(chis1[-1, :], chisr[-1, :]) * pp) * 
            Ysr[-1, :][np.newaxis, :] * J_alpha(
            -np.conj(chis1[-1, :][:, np.newaxis])+chisr[np.newaxis, -1]) +
            (np.outer(gk[indmode1], gk[indmoder]) + 
            np.outer(chis1[-1, :], chisr[-1, :]) * pp) *
            Xsr[-1, :][np.newaxis, :] * J_alpha(
            -np.conj(chis1[-1, :][:, np.newaxis])-chisr[np.newaxis, -1]))

    # Contributions from layers
    for il in range(1, d_array.size+1):
        mat = mat + eps_inv_mat[il][indmat] * ( \
        (np.outer(gk[indmode1], gk[indmoder]) + 
            np.outer(chis1[il, :], chisr[il, :]) * pp) * ( \
        np.outer(np.conj(As1[il, :]), Ysr[il, :])*I_alpha(chisr[il, :] - \
            np.conj(chis1[il, :][:, np.newaxis]), d_array[il-1]) + \
        np.outer(np.conj(Bs1[il, :]), Xsr[il, :])*I_alpha(-chisr[il, :] +\
            np.conj(chis1[il, :][:, np.newaxis]), d_array[il-1]) ) + \
        (np.outer(gk[indmode1], gk[indmoder]) - 
            np.outer(chis1[il, :], chisr[il, :]) * pp) * ( 
        np.outer(np.conj(As1[il, :]), Xsr[il, :])*I_alpha(-chisr[il, :] -\
            np.conj(chis1[il, :][:, np.newaxis]), d_array[il-1]) +\
        np.outer(np.conj(Bs1[il, :]), Ysr[il, :])*I_alpha(chisr[il, :] +\
            np.conj(chis1[il, :][:, np.newaxis]), d_array[il-1])  ) )

    return mat