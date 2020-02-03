import numpy as np

from legume.backend import backend as bd
from .slab_modes import I_alpha, J_alpha

"""
Everything is as defined in the legume manuscript
"""

def IJ_layer(il, Nl, arg, ds):
    """
    Integral along z in every layer computed analytically
    """
    if il == 0:
        return J_alpha(-arg)
    elif il == Nl-1:
        return J_alpha(arg)
    else:
        return I_alpha(arg, ds[il-1])

def mat_te_te(eps_array, d_array, eps_inv_mat, indmode1, oms1,
                    As1, Bs1, chis1, indmode2, oms2, As2, Bs2, 
                    chis2, qq):
    """
    Matrix block for TE-TE mode coupling
    """

    # Index matrix selecting the participating modes
    indmat = np.ix_(indmode1, indmode2)
    # Number of layers
    Nl = eps_array.size

    # Build the matrix layer by layer
    mat = bd.zeros((indmode1.size, indmode2.size))
    for il in range(0, Nl):
        mat = mat + eps_inv_mat[il][indmat] * \
        eps_array[il]**2 * \
        (   bd.outer(bd.conj(As1[il, :]), As2[il, :]) * 
            IJ_layer(il, Nl, chis2[il, :] - bd.conj(chis1[il, :][:, bd.newaxis]),
                d_array) + 
            bd.outer(bd.conj(Bs1[il, :]), Bs2[il, :]) * 
            IJ_layer(il, Nl, -chis2[il, :] + bd.conj(chis1[il, :][:, bd.newaxis]),
                d_array) +
            bd.outer(bd.conj(As1[il, :]), Bs2[il, :]) *
            IJ_layer(il, Nl, -chis2[il, :] - bd.conj(chis1[il, :][:, bd.newaxis]),
                d_array) + 
            bd.outer(bd.conj(Bs1[il, :]), As2[il, :]) *
            IJ_layer(il, Nl, chis2[il, :] + bd.conj(chis1[il, :][:, bd.newaxis]),
                d_array)
        )

    # Final pre-factor      
    mat = mat * bd.outer(oms1**2, oms2**2) * qq[indmat]
    return mat

def mat_tm_tm(eps_array, d_array, eps_inv_mat, gk, indmode1, oms1,
                    As1, Bs1, chis1, indmode2, oms2, As2, Bs2, 
                    chis2, pp):
    """
    Matrix block for TM-TM mode coupling
    """
    
    # Index matrix selecting the participating modes
    indmat = np.ix_(indmode1, indmode2)
    # Number of layers
    Nl = eps_array.size

    # Build the matrix layer by layer
    mat = bd.zeros((indmode1.size, indmode2.size))
    for il in range(0, Nl):
        mat = mat + eps_inv_mat[il][indmat]*( 
        (pp[indmat] * bd.outer(bd.conj(chis1[il, :]), chis2[il, :]) + \
        bd.outer(gk[indmode1], gk[indmode2])) * ( 
            bd.outer(bd.conj(As1[il, :]), As2[il, :]) * 
            IJ_layer(il, Nl, chis2[il, :] - bd.conj(chis1[il, :][:, bd.newaxis]),
                d_array) + 
            bd.outer(bd.conj(Bs1[il, :]), Bs2[il, :]) * 
            IJ_layer(il, Nl, -chis2[il, :] + bd.conj(chis1[il, :][:, bd.newaxis]),
                d_array)) - \
        (pp[indmat] * bd.outer(bd.conj(chis1[il, :]), chis2[il, :]) - \
        bd.outer(gk[indmode1], gk[indmode2])) * ( 
            bd.outer(bd.conj(As1[il, :]), Bs2[il, :]) *
            IJ_layer(il, Nl, -chis2[il, :] - bd.conj(chis1[il, :][:, bd.newaxis]),
                d_array) + 
            bd.outer(bd.conj(Bs1[il, :]), As2[il, :]) *
            IJ_layer(il, Nl, chis2[il, :] + bd.conj(chis1[il, :][:, bd.newaxis]),
                d_array))  )

    return mat

def mat_te_tm(eps_array, d_array, eps_inv_mat, indmode1, oms1,
                    As1, Bs1, chis1, indmode2, oms2, As2, Bs2, 
                    chis2, qp):
    """
    Matrix block for TE-TM mode coupling
    """
    
    # Index matrix selecting the participating modes
    indmat = np.ix_(indmode1, indmode2)
    # Number of layers
    Nl = eps_array.size

    # Build the matrix layer by layer
    mat = bd.zeros((indmode1.size, indmode2.size))
    # Contributions from layers
    for il in range(0, Nl):
        mat = mat + 1j * eps_inv_mat[il][indmat] * \
        eps_array[il] * chis2[il, :][bd.newaxis, :] * ( 
        -bd.outer(bd.conj(As1[il, :]), As2[il, :]) * 
            IJ_layer(il, Nl, chis2[il, :] - bd.conj(chis1[il, :][:, bd.newaxis]),
                d_array)  
        +bd.outer(bd.conj(Bs1[il, :]), Bs2[il, :]) * 
            IJ_layer(il, Nl, -chis2[il, :] + bd.conj(chis1[il, :][:, bd.newaxis]),
                d_array) 
        +bd.outer(bd.conj(As1[il, :]), Bs2[il, :]) *
            IJ_layer(il, Nl, -chis2[il, :] - bd.conj(chis1[il, :][:, bd.newaxis]),
                d_array)  
        -bd.outer(bd.conj(Bs1[il, :]), As2[il, :]) *
            IJ_layer(il, Nl, chis2[il, :] + bd.conj(chis1[il, :][:, bd.newaxis]),
                d_array)  )

    # Final pre-factor
    mat = mat * (oms1**2)[:, bd.newaxis] * qp[indmat]
    return mat

def mat_tm_te(eps_array, d_array, eps_inv_mat, indmode1, oms1,
                    As1, Bs1, chis1, indmode2, oms2, As2, Bs2, 
                    chis2, pq):
    """
    Matrix block for TM-TE mode coupling
    """
    
        # Index matrix selecting the participating modes
    indmat = np.ix_(indmode1, indmode2)
    # Number of layers
    Nl = eps_array.size

    # Build the matrix layer by layer
    mat = bd.zeros((indmode1.size, indmode2.size))
    for il in range(0, Nl):
        mat = mat + 1j * eps_inv_mat[il][indmat] * \
        eps_array[il] * bd.conj(chis1[il, :])[:, bd.newaxis] * ( 
        bd.outer(bd.conj(As1[il, :]), As2[il, :]) * 
            IJ_layer(il, Nl, chis2[il, :] - bd.conj(chis1[il, :][:, bd.newaxis]),
                d_array)  
        -bd.outer(bd.conj(Bs1[il, :]), Bs2[il, :]) * 
            IJ_layer(il, Nl, -chis2[il, :] + bd.conj(chis1[il, :][:, bd.newaxis]),
                d_array)  
        +bd.outer(bd.conj(As1[il, :]), Bs2[il, :]) *
            IJ_layer(il, Nl, -chis2[il, :] - bd.conj(chis1[il, :][:, bd.newaxis]),
                d_array)  
        -bd.outer(bd.conj(Bs1[il, :]), As2[il, :]) *
            IJ_layer(il, Nl, chis2[il, :] + bd.conj(chis1[il, :][:, bd.newaxis]),
                d_array)  )

    # Final pre-factor
    mat = mat * (oms2**2)[bd.newaxis, :] * pq[indmat]
    return mat