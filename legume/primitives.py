import numpy as np
import scipy.sparse as sp
from functools import partial
from .utils import toeplitz_block, get_value, fsolve

from autograd.extend import primitive, defvjp, vspace
from autograd import grad, vector_jacobian_product
import autograd.numpy as npa

''' Define here various primitives needed for the main code 
To use with both numpy and autograd backends, define the autograd primitive of 
a numpy function fnc as fnc_ag, and then define the vjp'''

def T(x): return np.swapaxes(x, -1, -2)

'''=========== NP.SQRT STABLE AROUND 0 =========== '''
sqrt_ag = primitive(np.sqrt)

def vjp_maker_sqrt(ans, x):
    def vjp(g):
        return g * 0.5 * (x + 1e-10)**0.5/(x + 1e-10)
        # return np.where(np.abs(x) > 1e-10, g * 0.5 * x**-0.5, 0.)
    return vjp

defvjp(sqrt_ag, vjp_maker_sqrt)

'''=========== TOEPLITZ-BLOCK =========== '''

toeplitz_block_ag = primitive(toeplitz_block)

def vjp_maker_TB_T1(Tmat, n, T1, T2):
    ''' Gives vjp for Tmat = toeplitz_block(n, T1, T2) w.r.t. T1'''
    def vjp(v):
        ntot = Tmat.shape[0]
        p = int(ntot/n) # Linear size of each block
        vjac = np.zeros(T1.shape, dtype=np.complex128)

        for ind1 in range(n):
            for ind2 in range(ind1, n):
                for indp in range(p):
                    vjac[(ind2-ind1)*p:(ind2-ind1+1)*p-indp] += \
                        v[ind1*p + indp, ind2*p+indp:(ind2+1)*p]

                    if ind2 > ind1:
                        vjac[(ind2-ind1)*p:(ind2-ind1+1)*p-indp] += \
                            np.conj(v[ind2*p+indp:(ind2+1)*p, ind1*p + indp])
        return vjac

    return vjp

def vjp_maker_TB_T2(Tmat, n, T1, T2):
    ''' Gives vjp for Tmat = toeplitz_block(n, T1, T2) w.r.t. T2'''
    def vjp(v):
        ntot = Tmat.shape[0]
        p = int(ntot/n) # Linear size of each block
        vjac = np.zeros(T2.shape, dtype=np.complex128)

        for ind1 in range(n):
            for ind2 in range(ind1, n):
                for indp in range(p):
                    vjac[(ind2-ind1)*p+1:(ind2-ind1+1)*p-indp] += \
                        v[ind1*p+indp+1:(ind1+1)*p, ind2*p+indp]

                    if ind2 > ind1:
                        vjac[(ind2-ind1)*p+1:(ind2-ind1+1)*p-indp] += \
                            np.conj(v[ind2*p+indp, ind1*p+indp+1:(ind1+1)*p])
        return vjac

    return vjp

defvjp(toeplitz_block_ag, None, vjp_maker_TB_T1, vjp_maker_TB_T2)

'''=========== NUMPY.LINALG.EIGH =========== '''

eigh_ag = primitive(np.linalg.eigh)

def vjp_maker_eigh(ans, x, UPLO='L'):
    """Gradient for eigenvalues and vectors of a hermitian matrix."""
    N = x.shape[-1]
    w, v = ans              # Eigenvalues, eigenvectors.
    vc = np.conj(v)
    
    def vjp(g):
        wg, vg = g          # Gradient w.r.t. eigenvalues, eigenvectors.
        w_repeated = np.repeat(w[:, np.newaxis], N, axis=-1)

        # Eigenvalue part
        vjp_temp = np.dot(vc * wg[np.newaxis, :], T(v)) 

        # Add eigenvector part only if non-zero backward signal is present.
        # This can avoid NaN results for degenerate cases if the function 
        # depends on the eigenvalues only.
        if np.any(vg):
            off_diag = np.ones((N, N)) - np.eye(N)
            F = off_diag / (T(w_repeated) - w_repeated + np.eye(N))
            vjp_temp += np.dot(np.dot(vc, F * np.dot(T(v), vg)), T(v))

        # eigh always uses only the lower or the upper part of the matrix
        # we also have to make sure broadcasting works
        reps = np.array(x.shape)
        reps[-2:] = 1

        if UPLO == 'L':
            tri = np.tile(np.tril(np.ones(N), -1), reps)
        elif UPLO == 'U':
            tri = np.tile(np.triu(np.ones(N), 1), reps)
        
        return np.real(vjp_temp)*np.eye(vjp_temp.shape[-1]) + \
            (vjp_temp + np.conj(T(vjp_temp))) * tri

    return vjp

defvjp(eigh_ag, vjp_maker_eigh)

'''=========== MATRIX INVERSE =========== '''
'''We define this here without the `einsum` notation that's used in autograd.
`einsum` allows broadcasting (which we don't care about), but is slower 
(which we do)
'''

inv_ag = primitive(np.linalg.inv)

def vjp_maker_inv(ans, x):
    return lambda g: -np.dot(np.dot(T(ans), g), T(ans))
defvjp(inv_ag, vjp_maker_inv)

'''=========== SCIPY.SPARSE.LINALG.EIGSH =========== '''

eigsh_ag = primitive(sp.linalg.eigsh)

def vjp_maker_eigsh(ans, x, numeig=10, sigma=0.):
    """Gradient for eigenvalues and vectors of a hermitian matrix."""
    N = x.shape[-1]
    w, v = ans              # Eigenvalues, eigenvectors.
    vc = np.conj(v)
    
    def vjp(g):
        wg, vg = g          # Gradient w.r.t. eigenvalues, eigenvectors.
        w_repeated = np.repeat(w[..., np.newaxis], numeig, axis=-1)

        # Eigenvalue part
        vjp_temp = np.dot(vc * wg[..., np.newaxis, :], T(v)) 

        # Add eigenvector part only if non-zero backward signal is present.
        # This can avoid NaN results for degenerate cases if the function 
        # depends on the eigenvalues only.
        if np.any(vg):
            off_diag = np.ones((numeig, numeig)) - np.eye(numeig)
            F = off_diag / (T(w_repeated) - w_repeated + np.eye(numeig))
            vjp_temp += np.dot(np.dot(vc, F * np.dot(T(v), vg)), T(v))

        return vjp_temp

    return vjp

defvjp(eigsh_ag, vjp_maker_eigsh)

'''=========== NUMPY.INTERP =========== '''
'''This implementation might not be covering the full scope of the numpy.interp
function, but it covers everything we need
'''

interp_ag = primitive(np.interp)

def vjp_maker_interp(ans, x, xp, yp):
    '''Construct the vjp of interp(x, xp, yp) w.r.t. yp
    '''

    def vjp(g):
        dydyp = np.zeros((x.size, xp.size))
        for ix in range(x.size):
            indx = np.searchsorted(xp, x[ix]) - 1
            dydyp[ix, indx] = 1 - (x[ix] - xp[indx])/(xp[indx+1] - xp[indx])
            dydyp[ix, indx+1] = (x[ix] - xp[indx])/(xp[indx+1] - xp[indx])
        return np.dot(g, dydyp)
    return vjp

defvjp(interp_ag, None, None, vjp_maker_interp)


'''=========== SOLVE OF f(x, y) = 0 W.R.T. X =========== '''
fsolve_ag = primitive(fsolve)

def vjp_maker_fsolve(f, gradf, Nargs):
    '''
    Gradient of dx/dargs where x is found through fsolve. The gradient of f 
    w.r.t. both x and each of the args must be computable with autograd
    '''
    vjp_makers = [None, None, None]
    dfdx = gradf[0]

    def vjp_single_arg(ia):

        def vjp_maker(ans, f, lb, ub, *args):

            def vjp(g):       
                dfdy = gradf[ia+1]
                return np.dot(g, -1/dfdx(ans, *args) * dfdy(ans, *args))

            return vjp

        return vjp_maker

    for ia in range(Nargs):
        vjp_makers.append(vjp_single_arg(ia=ia))
    return tuple(vjp_makers)


