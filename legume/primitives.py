import numpy as np
from functools import partial
from .utils import toeplitz_block, get_value, fsolve

from autograd.extend import primitive, defvjp, vspace
from autograd import grad, vector_jacobian_product
import autograd.numpy as npa

''' Define here various primitives needed for the main code 
To use with both numpy and autograd backends, define the autograd primitive of 
a numpy function fnc as fnc_ag, and then define the vjp'''

def T(x): return np.swapaxes(x, -1, -2)
_dot = partial(np.einsum, '...ij,...jk->...ik')

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

'''=========== NUMPY.EIGH =========== '''

eigh_ag = primitive(np.linalg.eigh)

def vjp_maker_eigh(ans, x, UPLO='L'):
    """Gradient for eigenvalues and vectors of a hermitian matrix."""
    N = x.shape[-1]
    w, v = ans              # Eigenvalues, eigenvectors.
    vc = np.conj(v)
    
    def vjp(g):
        wg, vg = g          # Gradient w.r.t. eigenvalues, eigenvectors.
        w_repeated = np.repeat(w[..., np.newaxis], N, axis=-1)

        # Eigenvalue part
        vjp_temp = _dot(vc * wg[..., np.newaxis, :], T(v)) 

        # Add eigenvector part only if non-zero backward signal is present.
        # This can avoid NaN results for degenerate cases if the function depends
        # on the eigenvalues only.
        if np.any(vg):
            off_diag = np.ones((N, N)) - np.eye(N)
            F = off_diag / (T(w_repeated) - w_repeated + np.eye(N))
            vjp_temp += _dot(_dot(vc, F * _dot(T(v), vg)), T(v))

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

@primitive
def fsolve_grad(f, gradf, lb, ub, *args):
    '''Extend fsolve to also accept the gradient of the function with respect
    to all input parameters x and args
    '''
    return fsolve(f, lb, ub, *args)

def vjp_maker_fsolve_vjp(f, vjpf, Nargs):
    '''
    Gradient of dx/dargs where x is found through fsolve. The gradient of f 
    w.r.t. both x and each of the args must be computable with autograd
    '''
    vjp_makers = [None, None, None]

    def vjp_single_arg(ia):

        def vjp_maker(ans, f, lb, ub, *args):

            def vjp(g):
                dfdx = vjpf[0](vspace(ans).ones(), *args)[1]
                dfdy = vjpf[ia+1](vspace(ans).ones(), *args)[1]
                # print(dfdx, dfdy)
                return np.dot(g, -1/dfdx * dfdy)

            return vjp

        return vjp_maker

    for ia in range(Nargs):
        vjp_makers.append(vjp_single_arg(ia=ia))
    return tuple(vjp_makers)


