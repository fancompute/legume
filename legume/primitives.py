import numpy as np
from functools import partial
from .utils import toeplitz_block, get_value, fsolve

from autograd.extend import primitive, defvjp
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
"""Implementation taken from an un-resolved pull request
https://github.com/HIPS/autograd/pull/194
"""

def interp_ag(x, xp, yp, left=None, right=None, period=None):
    """ A partial rewrite of interp that is differentiable against yp """
    if period is not None:
        xp = npa.concatenate([[xp[-1] - period], xp, [xp[0] + period]])
        yp = npa.concatenate([npa.array([yp[-1]]), yp, npa.array([yp[0]])])
        return _interp(x % period, xp, yp, left, right, None)

    if left is None: left = yp[0]
    if right is None: right = yp[-1]

    xp = npa.concatenate([[xp[0]], xp, [xp[-1]]])

    yp = npa.concatenate([npa.array([left]), yp, npa.array([right])])
    m = make_matrix(x, xp)
    y = npa.inner(m, yp)
    return y

# The following are internal functions
def W(r, D):
    """ Convolution kernel for linear interpolation.
        D is the differences of xp.
    """
    mask = D == 0
    D[mask] = 1.0
    Wleft = 1.0 + r[1:] / D
    Wright = 1.0 - r[:-1] / D
    # edges
    Wleft = np.where(mask, 0, Wleft)
    Wright = np.where(mask, 0, Wright)
    Wleft = np.concatenate([[0], Wleft])
    Wright = np.concatenate([Wright, [0]])
    W = np.where(r < 0, Wleft, Wright)
    W = np.where(r == 0, 1.0, W)
    W = np.where(W < 0, 0, W)
    return W

def make_matrix(x, xp):
    D = np.diff(xp)
    w = []
    v0 = np.zeros(len(xp))
    v0[0] = 1.0
    v1 = np.zeros(len(xp))
    v1[-1] = 1.0
    for xi in x:
        # left, use left
        if xi < xp[0]: v = v0
        # right , use right
        elif xi > xp[-1]: v = v1
        else:
            v = W(xi - xp, D)
            v[0] = 0
            v[-1] = 0
        w.append(v)
    return np.array(w)


'''=========== SOLVE OF f(x, y) = 0 W.R.T. X =========== '''
fsolve_ag = primitive(fsolve)

def vjp_maker_fsolve(Nargs=3):
    '''
    Gradient of dx/dargs where x is found through fsolve. The gradient of f 
    w.r.t. both x and each of the args must be computable with autograd
    '''
    vjp_makers = [None, None, None]

    def vjp_single_arg(ia):

        def vjp_maker(ans, f, lb, ub, *args):
            dfdx = grad(f, 0)

            def vjp(g):       
                dfdy = grad(f, ia + 1)
                return np.dot(g, 1/dfdx(ans, *args) * dfdy(ans, *args))

            return vjp

        return vjp_maker

    for ia in range(Nargs):
        vjp_makers.append(vjp_single_arg(ia=ia))
    return tuple(vjp_makers)

defvjp(fsolve_ag, *vjp_maker_fsolve())
