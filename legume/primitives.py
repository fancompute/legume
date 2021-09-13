import numpy as np
import scipy.sparse as sp
from functools import partial
from .utils import toeplitz_block, get_value, fsolve, extend

from autograd.extend import primitive, defvjp, vspace
from autograd import grad, vector_jacobian_product
import autograd.numpy as npa
""" Define here various primitives needed for the main code 
To use with both numpy and autograd backends, define the autograd primitive of 
a numpy function fnc as fnc_ag, and then define the vjp"""


def T(x):
    return np.swapaxes(x, -1, -2)


"""=========== EXPAND ARRAY TO A GIVEN SHAPE =========== """

# extend(vals, inds, shape) makes an array of shape `shape` where indices
# `inds` have values `vals`
extend_ag = primitive(extend)


def vjp_maker_extend(ans, vals, inds, shape):
    def vjp(g):
        return g[inds]

    return vjp


defvjp(extend_ag, vjp_maker_extend, None, None)
"""=========== NP.SQRT STABLE AROUND 0 =========== """
sqrt_ag = primitive(np.sqrt)


def vjp_maker_sqrt(ans, x):
    def vjp(g):
        return g * 0.5 * (x + 1e-10)**0.5 / (x + 1e-10)
        # return np.where(np.abs(x) > 1e-10, g * 0.5 * x**-0.5, 0.)

    return vjp


defvjp(sqrt_ag, vjp_maker_sqrt)
"""=========== TOEPLITZ-BLOCK =========== """

toeplitz_block_ag = primitive(toeplitz_block)


def vjp_maker_TB_T1(Tmat, n, T1, T2):
    """ Gives vjp for Tmat = toeplitz_block(n, T1, T2) w.r.t. T1"""
    def vjp(v):
        ntot = Tmat.shape[0]
        p = int(ntot / n)  # Linear size of each block
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
    """ Gives vjp for Tmat = toeplitz_block(n, T1, T2) w.r.t. T2"""
    def vjp(v):
        ntot = Tmat.shape[0]
        p = int(ntot / n)  # Linear size of each block
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
"""=========== NUMPY.LINALG.EIGH =========== """

eigh_ag = primitive(np.linalg.eigh)


def vjp_maker_eigh(ans, x, UPLO='L'):
    """Gradient for eigenvalues and vectors of a hermitian matrix."""
    N = x.shape[-1]
    w, v = ans  # Eigenvalues, eigenvectors.
    vc = np.conj(v)

    def vjp(g):
        wg, vg = g  # Gradient w.r.t. eigenvalues, eigenvectors.
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
"""=========== MATRIX INVERSE =========== """
"""We define this here without the `einsum` notation that's used in autograd.
`einsum` allows broadcasting (which we don't care about), but is slower 
(which we do)
"""

inv_ag = primitive(np.linalg.inv)


def vjp_maker_inv(ans, x):
    return lambda g: -np.dot(np.dot(T(ans), g), T(ans))


defvjp(inv_ag, vjp_maker_inv)
"""=========== SCIPY.SPARSE.LINALG.EIGSH =========== """

eigsh_ag = primitive(sp.linalg.eigsh)

# def vjp_maker_eigsh(ans, x, **kwargs):
#     """Gradient for eigenvalues and vectors of a hermitian matrix."""
#     numeig = kwargs['k']
#     N = x.shape[-1]
#     w, v = ans              # Eigenvalues, eigenvectors.
#     vc = np.conj(v)

#     def vjp(g):
#         wg, vg = g          # Gradient w.r.t. eigenvalues, eigenvectors.
#         w_repeated = np.repeat(w[..., np.newaxis], numeig, axis=-1)

#         # Eigenvalue part
#         vjp_temp = np.dot(vc * wg[..., np.newaxis, :], T(v))

#         # Add eigenvector part only if non-zero backward signal is present.
#         # This can avoid NaN results for degenerate cases if the function
#         # depends on the eigenvalues only.
#         if np.any(vg):
#             off_diag = np.ones((numeig, numeig)) - np.eye(numeig)
#             F = off_diag / (T(w_repeated) - w_repeated + np.eye(numeig))
#             vjp_temp += np.dot(np.dot(vc, F * np.dot(T(v), vg)), T(v))

#         return vjp_temp

#     return vjp


def vjp_maker_eigsh(ans, mat, **kwargs):
    """Steven Johnson method extended to a Hermitian matrix
    https://math.mit.edu/~stevenj/18.336/adjoint.pdf
    """
    numeig = kwargs['k']
    N = mat.shape[0]

    def vjp(g):
        vjp_temp = np.zeros_like(mat)
        for iv in range(numeig):
            a = ans[0][iv]
            v = ans[1][:, iv]
            vc = np.conj(v)
            ag = g[0][iv]
            vg = g[1][:, iv]

            # Eigenvalue part
            vjp_temp += ag * np.outer(vc, v)

            # Add eigenvector part only if non-zero backward signal is present.
            # This can avoid NaN results for degenerate cases if the function
            # depends on the eigenvalues only.
            if np.any(vg):
                # Projection operator on space orthogonal to v
                P = np.eye(N, N) - np.outer(vc, v)
                Amat = T(mat - a * np.eye(N, N))
                b = P.dot(vg)

                # Initial guess orthogonal to v
                v0 = P.dot(np.random.randn(N))

                # Find a solution lambda_0 using conjugate gradient
                (l0, _) = sp.linalg.cg(Amat, b, x0=v0, atol=0)
                # Project to correct for round-off errors
                l0 = P.dot(l0)

                vjp_temp -= np.outer(l0, v)

        return vjp_temp

    return vjp


defvjp(eigsh_ag, vjp_maker_eigsh)
"""=========== NUMPY.INTERP =========== """
"""This implementation might not be covering the full scope of the numpy.interp
function, but it covers everything we need
"""

interp_ag = primitive(np.interp)


def vjp_maker_interp(ans, x, xp, yp):
    """Construct the vjp of interp(x, xp, yp) w.r.t. yp
    """
    def vjp(g):
        dydyp = np.zeros((x.size, xp.size))
        for ix in range(x.size):
            indx = np.searchsorted(xp, x[ix]) - 1
            dydyp[ix,
                  indx] = 1 - (x[ix] - xp[indx]) / (xp[indx + 1] - xp[indx])
            dydyp[ix,
                  indx + 1] = (x[ix] - xp[indx]) / (xp[indx + 1] - xp[indx])
        return np.dot(g, dydyp)

    return vjp


defvjp(interp_ag, None, None, vjp_maker_interp)
"""=========== SOLVE OF f(x, y) = 0 W.R.T. X =========== """
fsolve_ag = primitive(fsolve)
"""fsolve_ag(fun, lb, ub, *args) solves fun(x, *args) = 0 for lb <= x <= ub
    x and the output of fun are both scalar
    args can be anything
"""


def vjp_factory_fsolve(ginds):
    """
    Factory function defining the vjp_makers for a generic fsolve_ag with 
    multiple extra arguments

    Output: a list of vjp_makers for backproping through dx/darg where x is 
    found through fsolve_ag and arg is one of the function args. 
    Input: 
        - ginds : Boolean list defining which args will be differentiated.
        grad(f, gind) must exist for all gind==True in ginds
        grad(f, 0), i.e. the gradient w.r.t. x, must also exist
    """

    # Gradients w.r.t fun, lb and ub are not computed
    vjp_makers = [None, None, None]

    def vjp_single_arg(ia):
        def vjp_maker(ans, *args):
            f = args[0]
            fargs = args[3:]
            dfdx = grad(f, 0)(ans, *fargs)
            dfdy = grad(f, ia + 1)(ans, *fargs)

            def vjp(g):
                return np.dot(g, -1 / dfdx * dfdy)

            return vjp

        return vjp_maker

    for (ia, gind) in enumerate(ginds):
        if gind == True:
            vjp_makers.append(vjp_single_arg(ia=ia))
        else:
            vjp_makers.append(None)

    return tuple(vjp_makers)


# NB: This definition is for the specific fsolve with three arguments
# used for the guided modes!!!
defvjp(fsolve_ag, *vjp_factory_fsolve([False, True, True]))
"""=========== MAP FUNCTION EVALUATION =========== """
""" A variation of the `functools.map` function applied to a list of functions,
    defined as follows
        `fmap(fns, params) = map(lambda f: f(params), fns)`
        (the output is converted to a numpy array)

    We assume that each `f` in `fns` returns a scalar such that the output is an 
    array of the same size as `fns`.
"""


@primitive
def fmap(fns, params):
    """ autograd-ready version of functools.fmap applied to a list of functions
    `fns` taking the same parmeters `params`
    Arguments:
        `fns`: list of functions of `params` that return a scalar
        `params`: array of parameters feeding into each individual computation
    Returns:
        Numpy array of same size as the `fns` list
    """

    # use standard map function and convert to a Numpy array
    return np.array(list(map(lambda f: f(params), fns))).ravel()


def vjp_maker_fmap(ans, fns, params):
    # get the gradient of each function and stack along the 0-th dimension
    grads = np.stack(list(map(lambda f: grad(f)(params), fns)), axis=0)
    # this literally does the vector-jacobian product
    return lambda v: np.dot(v.T, grads)


defvjp(fmap, None, vjp_maker_fmap)
