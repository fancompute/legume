"""
Various utilities used in the main code.
NOTE: there should be no autograd functions here, only plain numpy/scipy
"""

import numpy as np
from scipy.linalg import toeplitz
from scipy.optimize import brentq


def ftinv(ft_coeff, gvec, xgrid, ygrid):
    """ 
    Returns the discrete inverse Fourier transform over a real-space mesh 
    defined by 'xgrid', 'ygrid', computed given a number of FT coefficients 
    'ft_coeff' defined over a set of reciprocal vectors 'gvec'.
    This could be sped up through an fft function but written like this it is 
    more general as we don't have to deal with grid and lattice issues.
    """
    (xmesh, ymesh) = np.meshgrid(xgrid, ygrid)
    ftinv = np.zeros(xmesh.shape, dtype=np.complex128)

    # Take only the unique components
    (g_unique, ind_unique) = np.unique(gvec, return_index=True, axis=1)

    for indg in ind_unique:
        ftinv += ft_coeff[indg]*np.exp(1j*gvec[0, indg]*xmesh + \
                            1j*gvec[1, indg]*ymesh)

    # # Do the x- and y-transforms separately 
    # # I wrote this but then realized it doesn't improve anything
    # (gx_u, indx) = np.unique(gvec[0, :], return_inverse=True)
    # for ix, gx in enumerate(gx_u):
    #   ind_match = np.where(indx==ix)[0]
    #   (gy_u, indy) = np.unique(gvec[1, ind_match], return_index=True)
    #   term = np.zeros(xmesh.shape, dtype=np.complex128)
    #   for iy, gy in enumerate(gy_u):
    #       # print(ft_coeff[indx[indy[iy]]])
    #       term += ft_coeff[ind_match[indy[iy]]]*np.exp(-1j*gy*ymesh)
    #   ftinv += term*np.exp(-1j*gx*xmesh)

    # # Can also be defined through a DFT matrix but it doesn't seem faster and 
    # # it's *very* memory intensive.
    # exp_matrix = xmesh.reshape((-1, 1)).dot(g_unique[[0], :]) + \
    #               ymesh.reshape((-1, 1)).dot(g_unique[[1], :])

    # dft_matrix = np.exp(1j*exp_matrix)
    # ftinv = dft_matrix.dot(ft_coeff[ind_unique]).reshape(xmesh.shape)
    # print(ftinv)
    return ftinv


def ft2square(lattice, ft_coeff, gvec):
    """
    Make a square array of Fourier components given a number of them defined 
    over a set of reciprocal vectors gvec.
    NB: function hasn't really been tested, just storing some code.
    """
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
    """ Numerically differentiate `fn` w.r.t. its argument `arg` 
    `arg` can be a numpy array of arbitrary shape
    `step_size` can be a number or an array of the same shape as `arg` """

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


def vjp_maker_num(fn, arg_inds, steps):
    """ Makes a vjp_maker for the numerical derivative of a function `fn`
    w.r.t. argument at position `arg_ind` using step sizes `steps` """

    def vjp_single_arg(ia):
        arg_ind = arg_inds[ia]
        step = steps[ia]

        def vjp_maker(fn_out, *args):
            shape = args[arg_ind].shape
            num_p = args[arg_ind].size
            step = steps[ia]

            def vjp(v):

                vjp_num = np.zeros(num_p)
                for ip in range(num_p):
                    args_new = list(args)
                    args_rav = args[arg_ind].flatten()
                    args_rav[ip] += step
                    args_new[arg_ind] = args_rav.reshape(shape)
                    dfn_darg = (fn(*args_new) - fn_out)/step
                    vjp_num[ip] = np.sum(v * dfn_darg)

                return vjp_num

            return vjp

        return vjp_maker

    vjp_makers = []
    for ia in range(len(arg_inds)):
        vjp_makers.append(vjp_single_arg(ia=ia))

    return tuple(vjp_makers)


def toeplitz_block(n, T1, T2):
    """
    Constructs a Hermitian Toeplitz-block-Toeplitz matrix with n blocks and 
    T1 in the first row and T2 in the first column of every block in the first
    row of blocks 
    """
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
    """
    This is for when using the 'autograd' backend and you want to detach an 
    ArrayBox and just convert it to a numpy array.
    """
    if str(type(x)) == "<class 'autograd.numpy.numpy_boxes.ArrayBox'>":
        return x._value
    else:
        return x


def fsolve(f, lb, ub, *args):
    """
    Solve for scalar f(x, *args) = 0 w.r.t. scalar x within lb < x < ub
    """
    args_value = tuple([get_value(arg) for arg in args])
    return brentq(f, lb, ub, args=args_value)


def find_nearest(array, value, N):
    """
    Find the indexes of the N elements in an array nearest to a given value
    (Not the most efficient way but this is not a coding interview...)
    """ 
    idx = np.abs(array - value).argsort()
    return idx[:N]


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
    return SAB


def extend(vals, inds, shape):
    """ Makes an array of shape `shape` where indices `inds` have vales `vals` 
    """
    z = np.zeros(shape, dtype=vals.dtype)
    z[inds] = vals
    return z
