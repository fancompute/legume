"""
Various utilities used in the main code.
NOTE: there should be no autograd functions here, only plain numpy/scipy
"""

import numpy as np
from scipy.linalg import toeplitz
from scipy.optimize import brentq
import numpy as np
from PIL import Image
import itertools


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
        ftinv += ft_coeff[indg] * np.exp(1j * gvec[0, indg] * xmesh + \
                                         1j * gvec[1, indg] * ymesh)

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
    nx = np.int_(np.abs(np.max(gvec[0, :]) / dgx))
    ny = np.int_(np.abs(np.max(gvec[1, :]) / dgy))
    nxtot = 2 * nx + 1
    nytot = 2 * ny + 1
    eps_ft = np.zeros((nxtot, nytot), dtype=np.complex128)
    gx_grid = np.arange(-nx, nx) * dgx
    gy_grid = np.arange(-ny, ny) * dgy

    for jG in range(gvec.shape[1]):
        nG = np.int_(gvec[:, jG] / [dgx, dgy])
        eps_ft[nx + nG1[0], ny + nG1[1]] = ft_coeff[jG]

    return (eps_ft, gx_grid, gy_grid)


def grad_num(fn, arg, step_size=1e-7):
    """ Numerically differentiate `fn` w.r.t. its argument `arg` 
    `arg` can be a numpy array of arbitrary shape
    `step_size` can be a number or an array of the same shape as `arg` """

    N = arg.size
    shape = arg.shape
    gradient = np.zeros((N, ))
    f_old = fn(arg)

    if type(step_size) == float:
        step = step_size * np.ones((N))
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
                    dfn_darg = (fn(*args_new) - fn_out) / step
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
    p = int(ntot / n)  # Linear size of each block
    Tmat = np.zeros((ntot, ntot), dtype=T1.dtype)
    for ind1 in range(n):
        for ind2 in range(ind1, n):
            toep1 = T1[(ind2 - ind1) * p:(ind2 - ind1 + 1) * p]
            toep2 = T2[(ind2 - ind1) * p:(ind2 - ind1 + 1) * p]
            Tmat[ind1 * p:(ind1 + 1) * p, ind2 * p:(ind2 + 1) * p] = \
                toeplitz(toep2, toep1)

    return np.triu(Tmat) + np.conj(np.transpose(np.triu(Tmat, 1)))

    return np.triu(Tmat) + np.conj(np.transpose(np.triu(Tmat, 1)))


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


def RedhefferStar(SA, SB):  # SA and SB are both 2x2 matrices;
    assert type(SA) == np.ndarray, 'not np.matrix'
    assert type(SB) == np.ndarray, 'not np.matrix'

    I = 1
    # once we break every thing like this, we should still have matrices
    SA_11 = SA[0, 0]
    SA_12 = SA[0, 1]
    SA_21 = SA[1, 0]
    SA_22 = SA[1, 1]
    SB_11 = SB[0, 0]
    SB_12 = SB[0, 1]
    SB_21 = SB[1, 0]
    SB_22 = SB[1, 1]

    D = 1.0 / (I - SB_11 * SA_22)
    F = 1.0 / (I - SA_22 * SB_11)

    SAB_11 = SA_11 + SA_12 * D * SB_11 * SA_21
    SAB_12 = SA_12 * D * SB_12
    SAB_21 = SB_21 * F * SA_21
    SAB_22 = SB_22 + SB_21 * F * SA_22 * SB_12

    SAB = np.array([[SAB_11, SAB_12], [SAB_21, SAB_22]])
    return SAB


def extend(vals, inds, shape):
    """ Makes an array of shape `shape` where indices `inds` have vales `vals` 
    """
    z = np.zeros(shape, dtype=vals.dtype)
    z[inds] = vals
    return z


def low_pass_down_sample(bitmap, factor):
    """
    Downsample a bitmap via DFT truncation.

    Parameters
    ----------
    bitmap : Input bitmap (np array)
    factor : If factor is a number will down sample array by this factor.

             I.E. Shape_new = floor(shape_old/factor)

             Factor can be a single scalar or a 2 length array_like object.

    Returns
    -------
     Down sampled bitmap array.
    """

    if np.array(factor).shape.__len__() == 0:
        factor = factor * np.ones(2)
    else:
        factor = np.array(factor)
    double_factor = factor * 2

    fft_eps = np.fft.fft2(bitmap) / (bitmap.shape[0] * bitmap.shape[1])
    x = int(fft_eps.shape[0] / double_factor[0])
    y = int(fft_eps.shape[1] / double_factor[1])

    top = np.vstack([fft_eps[0:x, 0:y], fft_eps[-x:, 0:y]])
    bot = np.vstack([fft_eps[0:x, -y:], fft_eps[-x:, -y:]])
    dft_trunc = np.hstack([top, bot])

    bitmap_lowpass = np.real(
        np.fft.ifft2(dft_trunc) * dft_trunc.shape[0] * dft_trunc.shape[1])

    return bitmap_lowpass


def import_eps_image(img_path, eps_map, tol=5):
    """
    Import an permittivity distribution from a color image.

    Parameters
    ----------
    img_path : Path to image. Image will be interpreted in greyscale.
    eps_map : List of eps values.
              Image should have this many colors within tolerance set by tol.

              Darkest points of image will be mapped to first eps in eps_map.
              Brightest will be converted to last point in eps_map.

    tol : Tolerance to deviations from a brightness.

    Returns
    -------
    np array of permittivity values
    """

    img = Image.open(img_path)

    img = img.quantize(colors=len(eps_map), dither=Image.FLOYDSTEINBERG)
    img = img.convert('L')

    img = np.asarray(img)
    shape = img.shape

    eps_map = np.sort(eps_map)

    eps_array = np.zeros(shape)
    for i, eps in enumerate(eps_map):
        mask = img > img.max() - tol
        eps_array = eps_array + eps * mask
        img = img * (1 - mask)

    return eps_array


def find_band_gaps(gme,
                   order,
                   sample_rate=10,
                   band_tol=0.1,
                   trim_lc=False,
                   lc_trim=0,
                   numeig=20):
    """
    Find band gaps from a guided mode expansion.

    Parameters
    ----------
    gme : guided mode expansion (this function will overwrite any results.)
    order : order of modes to calculate. (even TE, odd TM)
    sample_rate : Governs number of k points sampled
    band_tol : tolerance of what frequency difference counts as a band gap.
    trim_lc : If True removes points in light cone from consideration.
    lc_trim : Tolerance of light cone trim.
    numeig : Number of bands in frequency to calculate starting from zero.

    Returns
    -------
    bands_gaps: list of lists in form [top of gap, bottom of gap, middle of
    gap],
    k_air: k points of top of gap. (Bottom of air band)
    k_di: k point of bottom of gap. (Top of dielectric band)
    """

    lattice = gme.phc.lattice

    bz = lattice.get_irreducible_brioullin_zone_vertices()

    path = lattice.bz_path(bz, [sample_rate] * (len(bz) - 1))

    gme.run(kpoints=path['kpoints'],
            gmode_inds=order,
            numeig=numeig,
            compute_im=False,
            gradients='approx',
            verbose=False)

    k_abs = np.tile((gme.kpoints[0]**2 + gme.kpoints[1]**2)**(1 / 2),
                    (numeig, 1)).T
    if trim_lc:
        in_lc_freqs = gme.freqs[gme.freqs /
                                (np.abs(k_abs - lc_trim) + 1e-10) <= 1 /
                                (2 * np.pi)]

        freqs_flat = np.sort(in_lc_freqs)
    else:
        freqs_flat = np.sort(gme.freqs.flatten())

    gaps = np.diff(freqs_flat)
    band_gaps = []
    for i in range(gaps.size):
        if gaps[i] >= band_tol:
            band_gaps.append([
                freqs_flat[i], freqs_flat[i + 1],
                (freqs_flat[i] + freqs_flat[i + 1]) / 2
            ])

    band_gaps = np.array(band_gaps)

    if band_gaps.size == 0:
        return [], [], []

    k_air_arg = np.array([
        np.argwhere(gme.freqs == band_gap[1])[0][0] for band_gap in band_gaps
    ])
    k_di_arg = np.array([
        np.argwhere(gme.freqs == band_gap[0])[0][0] for band_gap in band_gaps
    ])

    k_air = (gme.kpoints[0][k_air_arg], gme.kpoints[1][k_air_arg])
    k_di = (gme.kpoints[0][k_di_arg], gme.kpoints[1][k_di_arg])

    return band_gaps, k_air, k_di


def fold_K(k, supercell_size):
    """
    Fold k-points into a supercell_size Brillouin Zone.

    Parameters
    ----------
    k : array of k points to fold in 1 dimension.
    supercell_size : Number of primitive cells in supercell along
                     dimension of interest.

    Returns
    -------
    Folded array of k points.
    """

    a = np.pi / supercell_size
    fold = np.floor(k / a)
    s = np.mod(fold, 2)
    new_k = (-1)**s * np.mod(k, a) + s * a
    return new_k


def unfold_bands(super_gme, super_cell_size, branch_start=-np.pi):
    """
    Unfold bands of a guided mode expansion based on spectral density
    calculations.

    Parameters
    ----------
    super_gme : Guided mode expansion object of super cell with eigenvectors
                calculated.
    super_cell_size: Number of tiles along first and second lattice vectors.
    branch_start: Leading value of interval to unfold onto.
                  Range: [branch_start, 2pi+branch_start),
                  default range [-pi, pi).

    Returns
    -------
    Dispersion relation of unfolded band structure of super cell calculation
    and normalized spectral density for each eigenvector.

    Note k returned in units of 1/a, a being the primitive lattice constant.
    """
    # Collect g-vectors of primitive and super_cell_size expansions.

    gvecs = super_gme.gvec.reshape(2, super_gme.n1g, super_gme.n2g)
    prim_gvecs = super_gme.gvec.reshape(
        2, super_gme.n1g, super_gme.n2g) * np.array(super_cell_size).reshape(
            2, 1, 1)

    mask = np.logical_and(np.isin(gvecs[0], prim_gvecs[0]),
                          np.isin(gvecs[1], prim_gvecs[1]))

    # Determine periodicity of the crystal.

    lattice = super_gme.phc.lattice

    b1 = lattice.b1
    b2 = lattice.b2

    N1, N2 = super_cell_size

    # Pad the mask so that when we roll the mask we don't create artifacts.
    # TODO We lose prim vectors on sides.

    pad_mask = np.pad(mask, ((N1 - 1, 0), (N2 - 1, 0)))

    unfolded_kpoints = np.array(
        [np.empty(super_gme.freqs.shape),
         np.empty(super_gme.freqs.shape)])

    probabilities = np.empty(
        (super_gme.freqs.shape[0], super_gme.freqs.shape[1], N1, N2))

    for k in range(len(super_gme.kpoints.transpose())):
        for w in range(len(super_gme.eigvecs[k].transpose())):

            eig = super_gme.eigvecs[k].transpose()[w].reshape(
                (super_gme.n1g, super_gme.n2g))
            # Roll padded mask and truncate to shape of eigen vector.
            probability = np.empty((N1, N2))
            for i, j in itertools.product(range(N1), range(N2)):
                probability[i, j] = np.sum(
                    np.square(
                        np.abs(eig[np.roll(pad_mask, (i, j),
                                           axis=(0, 1))[N1 - 1:, N2 - 1:]])))

            # Normalize probability
            probability = probability / np.sum(probability)

            args = np.argwhere(probability == probability.max())[0]

            trans = (args[0]) * b1 + (args[1]) * b2

            new_k = super_gme.kpoints.T[k] + trans
            # Shift K to appropriate branch of mod function.
            new_k = np.mod(new_k - branch_start, 2 * np.pi) + branch_start

            unfolded_kpoints[0][k, w] = new_k[0]
            unfolded_kpoints[1][k, w] = new_k[1]
            probabilities[k, w, :, :] = probability

    return unfolded_kpoints, probabilities


def fixed_point_cluster(mean,
                        diff,
                        N,
                        kernel='gaussian',
                        u=None,
                        reflect=False):
    """
    Generates a 1d cluster of points.

    Useful for generating a cluster of test points around a single k vector.

    Parameters
    ----------
    mean : Mean to generate points around
    diff : Difference from mean. Distribution is anchored at these points.
    N : Number of points to generate.
    kernel : Function governing cluster. Dilates a linearspace

             Should be non-negative and continuous.
             If callable object given runs that function.
             If string given looks up function from preset table.

             Preset Table: 'binomial'
                           'id'

    u : single parameter governing preset kernels
    reflect : Reflects the cluster about the zero axis.

    Returns
    -------
    1d array of points.
    """

    kern_dict = {
        "binomial": lambda x: np.abs(x + (u - 1) / (3 * diff**2) * x**3),
        "id": lambda x: np.abs(x)
    }

    if isinstance(kernel, str):
        kernel = kern_dict[kernel]
    elif callable(kernel):
        kernel = kernel
    else:
        raise ValueError("Kernel not callable or string.")

    lin_array = np.linspace(mean - diff, mean + diff, N)
    clustered_array = np.sign(lin_array - mean) * diff * kernel(
        lin_array - mean) / kernel(diff) + mean

    if reflect:
        clustered_array = np.concatenate((-clustered_array, clustered_array))

    return clustered_array


def isolate_bands(kpoints, freqs):
    """
    Isolate bands from band structure.

    Parameters
    ----------
    kpoints : array of k points
    freqs : list of lists of frequencies corresponding to k points.
            Each entry corresponds to kpoints

    Returns
    -------
    final_bands: list of lists each corresponding to a band.

                Each list contains array of freqs
                and array of kpoints.
    final_arg_bands: list of lists each corresponding to a band.

                Each list contains array of arguments for freqs
                and array of arguments for kpoints.
    """

    args = np.argsort(kpoints)

    kpoints_sorted, freqs_sorted = kpoints[args], freqs[args][:]

    low_arg_bands = []
    arg_bands = []
    high_arg_bands = []

    bands = []
    low_bands = []
    high_bands = []

    for find, freq in enumerate(freqs_sorted[0]):
        bands.append([[kpoints_sorted[0]], [freq]])
        arg_bands.append([[args[0]], [find]])

    for kind, k in enumerate(kpoints_sorted[1:]):
        k_freqs = freqs_sorted[kind + 1]

        error_cont = 0
        error_plus = 0
        error_neg = 0

        for bind, band in enumerate(bands[:]):
            error_cont += (k_freqs[bind] - band[1][-1])**2

        for bind, band in enumerate(bands[1:]):
            error_plus += (k_freqs[:-1][bind] - band[1][-1])**2

        for bind, band in enumerate(bands[:-1]):
            error_neg += (k_freqs[1:][bind] - band[1][-1])**2

        if error_plus == min([error_cont, error_plus, error_neg]):
            low_bands.insert(0, bands[0])
            low_arg_bands.insert(0, arg_bands[0])

            bands = bands[1:]
            bands.append([[], []])
            arg_bands = arg_bands[1:]
            arg_bands.append([[], []])

        elif error_neg == min([error_cont, error_plus, error_neg]):
            high_bands.append(bands[-1])
            high_arg_bands.append(arg_bands[-1])

            bands = bands[:-1]
            bands.insert(0, [[], []])
            arg_bands = arg_bands[:-1]
            arg_bands.insert(0, [[], []])

        for find, freq in enumerate(k_freqs):
            bands[find][1].append(freq)
            bands[find][0].append(k)
            arg_bands[find][1].append(find)
            arg_bands[find][0].append(args[kind + 1])

    final_bands = low_bands + bands + high_bands
    final_bands = [np.array(band) for band in final_bands]

    final_arg_bands = low_arg_bands + arg_bands + high_arg_bands
    final_arg_bands = [np.array(band) for band in final_arg_bands]

    return final_bands, final_arg_bands
