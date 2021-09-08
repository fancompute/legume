"""
Library of useful functions for chickpea
"""
import numpy as np
from PIL import Image, ImagePalette
import scipy as sp
import legume
from operator import itemgetter
import itertools

def lowpass_downsample(bitmap, factor):
    """
    Downsample a bitmap via DFT truncation.

    :param bitmap: Input bitmap (np array)
    :param factor: Will downsample array by this factor.
    :return: Downsampeld bitmap array.
    """
    if np.array(factor).shape.__len__() == 0:
        factor = factor * np.ones(2)
    else:
        factor = np.array(factor)
    double_factor = factor * 2



    fft_eps = np.fft.fft2(bitmap) / (bitmap.shape[0] * bitmap.shape[1])
    X = int(fft_eps.shape[0] / double_factor[0])
    Y = int(fft_eps.shape[1] / double_factor[1])
    fft_trunc = fft_eps[0:X, 0:X]

    top = np.vstack([fft_eps[0:X, 0:Y], fft_eps[-X:, 0:Y]])
    bot = np.vstack([fft_eps[0:X, -Y:], fft_eps[-X:, -Y:]])
    dft_trunc = np.hstack([top, bot])

    bitmap_lowpass = np.real(np.fft.ifft2(dft_trunc) * dft_trunc.shape[0] * dft_trunc.shape[1])

    return bitmap_lowpass


def import_eps_image(img_path, eps_map, tol=5):
    """

    :param img_path: Path to image
    :param eps_map: List of eps values. Image should have this many colors within tolerance set by tol.

                    Darkest points of images will be converted to first eps in eps_map.
                    Brightest will be converted to last point in eps_map.
    :param tol: tolerance to deviations from a color.

    :return: np array of permitivity values
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


def find_band_gaps(gme, order, sample_rate=10, band_tol=0.1, trim_lc=False, lc_trim=0, numeig=20):
    """
    Find band gaps from a guided mode expansion.
    :param gme: guided mode expansion (this function will overwrite any results.)
    :param order: order of modes to calculate. (even TE, odd TM)
    :param sample_rate: Governs number of k points sampled
    :param band_tol: tolerance of what frequency difference counts as a band gap.
    :param trim_lc: If True removes points in light cone from consideration.
    :param lc_trim: Tolerance of light cone trim.
    :param numeig: Number of bands in frequency to calculate starting from zero.

    :return: bands_gaps: list of lists in form [top of gap, bottom of gap, middle of gap],
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

    k_abs = np.tile((gme.kpoints[0] ** 2 + gme.kpoints[1] ** 2) ** (1 / 2), (numeig, 1)).T
    if trim_lc:
        in_lc_freqs = gme.freqs[
            gme.freqs / (np.abs(k_abs - lc_trim) + 1e-10) <= 1 / (2 * np.pi)]

        freqs_flat = np.sort(in_lc_freqs)
    else:
        freqs_flat = np.sort(gme.freqs.flatten())

    gaps = np.diff(freqs_flat)
    band_gaps = []
    for i in range(gaps.size):
        if gaps[i] >= band_tol:
            band_gaps.append([freqs_flat[i], freqs_flat[i + 1], (freqs_flat[i] + freqs_flat[i + 1]) / 2])

    band_gaps = np.array(band_gaps)

    if band_gaps.size == 0:
        return [], [], []

    k_air_arg = np.array([np.argwhere(gme.freqs == band_gap[1])[0][0] for band_gap in band_gaps])
    k_di_arg = np.array([np.argwhere(gme.freqs == band_gap[0])[0][0] for band_gap in band_gaps])

    k_air = (gme.kpoints[0][k_air_arg], gme.kpoints[1][k_air_arg])
    k_di = (gme.kpoints[0][k_di_arg], gme.kpoints[1][k_di_arg])

    return band_gaps, k_air, k_di

def fold_K(k, supercell_size):
    """
    Fold kpoints into a supercell_size Brillouin Zone.

    :param k: array of k points to fold in 1 dimension.
    :param supercell_size: number of primitive cells in supercell along dimension of interest.
    :return: folded array of k points.
    """
    a= np.pi / supercell_size
    fold=np.floor(k/a)
    s=np.mod(fold,2)
    new_k = (-1)**s *np.mod(k,a)+s*a
    return new_k



def unfold_bands(super_gme, supercell_size, branch_start=-np.pi):
    """
    Unfold bands of a guided mode expansion based on spectral density calculations.

    :param super_gme: Guided mode expansion object of super cell with eigenvectors calculated.
                      Must be rectangular tiling of primitive cells.

    :param supercell_size: Number of tiles along x and y axis.

    :param branch_start: range of unfolding. Range: [branch_start, 2pi+branch_start), default range [-pi, pi).
    :return: Dispersion relation of unfolded band structure of super cell calculation and normalized spectral density for each eigenvector.
             Note k returned in units of 1/a, a being the primitive lattice constant.

    TODO: This funciton can be generalized to plane wave expansion.
    """
    # Collect g-vectors of primitive and supercell_size expansions.

    gvecs = super_gme.gvec.reshape(2, super_gme.n1g, super_gme.n2g)
    prim_gvecs = super_gme.gvec.reshape(2, super_gme.n1g, super_gme.n2g) * np.array(supercell_size).reshape(2,1,1)

    mask = np.logical_and(np.isin(gvecs[0], prim_gvecs[0]), np.isin(gvecs[1], prim_gvecs[1]))

    # Determine periodicity of the crystal.

    lattice = super_gme.phc.lattice

    b1 = lattice.b1
    b2 = lattice.b2

    N1, N2 = supercell_size

    # Pad the mask so that when we roll the mask we don't create artifacts.
    # TODO We lose prim vectors on sides.

    pad_mask = np.pad(mask, ((N1-1, 0), (N2-1, 0)))

    unfolded_kpoints = np.array([np.empty(super_gme.freqs.shape),np.empty(super_gme.freqs.shape)])
    probabilities = np.empty((super_gme.freqs.shape[0], super_gme.freqs.shape[1], N1, N2))


    for k in range(len(super_gme.kpoints.transpose())):
        for w in range(len(super_gme.eigvecs[k].transpose())):

            eig = super_gme.eigvecs[k].transpose()[w].reshape((super_gme.n1g, super_gme.n2g))
            # Roll padded mask and truncate to shape of eigen vector.
            probability = np.empty((N1,N2))
            for i,j in itertools.product(range(N1),range(N2)): probability[i,j] = \
                np.sum(np.square(np.abs(eig[np.roll(pad_mask, (i,j), axis=(0, 1))[N1-1:, N2-1:]])))

            # Normalize probability
            probability = probability / np.sum(probability)

            args = np.argwhere(probability == probability.max())[0]

            trans = (args[0]) * b1 + (args[1]) * b2

            new_k = super_gme.kpoints.T[k] + trans
            # Shift K to appropriate branch of mod function.
            new_k = np.mod(new_k - branch_start, 2 * np.pi) + branch_start

            unfolded_kpoints[0][k,w] = new_k[0]
            unfolded_kpoints[1][k,w] = new_k[1]
            probabilities[k,w,:,:] = probability

    return unfolded_kpoints, probabilities


def fixed_point_cluster(mean, diff, N, kernel='gaussian', u=None, reflect=False):
    """
    Generates a cluster of points.
    :param mean: Mean to generate points around
    :param diff: difference from mean. Distribution is anchored at these points.
    :param N: Number of points to generate.
    :param kernel: Function governing cluster. Should be non-negative and continuous.
                   If callable object given runs that function.
                   If string given looks up function from preset table.
    :param u: single parameter governing preset kernels
    :param reflect: Reflect the cluster around the zero axis.
    :return: cluster of points.
    """

    kern_dict = {"binomial": lambda x: np.abs(x + (u - 1) / (3 * diff ** 2) * x ** 3),
                 "id": lambda x: np.abs(x)}

    if isinstance(kernel, str):
        kernel = kern_dict[kernel]
    elif callable(kernel):
        kernel = kernel
    else:
        raise ValueError("Kernel not callable or string.")

    lin_array = np.linspace(mean - diff, mean + diff, N)
    clustered_array = np.sign(lin_array - mean) * diff * kernel(lin_array - mean) / kernel(diff) + mean

    if reflect:
        clustered_array = np.concatenate((-clustered_array, clustered_array))

    return clustered_array

def isolate_bands(kpoints, freqs):
    """
    Isolate bands from band structure.

    :param k: array of k points
    :param freqs: list of lists of frequencies corresponding to k points. Each entry corresponds to kpoints
    :return: List of lists each entry contains array of freqs and array of k for bands.
    """


    args=np.argsort(kpoints)

    kpoints_sorted, freqs_sorted= kpoints[args], freqs[args][:]

    low_arg_bands=[]
    arg_bands=[]
    high_arg_bands=[]

    bands = []
    low_bands = []
    high_bands = []
    base_ind=0
    num_bands = freqs_sorted.shape[1]

    for find, freq in enumerate(freqs_sorted[0]):
        bands.append([[kpoints_sorted[0]],[freq]])
        arg_bands.append([[args[0]],[find]])

    for kind, k in enumerate(kpoints_sorted[1:]):
        k_freqs = freqs_sorted[kind + 1]

        error_cont = 0
        error_plus = 0
        error_neg = 0

        for bind, band in enumerate(bands[:]):
            error_cont += (k_freqs[bind]-band[1][-1])**2

        for bind, band in enumerate(bands[1:]):
            error_plus += (k_freqs[:-1][bind]-band[1][-1])**2

        for bind, band in enumerate(bands[:-1]):
            error_neg += (k_freqs[1:][bind]-band[1][-1])**2

        if error_plus == min([error_cont,error_plus, error_neg]):
            low_bands.insert(0, bands[0])
            low_arg_bands.insert(0, arg_bands[0])

            bands = bands[1:]
            bands.append([[], []])
            arg_bands = arg_bands[1:]
            arg_bands.append([[], []])

        elif error_neg == min([error_cont,error_plus, error_neg]):
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
            arg_bands[find][0].append(args[kind+1])

    final_bands=low_bands+bands+high_bands
    final_bands = [np.array(band) for band in final_bands]

    final_arg_bands=low_arg_bands+arg_bands+high_arg_bands
    final_arg_bands = [np.array(band) for band in final_arg_bands]


    return final_bands, final_arg_bands






