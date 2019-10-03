""" This attempts to perform topology optimization of a photonic crystal slab band gap through a density
defined across a grid of polygons
"""

import argparse

import autograd
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import scipy
from autograd import grad

import legume
from legume.backend import backend as bd
from legume.optimizers import adam_optimize

parser = argparse.ArgumentParser()

parser.add_argument('--initialize', action='store_true')
parser.add_argument('--optimize', action='store_true')

parser.add_argument('--verbose', action='store_true')

parser.add_argument('-init', default='soft', type=str)
parser.add_argument('-epochs', default=10, type=int)
parser.add_argument('-lr', default=0.1, type=float)
parser.add_argument('-gmax', default=5, type=float)
parser.add_argument('-gmode_npts', default=2000, type=int)
parser.add_argument('-neig', default=10, type=int)
parser.add_argument('-N_polygons', default=40, type=int)
parser.add_argument('-D', default=0.50, type=float)
parser.add_argument('-eta', default=0.5, type=float)
parser.add_argument('-beta', default=10, type=float)
parser.add_argument('-eps_b', default=12, type=float)
parser.add_argument('-r0', default=0.2, type=float)
args = parser.parse_args()

options = {'gmode_inds': np.array([0, 3]),
           'gmode_npts': args.gmode_npts,
           'numeig': args.neig,
           'verbose': args.verbose}

lattice = legume.Lattice('hexagonal')
a2 = lattice.a1
lattice.a1 = lattice.a2
lattice.a2 = a2
# lattice_hardcoded = legume.Lattice('square') # Hardcoded to squares to mimic old version (periodicity is still actually hex)

# _,_,sq_Xe,sq_Ye = generate_grid(legume.Lattice('square'), 1, extent_min=-0.5, extent_max=0.5)
# lat = legume.Lattice('hexagonal')
# a2 = lat.a1
# lat.a1 = lat.a2
# lat.a2 = a2
# _,_,hex_Xe,hex_Ye = generate_grid(lat, 1, extent_min=-0.5, extent_max=0.5)
# plt.figure();
# plt.plot(sq_Xe, sq_Ye, 'k-');
# plt.plot(sq_Xe[0], sq_Ye[0], 'ko');
# plt.plot(hex_Xe, hex_Ye, 'r-');
# plt.plot(hex_Xe[0], hex_Ye[0], 'ro');
# plt.axis('image')
# plt.show()

path = lattice.bz_path(['G', 'M', 'K', 'G'], [10, 10, 10])
path_opt = path # lattice.bz_path(['M', 'K'], [5])

def init_hole(lattice, N, r0, mode='soft'):
    """Initialize the density distribution across N**2 polygons projected onto `lattice`
    """
    x, y, _, _ = generate_grid(lattice, N, extent_min=-0.5, extent_max=0.5)
    r = np.sqrt(np.square(x) + np.square(y))
    rho = np.zeros(r.shape)

    if mode=='hard':
        rho[r<r0] = 1.0
    elif mode =='soft':
        rho += np.exp(-r**2/r0**2)
    else:
        raise ValueError('Invalid init mode = %s' % init)

    return rho

def projection(rho, eta=0.5, beta=100):
    """Density projection operator
    """
    return bd.divide(bd.tanh(beta * eta) + bd.tanh(beta * (rho - eta)), bd.tanh(beta * eta) + bd.tanh(beta * (1 - eta)))

def make_simulation(polygons):
    """Create the gme object and build the phc from `polygons`
    """
    phc = legume.PhotCryst(lattice)
    phc.add_layer(d=args.D, eps_b=args.eps_b)
    for polygon in polygons: phc.layers[-1].add_shape(polygon)
    gme = legume.GuidedModeExp(phc, gmax=args.gmax)
    return gme

def generate_grid(lattice, N, extent_min=-0.5, extent_max=0.5):
    """Generates the center and edge coords for N**2 polygons projected onto `lattice`

    """
    # Set the extent of the grid 
    extent = extent_max - extent_min

    # Grid cell size
    h = extent / N

    # Coordinate transform matrix
    T = np.hstack((lattice.a1[:, np.newaxis], lattice.a2[:, np.newaxis]))

    Xc = np.linspace(extent_min+h/2, extent_max-h/2, N) # Grid cell center x coords
    Yc = np.linspace(extent_min+h/2, extent_max-h/2, N) # Grid cell center y coords
    Xc, Yc = np.meshgrid(Xc, Yc)
    Xc = Xc.reshape(-1)
    Yc = Yc.reshape(-1)

    XYc =  T @ np.stack((Xc, Yc)) 

    # Make *relative* polygon vertex coordinates
    corner_x_rel = [-h/2, +h/2, +h/2, -h/2]
    corner_y_rel = [-h/2, -h/2, +h/2, +h/2]

    # Make absolute polygon vertex coordinates
    coords = []
    for i, (x_rel, y_rel) in enumerate(zip(corner_x_rel, corner_y_rel)):
        coords.append(Xc+x_rel)
        coords.append(Yc+y_rel)

    # Project into the lattice vectors
    # We use block_diag to do the vertex coordinates in parallel
    XYe = scipy.sparse.block_diag((T,T,T,T)) @ np.stack(coords)

    Xe = XYe[0::2,:]
    Ye = XYe[1::2,:]
    Xc = XYc[0,:]
    Yc = XYc[1,:]

    return Xc, Yc, Xe, Ye


def generate_polygons(lattice, rho, eta=0.5, beta=10):
    """Converts the density, `rho`, into legume polygons.

    This performs a projection of the density along the way. 
    `lattice` can be the actual lattice of the crystal or can be hardcoded as 'square'
    if we want to just use a rectangular grid, pray to god, and hope for the best...
    """

    # TODO(ian): perhaps add a check here that len(rho) == N**2 in case of user error
    N = int(np.sqrt(len(rho)))
    _, _, Xe, Ye = generate_grid(lattice, N, extent_min=-0.5, extent_max=0.5)
    
    rho_proj = projection(rho, eta, beta)

    polygons = []
    for i in range(len(rho)):
        eps = args.eps_b + (1 - args.eps_b) * rho_proj[i]
        polygon = legume.Poly(eps=eps,
                              x_edges=Xe[:,i],
                              y_edges=Ye[:,i])
        polygons.append(polygon)

    return polygons

def objective(rho):
    """The objective function.

    Takes rho -> polygons
    Makes gme object
    Runs gme
    Computes size of bandgap
    """
    polygons = generate_polygons(lattice, rho, eta=args.eta, beta=args.beta)
    gme = make_simulation(polygons)
    gme.run(kpoints=path_opt.kpoints, **options)
    band_up = gme.freqs[:,1:]
    band_dn = gme.freqs[:,0]

    # eps_clad = [gme.phc.claddings[0].eps_avg, gme.phc.claddings[-1].eps_avg]
    # vec_LL = bd.sqrt(np.square(gme.kpoints[0, :]) + bd.square(gme.kpoints[1, :])) \
    #     / 2 / np.pi / np.sqrt(bd.max(eps_clad))

    # intersection_up_LL = bd.min(bd.vstack( (band_up, vec_LL) ), axis=0)
    intersection_up_LL = band_up
    gap_width = bd.min(intersection_up_LL)-bd.max(band_dn)

    return gap_width


def summarize_results(gme):
    """Summarize the results of a gme run.

    Plots the bands and the real space structure
    """
    gap_size = bd.min(gme.freqs[:,1])-bd.max(gme.freqs[:,0])
    gap_mid = 0.5*bd.min(gme.freqs[:,1]) + 0.5* bd.max(gme.freqs[:,0])

    print("Gap size (relative): %.2f (%.2f)" % (gap_size, gap_size/gap_mid))

    fig = plt.figure(constrained_layout=True)
    gs = fig.add_gridspec(ncols=2, nrows=1)
    ax1 = fig.add_subplot(gs[0])
    legume.viz.bands(gme, ax=ax1)
    gme.phc.plot_overview(fig=fig, gridspec=gs[1], cbar=False)

legume.set_backend('numpy')

# Initialize the polygon slab

rho_0 = init_hole(lattice, args.N_polygons, args.r0, mode=args.init)

if args.initialize:
    polygons = generate_polygons(lattice, rho_0, eta=args.eta, beta=args.beta)
    gme = make_simulation(polygons)
    gme.run(kpoints=path.kpoints, **options)
    summarize_results(gme)

# Optimize
if args.optimize:
    legume.set_backend('autograd')
    objective_grad = grad(objective)
    (rho_opt, ofs) = adam_optimize(objective, rho_0, objective_grad, step_size=args.lr, Nsteps=args.epochs,
                                   options={'direction': 'max', 'disp': ['of']})

    legume.set_backend('numpy')
    polygons = generate_polygons(lattice, rho_opt, eta=args.eta, beta=args.beta)
    gme = make_simulation(polygons)
    gme.run(kpoints=path.kpoints, **options)
    summarize_results(gme)

if False:
    # WIP on converting to an exportable format
    import skimage

    proj_rho_opt = projection(rho_opt)
    rho_contours = skimage.measure.find_contours(proj_rho_opt, 0.5)

    fig, ax = plt.subplots()
    ax.imshow(proj_rho_opt, cmap=plt.cm.gray)
    for n, contour in enumerate(rho_contours):
        ax.plot(contour[:, 1], contour[:, 0], linewidth=2)

    ax.axis('image')
    ax.set_xticks([])
    ax.set_yticks([])
    plt.show()


    N = args.N_polygons
    vertices = []
    edges = []
    for n, contour in enumerate(rho_contours):
        contour_scaled = contour/(N-1) - 0.5
        vertices.append(contour_scaled)
        edges.append(np.arange(contour_scaled.shape[0]))
