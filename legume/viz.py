import matplotlib.pyplot as plt
import numpy as np


# TODO: Make this more general
def bands(gme, lightcone=True, ax=None, figsize=(4,5), ls='o'):

    if np.all(gme.kpoints[0,:]==0) and not np.all(gme.kpoints[1,:]==0) \
        or np.all(gme.kpoints[1,:]==0) and not np.all(gme.kpoints[0,:]==0):
        X = np.sqrt(np.square(gme.kpoints[0,:]) + 
                np.square(gme.kpoints[1,:])) / 2 / np.pi
    else:
        X = np.arange(len(gme.kpoints[0, :]))

    if ax is None:
        fig, ax = plt.subplots(1, 1, constrained_layout=True, figsize=figsize)

    ax.plot(X, gme.freqs, ls, c="#1f77b4", label="", ms=4, mew=1)

    if lightcone:
        eps_clad = [gme.phc.claddings[0].eps_avg, gme.phc.claddings[-1].eps_avg]
        vec_LL = np.sqrt(np.square(gme.kpoints[0, :]) + 
            np.square(gme.kpoints[1, :])) / 2 / np.pi / np.sqrt(max(eps_clad))
        ax.fill_between(X, vec_LL,  max(vec_LL.max(), gme.freqs[:].max()), 
                        facecolor="#cccccc", zorder=4, alpha=0.5)

    ax.set_xlim(left=0, right=max(X))
    ax.set_ylim(bottom=0.0, top=gme.freqs[:].max())
    # ax.set_xticks([])
    ax.set_xlabel('Wave vector')
    ax.set_ylabel('Frequency')

    plt.show()

    return ax

def plot_eps(eps_r, clim=None, ax=None, extent=None, cmap='Greys', cbar=False):

    if ax is None:
        fig, ax = plt.subplots(1, constrained_layout=True)

    im = ax.imshow(eps_r, cmap=cmap, origin='lower', extent=extent)
    if clim:
        im.set_clim(vmin=clim[0], vmax=clim[1])

    if cbar:
        plt.colorbar(im, ax=ax)
        
    return im

def plot_xz(phc, y=0, Nx=100, Nz=50, ax=None, clim=None,
             cbar=False, cmap='Greys'):
    '''
    Plot an xz-cross section showing all the layers and shapes
    '''
    (xgrid, zgrid) = (phc.lattice.xy_grid(Nx=Nx)[0], phc.z_grid(Nz=Nz))

    [xmesh, zmesh] = np.meshgrid(xgrid, zgrid)
    ymesh = y*np.ones(xmesh.shape)

    eps_r = phc.get_eps((xmesh, ymesh, zmesh))
    extent = [xgrid[0], xgrid[-1], zgrid[0], zgrid[-1]]

    plot_eps(eps_r, clim=clim, ax=ax, extent=extent, cbar=cbar, cmap=cmap)


def plot_xy(phc, z=0, Nx=100, Ny=100, ax=None, clim=None,
             cbar=False, cmap='Greys'):
    '''
    Plot an xy-cross section showing all the layers and shapes
    '''
    (xgrid, ygrid) = phc.lattice.xy_grid(Nx=Nx, Ny=Ny)
    [xmesh, ymesh] = np.meshgrid(xgrid, ygrid)
    zmesh = z*np.ones(xmesh.shape)

    eps_r = phc.get_eps((xmesh, ymesh, zmesh))
    extent = [xgrid[0], xgrid[-1], ygrid[0], ygrid[-1]]

    plot_eps(eps_r, clim=clim, ax=ax, extent=extent, cbar=cbar, cmap=cmap)


def plot_yz(phc, x=0, Ny=100, Nz=50, ax=None, clim=None,
             cbar=False, cmap='Greys'):
    '''
    Plot a yz-cross section showing all the layers and shapes
    '''
    (ygrid, zgrid) = (phc.lattice.xy_grid(Ny=Ny)[1], phc.z_grid(Nz=Nz))
    [ymesh, zmesh] = np.meshgrid(ygrid, zgrid)
    xmesh = x*np.ones(ymesh.shape)

    eps_r = phc.get_eps((xmesh, ymesh, zmesh))
    extent = [ygrid[0], ygrid[-1], zgrid[0], zgrid[-1]]

    plot_eps(eps_r, clim=clim, ax=ax, extent=extent, cbar=cbar, cmap=cmap)


def plot_reciprocal(gme):
    '''
    Plot the reciprocal lattice of a GME object
    '''
    fig, ax = plt.subplots(1, constrained_layout=True)
    plt.plot(gme.gvec[0, :], gme.gvec[1, :], 'bx')
    ax.set_title("Reciprocal lattice")
    plt.show()
