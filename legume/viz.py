import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np


# TODO: Make this more general
def bands(gme, lightcone=True, ax=None, figsize=(4,5), ls='o', Q=False, 
    cmap='viridis', size=20, edgecolor='w', Q_clip=1e10):

    if np.all(gme.kpoints[0,:]==0) and not np.all(gme.kpoints[1,:]==0) \
        or np.all(gme.kpoints[1,:]==0) and not np.all(gme.kpoints[0,:]==0):
        X0 = np.sqrt(np.square(gme.kpoints[0,:]) + 
                np.square(gme.kpoints[1,:])) / 2 / np.pi
    else:
        X0 = np.arange(len(gme.kpoints[0, :]))

    X = np.tile(X0.reshape(len(X0),1), (1, gme.freqs.shape[1]))

    if ax is None:
        fig, ax = plt.subplots(1, 1, constrained_layout=True, figsize=figsize)
    if Q:
        if len(gme.freqs_im) == 0:
            freqs_im = []
            for kind in range(len(X0)):
                (freq_im, _, _) = gme.compute_rad(kind=kind, 
                                        minds=range(gme.numeig))
                freqs_im.append(freq_im)
        else:
            freqs_im = gme.freqs_im
        freqs_im = np.array(freqs_im).flatten() + 1e-16
        Q = gme.freqs.flatten()/2/freqs_im
        Q_max = np.max(Q[Q<Q_clip])

        p = ax.scatter(X.flatten(), gme.freqs.flatten(), 
                            c=Q, cmap=cmap, s=size, vmax=Q_max, 
                            norm=mpl.colors.LogNorm(), edgecolors=edgecolor)
        plt.colorbar(p, ax=ax, label="Radiative quality factor", extend="max")
    else:
        ax.plot(X, gme.freqs, ls, c="#1f77b4", label="", ms=4, mew=1)

    if lightcone:
        eps_clad = [gme.phc.claddings[0].eps_avg, gme.phc.claddings[-1].eps_avg]
        vec_LL = np.sqrt(np.square(gme.kpoints[0, :]) + 
            np.square(gme.kpoints[1, :])) / 2 / np.pi / np.sqrt(max(eps_clad))
        ax.fill_between(X0, vec_LL,  max(100, vec_LL.max(), gme.freqs[:].max()), 
                        facecolor="#eeeeee", zorder=0)

    ax.set_xlim(left=0, right=max(X0))
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

def plot_field(gme, field, kind, mind, x=None, y=None, z=None,
            component='xyz', val='re', N1=100, N2=100, cbar=True, eps=True):
    '''
    Plot the 'field' ('H' or 'D') at the plane defined by 'x', 'y', or 'z', 
    for mode number mind at k-vector kind. 
    'comp' can be: 'x', 'y', 'z' or a combination thereof, e.g. 'xz' (a 
    separate plot is created for each component)
    'val' can be: 're', 'im', 'abs'
    '''

    field = field.lower()
    val = val.lower()
    component = component.lower()

    # Get the field fourier components
    if z is not None and x is None and y is None:
        (fi, grid1, grid2) = gme.get_field_xy(field, kind, mind, z, 
                                            component, N1, N2)
        if eps==True:
            epsr = gme.phc.get_eps(np.meshgrid(
                            grid1, grid2, np.array(z))).squeeze()
        pl, o, v = 'xy', 'z', z
    elif x is not None and z is None and y is None:
        (fi, grid1, grid2) = gme.get_field_yz(field, kind, mind, x, 
                                            component, N1, N2)
        if eps==True:
            epsr = gme.phc.get_eps(np.meshgrid(
                            np.array(x), grid1, grid2)).squeeze().transpose()
        pl, o, v = 'yz', 'x', x
    elif y is not None and z is None and x is None:
        (fi, grid1, grid2) = gme.get_field_xz(field, kind, mind, y, 
                                            component, N1, N2)
        if eps==True:
            epsr = gme.phc.get_eps(np.meshgrid(
                            grid1, np.array(y), grid2)).squeeze().transpose()
        pl, o, v = 'xz', 'y', y
    else:
        raise ValueError("Specify exactly one of 'x', 'y', or 'z'.")

    print(epsr.shape, grid1.shape, grid2.shape)

    f1 = plt.figure()
    sp = len(component)
    for ic, comp in enumerate(component):
        f = fi[comp]
        
        extent = [grid1[0], grid1[-1], grid2[0], grid2[-1]]
        ax = f1.add_subplot(1, sp, ic+1)

        if val=='re' or val=='im':
            Z = np.real(f) if val=='re' else np.imag(f)
            cmap = 'RdBu'
            vmax = np.abs(Z).max()
            vmin = -vmax
        elif val=='abs':
            Z = np.abs(f)
            cmap='magma'
            vmax = Z.max()
            vmin = 0
        else:
            raise ValueError("'val' can be 'im', 're', or 'abs'")

        im = ax.imshow(Z, extent=extent, cmap=cmap, vmin=vmin, vmax=vmax,
                        origin='lower')

        if eps==True:
            ax.contour(grid1, grid2, epsr, 1, colors='k', linewidths=1)

        if cbar==True:
            f1.colorbar(im, ax=ax)
        ax.set_title("%s(%s_%s)" % (val, field, comp))
        f1.suptitle("%s-plane at %s = %1.4f" %(pl, o, v))
        plt.show()