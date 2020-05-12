import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from .gme import GuidedModeExp
from .phc import PhotCryst, Circle
from .pwe import PlaneWaveExp


def bands(
    gme, 
    Q=False, 
    Q_clip=1e10, 
    cone=True, 
    conecolor='#eeeeee', 
    ax=None,
    figsize=(4,5), 
    Q_cmap='viridis', 
    markersize=6, 
    markeredgecolor='w', 
    markeredgewidth=1.5
):
    """Plot photonic band structure from a GME simulation

    Note
    ----
    The bands must be solved for and stored in the `GuidedModeExp` or 
    `PlaneWaveExp` object prior to calling this function.

    Parameters
    ----------
    gme : GuidedModeExp
    Q : bool, optional
        Whether each point should be colored according to the quality factor. 
        Default is False.
    Q_clip : float, optional
        The clipping (vmax) value for the quality factor colormap. Default is
        1e10.
    cone : bool , optional
        Whether the the light cone region of the band structure should be 
        shaded. Default is True.
    conecolor : str, optional
        Color of the light cone region. Default is '#eeeeee' (light grey).
    ax : matplotlib axis object, optional
        Matplotlib axis object for plotting. If not provided, a new figure and 
        axis are automatically created.
    figsize : Tuple, optional
        Figure size for created figure. Default is (4,5).
    Q_cmap : str or matplotlib colormap object, optional
        Colormap used for the quality factor. Default is 'viridis'.
    markersize : float, optional
        Band marker size. Default is 6.
    markeredgecolor : str, optional
        Band marker edge border color. Default is white.
    markeredgewidth : float, optional
        Band marker edge border width. Default is 1.5.

    Returns
    -------
    ax : matplotlib axis object
        Axis object for the plot.
    """

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
            gme.run_im()
        freqs_im = np.array(gme.freqs_im).flatten() + 1e-16
        Q = gme.freqs.flatten()/2/freqs_im
        Q_max = np.max(Q[Q<Q_clip])

        p = ax.scatter(X.flatten(), gme.freqs.flatten(), 
                        c=Q, cmap=Q_cmap, s=markersize**2, vmax=Q_max, 
                        norm=mpl.colors.LogNorm(), edgecolors=markeredgecolor, 
                        linewidth=markeredgewidth)
        plt.colorbar(p, ax=ax, label="Radiative quality factor", extend="max")
    else:
        ax.plot(X, gme.freqs, 'o', c="#1f77b4", label="", ms=markersize, 
                    mew=markeredgewidth, mec=markeredgecolor)

    if cone:
        eps_clad = [gme.phc.claddings[0].eps_avg, gme.phc.claddings[-1].eps_avg]
        vec_LL = np.sqrt(np.square(gme.kpoints[0, :]) + 
            np.square(gme.kpoints[1, :])) / 2 / np.pi / np.sqrt(max(eps_clad))
        ax.fill_between(X0, vec_LL,  max(100, vec_LL.max(), gme.freqs[:].max()), 
                        facecolor=conecolor, zorder=0)

    ax.set_xlim(left=0, right=max(X0))
    ax.set_ylim(bottom=0.0, top=gme.freqs[:].max())
    ax.set_xlabel('Wave vector')
    ax.set_ylabel('Frequency')

    return ax

def _plot_eps(eps_r, clim=None, ax=None, extent=None, cmap='Greys', cbar=False):

    if ax is None:
        fig, ax = plt.subplots(1, constrained_layout=True)

    im = ax.imshow(eps_r, cmap=cmap, origin='lower', extent=extent)
    if clim:
        im.set_clim(vmin=clim[0], vmax=clim[1])

    if cbar:
        plt.colorbar(im, ax=ax)
        
    return im

def _plot_circle(x, y, r, ax=None, color='b', lw=1, npts=51):

    if ax is None:
        fig, ax = plt.subplots(1, constrained_layout=True)

    phi = np.linspace(0, 2*np.pi, npts)
    xs = x + r * np.cos(phi)
    ys = y + r * np.sin(phi)
    pl = ax.plot(xs, ys, c=color, lw=lw)

    return pl

def eps(layer, Nx=100, Ny=100, ax=None, clim=None,
             cbar=False, cmap='Greys'):
    """Plot the in-plane permittivity distribution of a Layer instance

    Parameters
    ----------
    layer : Layer
    Nx : int, optional
        Number of sample points to use in x-direction.
        Default is 100.
    Ny : int, optional
        Number of sample points to use in y-direction.
        Default is 100.
    ax : int, optional
        Matplotlib axis object to use for plot.
    clim : List[float], optional
        Matplotlib color limit to use for plot.
        Default is None.
    cbar : bool, optional
        Whether or not a colorbar should be added to the plot.
        Default is False.
    cmap : bool, optional
        Matplotlib colormap to use for plot
        Default is 'Greys'.
    """
    (xgrid, ygrid) = layer.lattice.xy_grid(Nx=Nx, Ny=Ny)
    [xmesh, ymesh] = np.meshgrid(xgrid, ygrid)

    eps_r = layer.get_eps((xmesh, ymesh))
    extent = [xgrid[0], xgrid[-1], ygrid[0], ygrid[-1]]

    _plot_eps(eps_r, clim=clim, ax=ax, extent=extent, cbar=cbar, cmap=cmap)

def eps_xz(phc, y=0, Nx=100, Nz=50, ax=None, clim=None,
             cbar=False, cmap='Greys', plot=True):
    """Plot permittivity cross section of a photonic crystal in an xz plane

    Parameters
    ----------
    phc : PhotCryst
    y : float, optional
        The y-coordinate of the xz plane.
        Default is 0.
    Nx : int, optional
        Number of sample points to use in x-direction.
        Default is 100.
    Nz : int, optional
        Number of sample points to use in z-direction.
        Default is 50.
    ax : int, optional
        Matplotlib axis object to use for plot.
    clim : List[float], optional
        Matplotlib color limit to use for plot.
        Default is None.
    cbar : bool, optional
        Whether or not a colorbar should be added to the plot.
        Default is False.
    cmap : bool, optional
        Matplotlib colormap to use for plot
        Default is 'Greys'.
    plot : bool, optional
        Whether or not the a plot should be generated. Useful for cases where
        only the array values are needed, e.g. for logging during optimization.
        Default is True.

    Returns
    -------
    eps_r : np.ndarray
        Array containing permittivity values
    """
    (xgrid, zgrid) = (phc.lattice.xy_grid(Nx=Nx)[0], phc.z_grid(Nz=Nz))

    [xmesh, zmesh] = np.meshgrid(xgrid, zgrid)
    ymesh = y*np.ones(xmesh.shape)

    eps_r = phc.get_eps((xmesh, ymesh, zmesh))
    extent = [xgrid[0], xgrid[-1], zgrid[0], zgrid[-1]]

    if plot:
        _plot_eps(eps_r, clim=clim, ax=ax, extent=extent, cbar=cbar, cmap=cmap)

    return eps_r

def eps_xy(phc, z=0, Nx=100, Ny=100, ax=None, clim=None,
             cbar=False, cmap='Greys', plot=True):
    """Plot permittivity cross section of a photonic crystal in an xy plane

    Parameters
    ----------
    phc : PhotCryst
    z : float, optional
        The z-coordinate of the xz plane.
        Default is 0.
    Nx : int, optional
        Number of sample points to use in x-direction.
        Default is 100.
    Ny : int, optional
        Number of sample points to use in y-direction.
        Default is 100.
    ax : int, optional
        Matplotlib axis object to use for plot.
    clim : List[float], optional
        Matplotlib color limit to use for plot.
        Default is None.
    cbar : bool, optional
        Whether or not a colorbar should be added to the plot.
        Default is False.
    cmap : bool, optional
        Matplotlib colormap to use for plot
        Default is 'Greys'.
    plot : bool, optional
        Whether or not the a plot should be generated. Useful for cases where
        only the array values are needed, e.g. for logging during optimization.
        Default is True.

    Returns
    -------
    eps_r : np.ndarray
        Array containing permittivity values
    """
    (xgrid, ygrid) = phc.lattice.xy_grid(Nx=Nx, Ny=Ny)
    [xmesh, ymesh] = np.meshgrid(xgrid, ygrid)
    zmesh = z*np.ones(xmesh.shape)

    eps_r = phc.get_eps((xmesh, ymesh, zmesh))
    extent = [xgrid[0], xgrid[-1], ygrid[0], ygrid[-1]]

    if plot:
        _plot_eps(eps_r, clim=clim, ax=ax, extent=extent, cbar=cbar, cmap=cmap)

    return eps_r


def eps_yz(phc, x=0, Ny=100, Nz=50, ax=None, clim=None,
             cbar=False, cmap='Greys', plot=True):
    """Plot permittivity cross section of a photonic crystal in an yz plane

    Parameters
    ----------
    phc : PhotCryst
    x : float, optional
        The x-coordinate of the xz plane.
        Default is 0.
    Ny : int, optional
        Number of sample points to use in y-direction.
        Default is 100.
    Nz : int, optional
        Number of sample points to use in z-direction.
        Default is 50.
    ax : int, optional
        Matplotlib axis object to use for plot.
    clim : List[float], optional
        Matplotlib color limit to use for plot.
        Default is None.
    cbar : bool, optional
        Whether or not a colorbar should be added to the plot.
        Default is False.
    cmap : bool, optional
        Matplotlib colormap to use for plot
        Default is 'Greys'.
    plot : bool, optional
        Whether or not the a plot should be generated. Useful for cases where
        only the array values are needed, e.g. for logging during optimization.
        Default is True.

    Returns
    -------
    eps_r : np.ndarray
        Array containing permittivity values
    """
    (ygrid, zgrid) = (phc.lattice.xy_grid(Ny=Ny)[1], phc.z_grid(Nz=Nz))
    [ymesh, zmesh] = np.meshgrid(ygrid, zgrid)
    xmesh = x*np.ones(ymesh.shape)

    eps_r = phc.get_eps((xmesh, ymesh, zmesh))
    extent = [ygrid[0], ygrid[-1], zgrid[0], zgrid[-1]]

    if plot:
        _plot_eps(eps_r, clim=clim, ax=ax, extent=extent, cbar=cbar, cmap=cmap)

    return eps_r

def shapes(layer, ax=None, npts=101, color='k', lw=1, pad=True):
    """Plot all shapes of Layer
    """

    (xext, yext) = layer.lattice.xy_grid(Nx=2, Ny=2)
    if ax is None:
        fig, ax = plt.subplots(1, constrained_layout=True)

    if pad == True:
        a1 = layer.lattice.a1
        a2 = layer.lattice.a2
        xy_p = [a1, -a1, a2, -a2]

    for shape in layer.shapes:
        if type(shape) == Circle:
            x = shape.x_cent
            y = shape.y_cent
            r = shape.r
            _plot_circle(x, y, r, ax=ax, color=color, lw=lw, npts=npts)
            if pad == True:
                for (x_p, y_p) in xy_p:
                    _plot_circle(x + x_p, y + y_p, r,
                                ax=ax, color=color, lw=lw, npts=npts)
        else:
            # Everything else should be a Poly subclass
            ax.plot(shape.x_edges, shape.y_edges, c=color, lw=lw)
            if pad == True:
                for (x_p, y_p) in xy_p:
                    ax.plot(shape.x_edges + x_p, shape.y_edges + y_p,
                            c=color, lw=lw)
    ax.set_xlim(xext)
    ax.set_ylim(yext)
    ax.set_aspect('equal')


def structure(struct, Nx=100, Ny=100, Nz=50, cladding=False, cbar=True, 
                cmap='Greys', gridspec=None, fig=None, figsize=(4,8)):
    """Plot permittivity for all cross sections of a photonic crystal

    Parameters
    ----------
    struct : GuidedModeExp or PlaneWaveExp 
    Nx : int, optional
        Number of sample points to use in x-direction.
        Default is 100.
    Ny : int, optional
        Number of sample points to use in y-direction.
        Default is 100.
    Nz : int, optional
        Number of sample points to use in z-direction.
        Default is 50.
    cladding : bool, optional
        Whether or not the cladding should be plotted.
        Default is False.
    cbar : bool, optional
        Whether or not a colorbar should be added to the plot.
        Default is False.
    cmap : bool, optional
        Matplotlib colormap to use for plot
        Default is 'Greys'.
    gridspec : Matplotlib gridspec object, optional
        Gridspec to use for creating the plots.
        Default is None.
    fig : Matplotlib figure object, optional
        Figure to use for creating the plots.
        Default is None.
    figsize : Tuple, optional
        Size of Matplotlib figure to create.
        Default is (4,8).
    """
    if isinstance(struct, GuidedModeExp):
        phc = struct.phc
    elif isinstance(struct, PhotCryst):
        phc = struct
    else:
        raise ValueError("'struct' should be a 'PhotCryst' or a "
                                "'GuidedModeExp' instance")

    (eps_min, eps_max) = phc.get_eps_bounds()

    if cladding==True:
        all_layers = [phc.claddings[0]] + phc.layers + [phc.claddings[1]]
    else:
        all_layers = phc.layers
    N_layers = len(all_layers)

    if gridspec is None and fig is None:
        fig = plt.figure(constrained_layout=True, figsize=figsize)
        gs = mpl.gridspec.GridSpec(N_layers+1, 2, figure=fig)
    elif gridspec is not None and fig is not None:
        gs = mpl.gridspec.GridSpecFromSubplotSpec(N_layers+1, 2, gridspec)
    else:
        raise ValueError("Parameters gridspec and fig should be both specified "
                            "or both unspecified")

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax = []
    for i in range(N_layers):
        ax.append(fig.add_subplot(gs[1+i, :]))

    eps_xz(phc, ax=ax1, Nx=Nx, Nz=Nz,
                clim=[eps_min, eps_max], cbar=False, cmap=cmap)
    ax1.set_title("xz at y = 0")
    eps_yz(phc, ax=ax2, Ny=Ny, Nz=Nz,
                clim=[eps_min, eps_max], cbar=cbar, cmap=cmap)
    ax2.set_title("yz at x = 0")

    for indl in range(N_layers):
        zpos = (all_layers[indl].z_max + all_layers[indl].z_min)/2
        eps_xy(phc, z=zpos, ax=ax[indl], Nx=Nx, Ny=Ny,
                clim=[eps_min, eps_max], cbar=False, cmap=cmap)
        if cladding==True:
            if indl > 0 and indl < N_layers-1:
                ax[indl].set_title("xy in layer %d" % indl)
            elif indl==N_layers-1:
                ax[0].set_title("xy in lower cladding")
                ax[-1].set_title("xy in upper cladding")
        else:
            ax[indl].set_title("xy in layer %d" % indl)


def eps_ft(struct, Nx=100, Ny=100, cladding=False, cbar=True, 
                cmap='Greys', gridspec=None, fig=None, figsize=(4,8)):
    """Plot a permittivity cross section computed from an inverse FT

    The Fourier transform is computed with respect to the GME reciprocal
    lattice vectors.

    Parameters
    ----------
    struct : GuidedModeExp or PlaneWaveExp 
    Nx : int, optional
        Number of sample points to use in x-direction.
        Default is 100.
    Ny : int, optional
        Number of sample points to use in y-direction.
        Default is 100.
    cladding : bool, optional
        Whether or not the cladding should be plotted.
        Default is False.
    cbar : bool, optional
        Whether or not a colorbar should be added to the plot.
        Default is False.
    cmap : bool, optional
        Matplotlib colormap to use for plot
        Default is 'Greys'.
    gridspec : Matplotlib gridspec object, optional
        Gridspec to use for creating the plots.
        Default is None.
    fig : Matplotlib figure object, optional
        Figure to use for creating the plots.
        Default is None.
    figsize : Tuple, optional
        Size of Matplotlib figure to create.
        Default is (4,8).
    """

    # Do some parsing of the inputs 
    if isinstance(struct, GuidedModeExp):
        str_type = 'gme'
    elif isinstance(struct, PlaneWaveExp):
        str_type = 'pwe'
    else:
        raise ValueError("'struct' should be a 'PlaneWaveExp' or a "
                                "'GuidedModeExp' instance")

    if cladding==True:
        if str_type == 'pwe':
            print("Warning: ignoring 'cladding=True' for PlaneWaveExp "
                   "structure.")
            all_layers = [struct.layer]
        else:
            all_layers = [struct.phc.claddings[0]] + struct.phc.layers + \
                        [struct.phc.claddings[1]]
    else:
        all_layers = struct.phc.layers if str_type == 'gme' else [struct.layer]
    N_layers = len(all_layers)

    # Initialize gridspec and figure
    if gridspec is None and fig is None:
        fig = plt.figure(constrained_layout=True, figsize=figsize)
        gs = mpl.gridspec.GridSpec(N_layers, 1, figure=fig)
    elif gridspec is not None and fig is not None:
        gs = mpl.gridspec.GridSpecFromSubplotSpec(N_layers, 1, gridspec)
    else:
        raise ValueError("Parameters gridspec and fig should be both specified "
                            "or both unspecified")

    ax = []
    for i in range(N_layers):
        ax.append(fig.add_subplot(gs[i, :]))

    (eps_min, eps_max) = (all_layers[0].eps_b, all_layers[0].eps_b)
    ims = []
    for (indl, layer) in enumerate(all_layers):
        (eps_r, xgrid, ygrid) = struct.get_eps_xy(Nx=Nx, Ny=Ny,
                                z=(layer.z_min + layer.z_max)/2)

        eps_min = min([eps_min, np.amin(np.real(eps_r))])
        eps_max = max([eps_max, np.amax(np.real(eps_r))])
        extent = [xgrid[0], xgrid[-1], ygrid[0], ygrid[-1]]

        im = _plot_eps(np.real(eps_r), ax=ax[indl], extent=extent, 
                        cbar=False)
        ims.append(im)
        if cladding:
            if indl > 0 and indl < N_layers-1:
                ax[indl].set_title("xy in layer %d" % indl)
            elif indl==N_layers-1:
                ax[0].set_title("xy in lower cladding")
                ax[-1].set_title("xy in upper cladding")
        else:
            ax[indl].set_title("xy in layer %d" % indl)
    
    for il in range(N_layers):
        ims[il].set_clim(vmin=eps_min, vmax=eps_max)
    plt.colorbar(ims[-1])


def reciprocal(struct):
    """Plot the reciprocal lattice of a GME or PWE object

    Parameters
    ----------
    struct : GuidedModeExp or PlaneWaveExp 
    """
    fig, ax = plt.subplots(1, constrained_layout=True)
    plt.plot(struct.gvec[0, :], struct.gvec[1, :], 'bx')
    ax.set_title("Reciprocal lattice")


def field(
    struct, 
    field, 
    kind, 
    mind, 
    x=None, 
    y=None, 
    z=None, 
    periodic=True,
    component='xyz', 
    val='re', 
    N1=100, 
    N2=100, 
    cbar=True, 
    eps=True,
    eps_levels=None
):
    """Visualize mode fields over a 2D slice in x, y, or z

    Note
    ----
    The fields must be solved for and stored in the `GuidedModeExp` or 
    `PlaneWaveExp` object prior to calling this function.

    Parameters
    ----------
    struct : GuidedModeExp or PlaneWaveExp 
    field : {'H', 'D', 'E'}
        The field quantity to plot
    kind : int
        The wave vector index to plot
    mind : int
        The mode index to plot
    x, y, z : float
        Coordinate of the slice in either x, y, or z. One and 
        only one of these should be specified
    periodic : bool, optional
        Whether the periodic portion or the full field should be plotted
    component : str, optional
        Component of the vector field to plot
    val : {'re', 'im', 'abs'}, optional
        Field value to plot
    N1 : int, optional
        Number of grid points to sample in first spatial dim
    N2 : int, optional
        Number of grid points to sample in second spatial dim
    cbar : bool, optional
        Whether to include a colorbar
    eps : bool, optional
        Whether an outline of the permittivity should be overlaid
    eps_levels : List, optional
        A list of contour levels for the permittivity

    Returns
    -------
    fig : matplotlib figure object
        Figure object for the plot.
    """

    if isinstance(struct, GuidedModeExp):
        str_type = 'gme'
    elif isinstance(struct, PlaneWaveExp):
        str_type = 'pwe'
    else:
        raise ValueError("'struct' should be a 'PlaneWaveExp' or a "
                                "'GuidedModeExp' instance")

    field = field.lower()
    val = val.lower()
    component = component.lower()

    # Get the field fourier components
    if (x is None and y is None and z is None and str_type == 'pwe') or \
        (z is not None and x is None and y is None):

        zval = 0. if z == None else z
        (fi, grid1, grid2) = struct.get_field_xy(field, kind, mind, z,
                                        component=component, Nx=N1, Ny=N2)
        if eps == True:
            if str_type == 'pwe':
                epsr = struct.layer.get_eps(np.meshgrid(
                            grid1, grid2)).squeeze()

            else:
                epsr = struct.phc.get_eps(np.meshgrid(
                            grid1, grid2, np.array(z))).squeeze()

        pl, o, v = 'xy', 'z', zval
        if periodic==False:
            kenv = np.exp(1j*grid1*struct.kpoints[0, kind] + 
                            1j*grid2*struct.kpoints[1, kind])

    elif x is not None and z is None and y is None:
        if str_type == 'pwe':
            raise NotImplementedError("Only plotting in the xy-plane is "
                "supported for PlaneWaveExp structures.")

        (fi, grid1, grid2) = struct.get_field_yz(field, kind, mind, x, 
                                            component=component, Ny=N1, Nz=N2)
        if eps==True:
            epsr = struct.phc.get_eps(np.meshgrid(
                            np.array(x), grid1, grid2)).squeeze().transpose()
        pl, o, v = 'yz', 'x', x
        if periodic==False:
            kenv = np.exp(1j*grid1*struct.kpoints[1, kind] +
                            1j*x*struct.kpoints[0, kind])
    elif y is not None and z is None and x is None:
        if str_type == 'pwe':
            raise NotImplementedError("Only plotting in the xy-plane is "
                "supported for PlaneWaveExp structures.")

        (fi, grid1, grid2) = struct.get_field_xz(field, kind, mind, y, 
                                            component=component, Nx=N1, Nz=N2)
        if eps==True:
            epsr = struct.phc.get_eps(np.meshgrid(
                            grid1, np.array(y), grid2)).squeeze().transpose()
        pl, o, v = 'xz', 'y', y
        if periodic==False:
            kenv = np.exp(1j*grid1*struct.kpoints[0, kind] +
                            1j*y*struct.kpoints[1, kind])
    else:
        raise ValueError("Specify exactly one of 'x', 'y', or 'z'.")

    sp = len(component)
    f1, axs = plt.subplots(1, sp, constrained_layout=True)
    for ic, comp in enumerate(component):
        f = fi[comp]
        if periodic==False:
            f *= kenv
        
        extent = [grid1[0], grid1[-1], grid2[0], grid2[-1]]
        if sp > 1:
            ax = axs[ic]
        else:
            ax = axs

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
            lcs = 'k' if val.lower() in ['re', 'im'] else 'w'
            ax.contour(grid1, grid2, epsr, 0 if eps_levels is None else \
                        eps_levels, colors=lcs, linewidths=1, alpha=0.5)

        if cbar==True:
            f1.colorbar(im, ax=ax, shrink=0.5)

        title_str = ""

        title_str += "%s$(%s_{%s%d})$ at $k_{%d}$\n" % (val.capitalize(), 
                                    field.capitalize(), comp, mind, kind)
        title_str += "%s-plane at $%s = %1.2f$\n" % (pl, o, v)
        title_str += "$f = %.2f$" % (struct.freqs[kind, mind])
        if str_type == 'gme':
            if struct.freqs_im != []:
                if np.abs(struct.freqs_im[kind, mind]) > 1e-20:
                    title_str += " $Q = %.2E$\n" % (struct.freqs[kind, mind]/2/
                                            struct.freqs_im[kind, mind])
                else:
                    title_str += " $Q = Inf\n"

        ax.set_title(title_str)

    return f1
