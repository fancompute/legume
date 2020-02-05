import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from .gme import GuidedModeExp
from .phc import PhotCryst, Circle
from .pwe import PlaneWaveExp


def bands(gme, Q=False, Q_clip=1e10, cone=True, conecolor='#eeeeee', ax=None, 
    figsize=(4,5), 
    Q_cmap='viridis', markersize=6, markeredgecolor='w', markeredgewidth=1.5):
    """Visualize the computed bands and, optionally, the quality factor from a 
    GME object

    Required arguments:
    gme             -- The GME object whose bands should be visualized

    Keyword arguments:
    cone            -- Boolean specifying whether the light cone should be 
                        shaded
    conecolor       -- Color string specifying the color of the light cone
    Q               -- Boolean specifying whether the quality factor should be 
                        visualized on the bands
    Q_clip          -- Value to clip the Q colormap
    ax              -- Matplotlib axis handle for plotting, if None, a new 
                        figure is created
    figsize         -- If creating a new figure, this size is used
    Q_cmap          -- Colormap used to visualize quality factor
    markersize      -- Band marker size
    markeredgecolor -- Band marker edge border color
    markeredgewidth -- Band marker edge border width
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

    # plt.show()

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

def plot_circle(x, y, r, ax=None, color='b', lw=1, npts=51):

    if ax is None:
        fig, ax = plt.subplots(1, constrained_layout=True)

    phi = np.linspace(0, 2*np.pi, npts)
    xs = x + r * np.cos(phi)
    ys = y + r * np.sin(phi)
    pl = ax.plot(xs, ys, c=color, lw=lw)

    return pl

def eps(layer, Nx=100, Ny=100, ax=None, clim=None,
             cbar=False, cmap='Greys'):
    '''
    Plot the in-plane permittivity distribution of a Layer instance
    '''
    (xgrid, ygrid) = layer.lattice.xy_grid(Nx=Nx, Ny=Ny)
    [xmesh, ymesh] = np.meshgrid(xgrid, ygrid)

    eps_r = layer.get_eps((xmesh, ymesh))
    extent = [xgrid[0], xgrid[-1], ygrid[0], ygrid[-1]]

    plot_eps(eps_r, clim=clim, ax=ax, extent=extent, cbar=cbar, cmap=cmap)

def eps_xz(phc, y=0, Nx=100, Nz=50, ax=None, clim=None,
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

def eps_xy(phc, z=0, Nx=100, Ny=100, ax=None, clim=None,
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


def eps_yz(phc, x=0, Ny=100, Nz=50, ax=None, clim=None,
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

def shapes(layer, ax=None, npts=101, color='k', lw=1, pad=True):
    '''Plot all the shapes in a ShapesLayer object 'layer'
    npts      -- number of points for discretization of circles
    pad       -- if True, will add an extra elementary cell on each side
    '''

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
            plot_circle(x, y, r, ax=ax, color=color, lw=lw, npts=npts)
            if pad == True:
                for (x_p, y_p) in xy_p:
                    plot_circle(x + x_p, y + y_p, r,
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
    # plt.show()

def structure(struct, Nx=100, Ny=100, Nz=50, cladding=False, cbar=True, 
                cmap='Greys', gridspec=None, fig=None, figsize=(4,8)):
    '''
    Plot the permittivity of the PhC cross-sections
    '''
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
    # plt.show()

def eps_ft(struct, Nx=100, Ny=100, cladding=False, cbar=True, 
                cmap='Greys', gridspec=None, fig=None, figsize=(4,8)):
    '''
    Plot the permittivity of the PhC cross-sections as computed from an 
    inverse Fourier transform with the GME reciprocal lattice vectors.
    '''

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

        im = plot_eps(np.real(eps_r), ax=ax[indl], extent=extent, 
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
    # plt.show()

def reciprocal(exp):
    '''
    Plot the reciprocal lattice of a GME or PWE object
    '''
    fig, ax = plt.subplots(1, constrained_layout=True)
    plt.plot(exp.gvec[0, :], exp.gvec[1, :], 'bx')
    ax.set_title("Reciprocal lattice")
    # plt.show()

def field(struct, field, kind, mind, x=None, y=None, z=None, periodic=True,
            component='xyz', val='re', N1=100, N2=100, cbar=True, eps=True, 
            eps_levels=None):
    """Visualize the field components of a mode over a slice in x, y, or z

    Required arguments:
    struct          -- A GME or PWE object
    field           -- The field quantity, should be one of 'H', 'D', or 'E'
    kind            -- The wave vector index of the mode
    mind            -- The mode index
    x, y, or z      -- Coordinate of the slice in either x, y, or z. One and 
                        only one of these should be specified

    Keyword arguments:
    periodic        -- Whether the periodic portion or the full field should be 
                        plotted
    component       -- Component of the vector field to plot
    val             -- Field value to plot, either 're', 'im', or 'abs'
    N1              -- Number of grid points to sample in first spatial dim
    N2              -- Number of grid points to sample in second spatial dim
    cbar            -- Whether to include a colorbar
    eps             -- Whether an outline of the permittivity should be overlaid
    eps_levels      -- The contour levels for the permittivity
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
                                         component, N1, N2)
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
                                            component, N1, N2)
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
                                            component, N1, N2)
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
                title_str += " $Q = %.2E$\n" % (struct.freqs[kind, mind]/2/
                                            struct.freqs_im[kind, mind])

        ax.set_title(title_str)
    # plt.show()

    return f1
