import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

import legume.viz as viz
from legume.backend import backend as bd
import legume.utils as utils
from . import ShapesLayer, FreeformLayer, Lattice

class PhotCryst(object):
    '''
    Class for a photonic crystal which can contain a number of layers
    '''
    def __init__(self, lattice, eps_l=1, eps_u=1):
        # Define permittivity of lower and upper cladding
        # For now claddings are just ShapesLayer, maybe extend to freeform
        self.claddings = []
        self.claddings.append(ShapesLayer(lattice, eps_b=eps_l, z_min=-1e50))
        self.claddings.append(ShapesLayer(lattice, eps_b=eps_u, z_max=1e50))
        
        self.lattice = lattice

        # Initialize an empty list of layers
        self.layers = []

    def __repr__(self):
        rep = 'PhotonicCrystal('
        rep += '\n' + repr(self.lattice)
        for i, layer in enumerate(self.layers):
            rep += '\n%d: ' % i + repr(layer)
        rep += '\n)' if len(self.layers) > 0 else ')'
        return rep

    def z_grid(self, Nz=100, dist=1):
        ''' 
        Define a z-grid for visualization purposes once some layers have been 
        added
        '''
        zmin = self.layers[0].z_min - dist
        zmax = self.layers[-1].z_max + dist
    
        return np.linspace(zmin, zmax, Nz)

    def add_layer(self, d, eps_b=1, layer_type='shapes'):
        '''
        Add a layer with thickness d and background permittivity eps_b
        '''
        if self.layers == []:
            z_min = 0
        else:
            z_min = self.layers[-1].z_max

        if layer_type.lower() == 'shapes':
            layer = ShapesLayer(self.lattice, z_min, z_min + d, eps_b=eps_b)
        elif layer_type.lower() == 'freeform':
            layer = FreeformLayer(self.lattice, z_min, z_min + d, eps_b=eps_b)
        else:
            raise ValueError("'layer_type' must be 'shapes' or 'freeform'")

        self.claddings[1].z_min = z_min + d
        self.layers.append(layer)

    def add_shape(self, *args, **kwargs):
        '''
        Add a shape to layer number layer_ind
        '''
        cladding = kwargs.get('cladding', None)
        layer = kwargs.get('layer', -1)
        if cladding is not None:
            if cladding==0 or cladding.lower()=='l':
                lay = self.claddings[0]
            elif cladding==1 or cladding.lower()=='u':
                lay = self.claddings[1]
            else:
                raise ValueError("'cladding' must be 0 or 'l' for lower" \
                    "cladding and 1 or 'u' for upper cladding")
        else:
            if layer >= len(self.layers):
                raise ValueError("Layer index larger than total number of "\
                    "layers")
            else:
                lay = self.layers[layer]
        if lay.layer_type == 'shapes':
            lay.add_shape(*args)
        else:
            raise TypeError("Shapes can only be added to a ShapesLayer")

    def get_eps(self, points):
        '''
        Compute the permittivity of the PhC at a set of points defined by
        a tuple of x, y, z positions which are same-size arrays
        '''
        (xmesh, ymesh, zmesh) = points
        a_shape = xmesh.shape
        if (ymesh.shape != a_shape) or (ymesh.shape != a_shape):
            raise ValueError(
                    "xmesh, ymesh and zmesh must have the same shape")

        eps_r = np.zeros(a_shape)

        # eps_r[zmesh < self.layers[0].z_min] = self.eps_l
        # eps_r[zmesh >= self.layers[-1].z_max] = self.eps_u
        a1 = self.lattice.a1
        a2 = self.lattice.a2

        for layer in self.layers + self.claddings:
            zlayer = (zmesh >= layer.z_min) * (zmesh < layer.z_max)
            eps_r[zlayer] = layer.eps_b

            # Slightly hacky way to include the periodicity
            a_p = min([np.linalg.norm(a1), 
                       np.linalg.norm(a2)])
            nmax = np.int_(np.sqrt(np.square(np.max(abs(xmesh))) + 
                            np.square(np.max(abs(ymesh))))/a_p) + 1

            for shape in layer.shapes:
                for n1 in range(-nmax, nmax):
                    for n2 in range(-nmax, nmax):
                        in_shape = shape.is_inside(xmesh + 
                            n1*a1[0] + n2*a2[0], ymesh + 
                            n1*a1[1] + n2*a2[1])
                        eps_r[in_shape*zlayer] = utils.get_value(shape.eps)

        return eps_r

    def get_eps_bounds(self):
        # Returns the minimum and maximum permittivity of the structure
        eps_min = self.claddings[0].eps_b
        eps_max = self.claddings[0].eps_b

        for layer in self.layers + self.claddings:
            eps_min = min([eps_min, layer.eps_b])
            eps_max = max([eps_max, layer.eps_b])
            for shape in layer.shapes:
                eps_min = min([eps_min, shape.eps])
                eps_max = max([eps_max, shape.eps])

        return (eps_min, eps_max)

    def plot_cross(self, cross_section='xy', pos=0, Npts=[100, 100]):
        '''
        Plot a cross-section of the PhC at position pos along the third axis
        '''
        if cross_section == 'xy':
            viz.plot_xy(self, z=pos, Nx=Npts[0], Ny=Npts[1])
        elif cross_section == 'xz':
            viz.plot_xz(self, y=pos, Nx=Npts[0], Nz=Npts[1])
        elif cross_section == 'yz':
            viz.plot_yz(self, x=pos, Ny=Npts[0], Nz=Npts[1])
        else:
            raise ValueError("Cross-section must be in {'xy', 'yz', 'xz'}")

    def plot_overview(self, Nx=100, Ny=100, Nz=50, cladding=False, cbar=True, cmap='Greys', gridspec=None, fig=None, figsize=(4,8)):
        '''
        Plot an overview of PhC cross-sections
        '''

        (eps_min, eps_max) = self.get_eps_bounds()

        if cladding:
            all_layers = [self.claddings[0]] + self.layers + [self.claddings[1]]
        else:
            all_layers = self.layers
        N_layers = len(all_layers)

        if gridspec is None and fig is None:
            fig = plt.figure(constrained_layout=True, figsize=figsize)
            gs = mpl.gridspec.GridSpec(N_layers+1, 2, figure=fig)
        elif gridspec is not None and fig is not None:
            gs = mpl.gridspec.GridSpecFromSubplotSpec(N_layers+1, 2, gridspec)
        else:
            raise ValueError("Parameters gridspec and fig should be both specified or both unspecified")

        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])
        ax = []
        for i in range(N_layers):
            ax.append(fig.add_subplot(gs[1+i, :]))

        viz.plot_xz(self, ax=ax1, Nx=Nx, Nz=Nz,
                    clim=[eps_min, eps_max], cbar=False, cmap=cmap)
        ax1.set_title("xz at y = 0")
        viz.plot_yz(self, ax=ax2, Ny=Ny, Nz=Nz,
                    clim=[eps_min, eps_max], cbar=cbar, cmap=cmap)
        ax2.set_title("yz at x = 0")

        for indl in range(N_layers):
            zpos = (all_layers[indl].z_max + all_layers[indl].z_min)/2
            viz.plot_xy(self, z=zpos, ax=ax[indl], Nx=Nx, Ny=Ny,
                    clim=[eps_min, eps_max], cbar=False, cmap=cmap)
            if cladding==True:
                if indl > 0 and indl < N_layers-1:
                    ax[indl].set_title("xy in layer %d" % indl)
                elif indl==N_layers-1:
                    ax[0].set_title("xy in lower cladding")
                    ax[-1].set_title("xy in upper cladding")
            else:
                ax[indl].set_title("xy in layer %d" % indl)
        plt.show()