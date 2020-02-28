import numpy as np

from legume.backend import backend as bd
import legume.utils as utils
from . import ShapesLayer, FreeformLayer, Lattice

class PhotCryst(object):
    """
    Class for a photonic crystal which can contain a number of layers.
    """
    def __init__(self, lattice, eps_l: float=1, eps_u: float=1):
        """Initialize a photonic crystal.
        
        Parameters
        ----------
        lattice : Lattice
            A lattice defining the 2D periodicity.
        eps_l : float, optional
            Permittivity of the lower cladding.
        eps_u : float, optional
            Permittivity of the upper cladding.
        """
        # Define permittivity of lower and upper cladding
        # For now claddings are just ShapesLayer
        self.claddings = []
        self.claddings.append(ShapesLayer(lattice, eps_b=eps_l, z_min=-1e10))
        self.claddings.append(ShapesLayer(lattice, eps_b=eps_u, z_max=1e10))
        
        # Initialize underlying lattice
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
        """ 
        Define a z-grid for visualization purposes once some layers have been 
        added.
        """
        zmin = self.layers[0].z_min - dist
        zmax = self.layers[-1].z_max + dist
    
        return np.linspace(zmin, zmax, Nz)

    def add_layer(self, d: float, eps_b: float=1, layer_type: str='shapes'):
        """
        Add a layer to the photonic crystal, on top of all currently existing
        layers.
        
        Parameters
        ----------
        d : float
            Layer thickness.
        eps_b : float, optional
            Layer background permittivity.
        layer_type : {'shapes'}, optional
            Currently only the ShapesLayer subclass is implemented.
        """
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

    def add_shape(self, shapes, layer=-1, cladding=None):
        """
        Add a list of shapes to a given layer.
        
        Parameters
        ----------
        shapes : Shape
            A Shape object, or a list of such objects.
        layer : int, optional
            Layer index to which the shape is added. Default is last layer.
        cladding : None, optional
            If ``cladding == 0 or 'l'``, add the shapes to the lower cladding.
            If ``cladding == 1 or 'u'``, add the shapes to the lower cladding.
        """
        if cladding is not None:
            if cladding==0 or cladding=='l':
                lay = self.claddings[0]
            elif cladding==1 or cladding=='u':
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
            lay.add_shape(shapes)
        else:
            raise TypeError("Shapes can only be added to a ShapesLayer")

    def get_eps(self, points):
        """
        Compute the permittivity of the PhC at a set of points.
        
        Parameters
        ----------
        points : tuple of np.ndarray
            The `x`, `y`, and `z` positions of the points, arrays of same shape.
        
        Returns
        -------
        eps_r : np.ndarray
            Array of same shape with the permittivity at each point.
        """
        (xmesh, ymesh, zmesh) = points
        a_shape = xmesh.shape
        if (ymesh.shape != a_shape) or (zmesh.shape != a_shape):
            raise ValueError(
                    "xmesh, ymesh and zmesh must have the same shape")

        eps_r = np.zeros(a_shape)

        a1 = self.lattice.a1
        a2 = self.lattice.a2

        for layer in self.layers + self.claddings:
            zlayer = (zmesh >= layer.z_min) * (zmesh < layer.z_max)
            if np.sum(zlayer) > 0:
                eps_r[zlayer] = layer.get_eps((xmesh[zlayer], ymesh[zlayer]))

        return eps_r

    def get_eps_bounds(self):
        """Returns the minimum and maximum permittivity of the structure.
        """
         
        eps_min = self.claddings[0].eps_b
        eps_max = self.claddings[0].eps_b

        for layer in self.layers + self.claddings:
            eps_min = min([eps_min, layer.eps_b])
            eps_max = max([eps_max, layer.eps_b])
            for shape in layer.shapes:
                eps_min = min([eps_min, shape.eps])
                eps_max = max([eps_max, shape.eps])

        return (eps_min, eps_max)