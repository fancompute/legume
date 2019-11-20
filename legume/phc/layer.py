import numpy as np
from legume.backend import backend as bd
import legume.utils as utils
from .shapes import Shape, Circle, Poly, Square

class Layer(object):
    '''
    Class for a single layer in the potentially multi-layer PhC
    '''
    def __init__(self, lattice, z_min=0, z_max=0):
        # Define beginning and end in z-direction
        self.z_min = z_min
        self.z_max = z_max
        self.d = z_max - z_min

        self.lattice = lattice

    def __repr__(self):
        return 'Layer'

    def compute_ft(self, gvec):
        '''
        Compute fourier transform over gvec: [2 x Ng] numpy array
        '''
        raise NotImplementedError("compute_ft() needs to be implemented by"
            "Layer subclasses")

class ShapesLayer(Layer):
    '''
    Layer with permittivity defined by Shape objects
    '''
    def __init__(self, lattice, z_min=0, z_max=0, eps_b=1):
        super().__init__(lattice, z_min, z_max)

        # Define background permittivity
        self.eps_b = eps_b

        # Initialize average permittivity - needed for guided-mode computation
        self.eps_avg = np.array(eps_b)

        # Initialize an empty list of shapes
        self.layer_type = 'shapes'
        self.shapes = []

    def __repr__(self):
        rep = 'ShapesLayer(eps_b = %.2f, d = %.2f' % (self.eps_b, self.d)
        rep += ',' if len(self.shapes) > 0 else ''
        for shape in self.shapes:
            rep += '\n' + repr(shape)
        rep += '\n)' if len(self.shapes) > 0 else ')'
        return rep

    def add_shape(self, *args):
        '''
        Add a shape to the layer
        '''

        for shape in args:
            if isinstance(shape, Shape):
                self.shapes.append(shape)
                self.eps_avg = self.eps_avg + (shape.eps - self.eps_b) * \
                                shape.area/self.lattice.ec_area
            else:
                raise ValueError("Arguments to add_shape must be an instance"
                    "of legume.Shape (e.g legume.Circle or legume.Poly)")

    def compute_ft(self, gvec):
        '''
        Compute fourier transform over gvec: [2 x Ng] numpy array
        '''
        FT = bd.zeros(gvec.shape[1])
        for shape in self.shapes:
            # Note: compute_ft() returns the FT of a function that is one 
            # inside the shape and zero outside
            FT = FT + (shape.eps - self.eps_b)*shape.compute_ft(gvec)

        # Apply some final coefficients
        # Note the hacky way to set the zero element so as to work with
        # 'autograd' backend
        ind0 = bd.abs(gvec[0, :]) + bd.abs(gvec[1, :]) < 1e-10  
        FT = FT / self.lattice.ec_area
        FT = FT*(1-ind0) + self.eps_avg*ind0

        return FT

    def get_eps(self, points):
        '''
        Compute the permittivity of the layer over a 'points' tuple containing
        a meshgrid in x, y defined by arrays of same shape
        '''
        xmesh, ymesh = points
        if ymesh.shape != xmesh.shape:
            raise ValueError(
                    "xmesh and ymesh must have the same shape")

        eps_r = self.eps_b * bd.ones(xmesh.shape)

        # Slightly hacky way to include the periodicity
        a1 = self.lattice.a1
        a2 = self.lattice.a2

        a_p = min([np.linalg.norm(a1), 
                   np.linalg.norm(a2)])
        nmax = np.int_(np.sqrt(np.square(np.max(abs(xmesh))) + 
                        np.square(np.max(abs(ymesh))))/a_p) + 1

        for shape in self.shapes:
            for n1 in range(-nmax, nmax+1):
                for n2 in range(-nmax, nmax+1):
                    in_shape = shape.is_inside(xmesh + 
                        n1*a1[0] + n2*a2[0], ymesh + 
                        n1*a1[1] + n2*a2[1])
                    eps_r[in_shape] = utils.get_value(shape.eps)

        return eps_r

class FreeformLayer(Layer):
    '''
    Layer with permittivity defined by a freeform distribution on a grid
    '''
    def __init__(self, lattice, z_min=0, z_max=0, eps_init=1, res=10):
        super().__init__(lattice, z_min, z_max, eps_b)

        # Initialize average permittivity - needed for guided-mode computation
        self.eps_avg = np.array(eps_init)

        # Initialize an empty list of shapes
        self.layer_type = 'freeform'
        self._init_grid(res)

    def _init_grid(res):
        '''
        Initialize a grid with resolution res, with res[0] pixels along the 
        lattice.a1 direction and res[1] pixels along the lattice.a2 direction
        '''
        res = np.array(res)
        if res.size == 1: 
            res = res * np.ones((2,))

    def compute_ft(self, gvec):
        '''
        Compute fourier transform over gvec: [2 x Ng] numpy array
        '''
        raise NotImplementedError("compute_ft() is not yet imlemented for"
            "the free form layer")