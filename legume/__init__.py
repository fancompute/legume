'''
Photonic crystal PWE/GME simulator with autograd support
Monkey-patching numpy/autograd backend inspired by Floris Laporte's FDTD
package at github.com/flaport/fdtd
'''
from .phc import Shape, Circle, Ellipse, Poly, Square, Hexagon, FourierShape
from .phc import PhotCryst, Layer, ShapesLayer, FreeformLayer, Lattice

from . import gds
from . import viz
from . import constants

from .pol import HopfieldPol
from .exc import ExcitonSchroedEq
from .pwe import PlaneWaveExp
from .gme import GuidedModeExp
from .gme.slab_modes import guided_modes, rad_modes
from .backend import backend, set_backend
from .print_backend import print_backend, set_print_backend

__all__ = [
    'GuidedModeExp', 'PlaneWaveExp', 'ExcitonSchroedEq', 'HopfieldPol',
    'PhotCryst', 'ShapesLayer', 'FreeformLayer', 'Lattice', 'Shape', 'Circle',
    'Poly', 'Square', 'Hexagon', 'Ellipse'
]

__version__ = '1.0.1'
