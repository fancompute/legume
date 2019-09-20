'''
Photonic crystal PWE/GME simulator with autograd support
Monkey-patching numpy/autograd backend inspired by Floris Laporte's FDTD
package at github.com/flaport/fdtd
'''

from . import viz
from .phc import PhotCryst, ShapesLayer, FreeformLayer, Lattice
from .pwe import PlaneWaveExp
from .gme import GuidedModeExp
from .shapes import Shape, Circle, Poly, Square, Hexagon
from .backend import backend
from .backend import set_backend
from .slab_modes import guided_modes