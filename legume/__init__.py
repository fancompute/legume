'''
Photonic crystal PWE/GME simulator with autograd support
Monkey-patching numpy/autograd backend inspired by Floris Laporte's FDTD
package at github.com/flaport/fdtd
'''
from .phc import Shape, Circle, Poly, Square, Hexagon
from .phc import PhotCryst, ShapesLayer, FreeformLayer, Lattice

from . import viz

from .pwe import PlaneWaveExp
from .gme import GuidedModeExp
from .gme.slab_modes import guided_modes, rad_modes
from .backend import backend, set_backend
