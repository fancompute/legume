'''
Photonic crystal PWE/GME simulator with autograd support
Monkey-patching numpy/autograd backend inspired by Floris Laporte's FDTD
package at github.com/flaport/fdtd
'''

from .phc import PhotCryst, Layer, Lattice
from .pwe import PlaneWaveExp
from .gme import GuidedModeExp
from .shapes import Shape, Circle, Poly, Square, Hexagon
from .backend import backend
from .backend import set_backend
from .guided_modes import guided_modes