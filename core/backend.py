'''
Backend for the simulations. Available backends:
 - numpy [default]
 - autograd
A backend can be set with the 'set_backend'
	import core
	core.set_backend("autograd")

Numpy is still used with some functionalities; if autograd backend is set, 
it is used when needed only so as not to slow down everything
'''

# Numpy must be present
import numpy as np
import scipy as sp

# Import some specially written functions
from .utils import toeplitz_block
from .primitives import toeplitz_block_ag

# Import autograd if available
try:
	import autograd.numpy as npa
	import autograd.scipy as spa
	AG_AVAILABLE = True
except ImportError:
	AG_AVAILABLE = False

class Backend(object):
	'''
	Backend Base Class 
	'''

	# types
	int = np.int64
	float = np.float64
	complex = np.complex128

	def __repr__(self):
		return self.__class__.__name__

class NumpyBackend(Backend):
	""" Numpy Backend """

	# methods
	sum = staticmethod(np.sum)
	hstack = staticmethod(np.hstack)
	norm = staticmethod(np.linalg.norm)
	dot = staticmethod(np.dot)
	cross = staticmethod(np.cross)
	real = staticmethod(np.real)
	inv = staticmethod(np.linalg.inv)
	transpose = staticmethod(np.transpose)
	toeplitz_block = staticmethod(toeplitz_block)
	eigh = staticmethod(np.linalg.eigh)
	outer = staticmethod(np.outer)

	# math functions
	exp = staticmethod(np.exp)
	bessel1 = staticmethod(sp.special.j1)
	sqrt = staticmethod(np.sqrt)
	abs = staticmethod(np.abs)
	square = staticmethod(np.square)

	def is_array(self, arr):
		""" check if an object is an array """
		return isinstance(arr, np.ndarray)

	# constructors
	array = staticmethod(np.array)
	ones = staticmethod(np.ones)
	zeros = staticmethod(np.zeros)
	eye = staticmethod(np.eye)
	linspace = staticmethod(np.linspace)
	arange = staticmethod(np.arange)
	newaxis = staticmethod(np.newaxis)

class AutogradBackend(Backend):
	""" Autograd Backend """

	# methods
	sum = staticmethod(npa.sum)
	hstack = staticmethod(npa.hstack)
	cross = staticmethod(npa.cross)
	norm = staticmethod(npa.linalg.norm)
	dot = staticmethod(npa.dot)
	real = staticmethod(npa.real)
	inv = staticmethod(npa.linalg.inv)
	transpose = staticmethod(npa.transpose)
	toeplitz_block = staticmethod(toeplitz_block_ag)
	eigh = staticmethod(npa.linalg.eigh)
	outer = staticmethod(npa.outer)

	# math functions
	exp = staticmethod(npa.exp)
	bessel1 = staticmethod(spa.special.j1)
	sqrt = staticmethod(npa.sqrt)
	abs = staticmethod(npa.abs)
	square = staticmethod(npa.square)

	# constructors
	array = staticmethod(npa.array)
	ones = staticmethod(npa.ones)
	zeros = staticmethod(npa.zeros)
	eye = staticmethod(npa.eye)
	linspace = staticmethod(npa.linspace)
	arange = staticmethod(npa.arange)
	newaxis = staticmethod(npa.newaxis)

backend = NumpyBackend()

def set_backend(name: str):
	'''
	Set the backend for the simulations
	This function monkeypatches the backend object by changing its class.
	This way, all methods of the backend object will be replaced.
	Args:
		name: name of the backend. Allowed backend names:
			- 'numpy'
			- 'autograd' 
	'''
	# perform checks
	if name == 'autograd' and not AG_AVAILABLE:
		raise ValueError("Autograd backend is not available, autograd must \
			be installed.")

	# change backend by monkeypatching
	if name == 'numpy':
		backend.__class__ = NumpyBackend
	elif name == 'autograd':
		backend.__class__ = AutogradBackend
	else:
		raise ValueError(f"unknown backend '{name}'")