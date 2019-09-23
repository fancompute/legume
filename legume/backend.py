'''
Backend for the simulations. Available backends:
 - numpy [default]
 - autograd
A backend can be set with the 'set_backend'
	import legume
	legume.set_backend("autograd")

Numpy is still used with some functionalities; if autograd backend is set, 
it is used when needed only so as not to slow down everything
'''

# Numpy must be present
import numpy as np
import scipy as sp

# Import some specially written functions
from .utils import toeplitz_block
from .primitives import toeplitz_block_ag, eigh_ag

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
	stack = staticmethod(np.stack)
	hstack = staticmethod(np.hstack)
	vstack = staticmethod(np.vstack)
	transpose = staticmethod(np.transpose)
	toeplitz_block = staticmethod(toeplitz_block)
	roll = staticmethod(np.roll)
	where = staticmethod(np.where)
	triu = staticmethod(np.triu)
	amax = staticmethod(np.amax)
	sort = staticmethod(np.sort)

	# math functions
	exp = staticmethod(np.exp)
	bessel1 = staticmethod(sp.special.j1)
	sqrt = staticmethod(np.sqrt)
	divide = staticmethod(np.divide)
	abs = staticmethod(np.abs)
	square = staticmethod(np.square)
	sin = staticmethod(np.sin)
	cos = staticmethod(np.cos)
	tanh = staticmethod(np.tanh)
	norm = staticmethod(np.linalg.norm)
	dot = staticmethod(np.dot)
	cross = staticmethod(np.cross)
	real = staticmethod(np.real)
	imag = staticmethod(np.imag)
	inv = staticmethod(np.linalg.inv)
	eigh = staticmethod(np.linalg.eigh)
	outer = staticmethod(np.outer)
	conj = staticmethod(np.conj)
	var = staticmethod(np.var)

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
	stack = staticmethod(npa.stack)
	hstack = staticmethod(npa.hstack)
	vstack = staticmethod(npa.vstack)
	transpose = staticmethod(npa.transpose)
	toeplitz_block = staticmethod(toeplitz_block_ag)
	roll = staticmethod(npa.roll)
	where = staticmethod(npa.where)
	triu = staticmethod(npa.triu)
	amax = staticmethod(npa.amax)
	sort = staticmethod(npa.sort)

	# math functions
	exp = staticmethod(npa.exp)
	bessel1 = staticmethod(spa.special.j1)
	sqrt = staticmethod(npa.sqrt)
	divide = staticmethod(npa.divide)
	abs = staticmethod(npa.abs)
	square = staticmethod(npa.square)
	sin = staticmethod(npa.sin)
	cos = staticmethod(npa.cos)
	tanh = staticmethod(npa.tanh)
	cross = staticmethod(npa.cross)
	norm = staticmethod(npa.linalg.norm)
	dot = staticmethod(npa.dot)
	real = staticmethod(npa.real)
	imag = staticmethod(npa.imag)
	inv = staticmethod(npa.linalg.inv)
	eigh = staticmethod(eigh_ag)
	outer = staticmethod(npa.outer)
	conj = staticmethod(npa.conj)
	var = staticmethod(npa.var)

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
