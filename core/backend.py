'''
Backend for the simulations
Three backends are available:
 - numpy [default]
 - autograd
A backend can be set with the 'set_backend'
	import core
	core.set_backend("autograd")
'''

# Numpy must be present
import numpy

# Import autograd if available
try:
	import autograd as ag
	AG_AVAILABLE = True
except ImportError:
	AG_AVAILABLE = False

class Backend(object):
	'''
	Backend Base Class 
	'''

class NumpyBackend(Backend):
	""" Numpy Backend """

	# types
	int = numpy.int64
	float = numpy.float64

	# methods
	exp = staticmethod(numpy.exp)
	sin = staticmethod(numpy.sin)
	cos = staticmethod(numpy.cos)
	sum = staticmethod(numpy.sum)
	stack = staticmethod(numpy.stack)
	transpose = staticmethod(numpy.transpose)
	reshape = staticmethod(numpy.reshape)
	squeeze = staticmethod(numpy.squeeze)
	cross = staticmethod(numpy.cross)

	def is_array(self, arr):
		""" check if an object is an array """
		return isinstance(arr, numpy.ndarray)

	# constructors
	array = staticmethod(numpy.array)
	ones = staticmethod(numpy.ones)
	zeros = staticmethod(numpy.zeros)
	linspace = staticmethod(numpy.linspace)
	arange = staticmethod(numpy.arange)
	numpy = staticmethod(numpy.asarray)

class AutogradBackend(Backend):
	""" Autograd Backend """

	# types
	int = numpy.int64
	float = numpy.float64

	# methods
	exp = staticmethod(numpy.exp)
	sin = staticmethod(numpy.sin)
	cos = staticmethod(numpy.cos)
	sum = staticmethod(ag.numpy.sum)
	stack = staticmethod(numpy.stack)
	transpose = staticmethod(numpy.transpose)
	reshape = staticmethod(numpy.reshape)
	squeeze = staticmethod(numpy.squeeze)
	cross = staticmethod(ag.numpy.cross)
	norm = staticmethod(ag.numpy.linalg.norm)
	dot = staticmethod(ag.numpy.dot)

	def is_array(self, arr):
		""" check if an object is an array """
		return isinstance(arr, numpy.ndarray)

	# constructors
	array = staticmethod(numpy.array)
	ones = staticmethod(numpy.ones)
	zeros = staticmethod(numpy.zeros)
	linspace = staticmethod(numpy.linspace)
	arange = staticmethod(numpy.arange)
	numpy = staticmethod(numpy.asarray)

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