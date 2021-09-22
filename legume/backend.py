"""
Backend for the simulations. Available backends:
 - numpy [default]
 - autograd
A backend can be set with the 'set_backend'
    import legume
    legume.set_backend("autograd")

Numpy is still used with some functionalities; if autograd backend is set, 
it is used when needed only so as not to slow down everything
"""

# Numpy must be present
import numpy as np
import scipy as sp

# Import some specially written functions
from .utils import toeplitz_block, fsolve, extend

# Import autograd if available
try:
    import autograd.numpy as npa
    import autograd.scipy as spa
    from .primitives import (toeplitz_block_ag, eigh_ag, interp_ag, fsolve_ag,
                             eigsh_ag, inv_ag, sqrt_ag, extend_ag)
    AG_AVAILABLE = True
except ImportError:
    AG_AVAILABLE = False


class Backend(object):
    """
    Backend Base Class 
    """
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
    reshape = staticmethod(np.reshape)
    toeplitz_block = staticmethod(toeplitz_block)
    roll = staticmethod(np.roll)
    where = staticmethod(np.where)
    argwhere = staticmethod(np.argwhere)
    triu = staticmethod(np.triu)
    amax = staticmethod(np.amax)
    max = staticmethod(np.max)
    min = staticmethod(np.min)
    sort = staticmethod(np.sort)
    argsort = staticmethod(np.argsort)
    interp = staticmethod(np.interp)
    fsolve_D22 = staticmethod(fsolve)
    extend = staticmethod(extend)

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
    eigsh = staticmethod(sp.sparse.linalg.eigsh)
    outer = staticmethod(np.outer)
    conj = staticmethod(np.conj)
    var = staticmethod(np.var)
    power = staticmethod(np.power)

    def is_array(self, arr):
        """ check if an object is an array """
        return isinstance(arr, np.ndarray)

    # constructors
    diag = staticmethod(np.diag)
    array = staticmethod(np.array)
    ones = staticmethod(np.ones)
    zeros = staticmethod(np.zeros)
    eye = staticmethod(np.eye)
    linspace = staticmethod(np.linspace)
    arange = staticmethod(np.arange)
    newaxis = staticmethod(np.newaxis)


if AG_AVAILABLE:

    class AutogradBackend(Backend):
        """ Autograd Backend """
        # methods
        sum = staticmethod(npa.sum)
        stack = staticmethod(npa.stack)
        hstack = staticmethod(npa.hstack)
        vstack = staticmethod(npa.vstack)
        transpose = staticmethod(npa.transpose)
        reshape = staticmethod(npa.reshape)
        toeplitz_block = staticmethod(toeplitz_block_ag)
        roll = staticmethod(npa.roll)
        where = staticmethod(npa.where)
        argwhere = staticmethod(npa.argwhere)
        triu = staticmethod(npa.triu)
        amax = staticmethod(npa.amax)
        max = staticmethod(npa.max)
        min = staticmethod(npa.min)
        sort = staticmethod(npa.sort)
        argsort = staticmethod(npa.argsort)
        interp = staticmethod(interp_ag)
        fsolve_D22 = staticmethod(fsolve_ag)
        extend = staticmethod(extend_ag)

        # math functions
        exp = staticmethod(npa.exp)
        bessel1 = staticmethod(spa.special.j1)
        sqrt = staticmethod(sqrt_ag)
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
        inv = staticmethod(inv_ag)
        eigh = staticmethod(eigh_ag)
        eigsh = staticmethod(eigsh_ag)
        outer = staticmethod(npa.outer)
        conj = staticmethod(npa.conj)
        var = staticmethod(npa.var)
        power = staticmethod(npa.power)

        # constructors
        diag = staticmethod(npa.diag)
        array = staticmethod(npa.array)
        ones = staticmethod(npa.ones)
        zeros = staticmethod(npa.zeros)
        eye = staticmethod(npa.eye)
        linspace = staticmethod(npa.linspace)
        arange = staticmethod(npa.arange)
        newaxis = staticmethod(npa.newaxis)


backend = NumpyBackend()


def set_backend(name):
    """
    Set the backend for the simulations.
    This function monkey-patches the backend object by changing its class.
    This way, all methods of the backend object will be replaced.
    
    Parameters
    ----------
    name : {'numpy', 'autograd'}
        Name of the backend. HIPS/autograd must be installed to use 'autograd'.
    """
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
