"""
Backend for printing options. Available backends:
 - rich [default if rich is installed]
 - base
A backend can be set with the 'set_print_backend'
    import legume
    legume.set_print_backend("base")

"""
from .print_utils import *

# Import autograd if available
import sys
import numpy as np
# Import rich if available

try:
    from rich.console import Console
    from rich.table import Table
    from rich import box
    from rich import print
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False


class PrintBackend(object):
    """
    Backend Base Class 
    """

    def __repr__(self):
        return self.__class__.__name__


if RICH_AVAILABLE:

    class RichPrintBackend(PrintBackend):
        """ rich Backend """
        # methods
        GME_report = staticmethod(print_GME_report_rich)
        GME_im_report = staticmethod(print_GME_im_report_rich)
        ESE_report = staticmethod(print_ESE_report_rich)
        HP_report = staticmethod(print_HP_report_rich)
        update_prog = staticmethod(update_prog)


class BasePrintBackend(PrintBackend):
    """ Base print Backend """
    # methods
    GME_report = staticmethod(print_GME_report)
    GME_im_report = staticmethod(print_GME_im_report)
    ESE_report = staticmethod(print_ESE_report)
    HP_report = staticmethod(print_HP_report)
    update_prog = staticmethod(update_prog)


if RICH_AVAILABLE:
    print_backend = RichPrintBackend()
else:
    print_backend = BasePrintBackend()


def set_print_backend(name):
    """
    Set the print backend for legume.
    This function monkey-patches the backend object by changing its class.
    This way, all methods of the backend object will be replaced.
    
    Parameters
    ----------
    name : {'rich', 'base'}
        Name of the backend. rich must be installed to use 'rich'.
    """
    # perform checks
    if name == 'rich' and not RICH_AVAILABLE:
        raise ValueError("rich backend is not available, rich must \
            be installed.")

    # change backend by monkeypatching
    if name == 'rich':
        print_backend.__class__ = RichPrintBackend
    elif name == 'base':
        print_backend.__class__ = BasePrintBackend
    else:
        raise ValueError(f"unknown backend '{name}'")
