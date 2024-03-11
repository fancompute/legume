Installation
============

Required dependencies
---------------------

- Python (>= 3.6)
- `numpy <http://www.numpy.org/>`__ (>= 1.16)
- `scipy <http://www.scipy.org/>`__ (>= 1.2.1)
- `matplotlib <http://www.matplotlib.org/>`__ (>= 3.0.3)

Optional dependencies
---------------------

- `autograd <https://github.com/HIPS/autograd>`__ (>= 1.2): For computing gradients
- `gdspy <https://gdspy.readthedocs.io/>`__ (>= 1.5): For GDS structure export
- `scikit-image <https://scikit-image.org/>`__ (>= 0.15): For GDS structure export via rasterization

Instructions for local installation
-----------------------------------

To install the Legume version provided for the CPC submission (1.0.1),
enter in the legume folder. There you should be able to locate
the ``setup.py`` file. From this directory::

    pip install -e ./


To run inverse design you have to install autograd with the command::

    pip install autograd



To run all examples you need to install jupyter notebook with the command::

    pip install notebook 

then use jupyter notebook and navigate to the docs/examples folder.
