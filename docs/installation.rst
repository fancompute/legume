Installation
============

Required dependencies
---------------------

- Python (>= 3.6)
- `numpy <http://www.numpy.org/>`__ (>= 1.16,<2.0.0)
- `scipy <http://www.scipy.org/>`__ (>= 1.2.1)
- `matplotlib <http://www.matplotlib.org/>`__ (>= 3.0.3)

Optional dependencies
---------------------

- `autograd <https://github.com/HIPS/autograd>`__ (>= 1.2): For computing gradients
- `gdspy <https://gdspy.readthedocs.io/>`__ (>= 1.5): For GDS structure export
- `scikit-image <https://scikit-image.org/>`__ (>= 0.15): For GDS structure export via rasterization
- `rich <https://rich.readthedocs.io/en/latest/introduction.html>`__ (>= 12.5): For displaying more readable verbose output

Instructions
------------

To install the latest version of Legume from PyPi::

    pip install legume-gme

Alternatively, you can ``git clone`` Legume from the Fan group GitHub, manually install all of the required dependencies, and add the path to the Legume in your python path environment variable::

    git clone https://github.com/fancompute/legume.git
    export PYTHONPATH=$PYTHONPATH:/path/to/the/location/of/legume