Installation
============

Dependencies
------------

- `numpy <http://www.numpy.org/>`__ (1.16 or later)
- `scipy <http://www.scipy.org/>`__ (1.16 or later)
- `matplotlib <http://www.matplotlib.org/>`__ (3.0.3 or later)
- `autograd <https://github.com/HIPS/autograd>`__ (1.2.1 or later)

Instructions
------------

From PyPi::

    pip install legume-gme

Alternatively just git clone Legume, make sure you have all the requirements installed, and add the path to the folder in your python path::

    export PYTHONPATH=$PYTHONPATH:/path/to/the/location/of/legume
    git clone https://github.com/fancompute/legume.git
    pip install -e legume
    pip install -r legume/requirements.txt
