[![Build Status](https://travis-ci.com/fancompute/legume.svg?token=ZCPktA3Ki2eYVXYnfbrz&branch=master)](https://travis-ci.com/fancompute/legume.svg?branch=master) [![Documentation Status](https://readthedocs.org/projects/legume/badge/?version=latest)](https://legume.readthedocs.io/en/latest/?badge=latest)
[![Code style: yapf pep8](https://img.shields.io/badge/code%20style-yapf-000000.svg)](https://github.com/google/yapf)


<img src="https://github.com/fancompute/legume/blob/master/docs/_static/legume-logo.png" align="middle" title="logo" alt="logo">

legume (le Guided-Mode Expansion) is a python implementation of the GME method for photonic crystal slabs, 
including multi-layer structures. Plane-wave expansion for purely 2D structures is also included.
Also, we have an `autograd` backend that allows gradients of all output values with respect to all input parameters
to be computed efficiently!

## Install

This is an experimental package, the recommended way of installing it would be `git clone` followed by

```
poetry install
```

## Documentation and examples

Go to our [documentation](https://legume.readthedocs.io/en/latest/index.html) to find a number of [examples](https://legume.readthedocs.io/en/latest/examples.html), as well as a detailed [API reference](https://legume.readthedocs.io/en/latest/api.html).

The examples can also be found in ipython notebook form in `/docs/examples`.

Here's an example of a computation of the photonic bands of a photonic crystal, compared to Fig. 2(b) in Chapter 8 of the photonic crystal bible, [Molding the Flow of Light](http://ab-initio.mit.edu/book/).

<img src="https://github.com/fancompute/legume/blob/master/img/phc_bands.png" title="photonic_bands" alt="Quasi-TE bands of a photonic crystal slab">

We have only computed the quasi-TE modes of the slab (positive symmetry w.r.t. the plane bisecting the slab), which should be compared to the red lines in the figure on the right. The agreement is very good! And, the guided-mode expansion allows us to also compute the quasi-guided modes above light-line, together with their associated quality factor. These modes are typically hard to filter out in first-principle simulations, so `legume` is great for studying those. 

## Autograd

<img src="https://github.com/fancompute/legume/blob/master/img/cavity_opt.gif" title="cavity_opt" alt="Optimizing the quality factor of a photonic crystal cavity">

One exciting feature of `legume` is the `autograd` backend that can be used to automatically compute the gradient of the eigenmodes and eigenfrequencies with respect to any input parameters! In the optimization shown above, we tune the positions of the holes of a cavity in order to increase the quality factor. As is common in photonic crystal resonators, small modifications lead to tremendous improvement. The gradient of the quality factor with respect to the positions of **all** holes is computed in parallel using reverse-modeautomatic differentiation. 

## Citing

If you find legume useful for your research, we would apprecite you citing our [paper](https://arxiv.org/abs/2003.00379). For your convenience, you can use the following BibTex entry:

```
@article{minkov2020inverse,
  title={Inverse design of photonic crystals through automatic differentiation},
  author={Minkov, Momchil and Williamson, Ian AD and Andreani, Lucio C and Gerace, Dario and Lou, Beicheng and Song, Alex Y and Hughes, Tyler W and Fan, Shanhui},
  journal={ACS Photonics},
  volume={7},
  number={7},
  pages={1729--1741},
  year={2020},
  publisher={American Chemical Society}
}
```

## Acknowledgements

Apart from all the contributors to this repository, all the authors of the paper cited above contributed in various ways with the development of this package. Our logo was made by [Nadine Gilmer](https://nadinegilmer.com/). The backend switching between `numpy` and `autograd` follows the implementation in the [fdfd](https://github.com/flaport/fdtd) package of Floris Laporte.

