# legume

<img src="/docs/_static/legume-logo.png" title="logo" alt="logo">

legume (le GUided Mode Expansion) is a python implementation of the GME method for photonic crystal slabs, including multi-layer structures. Plane-wave expansion for purely 2D structures is also included. Also, we have an `autograd` backend that allows gradients of all output values with respect to all input parameters to be computed efficiently!

## Install

Easiest way:

```
pip install legume-gme
```

Alternatively, just `git clone` this repository, and make sure you have all the requirements installed.

## Documentation and examples

Go to our documentation to find a number of examples, as well as a detailed API reference.

The examples can also be found in ipython notebook form in `/docs/examples`.

## Autograd

<img src="/img/cavity_opt.gif" title="cavity_opt" alt="Optimizing the quality factor of a photonic crystal cavity">

One exciting feature of `legume` is the `autograd` backend that can be used to automatically compute the gradient of the eigenmodes and eigenfrequencies with respect to any input parameters! In the optimization shown above, we tune the positions of the holes of a cavity in order to increase the quality factor. As is common in photonic crystal resonators, small modifications lead to tremendous improvement. The gradient of the quality factor with respect to the positions of **all** holes is computed in parallel using reverse-modeautomatic differentiation. 

## Citing

If you find legume useful for your research, we would apprecite you citing our paper. For your convenience, you can use the following BibTex entry:

```
@article{legume,
  title = {legume},
  author = { ... },
  year = {2020},
  month = feb,
  volume = { ... },
  pages = { ... },
  doi = { ... },
  journal = { ... },
  number = { ... }
}
```

## Acknowledgements

Apart from all the contributors to this repository, all the authors of the paper cited above contributed in various ways with the development of this package. Our logo was made by [Nadine Gilmer](https://nadinegilmer.com/). The backend switching between `numpy` and `autograd` follows the implementation in the [fdfd](https://github.com/flaport/fdtd) package of Floris Laporte.

