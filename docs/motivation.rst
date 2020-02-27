Why legume?
===========

**legume** is a python implementation of the guided-mode expansion (GME) method 
for simulating photonic crystal slabs, i.e. for multi-layer structures that
look something like this

.. image:: _static/phc_schematic.png
  :width: 200
  :alt: Multi-layer photonic crystal

This 
is an extremely efficient method to obtain the photonic Bloch bands of such
structures and can be used to study bulk photonic crystals as well as devices 
like waveguides and cavities.

The guided-mode expansion is particularly useful for computing the 
quasi-guided modes above the light line, which are hard to isolate in 
first-principle finite-difference of finite-element methods. This can be 
invaluable for the study of the coupling of photonic crystal modes to the 
radiative environment, and of exotic phenomena like
bound states in the continuum.

The GME method can be super useful in itself, but on top of that we also 
provide a differentiable implementation through the `autograd` backend. This 
allows you to compute the gradient of an objective function that depends on any 
of the ouput parameters (eigenmode frequencies and losses, fields...) with 
respect to any of the input parameters (hole positions, sizes and shapes, 
slab thickness...) With this powerful addition, there's no end to the 
possibilities in using **legume** to optimize your devices!

Dive into the :ref:`examples` to see how this all works in practice!