# legume

legume (le GUided Mode Expansion) is a python implementation of the GME method for photonic crystal slabs, including support for multi-layer structures. Plane-wave expansion for purely 2D structures is also included. 

There is also an `autograd` backend (work in progress) that can be used to automatically compute the gradient of the eigenmodes and eigenfrequencies with respect to any input parameters!

## Install

Eventually this should be `pip` installable from PyPi, but for now hopefully this will work for you:

```
git clone https://github.com/fancompute/legume.git
pip install -e legume
pip install -r legume/requirements.txt
```

Alternatively just download and make sure you have all the requirements installed. 

## To do
(I've put some non-urgent things as Issues. Below are the things that have to be taken care of to have the minimum needed for what can be considered a functioning package)

- Test multi-layer GME
  - vs. some paper? 
  - vs. COMSOL?
  - vs. S4?
- Improve visualization methods in `viz.py` and add some new ones:
    - Q-factor visualization?

For the optimization part:
- Make Q-factor computation compatible with the autograd backend
- Implement constraints on shapes not crossing
- Implement topology optimization: pixels -> numerical FFT -> GME 

## Guided-mode computation options
`GuidedModeExp.run()` takes a number of optional argument. The default settings for these options are defined in `GuidedModeExp._run_options()`. Keys starting with `gmode_` refer to options in the guided mode computation, where we do the following: 

- Define a grid of g-points with `gmode_npts` number of points on it, from `g = 0` to `g = max(norm(G+k))`, where `G` is a reciprocal lattice vector.
- For every g-point, write a function `f(omega)` whose roots define the guided-mode frequencies.
- Define lower and upper `omega` bounds `om_lb` and `om_ub` within which the guided modes should lie. `f(omega)` becomes complex-valued outside so it's good to stick to those bounds.
- Start from `om_lb + gmode_tol` and look for a change of sign in `f(omega)`, with steps in omega defined by `gmode_step`. We add `gmode_tol` because `f(omega)` becomes unstable right at the bounds. We go up to `om_ub - gmode_tol`. 
- When a change of sign is found, use a SciPy root finding method to find the exact frequency.

If you think the GME run might not be fully converged, I suggest changing the parameters in the following order:
(**NB: all of these will increase the computation time in different ways!**)

- First and foremost, make sure you have a high enough `gmax`, which is defined upon initialization of `GuidedModeExp`.
- Then, `options['gmode_inds']` selects the indexes of the guided bands to be included in the calculation. These alternate between TE and TM defined with respect to the propagation direction. (Note: this classification is not valid in a 2D PhC but is still valid in a grating).
- Then, start tweaking the guided modes computation itself:
  - first try increasing `options['gmode_npts']`
  - then try decreasing `options['gmode_step']`
  - finally you could also try decreasing `options[gmode_tol]`
