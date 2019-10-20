# legume

legume (le GUided Mode Expansion) is a python implementation of the GME method for photonic crystal slabs, including support for multi-layer structures. Plane-wave expansion for purely 2D structures is also included. 

## Install

Eventually this should be `pip` installable from PyPi, but for now hopefully this will work for you:

```
git clone https://github.com/fancompute/legume.git
pip install -e legume
pip install -r legume/requirements.txt
```

Alternatively just `git clone` it, make sure you have all the requirements installed, and add the path to the folder in your python path: `export PYTHONPATH=$PYTHONPATH:/path/to/the/location/of/legume`.

## Autograd

There is also an `autograd` backend (work in progress) that can be used to automatically compute the gradient of the eigenmodes and eigenfrequencies with respect to any input parameters! Currently, to simplify matters, we are excluding the computation of the guided modes themselves from the backprop (they are treated as static). This is an approximation that works to a varying level of accuracy when compared to numerical gradients, depending on what you're looking at:

- Gradients with respect to hole positions should be **exact**.
- Gradients with respect to hole permittivity or size should be approximate, but within a few percent of the numerical gradient.
- Gradients with respect to layer thickness currently work for the **imaginary** parts of the frequencies **only**, and are again approximate (possibly worse than few percent error, but still good enough for optimization it seems). For the real part, you'll get a zero gradient. 

More technically:

- We currently store the effective permittivities `eps_array` and slab thicknesses `d_array` in the GME object, as well as versions `eps_array_val` and `d_array_val` that are detached from the gradient computation (converted from Autograd ArrayBox-es to plain Numpy Arrays). 
- `eps_array_val` and `d_array_val` are used to compute the guided modes, because that part is currently not Autograd compatible.
- Currently, `eps_array_val` and `d_array_val` are also used in computing the matrix elements for the real frequencies, because using `eps_array` and `d_array` gives worse results. Thus, the permittivity of the shapes enters the gradient only through the `eps_inv_mat` matrix (FT of 1/epsilon(r)), while the layer thickness does not enter. 
- For the imaginary parts, we use `eps_array` and `d_array` to compute the radiative modes, and `eps_array_val` and `d_array` in the matrix elements, so the thickness is (approximately) included. 

If the guided mode computation is made Autograd compatible, using `eps_array` and `d_array` everywhere should give exact gradients (as is already the case for changing hole positions). 

## To do
(I've put some non-urgent things as Issues. Below are the things that have to be taken care of to have the minimum needed for what can be considered a functioning package)

- Test multi-layer GME
  - vs. some paper? 
  - vs. COMSOL?
  - vs. S4?

For the optimization part:
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
