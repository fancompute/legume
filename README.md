# legume

legume (le GUided Mode Expansion) is a python implementation of the GME method for photonic crystal slabs, including multi-layer structures. Plane-wave expansion for purely 2D structures is also included. Also, we have an `autograd` backend that allows gradients of all output values with respect to all input parameters to be computed efficiently!

## Install

Easiest way:

```
pip install legume-gme
```

Alternatively, just `git clone` this repository, and make sure you have all the requirements installed.

## Autograd

<img src="/img/cavity_opt.gif" title="cavity_opt" alt="Optimizing the quality factor of a photonic crystal cavity">

One exciting feature is the `autograd` backend that can be used to automatically compute the gradient of the eigenmodes and eigenfrequencies with respect to any input parameters! When running GME, you can specify `'gradients' = 'exact'` (default), or `'gradients' = 'approx'` (faster). The latter discards the gradient due to the guided mode basis itself and only keeps the gradients from the diagonalization. Here are some rules of thumb on what to use:

- If you're optimizing hole positions (i.e. parameters that don't change the average permittivity), you're in luck! In this case, the `approx` gradients should actually be **exact**! 
- If you're optimizing dispersion (real part of eigenfrequencies) w.r.t. parameters that do **not** include the layer thicknesses, you could still try using `approx` gradients, as they might be within just a few percent of the exact ones. 
- If you're optimizing loss rates (imaginary part of eigenfrequencies) and/or if your parameters include the layer thicknesses, then the `approx` gradients could be significantly off, `exact` is recommended. 

## Guided-mode computation options
`GuidedModeExp.run()` takes a number of optional arguments. Below is a list of the most important ones, with a short explanation.

- `gmode_compute: {'exact'}, 'interp'` Define whether the guided modes are computed in the beginning and then interpolated, or whether they are computed exactly at every k-step.
- `gmode_inds: {[0]}, list/numpy array` List of indexes of the guided bands to be included in the expansion. These alternate between TE and TM defined with respect to the propagation direction.
- `numeig: {10}, int` Number of eigen-frequencies to be stored.
- `eig_sigma: {0.}, float` Target frequency, the closest `numeig` eigen-frequencies are stored.
- `compute_im: {True}, bool` Specifies if the imaginary parts of the frequencies should also be computed.
- `gradients: {'exact'}, 'approx'` Specifies whether to compute exact gradients, or approximate (faster)
- `eps_eff: {'average'}, 'background'` Using the 'average' or the 'background' permittivity of the layers in the guided mode computation.
- `verbose: {True}, bool` Print information at intermmediate steps.

## Convergence

If you think the GME run might not be fully converged, I suggest changing the parameters in the following order:

- First, make sure you have a high enough `gmax`, which is defined upon initialization of `GuidedModeExp`.
- Then, include higher-order modes by increasing the number of indexes included in `options['gmode_inds']`.
  - Note that after including more modes in `gmode_inds`, you should test again the convergence w.r.t. `gmax`.

Note that GME is, in the end, only an approximate method, so even if it is converged with respect to the above parameters but produces strange results, it might be that it's not that well-suited for the structure you are simulating.
