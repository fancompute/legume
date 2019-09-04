# pyGME

Python implementation of the Guided-Mode Expansion method for a multi-layer structure.

To do:
(I've put some non-urgent things as Issues. Below are the things that have to be taken care of to have the minimum needed for what can be considered a functioning package)

- Test single-layer, multi-mode GME with TE/TM coupling (asymmetric structure)
- Test multi-layer GME
  - vs. some paper? 
  - vs. COMSOL?
- Write the Q-factor computation
- Put various options in `gme.run` into an `options` dictionary
- Figure out how to store and visualize fields. Storing all the fields for all k-points is usually too much memory. **Note: this can be worked on for PWE already and then just ported to GME!**
  - Request which modes to store in `options`
  - Alternatively, request specific fields after the simulation has been run, by re-running the corresponding k-points (but the guided modes and the permittivity matrix are not re-computed)
  - Add some methods for field visualization including permittivity shapes overlay
- Write some nice tests and examples

For the optimization part:
- Make sure GME works with the autograd backend
- Implement constraints on shapes not crossing
- Implement topology optimization: pixels -> numerical FFT -> GME 

