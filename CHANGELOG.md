# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).



## [1.0.0] - 2024-06-04

### Added
#### GuidedModeExp
- `kz_symmetry` argument in `GuidedModeExp.set_run_options` to separate even and odd modes with respect the kz-plane, where k
    is the in-plane wavevector.
- `symm_thr` argument in `GuidedModeExp.set_run_options` to check that the Hamiltonian is effectively seprated into even and 
    odd blocks. 
- `use_sparse` argument in `GuidedModeExp.set_run_options` to use sparse matrices instead of dense when changing the basis
  to separate the Hamiltonian into even/odd blocks with respect to the kz symmetry plane.  
- `delta_gx` argument in `GuidedModeExp.set_run_options`, a small component added to the x-component of g-vectors to avoid
  problems at `g=k+G=0`
- `delta_gx` argument in `GuidedModeExp.set_run_options`, small shift of the absolute value of `g=k+G` to avoid
  problems at `g=0`.
- `only_gmodes` argument in `GuidedModeExp.set_run_options` to calculate the guided modes of the effective slabs.

- `unbalance_sp` property in `GuidedModeExp` to store unbalance of farfield coupling between s-polarized and
  p-ploarized component. 
- `kz_symms` property to `GuidedModeExp` to store the symmetry of the modes with respect the kz symmetry plane.

- `compute_rad_sp` method to `GuidedModeExp` to calculate losses and `unbalance_sp`.
- `_square_an`, `_hex_an`, `_rec_an` to `GuidedModeExp` to store in-plane angles defining
  possbile kz-planes for square, triangular (hexagonal) and rectangular lattices.
- `_construct_sym_mat` method to  `GuidedModeExp` to calculate the reflection matrix w.r.t.
  a plane parallel to z-axis and rotate by in-plane angle theta.
- `_ind_g` method to  `GuidedModeExp` to calculate the index of a vector in an array.
- `_separate_hamiltonian_dense` and `_separate_hamiltonian_sparse` methods to `GuidedModeExp` to separate the Hamiltonian into 
  even and odd blocks w.r.t. the kz symmetry plane.
- `_calculate_refl_mat` method to  `GuidedModeExp` to calculate all the reflection matrices neeed by
  `_separate_hamiltonian_dense` and `_separate_hamiltonian_sparse`.


#### ExcitonSchroedEq
- `ExcitonSchroedEq` class for solving 2D Schroedinger equation of excitons in a periodic potential.
#### HopfieldPol
- `HopfieldPol` class to calculate exciton-photon coupling giving rise to polariton eigenstates.

#### Photonic Crystal
-  `QuantumWellLayer` to `layer` module.

- `add_qw` method to `PhotCryst` to add a `QuantumWellLayer` (2D active layer) to the photonic crystal.
- `Ellipse` shape to `shapes` module.

- `angle` key to `lattice.bz_path` to store in-plane angles of wavevectros `(kx,ky)`.
- `k_indexes` key to `lattice.bz_path` to store normalized indexes in units of wavevector (`indexes` are just integer indexes).

#### Autograd
- `round`, `shape`, `concatenate`, `size`, `full`, `eig`, `matmul`, `tan`,
  `eigsh_ag`, `inv_ag`, `sqrt_ag`, `extend_ag`, `eig_ag`, `spdot_ag` functions added to numpy-autograd backend.


#### Visualization
- `eV` and `a` arguments to `viz.bands` to plot photonic bands in [eV] unit.
- `k_units` argument to `viz.bands` to plot x-axis in normalized unit proportional
  to the wavevector.
- `calculate_x` and `_calculate_LL` methods to `viz` module to calculate x-coordinate
  and light line of bands to plot.

#### Miscellaneous
- `constants` module with physical constants.
- `CHANGELOG` file.
- `PrintBackend` class and `print_utils` for printing options. The `PrintBackend` can be set to `base` or `rich` with `set_print_backend`. The backend can be set to `rich` only if the package [rich](https://rich.readthedocs.io/en/stable/index.html)
  is installed. 


- Examples to show new features usage.
- Test files for polariton module and symmetry separation.

### Changed
- When `truncate_abs=abs` we check if `gmax` is equal to one of the `|G|` in the reciprocal lattice.
  If that is the case, we increase `gmax+=1e-4` to avoid rounding problems.
- When `truncate_abs=abs` we calculate only the unique and not-redundant Fourier components of the 
  permittivity profile in each layer. 
- When `truncate_abs=abs` we calculate the Fourier transform of the permittivity only if the layer
  is patterned.
- `z_to_lind` is now defined in `utils`.
