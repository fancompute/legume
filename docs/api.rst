.. currentmodule:: legume

*************
API Reference
*************

This page provides an auto-generated summary of legume's API.


GuidedModeExp
=============

Creating a simulation
---------------------

.. autosummary::
   :toctree: generated/

   GuidedModeExp

Attributes
----------

.. autosummary::
   :toctree: generated/

   GuidedModeExp.freqs
   GuidedModeExp.freqs_im
   GuidedModeExp.eigvecs
   GuidedModeExp.coup_l
   GuidedModeExp.coup_u
   GuidedModeExp.kpoints
   GuidedModeExp.gvec


Methods
-------

.. autosummary::
   :toctree: generated/

   GuidedModeExp.run
   GuidedModeExp.run_im
   GuidedModeExp.compute_rad
   GuidedModeExp.get_eps_xy
   GuidedModeExp.ft_field_xy
   GuidedModeExp.get_field_xy
   GuidedModeExp.get_field_xz
   GuidedModeExp.get_field_yz
   GuidedModeExp.set_run_options
   

PlaneWaveExp
============

Creating a simulation
---------------------

.. autosummary::
   :toctree: generated/

   PlaneWaveExp

Attributes
----------

.. autosummary::
   :toctree: generated/

   PlaneWaveExp.eps_eff
   PlaneWaveExp.gmax

Methods
-------

.. autosummary::
   :toctree: generated/

   PlaneWaveExp.compute_ft
   PlaneWaveExp.get_eps_xy
   PlaneWaveExp.run
   PlaneWaveExp.compute_eps_inv


Photonic crystal
================

.. autosummary::
   :toctree: generated/

   Lattice
   PhotCryst
   ShapesLayer
   FreeformLayer


Geometry
========

.. autosummary::
   :toctree: generated/

   Circle
   Poly
   Square
   Hexagon


Visualization
=============

.. autosummary::
   :toctree: generated/

   viz.bands
   viz.plot_eps
   viz.structure
   viz.shapes
   viz.eps_xz
   viz.eps_xy
   viz.eps_yz
   viz.eps_ft
   viz.reciprocal
   viz.field

GDS
===

.. autosummary::
   :toctree: generated/

   gds.generate_gds
   gds.generate_gds_raster
