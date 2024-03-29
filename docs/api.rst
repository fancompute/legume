.. currentmodule:: legume

*************
API Reference
*************

This page provides an auto-generated summary of legume's API.

legume
======

   .. autosummary::
      :toctree: generated/

      set_backend


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
   GuidedModeExp.kpoints
   GuidedModeExp.gvec
   GuidedModeExp.rad_coup
   GuidedModeExp.rad_gvec


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

   PlaneWaveExp.freqs
   PlaneWaveExp.eigvecs
   PlaneWaveExp.kpoints
   PlaneWaveExp.gvec

Methods
-------

.. autosummary::
   :toctree: generated/

   PlaneWaveExp.run
   PlaneWaveExp.get_eps_xy
   PlaneWaveExp.ft_field_xy
   PlaneWaveExp.get_field_xy


Photonic crystal
================

.. autosummary::
   :toctree: generated/

   Lattice
   Lattice.bz_path
   PhotCryst
   Layer
   ShapesLayer


Geometry
========

.. autosummary::
   :toctree: generated/

   Circle
   Poly
   Square
   Hexagon
   FourierShape


Visualization
=============

.. autosummary::
   :toctree: generated/

   viz.bands
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
