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

ExcitonSchroedEq
================

Creating a simulation
---------------------

.. autosummary::
   :toctree: generated/

   ExcitonSchroedEq

Attributes
----------

.. autosummary::
   :toctree: generated/

   ExcitonSchroedEq.eners
   ExcitonSchroedEq.eigvecs
   ExcitonSchroedEq.kpoints
   ExcitonSchroedEq.gvec

Methods
-------

.. autosummary::
   :toctree: generated/

   ExcitonSchroedEq.run
   ExcitonSchroedEq.get_pot_xy
   ExcitonSchroedEq.ft_wavef_xy
   ExcitonSchroedEq.get_wavef_xy

HopfieldPol
===========

Creating a simulation
---------------------

.. autosummary::
   :toctree: generated/

   HopfieldPol

Attributes
----------

.. autosummary::
   :toctree: generated/

   HopfieldPol.eners
   HopfieldPol.eners_im
   HopfieldPol.eigvecs
   HopfieldPol.fractions_ex
   HopfieldPol.fractions_ph
   HopfieldPol.kpoints
   HopfieldPol.gvec

Methods
-------

.. autosummary::
   :toctree: generated/

   HopfieldPol.run


Photonic crystal
================

.. autosummary::
   :toctree: generated/

   Lattice
   Lattice.bz_path
   PhotCryst
   PhotCryst.add_layer
   PhotCryst.add_qw
   Layer
   ShapesLayer


Geometry
========

.. autosummary::
   :toctree: generated/

   Circle
   Ellipse
   Poly
   Square
   Hexagon
   FourierShape


Visualization
=============

.. autosummary::
   :toctree: generated/

   viz.bands
   viz.pol_bands
   viz.structure
   viz.shapes
   viz.eps_xz
   viz.eps_xy
   viz.eps_yz
   viz.eps_ft
   viz.pot_ft
   viz.reciprocal
   viz.field
   viz.wavef

GDS
===

.. autosummary::
   :toctree: generated/

   gds.generate_gds
   gds.generate_gds_raster
