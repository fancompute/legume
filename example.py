import numpy as np
import legume

import matplotlib.pyplot as plt

# Grating parameters as for Fig. 3 (b)
ymax = 0.1      # ficticious supercell length in y-direction, should be smaller than 1/gmax below
W = 0.45        # width of dielectric rods
H = 1.5         # total height of grating
D = 0.1         # thickness of added parts
Wa = (1-W)/2    # width of added parts
epss = 1.45**2  # permittivity of the rods
epsa = 1.1**2   # permittivity of the added parts

# Initialize the lattice and the PhC
lattice = legume.Lattice([1, 0], [0, ymax])
phc = legume.PhotCryst(lattice)

# First layer
phc.add_layer(d=D, eps_b=epss)
rect_add = legume.Poly(eps=epsa, x_edges=np.array([-0.5, -0.5, -W / 2, -W / 2]),
                     y_edges=np.array([0.5, -0.5, -0.5, 0.5]) * ymax)
rect_air = legume.Poly(eps=1, x_edges=np.array([W / 2, W / 2, 0.5, 0.5]),
                     y_edges=np.array([0.5, -0.5, -0.5, 0.5]) * ymax)
phc.add_shape([rect_add, rect_air])

# Second layer
phc.add_layer(d=H-2*D, eps_b=epss)
rect_air1 = legume.Poly(eps=1, x_edges=np.array([-0.5, -0.5, -W / 2, -W / 2]),
                     y_edges=np.array([0.5, -0.5, -0.5, 0.5]) * ymax)
rect_air2 = legume.Poly(eps=1, x_edges=np.array([W / 2, W / 2, 0.5, 0.5]),
                     y_edges=np.array([0.5, -0.5, -0.5, 0.5]) * ymax)
phc.add_shape([rect_air1, rect_air2])

# Third layer
phc.add_layer(d=D, eps_b=epss)
rect_air = legume.Poly(eps=1, x_edges=np.array([-0.5, -0.5, -W / 2, -W / 2]),
                     y_edges=np.array([0.5, -0.5, -0.5, 0.5]) * ymax)
rect_add = legume.Poly(eps=epsa, x_edges=np.array([W / 2, W / 2, 0.5, 0.5]),
                     y_edges=np.array([0.5, -0.5, -0.5, 0.5]) * ymax)
phc.add_shape([rect_add, rect_air])

# Make a BZ path along the G-X direction
path = phc.lattice.bz_path(['G', np.array([np.pi, 0])], [30])

neig = 7 # number of Bloch bands to store
gmax = 4 # truncation of reciprocal lattice vectors

# Initialize GME
gme = legume.GuidedModeExp(phc, gmax=gmax)

# Set some of the running options
options = {'gmode_inds': [1, 3, 5, 7], # Take only the modes with H in the xy-plane
           'numeig': neig,
           'verbose': False
            }

# Run the simulation
gme.run(kpoints=path['kpoints'], **options)

# Visualize the bands
ax = legume.viz.bands(gme)
plt.show()
