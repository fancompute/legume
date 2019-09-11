import matplotlib.pyplot as plt
import numpy as np

import pygme

y_max = 1.0
gmax = 15

W = 0.5
H = 1 / 3
D = 0.1

epsr_bot = 1.07
epsr_top = 11.25

lattice = pygme.Lattice([1, 0], [0, 1])
phc = pygme.PhotCryst(lattice)

# Substrate
phc.add_layer(d=D, eps_b=epsr_bot)

# Grating
phc.add_layer(d=H, eps_b=1)

grating = pygme.Poly(eps=epsr_top, x_edges=[-W / 2, -W / 2, +W / 2, +W / 2], y_edges=[0.5, -0.5, -0.5, 0.5])
phc.layers[-1].add_shape(grating)

gme = pygme.GuidedModeExp(phc, gmax=7)
gme.plot_overview_ft(Ny=500)

path = phc.lattice.bz_path(['G', np.array([np.pi, 0])], [50])

gme.run(kpoints=path.kpoints, gmode_inds=[0], N_g_array=500, verbose=False)

fig, ax = plt.subplots(1, constrained_layout=True)
plt.plot(gme.freqs, 'o-')
ax.set_xlim([0, gme.freqs.shape[0] - 1])
plt.show()
