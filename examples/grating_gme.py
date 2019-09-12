import matplotlib.pyplot as plt
import numpy as np

import pygme

gmax = 5
y_max = 1e-1 # Generally anything < 1/gmax is fine, too small is bad for visualization

W = 0.5
H = 1 / 3
D = 0.1

epsr_bot = 1.07
epsr_top = 11.25

lattice = pygme.Lattice([1, 0], [0, y_max])
phc = pygme.PhotCryst(lattice)

# Substrate
phc.add_layer(d=D, eps_b=epsr_bot)

# Grating
phc.add_layer(d=H, eps_b=1)

grating = pygme.Poly(eps=epsr_top, x_edges=[-W / 2, -W / 2, +W / 2, +W / 2], y_edges=np.array([0.5, -0.5, -0.5, 0.5])*y_max)
phc.layers[-1].add_shape(grating)

gme = pygme.GuidedModeExp(phc, gmax=gmax)
# gme.plot_overview_ft()
# phc.plot_overview()

path = phc.lattice.bz_path(['G', np.array([np.pi, 0])], [50])
options = {'gmode_inds': [0], 'gmode_npts':500, 'numeig':10, 'verbose':False}
gme.run(kpoints=path.kpoints, options=options)

fig, ax = plt.subplots(1, constrained_layout=True)
plt.plot(path.kpoints[0, :], gme.freqs, 'o')
ax.set_xlim([0, path.kpoints[0, -1]])
plt.show()
