import matplotlib.pyplot as plt
import numpy as np

import legume

y_max = 1.0
gmax = 7

W = 0.5
H = 1 / 3
D = 0.1

epsr_bot = 1.07
epsr_top = 11.25

lattice = legume.Lattice([1, 0], [0, 3])
# lattice = legume.Lattice('square')
layer = legume.Layer(lattice)

grating_bot = legume.Poly(eps=epsr_bot, x_edges=[-0.5, -0.5, +0.5, +0.5], y_edges=[0, -D, -D, 0])
grating_top = legume.Poly(eps=epsr_top, x_edges=[-W / 2, -W / 2, +W / 2, +W / 2], y_edges=[H, 0, 0, H])

layer.add_shape(grating_top, grating_bot)

pwe = legume.PlaneWaveExp(layer, gmax=gmax)

# path = layer.lattice.bz_path(['G', 'X'], [20])
path = layer.lattice.bz_path([np.array([0.0, 0.0]), np.array([np.pi, 0])], [9])

pwe.run(kpoints=path.kpoints, pol='tm')
freqs_tm = pwe.freqs

pwe.run(kpoints=path.kpoints, pol='te')
freqs_te = pwe.freqs

K = path.kpoints[0] / np.pi

fig, axs = plt.subplots(1, 2, figsize=(3, 4))
ax = axs[1]
ax.plot(K, freqs_tm, "o-", c="#1f77b4")
ax.plot(K, freqs_te, "o-", c="#d62728")
ax.fill_between(K, K, y_max, facecolor="#cccccc", zorder=4, alpha=0.5)
ax.set_ylim([0, y_max])
ax.set_xlim([0, 1.0])
ax.set_xlabel('Wave vector')
ax.set_ylabel('Frequency')

pwe.plot_overview_ft(ax=axs[0])

plt.show()
