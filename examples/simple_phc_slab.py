import legume

from legume import GuidedModeExp, Circle, PhotCryst, Lattice

Ny = 10
ra = 0.3
lattice = Lattice('hexagonal')

phc = PhotCryst(lattice)
phc.add_layer(d=0.5, eps_b=12.0)

phc.layers[-1].add_shape(Circle(x_cent=0.0, y_cent=0.0, r=ra))
# phc.plot_overview(cladding='True', Ny=500)

gme = GuidedModeExp(phc, gmax=3)
# gme.plot_overview_ft(Ny=500)

path = phc.lattice.bz_path(['G', 'M', 'K', 'G'], [20, 20, 20])

gme.run(kpoints=path.kpoints, gmode_inds=[0], verbose=False, numeig=10)

legume.viz.bands(gme)
