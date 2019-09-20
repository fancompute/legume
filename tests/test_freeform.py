
import pygme
import numpy as np
import matplotlib.pyplot as plt

#
def generate_grid(lattice, Npts):
	xn = np.arange(-0.5, +0.5, 1/Npts)
	yn = np.arange(-0.5, +0.5, 1/Npts)
	(xn_mg, yn_mg) = np.meshgrid(xn, yn)
	xn_v = xn_mg.reshape(-1)
	yn_v = yn_mg.reshape(-1)
	xyn = np.stack((xn_v, yn_v))

	Tl = np.hstack((lattice.a1[:, np.newaxis], lattice.a2[:, np.newaxis]))
	return Tl @ xyn

# Variables
radius = 0.4
lattice = pygme.Lattice([1, 0], [0, 1])
# lattice = pygme.Lattice([0.5, 0.8660254], [ 0.5, -0.8660254])

gmaxs = [5, 10, 15, 20, 25, 30]
gmaxs = [10]
err = []

for i, gmax in enumerate(gmaxs):
	phc = pygme.PhotCryst(lattice)
	phc.add_layer(d=0.5, eps_b=0)
	circ = pygme.Circle(eps=1, x_cent=0, y_cent=0, r=radius)
	phc.layers[-1].add_shape(circ)
	gme = pygme.GuidedModeExp(phc, gmax=gmax)

	gvec = gme.gvec
	Npts = int(np.sqrt(gvec.shape[1]))

	### Shapes
	ft_shape = phc.layers[0].compute_ft(gvec)
	ft_shape_2D = np.fft.ifftshift(ft_shape.reshape((Npts,Npts),order="F"))

	### Freeform

	xy = generate_grid(lattice, Npts)
	r = np.sqrt(np.square(xy[0,:]) + np.square(xy[1,:]))
	epsr = np.zeros(xy.shape[1])
	epsr[r<=radius] = 1.0

	ft = np.fft.fft2(epsr.reshape(Npts,Npts)) / Npts**2
	assert np.all(np.abs(np.imag(ft)) < 1e-6)

	fig, ax = plt.subplots(1, 2, constrained_layout=True, figsize=(6, 3))
	h = ax[0].imshow(np.abs(ft_shape_2D))
	plt.colorbar(h, ax=ax[0], shrink=0.5)
	h = ax[1].imshow(abs(ft))
	plt.colorbar(h, ax=ax[1], shrink=0.5)
	plt.show()

	err.append(np.linalg.norm(ft_shape_2D - ft))

plt.figure(constrained_layout=True)
plt.plot(gmaxs, err, 'o-')
plt.xlabel('$g_{max}$')
plt.ylabel('|| fft - analytic ||$_2$')
plt.show()


(xgrid, ygrid) = lattice.xy_grid()
ft_gme_fmt = np.fft.ifftshift(ft).reshape(-1,order="F")
out  = pygme.utils.ftinv(ft.reshape(-1,order="F"), gvec, xgrid, ygrid)

plt.figure()
plt.imshow(np.real(out))
plt.show()