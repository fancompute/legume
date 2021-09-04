import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from chickpea import utils
from legume import viz


class XYField:

    def __init__(self, res, z_dimension, polarization, modulation=np.array([0, 0]), renormalize=False, **kwargs):
        """
        General Parameters:
        :param res: resolution of field profile in xy. int or tuple of ints
        :param z_dimension: z dimension in crystal to generate field
        :param polarization: polarization of field. If 'TE' only calculate xy of e field and z of h field, likewise for 'TM',
                                                    None generates all components of both fields

        Constructor specific parameters:

        :param gme: Guided mode expansion to construct fields from. Alternatively provide field directly.
            :param modulation: K vector modulation applied to generated field. Default: [0,0]
            :param renormalize: If True normalizes field such that maximum occurs at t=0 with vector magnitude 1.
            :param kind: k index for guided mode expansion field
            :param mind: mode index for guided mode expansion field

        :param field: Field directly provided to construct: Should be function time -> np.array([[E_x, E_y,E_z],
                                                                                                 [H_x, H_y, H_z]])
                      with shape of E_i and H_i equal [res, res]
            :param phc: photonic crystal field was generated over.
            :param meshgrid: Meshgrid associated with field where each entry of E and H assigned to meshgrid.
        """
        self._eps_dist=None
        if np.array(res).__len__() == 1:
            self.res = np.array((res, res))

        elif np.array(res).__len__() == 2:
            self.res = np.array(res)
        else:
            raise ValueError("Resolution must be int or 2 entry array of ints.")

        self.polarization = polarization
        self.z_dimension = z_dimension

        if 'field' in kwargs:
            try:
                self.phc = kwargs['phc']
                self.meshgrid = kwargs['meshgrid']
                self._field = kwargs['field']
            except KeyError():
                raise ValueError("When giving a field as a keyword argument "
                                 "additional arguments \"phc\" and \"meshgrid\" are required.")

        elif 'gme' in kwargs:
            try:
                gme = kwargs['gme']
                self.gme = repr(gme)
                self.kind = kwargs['kind']
                self.mind = kwargs['mind']
            except KeyError():
                raise ValueError("When passing a gme object as a variable "
                                 "additional arguments \"kind\" and \"mind\" are required")

            self.phc = gme.phc
            self.lattice = self.phc.lattice

            e_comp, h_comp = {'TE': ['xy', 'z'],
                              'TM': ['z', 'xy'],
                              'None': ['xyz', 'xyz']}[polarization]

            fe, xgrid, ygrid = gme.get_field_xy(field='e', component=e_comp, z=z_dimension, kind=self.kind,
                                                mind=self.mind,
                                                Nx=self.res[0], Ny=self.res[1])

            fh, _, _ = gme.get_field_xy(field='h', component=h_comp, z=z_dimension, kind=self.kind, mind=self.mind,
                                        Nx=self.res[0], Ny=self.res[1])

            if polarization == 'TE':
                E = np.array([fe['x'], fe['y'], np.zeros(self.res[::-1])])
                H = np.array([np.zeros(self.res[::-1]), np.zeros(self.res[::-1]), fh['z']])

            elif polarization == 'TM':
                E = np.array([np.zeros(self.res[::-1]), np.zeros(self.res[::-1]), fe['z']])
                H = np.array([fh['x'], fh['y'], np.zeros(self.res[::-1])])

            elif polarization == 'None':
                E = np.array([fe['x'], fe['y'], fe['z']])
                H = np.array([fh['x'], fh['y'], fh['z']])
            else:
                raise ValueError("Polarization should be 'TE', 'TM' or 'None'")
            self.meshgrid = np.meshgrid(xgrid, ygrid)

            # Apply modulation
            E = E * np.exp(1j * (self.meshgrid[0] * modulation[0] + self.meshgrid[1] * modulation[1]))
            H = H * np.exp(1j * (self.meshgrid[1] * modulation[0] + self.meshgrid[1] * modulation[1]))

            if renormalize:
                field_0 = np.array([np.linalg.norm(E, axis=0), np.linalg.norm(H, axis=0)])

                argmax = np.unravel_index(np.argmax(np.abs(field_0)), field_0.shape)
                E = E / field_0[argmax]
                H = H / field_0[argmax]

            freq = gme.freqs[self.kind][self.mind]
            self._freq = freq
            self._kvec = modulation
            self._field = lambda t: np.array([E * np.exp(1j * freq * t), H * np.exp(1j * freq * t)])

        else:
            raise ValueError("Missing initializing variable \"gme\" or \"field\".")

        for layer in self.phc.layers + self.phc.claddings:
            zlayer = (self.z_dimension >= layer.z_min) * (self.z_dimension < layer.z_max)
            if np.sum(zlayer) > 0:
                self.layer = layer
                break

    @property
    def freq(self):
        if hasattr(self, '_freq'):
            return self._freq
        else:
            raise ValueError("Mode not derived from gme and has no defined frequency.")

    @property
    def kvec(self):
        if hasattr(self, '_kvec'):
            return self._kvec
        else:
            raise ValueError("Mode not constructed from gme and has no defined wavevector.")

    def poynting_vector(self, time=0):
        self._poynting_vector = np.cross(np.real(self._field(time)[0]), np.real(self._field(time)[1]), axis=0)
        return self._poynting_vector

    def coarse_poynting_vector(self, time=0, pv_coarseness=2):
        S = self.poynting_vector(time)
        S_x = S[0]
        S_y = S[1]
        S_z = S[2]

        S_x = utils.lowpass_downsample(S_x, pv_coarseness)
        S_y = utils.lowpass_downsample(S_y, pv_coarseness)
        S_z = utils.lowpass_downsample(S_z, pv_coarseness)
        return np.array([S_x, S_y, S_z])

    def chirality(self, field='e', time=0):
        if field == 'e':
            E = self._field(time)[0]
            self.e_spin_field = 1j * np.cross(E, np.conj(E), axis=0)
            return self.e_spin_field
        elif field == 'h':
            H = self._field(time)[1]
            self.h_spin_field = 1j * np.cross(H, np.conj(H), axis=0)
            return self.h_spin_field
        else:
            raise (ValueError("Field must be \'e\' or \'h\'"))

    def field(self, time=0):
        return self._field(time)

    def eps_dist(self):
        if self._eps_dist is None:
            self._eps_dist = self.phc.get_eps(
                (self.meshgrid[0], self.meshgrid[1], self.z_dimension * np.ones(self.meshgrid[0].shape)))
        return self._eps_dist

    def visualize_field(self, field, component, time=0, profile=True, poynting_vector=False, pv_coarseness=1,
                        eps=True, val='re', normalize=False, fig=None, figsize=None):
        comp_map = {'x': 0, 'y': 1, 'z': 2}
        field_ind = {'e': 0, 'h': 1}[field]
        if fig is None:
            fig = plt.figure(constrained_layout=True, figsize=figsize)

        if normalize:
            norm_field = np.zeros(self.res[::-1])
            for comp in component:
                field_comp = self.field(time)[field_ind][comp_map[comp]]
                norm_field = norm_field + np.abs(field_comp) ** 2
            norm_field = norm_field ** (1 / 2)
            vmax = norm_field.max()
            vmin = 0
            ax = fig.add_subplot()
            cmap = 'magma'
            self._field_ax_builder(ax=ax, field=norm_field, time=time, cmap=cmap, vmax=vmax, vmin=vmin, eps=eps,
                                   poynting_vector=poynting_vector, profile=profile, pv_coarseness=pv_coarseness)

        else:
            i = 1
            for comp in component:
                ax = fig.add_subplot(1, component.__len__(), i)

                field_comp = self.field(time)[field_ind][comp_map[comp]]
                if val == 're':
                    field_comp = np.real(field_comp)
                    vmax = np.abs(field_comp).max()
                    vmin = -vmax
                    cmap = 'RdBu'

                elif val == 'im':
                    field_comp = np.imag(field_comp)
                    vmax = np.abs(field_comp).max()
                    vmin = -vmax
                    cmap = 'RdBu'

                elif val == 'abs':
                    field_comp = np.abs(field_comp)
                    vmax = field_comp.max()
                    vmin = 0
                    cmap = 'magma'

                self._field_ax_builder(ax=ax, field=field_comp, time=time, vmax=vmax, vmin=vmin, cmap=cmap, eps=eps,
                                       poynting_vector=poynting_vector, profile=profile, pv_coarseness=pv_coarseness)
                i += 1
        return fig

    def _field_ax_builder(self, ax, field, time, vmax, vmin, cmap, eps, profile, pv_coarseness, poynting_vector):

        extent = self.meshgrid[0].min(), self.meshgrid[0].max(), self.meshgrid[1].min(), self.meshgrid[1].max()



        if profile:
            im=ax.imshow(field, extent=extent, vmax=vmax, vmin=vmin,
                      cmap=cmap, origin='lower')



        if eps:
            viz.eps_shapes(layer=self.layer, ax=ax, extent=extent, alpha=0.3)


        if profile:
            im=ax.imshow(field, extent=extent, vmax=vmax, vmin=vmin,
                      cmap=cmap, origin='lower', alpha=1)




        if poynting_vector:
            S = self.coarse_poynting_vector(time, pv_coarseness)
            S_0 = S[0]
            S_1 = S[1]

            vector_meshgrid = np.meshgrid(np.linspace(extent[0], extent[1], S_0.shape[1]),
                                          np.linspace(extent[2], extent[3], S_0.shape[0]))
            xgrid, ygrid = vector_meshgrid

            cell_size = min((np.abs((extent[0] - extent[1]) / S_0.shape[1])),
                            np.abs((extent[2] - extent[3]) / S_0.shape[0]))

            magnitude = ((S_0 ** 2) + (S_1 ** 2)) ** (1 / 2)

            colormap = matplotlib.cm.afmhot
            norm = Normalize()
            colors = magnitude
            norm.autoscale(colors)

            # Normalize length to cell size. Arrows should fill their assigned cell independent of pv_coarseness
            unit_s_0 = S_0 / magnitude * cell_size * 0.75
            unit_s_1 = S_1 / magnitude * cell_size * 0.75

            ax.quiver(np.ravel(xgrid), np.ravel(ygrid), np.ravel(unit_s_0), np.ravel(unit_s_1),
                      color=(colormap(norm(np.ravel(magnitude)))), pivot='mid',
                      angles='xy', scale_units='xy', scale=1, width=0.0015*pv_coarseness, # Fudge factor on width to make arrows the same shape independent of pv_coarseness.
                            alpha=0.8)

            # The parameters for the quiver
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])
        return ax

    def vizualize_chirality(self, field, component='z', time=0, fig=None, figsize=None, eps=True):
        extent = self.meshgrid[0].min(), self.meshgrid[0].max(), self.meshgrid[1].min(), self.meshgrid[1].max()
        comp_map = {'x': 0, 'y': 1, 'z': 2}
        chirality = self.chirality(field, time)[comp_map[component]]
        vmax = np.abs(chirality).max()
        vmin = -vmax
        if fig is None:
            fig = plt.figure(constrained_layout=True, figsize=figsize)

        ax = fig.add_subplot()

        ax.imshow(np.real(chirality), extent=extent, vmax=vmax, vmin=vmin,
                  cmap='RdBu', origin='lower')
        if eps:
            viz.eps_shapes(layer=self.layer, ax=ax, extent=extent, alpha=0.15)
        return fig

    def generate_mp4(self, time_range, frames, val='re', eps=True, poynting_vector=True, pv_coarseness=1, **kwargs):
        """
        Generate an animation of a visualization over a time_range
        @param time_range:
        @param frames:
        @param val:
        @param eps:
        @param poynting_vector:
        @param pv_coarseness:
        @param kwargs:
        @return:
        """
        # \\TODO Implement this

        raise NotImplementedError("Not implemented yet")

    def time_avg_pv(self, period, N_samples, fig=None, eps=True, pv_coarseness=1, cbar=True,figsize=None):
        if fig is None:
            fig = plt.figure(constrained_layout=True,figsize=figsize)

        ax=fig.add_subplot()

        S_avg = np.zeros((3, self.res[1], self.res[0]))

        for t in np.linspace(0, period, N_samples):
            pv = self.poynting_vector(t)
            S_avg = S_avg + pv / N_samples

        extent = self.meshgrid[0].min(), self.meshgrid[0].max(), self.meshgrid[1].min(), self.meshgrid[1].max()

        S_0 = S_avg[0]
        S_1 = S_avg[1]
        S_0 = utils.lowpass_downsample(S_0, pv_coarseness)
        S_1 = utils.lowpass_downsample(S_1, pv_coarseness)

        vector_meshgrid = np.meshgrid(np.linspace(extent[0], extent[1], S_0.shape[1]),
                                      np.linspace(extent[2], extent[3], S_0.shape[0]))
        xgrid, ygrid = vector_meshgrid
        magnitude = ((S_0 ** 2) + (S_1 ** 2)) ** (1 / 2)

        colormap = matplotlib.cm.afmhot
        norm = Normalize()
        colors = magnitude
        norm.autoscale(colors)

        unit_s_0 = S_0 / magnitude
        unit_s_1 = S_1 / magnitude

        if eps:
            viz.eps_shapes(self.phc.layers[-1], alpha=0.5, ax=ax, extent=extent)

        quiv = ax.quiver(np.ravel(xgrid), np.ravel(ygrid), np.ravel(unit_s_0), np.ravel(unit_s_1),
                         color=(colormap(norm(np.ravel(magnitude)))), pivot='mid',
                         angles='xy', scale_units='xy', scale=1, width=0.0015 *pv_coarseness, alpha=0.8)
        ax.set_facecolor(np.array([0, 0, 0, 1]))
        return fig

    def return_mode_volume(self)    :
        E = self._field(0)[0]
        eps = self.eps_dist()

        normed = np.linalg.norm(E, axis=0)
        return np.sum(normed*eps) / self.res[0]*self.res[1] / np.max(normed*eps) * np.linalg.norm(np.cross(self.lattice.a1, self.lattice.a2))

    def __add__(self, o):
        """
        Add two fields as time -> field_1(time) + field_2(time)
        @param o: Field to be added
        @return:
        """
        if self.phc != o.phc:
            raise (ValueError("Unable to add fields over different crystals."))
        if all(self.res != o.res):
            raise (ValueError("Unable to add fields with different resolutions."))
        if self.polarization != o.polarization:
            raise (ValueError("Unable to add fields with different polarizations."))
        if self.z_dimension != o.z_dimension:
            raise (ValueError("Unable to add fields with different z dimension."))

        field = lambda t: self._field(t) + o._field(t)
        return XYField(res = self.res, phc=self.phc, z_dimension=self.z_dimension,
                                 polarization=self.polarization, meshgrid=self.meshgrid, field=field)

    def __sub__(self, o):

        if self.phc != o.phc:
            raise (ValueError("Unable to add fields over different crystals."))
        if all(self.res != o.res):
            raise (ValueError("Unable to add fields with different resolutions."))
        if self.polarization != o.polarization:
            raise (ValueError("Unable to add fields with different polarizations."))
        if self.z_dimension != o.z_dimension:
            raise (ValueError("Unable to add fields with different z dimension."))

        field = lambda t: self._field(t) - o._field(t)
        return XYField(res = self.res, phc=self.phc, z_dimension=self.z_dimension,
                                 polarization=self.polarization, meshgrid=self.meshgrid, field=field)

    def __mul__(self, o):
        field = lambda t: o * self._field(t)
        return XYField(res=self.res, phc=self.phc, z_dimension=self.z_dimension,
                         polarization=self.polarization, meshgrid=self.meshgrid, field=field)

    def __rmul__(self, o):
        field = lambda t: o * self._field(t)
        return XYField(res=self.res, phc=self.phc, z_dimension=self.z_dimension,
                         polarization=self.polarization, meshgrid=self.meshgrid, field=field)