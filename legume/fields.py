import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize

from legume import utils
from legume import viz


class XYField:

    def __init__(self, res, z_dimension, modulation=np.array([0, 0]),
                 renormalize=False, **kwargs):
        """

        General Parameters
        ----------
        res : resolution of field profile in xy. int or tuple of ints
        z_dimension : z dimension in crystal of generated field

        To Construct from Guided Mode Expansion (Recommended)
        ----------
        gme : Guided mode expansion to construct fields from.
        modulation : K vector modulation applied to generated field.
                     Default: [0,0]
        renormalize : If True normalizes field such that maximum occurs at t=0
                      with vector magnitude 1.
        kind : k index for guided mode expansion field
        mind : mode index for guided mode expansion field
        polarization :  Polarization of field.
                        If 'TE' only calculate xy of e field and z of h field,
                        likewise for 'TM',
                        None generates all components of both fields.

        To Construct from Field
        ----------
        field : Field directly provided to construct object:
                Should be function time -> np.array([[E_x, E_y,E_z],
                [H_x, H_y, H_z]])
        phc : Photonic crystal field was generated over.
        meshgrid : Meshgrid associated with field where each entry of E
                   and H assigned to point on meshgrid.
        """

        self._eps_dist = None
        if np.array(res).__len__() == 1:
            self.res = np.array((res, res))

        elif np.array(res).__len__() == 2:
            self.res = np.array(res)
        else:
            raise ValueError("Resolution must be int "
                             "or 2 length array_like of ints.")

        self.z_dimension = z_dimension

        if 'field' in kwargs:
            try:
                self.phc = kwargs['phc']
                self.meshgrid = kwargs['meshgrid']
                self._field = kwargs['field']
            except KeyError():
                raise ValueError("When giving a field object as the"
                                 "initializing argument additional arguments "
                                 "\"phc\" and \"meshgrid\" are required.")
            if 'freq' in kwargs:
                self._freq = kwargs['freq']

        elif 'gme' in kwargs:
            try:
                gme = kwargs['gme']
                self.gme = repr(gme)
                self.kind = kwargs['kind']
                self.mind = kwargs['mind']
                self.polarization = kwargs['polarization']

            except KeyError():
                raise ValueError("When passing a gme object as the"
                                 "initializing argument additional arguments "
                                 "\"kind\", \"mind\" and \"polarization\" "
                                 "are required.")

            self.phc = gme.phc

            e_comp, h_comp = {'TE': ['xy', 'z'],
                              'TM': ['z', 'xy'],
                              'None': ['xyz', 'xyz']}[self.polarization]

            fe, xgrid, ygrid = gme.get_field_xy(field='e', component=e_comp,
                                                z=z_dimension, kind=self.kind,
                                                mind=self.mind,
                                                Nx=self.res[0], Ny=self.res[1])

            fh, _, _ = gme.get_field_xy(field='h', component=h_comp,
                                        z=z_dimension,
                                        kind=self.kind, mind=self.mind,
                                        Nx=self.res[0], Ny=self.res[1])

            if self.polarization == 'TE':
                E = np.array([fe['x'], fe['y'], np.zeros(self.res[::-1])])
                H = np.array(
                    [np.zeros(self.res[::-1]), np.zeros(self.res[::-1]),
                     fh['z']])

            elif self.polarization == 'TM':
                E = np.array(
                    [np.zeros(self.res[::-1]), np.zeros(self.res[::-1]),
                     fe['z']])
                H = np.array([fh['x'], fh['y'], np.zeros(self.res[::-1])])

            elif self.polarization == 'None':
                E = np.array([fe['x'], fe['y'], fe['z']])
                H = np.array([fh['x'], fh['y'], fh['z']])
            else:
                raise ValueError("Polarization should be 'TE', 'TM' or 'None'")
            self.meshgrid = np.meshgrid(xgrid, ygrid)

            # Apply modulation
            E = E * np.exp(1j * (self.meshgrid[0] * modulation[0]
                                 + self.meshgrid[1] * modulation[1]))
            H = H * np.exp(1j * (self.meshgrid[0] * modulation[0]
                                 + self.meshgrid[1] * modulation[1]))

            if renormalize:
                field_0 = np.array([np.linalg.norm(E, axis=0),
                                    np.linalg.norm(H, axis=0)])

                argmax = np.unravel_index(np.argmax(np.abs(field_0)),
                                          field_0.shape)
                E = E / field_0[argmax]
                H = H / field_0[argmax]

            freq = gme.freqs[self.kind][self.mind]
            self._freq = freq
            self._kvec = modulation
            self._field = lambda t: np.array([E * np.exp(1j * freq * t),
                                              H * np.exp(1j * freq * t)])

        else:
            raise ValueError("Missing initializing argument \"gme\" "
                             "or \"field\".")

        self.lattice = self.phc.lattice
        self.cell_size = np.abs(max([self.lattice.a1[0],
                                     self.lattice.a2[0]])) / res[0], \
                         np.abs(max([self.lattice.a1[1],
                                     self.lattice.a2[1]])) / res[1]

        for layer in self.phc.layers + self.phc.claddings:
            zlayer = (self.z_dimension >= layer.z_min) * (
                        self.z_dimension < layer.z_max)
            if np.sum(zlayer) > 0:
                self.layer = layer
                break

    @property
    def freq(self):
        """
        Frequency of the mode. (Not necessarily well defined)
        """
        if hasattr(self, '_freq'):
            return self._freq
        else:
            # Modes not constructed from gme and have no defined frequency.
            return None

    @property
    def kvec(self):
        """
        Spatial frequency of the mode (Not necessarily well defined).
        """
        if hasattr(self, '_kvec'):
            return self._kvec
        else:
            # Modes not constructed from gme and have no defined wavevector.
            return None

    def Poynting_vector(self, time=0):
        """
        Return the Poynting vector field at time = time.

        Parameters
        ----------
        time : Instantaneous time of Poynting vector field.

        Returns
        -------
        Poynting vector field. np.array shape (3,self.res[0], self.res[1])
        """
        return np.cross(np.real(self._field(time)[0]),
                        np.real(self._field(time)[1]), axis=0)

    def coarse_Poynting_vector(self, time=0, pv_coarseness=1):
        """
        Return a lowpass downsampled Poynting vector field at time = time.

        Parameters
        ----------
        time : Instantaneous time of Poynting vector field.
        pv_coarseness : downscaling factor of vector field.
       
        Returns
        -------
        np.array rough shape (3,self.res[0]/pv_coarseness,
                                self.res[1]/pv_coarseness)
        """
        S = self.Poynting_vector(time)
        S_x = S[0]
        S_y = S[1]
        S_z = S[2]

        S_x = utils.low_pass_down_sample(S_x, pv_coarseness)
        S_y = utils.low_pass_down_sample(S_y, pv_coarseness)
        S_z = utils.low_pass_down_sample(S_z, pv_coarseness)
        return np.array([S_x, S_y, S_z])

    def chirality(self, field='e', time=0):
        """
        Return the chirality of the electric or magnetic field at time=time

        Parameters
        ----------
        field : chirality of the electric or magnetic field.
        time : Instantaneous time of the chirality field.
        
        Returns
        -------
        chirality vector field. (3,self.res[0], self.res[1])
        """
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
        """
        Return the field at time = time.

        Parameters
        ----------
        time : Instantaneous time of the field
        
        Returns
        -------  
        [[E_x, E_y, E_z], [H_x, H_y, H_z]] field 
        shape is (self.res[0],self.res[1])
        """
        return self._field(time)

    def eps_dist(self):
        """
        return the permittivity distribution matching the field generated.
        
        Returns
        -------
        bitmap permittivity distribution of the crystal shape [res[0],res[1]]
        """
        if self._eps_dist is None:
            self._eps_dist = self.phc.get_eps(
                (self.meshgrid[0], self.meshgrid[1],
                 self.z_dimension * np.ones(self.meshgrid[0].shape)))
        return self._eps_dist

    def visualize_field(self, field, component, time=0, eps=True, val='re',
                        normalize=False, fig=None, figsize=None):
        """
        Visualize the electric or magnetic field.

        Parameters
        ----------
        field : 'e' or 'h' electric or magnetic field respectively.
        component : 'x', 'y' and/or 'z' components of the field to include.
        time : instantaneous time of the field
        eps : If true includes a sketch of the permittivity distribution.
        val : 're', 'im' or 'abs' if the real, imaginary or absolute value of
               the field is visualized.
        normalize : if normalize is true the Frobenius norm of the included
                    components is taken and displayed.
        fig : a figure may be passed to plot the axis on,
              if None a figure is generated.
        figsize : the size of the figure (if generated).
        
        Returns
        -------
        fig, a matplotlib figure object structuring the visualization.
        """
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
            self._field_ax_builder(ax=ax, field=norm_field, cmap=cmap,
                                   vmax=vmax, vmin=vmin, eps=eps)

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
                else:
                    raise ValueError("Val should be re, im or abs")
                self._field_ax_builder(ax=ax, field=field_comp, vmax=vmax,
                                       vmin=vmin, cmap=cmap, eps=eps)
                i += 1
        return fig

    def _field_ax_builder(self, ax, field, vmax, vmin, cmap, eps):

        extent = self.meshgrid[0].min(), self.meshgrid[0].max(), self.meshgrid[
            1].min(), self.meshgrid[1].max()

        im = ax.imshow(field, extent=extent, vmax=vmax, vmin=vmin,
                       cmap=cmap, origin='lower')

        if eps:
            viz.shapes(layer=self.layer, ax=ax)

        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])
        return ax

    def vizualize_chirality(self, field, component='z', time=0, fig=None,
                            figsize=None, eps=True):
        """
        Visualize the chirality of the electric or magnetic field.

        Parameters
        ----------
        field : 'e' or 'h' electric or magnetic field respectively.
        component : 'x', 'y' or 'z' component of the chirality field to include.
        time : instantaneous time of the chirality field.
        eps : If true includes a sketch of the permittivity distribution.
        fig : a figure may be passed to plot the axis on,
              if None a figure is generated.
        figsize : the size of the figure (if generated).
        
        Returns
        -------
        fig, a matplotlib figure object structuring the visualization.
        """
        extent = self.meshgrid[0].min(), self.meshgrid[0].max(), self.meshgrid[
            1].min(), self.meshgrid[1].max()
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
            viz.shapes(layer=self.layer, ax=ax)
            ax.set_xlim(extent[0], extent[1])
            ax.set_ylim(extent[2], extent[3])
        return fig

    def generate_mp4(self, time_range, frames, val='re', eps=True,
                     Poynting_vector=True, pv_coarseness=1, **kwargs):
        """
        Generate an animation of a visualization over a time_range
        time_range :
        frames :
        val :
        eps :
        Poynting_vector :
        pv_coarseness :
        kwargs :
        @return:
        """
        # \\TODO Implement this

        raise NotImplementedError("Not implemented yet")

    def visualize_Poynting_vector(self, fig=None, eps=True, time=0,
                                  pv_coarseness=1, figsize=None):
        """
        Visualize the Poynting_vector of the field. Magnitude is color coded,
        arrows lengths are normalized.

        Parameters
        ----------
        time : instantaneous time of the Poynting vector field
        eps : If true includes a sketch of the permittivity distribution.
        pv_coarseness : factor by which the Poynting vector field is
                        low pass down sampled.
        fig : a figure may be passed to plot the axis on,
              if None a figure is generated.
        figsize : the size of the figure (if generated).
        
        Returns
        -------
        fig, a matplotlib figure object structuring the visualization.
        """
        pv = self.Poynting_vector(time)

        fig = self._plot_xy_vector_field(vector_field=pv, fig=fig, eps=eps,
                                         pv_coarseness=pv_coarseness,
                                         figsize=figsize)

        return fig
        # The parameters for the quiver

    def time_avg_pv(self, period, N_samples):
        """
        The time averaged Poynting vector field.

        Parameters
        ----------
        period : the period over which the average is taken
                 (Generally 2*np.pi/frequency).
        N_samples : the number of samples to take in the average.
        
        Returns
        -------
        The time averaged Poynting vector field. np.array shape (3,self.res[0],
                                                                 self.res[1])
        """
        S_avg = np.zeros((3, self.res[1], self.res[0]))

        for t in np.linspace(0, period, N_samples):
            pv = self.Poynting_vector(t)
            S_avg = S_avg + pv / N_samples

        return S_avg

    def visualize_time_avg_pv(self, period, N_samples, fig=None, eps=True,
                              pv_coarseness=1, figsize=None):
        """
        Visualize the time averaged Poynting vector field.

        Parameters
        ----------
        period : the period over which the average is taken
                 (Generally 2*np.pi/frequency).
        N_samples : the number of samples to take in the average.
        eps : If true includes a sketch of the permittivity distribution.
        pv_coarseness : factor by which the Poynting vector field is
                        low pass down sampled.
        fig : a figure may be passed to plot the axis on,
              if None a figure is generated.
        figsize : the size of the figure (if generated).
        
        Returns
        -------
        A matplotlib figure object structuring the visualization.

        """
        S_avg = self.time_avg_pv(period, N_samples)

        fig = self._plot_xy_vector_field(vector_field=S_avg, fig=fig, eps=eps,
                                         pv_coarseness=pv_coarseness,
                                         figsize=figsize)
        return fig

    def _plot_xy_vector_field(self, vector_field, fig=None, eps=True,
                              pv_coarseness=1, figsize=None):
        if fig is None:
            fig = plt.figure(constrained_layout=True, figsize=figsize)

        ax = fig.add_subplot()
        extent = self.meshgrid[0].min(), self.meshgrid[0].max(), self.meshgrid[
            1].min(), self.meshgrid[1].max()

        S_0 = vector_field[0]
        S_1 = vector_field[1]
        S_0 = utils.low_pass_down_sample(S_0, pv_coarseness)
        S_1 = utils.low_pass_down_sample(S_1, pv_coarseness)

        vector_meshgrid = np.meshgrid(
            np.linspace(extent[0], extent[1], S_0.shape[1]),
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
            viz.shapes(layer=self.layer, ax=ax, color='w')
            ax.set_xlim(extent[0], extent[1])
            ax.set_ylim(extent[2], extent[3])

        quiv = ax.quiver(np.ravel(xgrid), np.ravel(ygrid), np.ravel(unit_s_0),
                         np.ravel(unit_s_1),
                         color=(colormap(norm(np.ravel(magnitude)))),
                         pivot='mid',
                         angles='xy', scale_units='xy',
                         scale=1 / min(self.cell_size) / pv_coarseness,
                         width=0.01 * pv_coarseness,
                         alpha=0.8)
        ax.set_facecolor(np.array([0, 0, 0, 1]))
        return fig

    def __add__(self, o):
        """
        Add two fields as time -> field_1(time) + field_2(time)

        Parameters
        ----------
        o : Field to be added

        Returns
        -------
        New Field object composed of self and o.
        """
        if self.phc != o.phc:
            raise (ValueError("Unable to add fields over different crystals."))
        if all(self.res != o.res):
            raise (ValueError("Unable to add fields with different "
                              "resolutions."))
        if self.z_dimension != o.z_dimension:
            raise (ValueError("Unable to add fields with different "
                              "z dimension."))

        def field(t):
            return self._field(t) + o._field(t)

        if np.abs(self._freq - o._freq) < 1e-6:
            return XYField(res=self.res, phc=self.phc,
                           z_dimension=self.z_dimension,
                           polarization=self.polarization,
                           meshgrid=self.meshgrid, field=field,
                           freq=self._freq)
        else:
            return XYField(res=self.res, phc=self.phc,
                           z_dimension=self.z_dimension,
                           polarization=self.polarization,
                           meshgrid=self.meshgrid, field=field)

    def __sub__(self, o):
        """
        Subtract two fields as time -> field_1(time) - field_2(time)

        Parameters
        ----------
        o : Field to be subtracted

        Returns
        -------
        New Field object composed of self and o.
        """
        if self.phc != o.phc:
            raise (ValueError("Unable to add fields over different crystals."))
        if all(self.res != o.res):
            raise (ValueError("Unable to add fields with different "
                              "resolutions."))
        if self.z_dimension != o.z_dimension:
            raise (ValueError("Unable to add fields with different "
                              "z dimension."))

        def field(t):
            return self._field(t) - o._field(t)

        if np.abs(self._freq - o._freq) < 1e-6:
            return XYField(res=self.res, phc=self.phc,
                           z_dimension=self.z_dimension,
                           polarization=self.polarization,
                           meshgrid=self.meshgrid, field=field,
                           freq=self._freq)
        else:
            return XYField(res=self.res, phc=self.phc,
                           z_dimension=self.z_dimension,
                           polarization=self.polarization,
                           meshgrid=self.meshgrid, field=field)

    def __mul__(self, o):
        """
        Multiply a field by a scalar OR a scalar array broadcastable onto
        [[E_x, E_y, E_z], [H_x, H_y, H_z]].
        Parameters
        ----------
        o: scalar or array to multiply field by.

        Returns
        -------
        New Field object self * o.

        """
        def field(t):
            return o * self._field(t)

        if self._freq is not None:
            return XYField(res=self.res, phc=self.phc,
                           z_dimension=self.z_dimension,
                           polarization=self.polarization,
                           meshgrid=self.meshgrid, field=field,
                           freq=self._freq)
        else:
            return XYField(res=self.res, phc=self.phc,
                           z_dimension=self.z_dimension,
                           polarization=self.polarization,
                           meshgrid=self.meshgrid, field=field)