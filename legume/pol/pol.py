import numpy as np
from legume.utils import ftinv, z_to_lind, from_freq_to_e
from legume.print_utils import verbose_print
from legume.print_backend import print_backend as prbd
from legume.backend import backend as bd
import legume.constants as cs
from legume.gme import GuidedModeExp
from legume.exc import ExcitonSchroedEq
import time


class HopfieldPol(object):
    """Main simulation class of the generalized Hopfield matrix method.
    """

    def __init__(self, phc, gmax, truncate_g='abs'):
        """Initialize the Schroedinger equation expansion.
        
        Parameters
        ----------
        phc : PhotCryst
            Photonic crystal object to be simulated.
        gmax : float, optional
            Maximum reciprocal lattice wave-vector length in units of 2pi/a.
        truncate_g : {'tbt', 'abs'}
            Truncation of the reciprocal lattice vectors, ``'tbt'`` takes a 
            parallelogram in reciprocal space, while ``'abs'`` takes a circle.
        """

        self.gme = GuidedModeExp(phc, gmax, truncate_g=truncate_g)

        self.exc_list = []

        if len(phc.qws) == 0:
            raise ValueError(
                "There are no active layers in the PhotCryst"
                ", add them with 'add_qw()' or run a GuidedModeExp"
                " simulation.")

        for qw in phc.qws:  #loop over quantum wells added with add_qw
            layer_ind = z_to_lind(phc, qw.z)
            self.exc_list.append(
                ExcitonSchroedEq(phc=phc,
                                 z=qw.z,
                                 V_shapes=qw.V_shapes,
                                 a=qw.a,
                                 M=qw.M,
                                 E0=qw.E0,
                                 loss=qw.loss,
                                 osc_str=qw.osc_str,
                                 gmax=gmax,
                                 truncate_g=truncate_g))

        # Check that all excitonic layers have the same lattice constant
        a_array = np.array([exc_sch.a for exc_sch in self.exc_list])
        if np.all(np.abs(a_array - a_array[0]) < 1e-15):
            self.a = a_array[0]
        else:
            raise ValueError("All the quantum well layers passed to"
                             " HopfieldPol should have the same lattice"
                             " constant a.")

    def __repr__(self):
        rep = 'HopfieldPol(\n'
        rep += 'phc = PhotCryst object' + ', \n'
        GME_run_options = [
            'gmax', 'gmode_compute', 'gmode_inds', 'gmode_step', 'gradients',
            'eig_solver', 'eig_sigma', 'eps_eff'
        ]
        rep += "GME: \n"
        for option in GME_run_options:
            try:
                val = getattr(self.gme, option)
                rep += "\t" + option + ' = ' + repr(val) + ', \n'
            except:
                pass
        rep += "ESE at:\n"
        for active in self.exc_list:
            rep += "\t z = " + f"{active.z:.6f}" + ', \n'
        rep += ')'
        return rep

    @property
    def eners(self):
        """Energies of the eigenmodes computed by the Hopfield matrix diagonalisation.
        """
        if self._eners is None: self._eners = []
        return self._eners

    @property
    def eners_im(self):
        """Imaginary part of the frequencies of the eigenmodes computed by the 
        Hopfield matrix diagonalisation.
        """
        if self._eners_im is None: self._eners_im = []
        return self._eners_im

    @property
    def eigvecs(self):
        """Eigenvectors of the eigenmodes computed by the by the Hopfield matrix diagonalisation.
        """
        if self._eigvecs is None: self._eigvecs = []
        return self._eigvecs

    @property
    def fractions_ex(self):
        """Excitonic fractions of the polaritonic eigenmodes.
        """
        if self._fractions_ex is None: self._fractions_ex = []
        return self._fractions_ex

    @property
    def fractions_ph(self):
        """Photonic fractions of the polaritonic eigenmodes.
        """
        if self._fractions_ph is None: self._fractions_ph = []
        return self._fractions_ph

    @property
    def kpoints(self):
        """Numpy array of shape (2, Nk) with the [kx, ky] coordinates of the 
        k-vectors over which the simulation is run.
        """
        if self._kpoints is None: self._kpoints = []
        return self._kpoints

    @property
    def gvec(self):
        """Numpy array of shape (2, Ng) with the [gx, gy] coordinates of the 
        reciprocal lattice vectors over which the simulation is run.
        """
        if self._gvec is None: self._gvec = []
        return self._gvec

    def _calculate_fraction(self, eigenvectors):
        """
        Calculate the photonic and excitonic fraction of the bands starting from
        the polaritonic eigenvectors.

         """

        # Not pythonic, could be done better
        num_bands = bd.shape(eigenvectors)[1]
        frac_ex = bd.zeros((num_bands))
        frac_ph = bd.zeros((num_bands))

        for band in range(num_bands):
            frac_ph[band] = bd.sum(bd.abs(eigenvectors[0:self.N_max,band])**2)\
            +bd.sum(bd.abs(eigenvectors[self.N_max+self.M_max*self.num_QWs:2*self.N_max+self.M_max*self.num_QWs,band])**2)

            frac_ex[band] = bd.sum(bd.abs(eigenvectors[self.N_max:self.N_max+self.M_max*self.num_QWs,band])**2)\
            +bd.sum(bd.abs(eigenvectors[2*self.N_max+self.M_max*self.num_QWs:,band])**2)

        return frac_ex, frac_ph

    def _calculate_C_D(self, exc, kind):
        """C and D blocks of generalized Hopfield matrix,
        see Appendix of https://journals.aps.org/prb/abstract/10.1103/PhysRevB.75.235325,
        here we adopt SI units by adding the factor 1/(4*pi*epsilon_0). The factor 1/sqrt(a)
        comes from the normalisation of the fields and recovers the correct units.

        The input is an ExcitonSchroedEq run from which we recover the oscillator strength, the
        polarization unit vector and the exc. wavefunction. The prefactor of C
        is also multiplied by 1/e to get the energies in eV. Finally, we also recover the Fourier
        components of the Electric field from the GuidedModeExp.

        The Oscillator strength must be converted to 'float'.

         """
        pref = -1j * bd.sqrt(cs.hbar**2 * cs.e**2 /
                             (4 * cs.m_e * cs.epsilon_0)) / cs.e / bd.sqrt(
                                 self.a)
        C = bd.zeros((self.N_max, self.M_max), dtype="complex")
        #n: loop over photonic modes, nu: loop over excitonic modes
        for n in range(self.N_max):
            E_comp = self.gme.ft_field_xy("E", kind=kind, mind=n, z=exc.z)
            for nu in range(self.M_max):
                W_comp = exc.ft_wavef_xy(kind=kind, mind=nu)
                C[n, nu] = pref * bd.sum(
                    bd.dot(bd.sqrt(exc.osc_str.astype(float)), E_comp) *
                    bd.conj(W_comp))

        D = bd.zeros((self.N_max, self.N_max), dtype="complex")
        #n_1, n_2 loop over photonic modes (n and n' in the paper), nu loop over excitonic modes
        for n_1 in range(self.N_max):
            for n_2 in range(self.N_max):
                D[n_1, n_2] = bd.sum(
                    bd.conj(C[n_1, :]) * C[n_2, :] /
                    bd.real(exc.eners[kind, :]))

        return C, D

    def _construct_Hopfield(self, kind):
        """ Construct the generalised Hopfield matrix for given k point 

        """

        # Conversion factor: from dimensionless frequency to eV
        conv_fact = from_freq_to_e(self.a)

        #Initialise the list which contains all the C blocks, and the final D block
        C_blocks = [[] for i in range(self.num_QWs)]
        D_final_block = bd.zeros((self.N_max, self.N_max), dtype="complex")

        #Calculate the photonic diagonal block
        diag_phot = bd.zeros((self.N_max, self.N_max), dtype="complex")

        np.fill_diagonal(
            diag_phot, self.gme.freqs[kind, :] * conv_fact +
            1j * self.gme.freqs_im[kind, :] * conv_fact)

        for ind_ex, exc_sch in enumerate(self.exc_list):
            C, D = self._calculate_C_D(exc=exc_sch, kind=kind)
            D_final_block = D_final_block + D
            C_blocks[ind_ex] = C

        C_final_block = bd.hstack([c for c in C_blocks])
        C_dagger_final_block = bd.conj(C_final_block.T)

        #Initialise the excitonic block
        diag_exc = bd.zeros(
            (self.M_max * self.num_QWs, self.M_max * self.num_QWs),
            dtype="complex")
        exc_el = bd.concatenate(
            [exc_out.eners[kind] for exc_out in self.exc_list])

        np.fill_diagonal(diag_exc, exc_el)

        diag_phot = diag_phot + 2 * bd.real(D_final_block)

        row_0 = bd.hstack(
            (diag_phot, -1j * C_final_block, -2 * D, -1j * C_final_block))
        row_1 = bd.hstack((1j * C_dagger_final_block, diag_exc,
                           -1j * C_dagger_final_block, diag_exc * 0.))
        row_2 = bd.hstack(
            (2 * D, -1j * C_final_block, -diag_phot, -1j * C_final_block))
        row_3 = bd.hstack((-1j * C_dagger_final_block, diag_exc * 0.,
                           1j * C_dagger_final_block, -diag_exc))
        M = bd.vstack((row_0, row_1, row_2, row_3))

        return M

    def run(self,
            gme_options={},
            exc_options={},
            kpoints: np.ndarray = np.array([[0], [0]]),
            verbose=True):
        """
        Compute the eigenmodes of the photonic crystal taking
        into account light-matter interaction.
        
        The generalized Hopfield method implemented
        proceeds as follows:

            Iterate over the k points:

                Run the :meth:`GuidedModeExp.run` method for
                the input phc.

                Run :meth:`ExcitonSchroedEq.run` for each
                quantum well layer added with :meth:`PhotCrys.add_qw`
                method to the photonic crystal.

                Compute the photons-excitons coupling terms and
                construct the Hopfield matrix for diagonalization.
                The properties of the eigenmodes are stored only
                for mode with positive energy. The energy modes are
                stored in :attr:`HopfieldPol.eners`. The losses are stored
                in :attr:`HopfieldPol.eners_im`. From the eigenvectors
                we calulate the photonic (excitonic) fractions of
                the polaritonic modes which are stored in
                :attr:`HopfieldPol.fractions_ph` (:attr:`HopfieldPol.fractions_ph`).

        
        Parameters
        ----------
        kpoints : np.ndarray, optional
            Numpy array of shape (2, Nk) with the [kx, ky] coordinates of the 
            k-vectors over which the simulation is run.
        """
        eners = []
        eners_im = []
        self._kpoints = kpoints
        self._eigvecs = []
        self._fractions_ex = []
        self._fractions_ph = []
        self.verbose = verbose
        self._gvec = self.gme.gvec

        #Force the same kpoints for gme and exc solvers
        gme_options['kpoints'] = self.kpoints
        exc_options['kpoints'] = self.kpoints

        #Run gme
        self.gme.run(**gme_options)

        self.num_QWs = np.shape(self.exc_list)[0]

        #Run all excitonic Sch. equations
        for exc_sch in self.exc_list:
            exc_sch.run(**exc_options)

        t_start = time.time()
        #Retrieve number of photonic/excitonic eigenvalues
        self.N_max = self.gme.numeig
        self.M_max = self.exc_list[0].numeig_ex
        num_k = kpoints.shape[1]  # Number of wavevectors

        for ik, k in enumerate(self.kpoints.T):
            prbd.update_prog(ik, num_k, self.verbose, "Running HP k-points:")

            # Construct the Hopfield matrix for diagonalization in eV
            mat = self._construct_Hopfield(kind=ik)
            self.numeig = np.shape(mat)[0]

            # NB: we shift the matrix by np.eye to avoid problems at the zero-
            # frequency mode at Gamma
            (ener2, evecs) = bd.eig(mat + bd.eye(mat.shape[0]))
            ener1 = ener2 - bd.ones(mat.shape[0])
            #Filter positive energies
            filt_pos = bd.real(ener1) >= 0
            ener1 = ener1[filt_pos]
            evecs = evecs[:, filt_pos]
            fractions_ex, fractions_ph = self._calculate_fraction(evecs)
            i_sort = bd.argsort(ener1)[0:int(
                self.numeig // 2 - 1
            )]  # Keep only np.shape(mat)[0]//2-1 eigenvalues, corresponding to positive energies

            ener = bd.real(ener1[i_sort])
            ener_im = bd.imag(ener1[i_sort])
            evec = evecs[:, i_sort]
            fraction_ex = fractions_ex[i_sort]
            fraction_ph = fractions_ph[i_sort]
            eners.append(ener)
            eners_im.append(ener_im)
            self._eigvecs.append(evec)
            self._fractions_ex.append(fraction_ex)
            self._fractions_ph.append(fraction_ph)

        # Store the energies
        self._fractions_ex = bd.array(self._fractions_ex)
        self._fractions_ph = bd.array(self._fractions_ph)
        self._eners = bd.array(eners)
        self._eners_im = bd.array(eners_im)
        self._eigvecs = bd.array(self._eigvecs)
        self.mat = mat

        self.total_time = time.time() - t_start

        prbd.HP_report(self)
