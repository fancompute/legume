import numpy as np
from legume.utils import ftinv
from legume.backend import backend as bd
import scipy.constants as cs
import sys, time


class ExcitonSchroedEq(object):
    """Main simulation class of the excitonic Schroedinger equation.
    """

    def __init__(self,
                 phc,
                 z,
                 V_shapes,
                 a,
                 M,
                 E0,
                 loss,
                 osc_str= None,
                 gmax: float = 3.,
                 truncate_g='abs'):
        """Initialize the Schroedinger equation expansion.
        
        Parameters
        ----------

        phc : PhotCryst
            Photonic crystal object to be simulated.
        gmax : float, optional
            Maximum reciprocal lattice wave-vector length in units of 2pi/a.
        a : float
            lattice constant [m]
        M : float
            exciton mass [kg]
        V_shapes: float
            potential of the shapes in the layer [eV]
        E0 : float
            free exciton energy [eV]
        z : float
            position of the excitonic layer in z direction.
            This cannot be in the claddings
        loss : float
            losses assumed to be contant [eV]
        osc_str : list or numpy array,optional
            oscillator strength in units [m^-2],
            it must have  3 components (x,y,z)
        truncate_g : {'tbt', 'abs'}
            Truncation of the reciprocal lattice vectors, ``'tbt'`` takes a 
            parallelogram in reciprocal space, while ``'abs'`` takes a circle.
        """

        self.phc = phc
        # Number of layers in the PhC
        self.N_layers = len(phc.layers)
        layer_index = self._z_to_lind(z)
        if layer_index == 0 or layer_index == self.N_layers+1:
            raise ValueError(f"ExcitonSchroedEq cannot be intilized in a cladding"
                            f" layer at z={z:.3f}, change the position 'z'.")
        else:
            # Note that layer_index=1 corresponds to the first layer phc.layers[0] 
            self.layer = phc.layers[layer_index-1]

        self.E0 = E0
        self.M = M
        self.a = a


        self.gmax = gmax
        self.loss = loss
        self.z = z
        self.V_shapes = V_shapes
        self.osc_str = osc_str
        self.truncate_g = truncate_g

        
        

        if osc_str is not None:
            if type(self.osc_str) == list:
                self.osc_str = np.asarray(self.osc_str)
            elif type(self.osc_str) == np.ndarray:
                pass
            else:
                raise TypeError("'osc_str' must be a list or a numpy array.")
            if np.shape(self.osc_str)[0] != 3:
                raise ValueError("'osc_str' must have 3 componets.")

        if self.truncate_g == 'tbt':
            self._init_reciprocal_tbt()
            self._compute_pot_ft_tbt()
        elif self.truncate_g == 'abs':
            self._init_reciprocal_abs()
            self._compute_pot_ft_abs()
        else:
            raise ValueError("'truncate_g' must be 'tbt' or 'abs'.")

    @property
    def eners(self):
        """Energies of the eigenmodes computed by the Schroedinger equation.
        """
        if self._eners is None: self._eners = []
        return self._eners

    @property
    def eigvecs(self):
        """Eigenvectors of the eigenmodes computed by the by the Schroedinger equation.
        """
        if self._eigvecs is None: self._eigvecs = []
        return self._eigvecs

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

    def _print(self, text, flush=False, end='\n'):
        """Print if verbose_ex==True
            """
        if self.verbose_ex == True:
            if flush == False:
                print(text, end=end)
            else:
                sys.stdout.write("\r" + text)
                sys.stdout.flush()

    def _z_to_lind(self, z):
        """
        Get a layer index corresponding to a position z. Claddings are included 
        as first and last layer
        """

        z_max = self.phc.claddings[0].z_max
        lind = 0  # Index denoting which layer (including claddings) z is in
        
        while z > z_max and lind < self.N_layers:
            lind += 1
            z_max = self.phc.layers[lind - 1].z_max
        if z > z_max and lind == self.N_layers: lind += 1

        return lind

    def _init_reciprocal_tbt(self):
        """
        Initialize reciprocal lattice vectors based on self.layer and self.gmax
        """
        n1max = np.int_(
            (2 * np.pi * self.gmax) / np.linalg.norm(self.layer.lattice.b1))
        n2max = np.int_(
            (2 * np.pi * self.gmax) / np.linalg.norm(self.layer.lattice.b2))

        # This constructs the reciprocal lattice in a way that is suitable
        # for Toeplitz-Block-Toeplitz inversion of the permittivity in the main
        # code. However, one caveat is that the hexagonal lattice symmetry is
        # not preserved.
        inds1 = np.tile(np.arange(-n1max, n1max + 1), (2*n2max + 1, 1))  \
                         .reshape((2*n2max + 1)*(2*n1max + 1), order='F')
        inds2 = np.tile(np.arange(-n2max, n2max + 1), 2 * n1max + 1)

        gvec = self.layer.lattice.b1[:, np.newaxis].dot(inds1[np.newaxis, :]) +\
                self.layer.lattice.b2[:, np.newaxis].dot(inds2[np.newaxis, :])

        # Save the reciprocal lattice vectors
        self._gvec = gvec

        # Save the number of vectors along the b1 and the b2 directions
        # Note: gvec.shape[1] = n1g*n2g
        self.n1g = 2 * n1max + 1
        self.n2g = 2 * n2max + 1

    def _init_reciprocal_abs(self):
        """
        Initialize reciprocal lattice vectors with circular truncation.
        """
        n1max = np.int_(
            (4 * np.pi * self.gmax) / np.linalg.norm(self.layer.lattice.b1))
        n2max = np.int_(
            (4 * np.pi * self.gmax) / np.linalg.norm(self.layer.lattice.b2))

        inds1 = np.tile(np.arange(-n1max, n1max + 1), (2*n2max + 1, 1))  \
                         .reshape((2*n2max + 1)*(2*n1max + 1), order='F')
        inds2 = np.tile(np.arange(-n2max, n2max + 1), 2 * n1max + 1)
        gvec = self.layer.lattice.b1[:, np.newaxis].dot(inds1[np.newaxis, :]) + \
                self.layer.lattice.b2[:, np.newaxis].dot(inds2[np.newaxis, :])
        gnorm = np.sqrt(gvec[0, :]**2 + gvec[1, :]**2)
        if self.gmax * 2 * np.pi in gnorm:
            gvec = gvec[:, gnorm <= 2 * np.pi * (self.gmax + 0.0001)]
            print(
                f"Warning: gmax={self.gmax} exactly equal to one of the g-vectors modulus"
                f", reciprocal lattice truncated with gmax={self.gmax+0.0001}"
                f" to avoid problems."
                f"\nPlane waves used in the expansion = {np.shape(gvec)[1]}.")
        else:
            gvec = gvec[:, gnorm <= 2 * np.pi * self.gmax]

        # Save the reciprocal lattice vectors
        self._gvec = gvec

    def _compute_pot_ft_tbt(self):
        """
        Compute the unique FT coefficients of the potential, V(g-g')
        """
        (n1max, n2max) = (self.n1g, self.n2g)
        G1 = -self.gvec + self.gvec[:, [0]]
        G2 = bd.zeros((2, n1max * n2max))

        for ind1 in range(n1max):
            G2[:, ind1*n2max:(ind1+1)*n2max] = - self.gvec[:, [ind1*n2max]] + \
                            self.gvec[:, range(n2max)]

        # Compute and store T1 and T2
        self.T1 = self.layer.compute_exc_ft(G1, self.V_shapes)
        self.T2 = self.layer.compute_exc_ft(G2, self.V_shapes)

        # Store the g-vectors to which T1 and T2 correspond
        self.G1 = G1
        self.G2 = G2

    def _compute_pot_ft_abs(self):
        """
        Compute the unique FT coefficients of the potential, V(g-g') for
         assuming abs-initialized reciprocal lattice
        """
        #self.pot_ft = []
        # Initialize the FT coefficient lists; in the end the length of these
        # will be equal to the total number of layers in the PhC
        self.T1 = []
        self.T2 = []

        T1 = self.layer.compute_exc_ft(self.gvec, self.V_shapes)
        T2 = self.layer.compute_exc_ft(self.gvec, self.V_shapes)

        # Store T1 and T2
        if bd.amax(bd.abs(bd.imag(T1))) < 1e-10 * bd.amax(bd.abs(bd.real(T1))):
            self.T1.append(bd.real(T1))
        else:
            self.T1.append(T1)
        if bd.amax(bd.abs(bd.imag(T2))) < 1e-10 * bd.amax(bd.abs(bd.real(T2))):
            self.T2.append(bd.real(T2))
        else:
            self.T2.append(T2)

        self.G1 = self.gvec
        self.G2 = self.gvec

        ggridx = (self.gvec[0, :][np.newaxis, :] -
                  self.gvec[0, :][:, np.newaxis]).ravel()
        ggridy = (self.gvec[1, :][np.newaxis, :] -
                  self.gvec[1, :][:, np.newaxis]).ravel()

        pot_ft = self.layer.compute_exc_ft(np.vstack((ggridx, ggridy)),
                                           self.V_shapes)
        self.pot_ft = bd.reshape(pot_ft,
                                 (self.gvec[0, :].size, self.gvec[0, :].size))

    def _compute_matrix(self):
        """
        Construct the FT matrix of the potential
        """
        if self.truncate_g == 'tbt':
            pot_mat = bd.toeplitz_block(self.n1g, self.T1, self.T2)
        elif self.truncate_g == 'abs':
            pot_mat = self.pot_ft
        self.pot_mat = pot_mat

    def set_run_options(self, numeig_ex: int = 10, verbose_ex: bool = True):
        """Set multiple options for the guided-mode expansion.
            
            Parameters
            numeig_ex : int, optional
            verbose_ex : bool, optional
                Print information at intermmediate steps. Default is True.
            """

        # Make a dictionary that stores all the options
        self._run_options = {'numeig_ex': numeig_ex, 'verbose_ex': verbose_ex}

        # Also store the options as separate attributes
        for (option, value) in self._run_options.items():
            # Set all the options as class attributes
            setattr(self, option, value)

    def run(self, kpoints=np.array([[0], [0]]), **kwargs):
        """
        Run the simulation. The computed eigen-frequencies are stored in
        :attr:`ExcitonSchroedEq.freqs`, and the corresponding eigenvectors - 
        in :attr:`ExcitonSchroedEq.eigvecs`.
        
        Parameters
        ----------
        kpoints : np.ndarray, optional
            Numpy array of shape (2, Nk) with the [kx, ky] coordinates of the 
            k-vectors over which the simulation is run.
        numeig_ex : int, optional
            Number of eigen-frequencies to be stored (starting from lowest).
        """
        self.set_run_options(**kwargs)

        self.t_eig = 0  # For timing of the diagonalization
        t_start = time.time()

        self._kpoints = kpoints
        # Change this if switching to a solver that allows for variable numeig_ex
        #self.numeig_ex = numeig_ex

        self._compute_matrix()

        eners = []
        self._eigvecs = []
        #self.verbose_ex = verbose_ex
        for ik, k in enumerate(kpoints.T):

            self._print(
                f"Running Exciton diagonalisation k-point {ik+1} of {kpoints.shape[1]}",
                flush=True)
            # Construct the matrix for diagonalization in eV
            diag_vec = np.asarray([[k[0] + self.gvec[0, :]],
                                   [k[1] + self.gvec[1, :]]]) / self.a
            diag = np.linalg.norm(diag_vec,
                                  axis=0)**2 * cs.hbar**2 / (2 * self.M * cs.e)
            mat = np.zeros((np.shape(self.gvec)[1], np.shape(self.gvec)[1]),
                           dtype="complex")
            np.fill_diagonal(mat, diag)
            mat = mat + self.pot_mat
            # Check if mat is Hermitian
            check = np.max(np.abs(mat - np.conjugate(mat.T)))
            if check > 1e-10:
                raise ValueError(
                    f"Excitonic Hamiltonian at {ik+1} of {kpoints.shape[1]}"
                    " k pointsis not Hermitian.")

            #    Gk = bd.sqrt(bd.square(k[0] + self.gvec[0, :]) + \
            #            bd.square(k[1] + self.gvec[1, :]))
            #    mat = bd.outer(Gk, Gk)
            #    mat = mat * self.eps_inv_mat

            # Diagonalize using numpy.linalg.eigh() for now; should maybe switch
            # to scipy.sparse.linalg.eigsh() in the future
            # NB: we shift the matrix by np.eye to avoid problems at the zero-
            # frequency mode at Gamma
            t_eig = time.time()
            (ener2, evecs) = bd.eigh(mat + bd.eye(mat.shape[0]))
            self.t_eig += time.time() - t_eig
            ener1 = ener2 - bd.ones(mat.shape[0])

            i_sort = bd.argsort(ener1)[0:self.numeig_ex]
            ener = ener1[i_sort]
            evec = evecs[:, i_sort]
            eners.append(ener)
            self._eigvecs.append(evec)

        self._print("", flush=True)
        self._print(
            f"{time.time()-t_start:.4f}s total time for real and imaginary energies, of which"
        )
        self._print(
            f"  {self.t_eig:.4f}s for diagonalization of the Hamiltonian")

        # Store the energies, E0 adds free exciton energy, loss are non-radiative losses
        self._eners = bd.array(eners) + self.E0 + 1j * self.loss
        self._eigvecs = bd.array(self._eigvecs)
        self.mat = mat

    def get_pot_xy(self, Nx=100, Ny=100):
        """
        Get the xy-plane potential of the layer as computed from 
        an inverse Fourier transform with the Schr Eq. reciprocal lattice vectors.
        
        Parameters
        ----------
        Nx : int, optional
            A grid of Nx points in the elementary cell is created.
        Ny : int, optional
            A grid of Ny points in the elementary cell is created.
        z : float
            Position of the xy-plane.
        
        Returns
        -------
        pot_r : np.ndarray
            The in-plane real-space potential.
        xgrid : np.ndarray
            The constructed grid in x.
        ygrid : np.ndarray
            The constructed grid in y.
        """
        # Layer index where z lies

        (xgrid, ygrid) = self.layer.lattice.xy_grid(Nx=Nx, Ny=Ny)

        ft_coeffs = np.hstack(
            (self.T1, self.T2, np.conj(self.T1), np.conj(self.T2)))
        gvec = np.hstack((self.G1, self.G2, -self.G1, -self.G2))
        #ft_coeffs[0] sincewe have only one layer
        pot_r = ftinv(ft_coeffs[0], gvec, xgrid, ygrid)
        return (pot_r, xgrid, ygrid)

    def ft_wavef_xy(self, kind, mind):
        """
        Compute the wavefunction Fourier components in the xy-plane

        kind : int
            The wavefunction of the mode at `ExcitonSchroedEq.kpoints[:, kind]` is 
            computed.
        mind : int
            The wavefunction of the `mind` mode at that kpoint is computed.
        ft: np.ndarray
            The Fourier transform of specified wavefunction.
        """
        evec = self.eigvecs[kind][:, mind]
        ft = evec
        return ft

    def get_wavef_xy(self, kind, mind, Nx=100, Ny=100):
        """
        Compute the wavefunction in the xy-plane.
        
        Parameters
        ----------
        kind : int
            The wavefunction of the mode at `ExcitonSchroedEq.kpoints[:, kind]` is 
            computed.
        mind : int
            The wavefunction of the `mind` mode at that kpoint is computed.
        Nx : int, optional
            A grid of Nx points in the elementary cell is created.
        Ny : int, optional
            A grid of Ny points in the elementary cell is created.
        
        Returns
        -------
        fi : dict
            A dictionary with the requested components, 'x', 'y', and/or 'z'.
        xgrid : np.ndarray
            The constructed grid in x.
        ygrid : np.ndarray
            The constructed grid in y.
        """

        # Make a grid in the x-y plane
        (xgrid, ygrid) = self.layer.lattice.xy_grid(Nx=Nx, Ny=Ny)

        # Get the wavefunction fourier components

        ft = self._eigvecs[kind, :, mind]

        fi = ftinv(ft, self.gvec, xgrid, ygrid)

        return (fi, xgrid, ygrid)
