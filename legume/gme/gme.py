import numpy as np

import time, sys
from itertools import zip_longest

from .slab_modes import guided_modes, rad_modes
from . import matrix_elements
from legume.backend import backend as bd
from legume.utils import get_value, ftinv, find_nearest


class GuidedModeExp(object):
    """
    Main simulation class of the guided-mode expansion.
    """
    def __init__(self, phc, gmax=3., truncate_g='tbt'):
        """Initialize the guided-mode expansion.
        
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

        self.phc = phc
        self.gmax = gmax
        self.truncate_g = truncate_g

        # Number of layers in the PhC
        self.N_layers = len(phc.layers)

        # Parameters below are defined when self.run() is called
        # Number of G points included for every mode, will be defined after run
        self.modes_numg = []
        # Total number of basis vectors (equal to np.sum(self.modes_numg))
        self.N_basis = []
        # Indexes of guided modes which are actually included in the computation
        # (in case gmode_inds includes modes that are above the gmax cutoff)
        self.gmode_include = []

        # Initialize all the attributes defined as properties below
        self._freqs = []
        self._freqs_im = []
        self._eigvecs = []
        self._rad_coup = {}
        self._rad_gvec = {}
        self._kpoints = []
        self._gvec = []

        # Initialize the reciprocal lattice vectors and compute the FT of all
        # the layers of the PhC
        if self.truncate_g == 'tbt':
            self._init_reciprocal_tbt()
            self._compute_ft_tbt()
        elif self.truncate_g == 'abs':
            self._init_reciprocal_abs()
            self._compute_ft_abs()
        else:
            raise ValueError("'truncate_g' must be 'tbt' or 'abs'.")

    def __repr__(self):
        rep = 'GuidedModeExp(\n'
        rep += 'phc = PhotCryst object' + ', \n'
        rep += 'gmax = ' + repr(self.gmax) + ', \n'
        run_options = [
            'gmode_compute', 'gmode_inds', 'gmode_step', 'gradients',
            'eig_solver', 'eig_sigma', 'eps_eff'
        ]
        for option in run_options:
            try:
                val = getattr(self, option)
                rep += option + ' = ' + repr(val) + ', \n'
            except:
                pass

        rep += ')'
        return rep

    @property
    def freqs(self):
        """Real part of the frequencies of the eigenmodes computed by the 
        guided-mode expansion.
        """
        return self._freqs

    @property
    def freqs_im(self):
        """Imaginary part of the frequencies of the eigenmodes computed by the 
        guided-mode expansion.
        """
        return self._freqs_im

    @property
    def eigvecs(self):
        """Eigenvectors of the eigenmodes computed by the guided-mode expansion.
        """
        return self._eigvecs

    @property
    def rad_coup(self):
        """Coupling to TE and TM radiative modes in the claddings.
        The dictionary has four keys: ``'l_te'``, ``'l_tm'``, ``'u_te'``, 
        ``'u_tm'``, denoting cladding (l/u) and polarization (te/tm, or S/P). 
        Each of these is a list of lists in which the first dimension 
        corresponds to the k-point index and the second dimension corresponds 
        to the mode index. Finally, the elements are numpy arrays with length 
        equal to all the allowed diffraction orders. The corresponding 
        reciprocal lattice vectors are stored in :attr:`GuidedModeExp.rad_gvec`.
        """
        return self._rad_coup

    @property
    def rad_gvec(self):
        """Reciprocal lattice vectos corresponding to the radiation emission 
        direction of the coupling constants stored in 
        :attr:`GuidedModeExp.rad_coup`.
        """
        return self._rad_gvec

    @property
    def kpoints(self):
        """Numpy array of shape (2, Nk) with the [kx, ky] coordinates of the 
        k-vectors over which the simulation is run.
        """
        return self._kpoints

    @property
    def gvec(self):
        """Numpy array of shape (2, Ng) with the [gx, gy] coordinates of the 
        reciprocal lattice vectors over which the simulation is run.
        """
        return self._gvec

    def _print(self, text, flush=False, end='\n'):
        """Print if verbose==True
            """
        if self.verbose == True:
            if flush == False:
                print(text, end=end)
            else:
                sys.stdout.write("\r" + text)
                sys.stdout.flush()

    def _init_reciprocal_tbt(self):
        """
        Initialize reciprocal lattice vectors with a parallelogram truncation
        such that the eps matrix is toeplitz-block-toeplitz
        """
        n1max = np.int_(
            (2 * np.pi * self.gmax) / np.linalg.norm(self.phc.lattice.b1))
        n2max = np.int_(
            (2 * np.pi * self.gmax) / np.linalg.norm(self.phc.lattice.b2))

        # This constructs the reciprocal lattice in a way that is suitable
        # for Toeplitz-Block-Toeplitz inversion of the permittivity in the main
        # code. This might be faster, but doesn't have a nice rotation symmetry
        # in the case of e.g. hexagonal lattice.
        inds1 = np.tile(np.arange(-n1max, n1max + 1), (2*n2max + 1, 1))  \
                         .reshape((2*n2max + 1)*(2*n1max + 1), order='F')
        inds2 = np.tile(np.arange(-n2max, n2max + 1), 2 * n1max + 1)

        gvec = self.phc.lattice.b1[:, np.newaxis].dot(inds1[np.newaxis, :]) + \
                self.phc.lattice.b2[:, np.newaxis].dot(inds2[np.newaxis, :])

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
            (4 * np.pi * self.gmax) / np.linalg.norm(self.phc.lattice.b1))
        n2max = np.int_(
            (4 * np.pi * self.gmax) / np.linalg.norm(self.phc.lattice.b2))

        inds1 = np.tile(np.arange(-n1max, n1max + 1), (2*n2max + 1, 1))  \
                         .reshape((2*n2max + 1)*(2*n1max + 1), order='F')
        inds2 = np.tile(np.arange(-n2max, n2max + 1), 2 * n1max + 1)

        gvec = self.phc.lattice.b1[:, np.newaxis].dot(inds1[np.newaxis, :]) + \
                self.phc.lattice.b2[:, np.newaxis].dot(inds2[np.newaxis, :])
        gnorm = np.sqrt(gvec[0, :]**2 + gvec[1, :]**2)
        gvec = gvec[:, gnorm <= 2 * np.pi * self.gmax]

        # Save the reciprocal lattice vectors
        self._gvec = gvec

    def _get_guided(self, gk, kind, mode):
        """
        Get all the guided mode parameters over 'gk' for mode number 'mode'
        Variable 'indmode' stores the indexes of 'gk' over which a guided
        mode solution was found
        """
        def interp_coeff(coeffs, il, ic, indmode, gs):
            """
            Interpolate the A/B coefficient (ic = 0/1) in layer number il
            """
            param_list = [coeffs[i][il, ic, 0] for i in range(len(coeffs))]
            c_interp = bd.interp(gk[indmode], gs, bd.array(param_list))
            return c_interp.ravel()

        def interp_guided(im, ik, omegas, coeffs):
            """
            Interpolate all the relevant guided mode parameters over gk
            """
            gs = self.g_array[ik][-len(omegas[ik][im]):]
            indmode = np.argwhere(gk > gs[0] - 1e-10).ravel()
            oms = bd.interp(gk[indmode], gs, bd.array(omegas[ik][im]))
            e_a = self.eps_array if self.gradients == 'exact' \
                                    else self.eps_array_val
            chis = self._get_chi(gk[indmode], oms, e_a)

            As, Bs = [], []
            for il in range(self.N_layers + 2):
                As.append(interp_coeff(coeffs[ik][im], il, 0, indmode, gs))
                Bs.append(interp_coeff(coeffs[ik][im], il, 1, indmode, gs))
            As = bd.array(As, dtype=bd.complex)
            Bs = bd.array(Bs, dtype=bd.complex)

            return (indmode, oms, As, Bs, chis)

        ik = 0 if self.gmode_compute.lower() == 'interp' else kind

        if mode % 2 == 0:
            (indmode, oms, As, Bs,
             chis) = interp_guided(mode // 2, ik, self.omegas_te,
                                   self.coeffs_te)
        else:
            (indmode, oms, As, Bs,
             chis) = interp_guided(mode // 2, ik, self.omegas_tm,
                                   self.coeffs_tm)
        return (indmode, oms, As, Bs, chis)

    def _get_chi(self, gk, oms, eps_array):
        """
        Function to get the z-direction wave-vectors chi for in-plane wave
        vectors gk and frequencies oms (can be either a single number or an 
        array of the same shape as gk)
        """
        chis = []
        for il in range(self.N_layers + 2):
            sqarg = bd.array(eps_array[il] * bd.square(oms) - bd.square(gk),
                             dtype=bd.complex)
            chi = bd.where(
                bd.real(sqarg) >= 0, bd.sqrt(sqarg), 1j * bd.sqrt(-sqarg))
            chis.append(chi)
        return bd.array(chis, dtype=bd.complex)

    def _get_rad(self, gkr, omr, pol, clad):
        """
        Get all the radiative mode parameters over 'gkr' at frequency 'omr' with
        polarization 'pol' and out-going in cladding 'clad'
        """
        chis = self._get_chi(gkr, omr, self.eps_array)
        (Xs, Ys) = rad_modes(omr, gkr, self.eps_array, self.d_array, pol, clad)

        return (Xs, Ys, chis)

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

    def _compute_guided(self, g_array):
        """
        Compute the guided modes using the slab_modes module, reshape the 
        results appropriately and store
        """

        # Expand boundaries a bit to make sure we get all the modes
        # Note that the 'exact' computation still uses interpolation,
        # but the grid is defined by the actual gk values
        g_array -= 1e-6
        g_array[-1] += 2e-6
        self.g_array.append(g_array)
        self.gmode_te = self.gmode_inds[np.remainder(self.gmode_inds, 2) == 0]
        self.gmode_tm = self.gmode_inds[np.remainder(self.gmode_inds, 2) != 0]
        reshape_list = lambda x: [list(filter(lambda y: y is not None, i)) \
                        for i in zip_longest(*x)]

        if self.gradients == 'exact':
            (e_a, d_a) = (self.eps_array, self.d_array)
        elif self.gradients == 'approx':
            (e_a, d_a) = (self.eps_array_val, self.d_array_val)

        if self.gmode_te.size > 0:
            (omegas_te,
             coeffs_te) = guided_modes(g_array,
                                       e_a,
                                       d_a,
                                       step=self.gmode_step,
                                       n_modes=1 + np.amax(self.gmode_te) // 2,
                                       tol=self.gmode_tol,
                                       pol='TE')
            omte = reshape_list(omegas_te)
            self.omegas_te.append(reshape_list(omegas_te))
            self.coeffs_te.append(reshape_list(coeffs_te))

        if self.gmode_tm.size > 0:
            (omegas_tm,
             coeffs_tm) = guided_modes(g_array,
                                       e_a,
                                       d_a,
                                       step=self.gmode_step,
                                       n_modes=1 + np.amax(self.gmode_tm) // 2,
                                       tol=self.gmode_tol,
                                       pol='TM')
            self.omegas_tm.append(reshape_list(omegas_tm))
            self.coeffs_tm.append(reshape_list(coeffs_tm))

    def _compute_ft_tbt(self):
        """
        Compute the unique FT coefficients of the permittivity, eps(g-g') for
        every layer in the PhC, assuming TBT-initialized reciprocal lattice
        """
        (n1max, n2max) = (self.n1g, self.n2g)
        G1 = -self.gvec + self.gvec[:, [0]]
        G2 = np.zeros((2, n1max * n2max))

        # Initialize the FT coefficient lists; in the end the length of these
        # will be equal to the total number of layers in the PhC
        self.T1 = []
        self.T2 = []

        for ind1 in range(n1max):
            G2[:, ind1*n2max:(ind1+1)*n2max] = - self.gvec[:, [ind1*n2max]] + \
                            self.gvec[:, range(n2max)]

        for layer in [self.phc.claddings[0]] + self.phc.layers + \
                            [self.phc.claddings[1]]:
            T1 = layer.compute_ft(G1)
            T2 = layer.compute_ft(G2)

            # Store T1 and T2
            if bd.amax(bd.abs(
                    bd.imag(T1))) < 1e-10 * bd.amax(bd.abs(bd.real(T1))):
                self.T1.append(bd.real(T1))
            else:
                self.T1.append(T1)
            if bd.amax(bd.abs(
                    bd.imag(T2))) < 1e-10 * bd.amax(bd.abs(bd.real(T2))):
                self.T2.append(bd.real(T2))
            else:
                self.T2.append(T2)

        # Store the g-vectors to which T1 and T2 correspond
        self.G1 = G1
        self.G2 = G2

    def _compute_ft_abs(self):
        """
        Compute the unique FT coefficients of the permittivity, eps(g-g') for
        every layer in the PhC, assuming abs-initialized reciprocal lattice
        """
        ggridx = (self.gvec[0, :][np.newaxis, :] -
                  self.gvec[0, :][:, np.newaxis]).ravel()
        ggridy = (self.gvec[1, :][np.newaxis, :] -
                  self.gvec[1, :][:, np.newaxis]).ravel()

        self.eps_ft = []
        for layer in [self.phc.claddings[0]] + self.phc.layers + \
                            [self.phc.claddings[1]]:
            eps_ft = layer.compute_ft(np.vstack((ggridx, ggridy)))
            self.eps_ft.append(
                bd.reshape(eps_ft,
                           (self.gvec[0, :].size, self.gvec[0, :].size)))

    def _construct_mat(self, kind):
        """
        Construct the Hermitian matrix for diagonalization for a given k
        """

        # G + k vectors
        gkx = self.gvec[0, :] + self.kpoints[0, kind] + 1e-10
        gky = self.gvec[1, :] + self.kpoints[1, kind]
        gk = np.sqrt(np.square(gkx) + np.square(gky))

        # Compute the guided modes over gk if using the 'exact' method
        if self.gmode_compute.lower() == 'exact':
            t = time.time()
            g_array = np.sort(gk)
            self._compute_guided(g_array)
            self.t_guided += time.time() - t

        # Unit vectors in the propagation direction; we add a tiny component
        # in the x-direction to avoid problems at gk = 0
        pkx = gkx / gk
        pky = gky / gk

        # Unit vectors in-plane orthogonal to the propagation direction
        qkx = gky / gk
        qky = -gkx / gk

        pp = np.outer(pkx, pkx) + np.outer(pky, pky)
        pq = np.outer(pkx, qkx) + np.outer(pky, qky)
        qq = np.outer(qkx, qkx) + np.outer(qky, qky)

        # Loop over modes and build the matrix block-by-block
        modes_numg = []

        # Find the gmode_inds that actually enter the computation (due to the
        # gmax cutoff, only a finite number of mode indexes can enter)
        # Note: we might need to have a gmode_include for every kind
        gmode_include = []
        ik = 0 if self.gmode_compute.lower() == 'interp' else kind
        for mode in self.gmode_inds:
            if (mode%2==0 and len(self.omegas_te[ik]) > mode//2) \
                or (mode%2==1 and len(self.omegas_tm[ik]) > mode//2):
                gmode_include.append(mode)
        if gmode_include == []:
            raise RuntimeError(
                "No guided modes were found for k-index %d. "
                "One possibility is "
                "that the effective permittivity of all layers is smaller than "
                "that of at least one cladding. Reconsider your structure, or "
                "try changing 'eps_eff' from 'average' to 'background' in "
                "the options to GuidedModeExp.run()." % kind)
        else:
            self.gmode_include.append(np.array(gmode_include))

        # We now construct the matrix block by block
        mat_blocks = [[] for i in range(self.gmode_include[-1].size)]

        if self.gradients == 'exact':
            (e_a, d_a) = (self.eps_array, self.d_array)
        elif self.gradients == 'approx':
            (e_a, d_a) = (self.eps_array_val, self.d_array)

        for im1 in range(self.gmode_include[-1].size):
            mode1 = self.gmode_include[-1][im1]
            (indmode1, oms1, As1, Bs1, chis1) = \
                        self._get_guided(gk, kind, mode1)
            modes_numg.append(indmode1.size)

            if len(modes_numg) > 1:
                mat_blocks[im1].append(
                    bd.zeros((modes_numg[-1], bd.sum(modes_numg[:-1]))))

            for im2 in range(im1, self.gmode_include[-1].size):
                mode2 = self.gmode_include[-1][im2]
                (indmode2, oms2, As2, Bs2, chis2) = \
                            self._get_guided(gk, kind, mode2)

                if mode1 % 2 + mode2 % 2 == 0:
                    mat_block = matrix_elements.mat_te_te(
                        e_a, d_a, self.eps_inv_mat, indmode1, oms1, As1, Bs1,
                        chis1, indmode2, oms2, As2, Bs2, chis2, qq)
                elif mode1 % 2 + mode2 % 2 == 2:
                    mat_block = matrix_elements.mat_tm_tm(
                        e_a, d_a, self.eps_inv_mat, gk, indmode1, oms1, As1,
                        Bs1, chis1, indmode2, oms2, As2, Bs2, chis2, pp)
                elif mode1 % 2 == 0 and mode2 % 2 == 1:
                    mat_block = matrix_elements.mat_te_tm(
                        e_a, d_a, self.eps_inv_mat, indmode1, oms1, As1, Bs1,
                        chis1, indmode2, oms2, As2, Bs2, chis2, pq.transpose())
                elif mode1 % 2 == 1 and mode2 % 2 == 0:
                    mat_block = matrix_elements.mat_tm_te(
                        e_a, d_a, self.eps_inv_mat, indmode1, oms1, As1, Bs1,
                        chis1, indmode2, oms2, As2, Bs2, chis2, pq)

                mat_blocks[im1].append(mat_block)

        # Store how many modes total were included in the matrix
        self.N_basis.append(np.sum(modes_numg))
        # Store a list of how many g-points were used for each mode index
        self.modes_numg.append(modes_numg)

        # Stack all the blocks together
        mat = bd.vstack([bd.hstack(mb) for mb in mat_blocks])
        """
        If the matrix is within numerical precision to real symmetric, 
        make it explicitly so. This will speed up the diagonalization and will
        often be the case, specifically when there is in-plane inversion
        symmetry in the PhC elementary cell
        """
        # if bd.amax(bd.abs(bd.imag(mat))) < 1e-10*bd.amax(bd.abs(bd.real(mat))):
        #     mat = bd.real(mat)
        #     print(mat.dtype)
        """
        Make the matrix Hermitian (note that only upper part of the blocks, i.e.
        (im2 >= im1) was computed
        """
        return bd.triu(mat, 1) + bd.transpose(bd.conj(bd.triu(mat, 1))) + \
                bd.real(bd.diag(bd.diag(mat)))

    def compute_eps_inv(self):
        """
        Construct the inverse FT matrices for the permittivity in each layer
        """
        try:
            self.eps_inv_mat
        except AttributeError:
            # List of inverse permittivity matrices for every layer
            self.eps_inv_mat = []

            if self.truncate_g == 'tbt':
                for it, T1 in enumerate(self.T1):
                    self.hom_layer = []
                    # For now we just use the numpy inversion. Later on we could
                    # implement the Toeplitz-Block-Toeplitz inversion (faster)
                    if bd.sum(bd.abs(T1[1:])) < 1e-10:
                        self.eps_inv_mat.append(
                            bd.eye(T1.size, T1.size) / T1[0])
                        self.hom_layer.append(True)
                    else:
                        eps_mat = bd.toeplitz_block(self.n1g, T1, self.T2[it])
                        self.eps_inv_mat.append(bd.inv(eps_mat))
                        self.hom_layer.append(False)
            elif self.truncate_g == 'abs':
                for eps_mat in self.eps_ft:
                    self.eps_inv_mat.append(bd.inv(eps_mat))

    def set_run_options(self,
                        gmode_compute='exact',
                        gmode_inds: list = [0],
                        gmode_npts: int = 1000,
                        gmode_step: float = 1e-2,
                        gmode_tol: float = 1e-10,
                        numeig: int = 10,
                        compute_im: bool = True,
                        gradients='exact',
                        eig_solver='eigh',
                        eig_sigma: float = 0.,
                        eps_eff='average',
                        verbose: bool = True):
        """Set multiple options for the guided-mode expansion.
            
            Parameters
            ----------
            gmode_compute : {'exact', 'interp'}, optional
                Define whether the guided modes are computed exactly at every 
                k-step ('exact'), or computed in the beginning and then 
                interpolated ('interp').
            gmode_inds : list or np.ndarray, optional
                Indexes of modes to be included in the expansion. 
                Default is [0].
            gmode_npts : int, optional
                Number of points over which guided modes are computed. 
                Default is 1000.
            gmode_step : float, optional
                Step in frequency in the search for guided mode solutions.
                Default is 1e-2.
            gmode_tol : float, optional
                Tolerance in the minimum and maximum frequency value when 
                looking for the guided-mode solutions. Default is 1e-10.
            numeig : int, optional
                Number of eigen-frequencies to be stored (starting from lowest).
                Default is 10.
            compute_im : bool, optional
                Should the imaginary parts of the frequencies also be computed.
                Default is True. 
            gradients : {'exact', 'approx'}, optional
                Whether to compute 'exact' gradients, or 'approx' (faster).
            eig_solver : {'eigh', 'eigsh'}, optional
                Eigenvalue solver, 'eigh' (numpy.linalg.eigh) or 'eigsh'
                (scipy.sparse.linalg.eigsh)
            eig_sigma : float, optional
                Target eigenvalue; numeig eigenvalues closest to eig_sigma are
                stored. Default is 0.
            eps_eff : {'average', 'background', 'custom'}, optional
                Using the 'average', 'background' or a 'custom' permittivity of 
                the layers in the guided mode computation. If 'custom', all the
                photonic crystal layers, including the claddings, must have a 
                pre-set ``eps_eff``.
            verbose : bool, optional
                Print information at intermmediate steps. Default is True.
            """

        # Make a dictionary that stores all the options
        self._run_options = {
            'gmode_compute': gmode_compute,
            'gmode_inds': gmode_inds,
            'gmode_npts': gmode_npts,
            'gmode_step': gmode_step,
            'gmode_tol': gmode_tol,
            'numeig': numeig,
            'compute_im': compute_im,
            'gradients': gradients,
            'eig_solver': eig_solver,
            'eig_sigma': eig_sigma,
            'eps_eff': eps_eff,
            'verbose': verbose
        }

        # Also store the options as separate attributes
        for (option, value) in self._run_options.items():
            # Make sure 'gmode_inds' is a numpy array
            if option.lower() == 'gmode_inds':
                value = np.array(value)
            # Set all the options as class attributes
            setattr(self, option, value)

    def run(self, kpoints: np.ndarray = np.array([[0], [0]]), **kwargs):
        """
        Compute the eigenmodes of the photonic crystal structure.
        
        The guided-mode expansion proceeds as follows:

            Compute the inverse matrix of the fourier transform of the 
            permittivity in every phc layer.
            
            Iterate over the k points:
            
                Compute the guided modes over all the (g + k) points.

                Compute the Hermitian matrix for diagonalization.

                Compute the real part of the eigenvalues, stored in 
                :attr:`GuidedModeExp.freqs`, and the corresponding eigenvectors,
                stored in :attr:`GuidedModeExp.eigvecs`.

            If compute_im=True (as is default), run :meth:`GuidedModeExp.run_im`.
        
        Parameters
        ----------
        kpoints : np.ndarray, optional
            numpy array of shape (2, :) with the [kx, ky] coordinates of the 
            k-vectors over which the simulation is run.
        **kwargs
            All the keyword arguments that can be passed here, as well as their 
            default values, are defined in 
            :meth:`GuidedModeExp.set_run_options`.
        """

        # Set the default options and then overwrite with the user supplied
        self.set_run_options(**kwargs)

        t_start = time.time()

        # Bloch momenta over which band structure is simulated
        self._kpoints = kpoints

        self.modes_numg = []
        self.N_basis = []
        self.gmode_include = []

        # Array of effective permittivity of every layer (including claddings)
        if self.eps_eff == 'average':
            layer_eps = 'eps_avg'
        elif self.eps_eff == 'background':
            layer_eps = 'eps_b'
        elif self.eps_eff == 'custom':
            layer_eps = 'eps_eff'
        else:
            raise ValueError("'eps_eff' can be 'average', 'background' or "
                             "'custom'")

        # Store an array of the effective permittivity for every layer
        #(including claddings)
        eps_array = bd.array([
            bd.array(getattr(layer, layer_eps), dtype=bd.float).ravel() for layer in
            [self.phc.claddings[0]] + self.phc.layers + [self.phc.claddings[1]]]).ravel()
        # A separate array where the values are converted from ArrayBox to numpy
        # array, if using the 'autograd' backend.
        eps_array_val = np.array([
            np.float64(get_value(getattr(layer, layer_eps))) for layer in
            [self.phc.claddings[0]] + self.phc.layers + [self.phc.claddings[1]]]).ravel()

        # Store an array of thickness of every layer (not including claddings)
        d_array = bd.array(list(layer.d for layer in \
            self.phc.layers), dtype=bd.float).ravel()
        # A separate array where the values are converted from ArrayBox to numpy
        # array, if using the 'autograd' backend.
        d_array_val = np.array(list(get_value(layer.d) for layer in \
            self.phc.layers), dtype=np.float64).ravel()

        (self.eps_array_val, self.eps_array, self.d_array_val, self.d_array) = \
                                (eps_array_val, eps_array, d_array_val, d_array)

        # Initialize attributes for guided-mode storing
        self.g_array = []
        """
        List dimensions in order: 
            - k-point index (length 1 if gmode_compute=='interp')
            - guided mode index
            - g_array index
        """
        self.omegas_te = []
        self.coeffs_te = []
        self.omegas_tm = []
        self.coeffs_tm = []

        # Pre-compute guided modes if the 'interp' option is used
        if self.gmode_compute.lower() == 'interp':
            t = time.time()
            kmax = np.amax(
                np.sqrt(np.square(kpoints[0, :]) + np.square(kpoints[1, :])))
            Gmax = np.amax(
                np.sqrt(
                    np.square(self.gvec[0, :]) + np.square(self.gvec[1, :])))

            # Array of g-points over which the guided modes will be computed
            g_array = np.linspace(0, Gmax + kmax, self.gmode_npts)
            self._compute_guided(g_array)
            self.t_guided = time.time() - t

        else:
            self.t_guided = 0
        self.t_eig = 0  # For timing of the diagonalization

        # Compute inverse matrix of FT of permittivity
        t = time.time()
        self.compute_eps_inv()
        t_eps_inv = time.time() - t

        # Loop over all k-points, construct the matrix, diagonalize, and compute
        # radiative losses for the modes requested by kinds_rad and minds_rad
        t_rad = 0
        freqs = []
        freqs_im = []
        self._eigvecs = []
        for ik, k in enumerate(kpoints.T):

            self._print("Running k-point %d of %d" %
                        (ik + 1, kpoints.shape[1]),
                        flush=True)
            mat = self._construct_mat(kind=ik)
            if self.numeig > mat.shape[0]:
                raise ValueError(
                    "Requested number of eigenvalues 'numeig' "
                    "larger than total size of basis set. Reduce 'numeig' or "
                    "increase 'gmax'")

            # NB: we shift the matrix by np.eye to avoid problems at the zero-
            # frequency mode at Gamma
            t_eig = time.time()
            if self.eig_solver == 'eigh':
                (freq2, evecs) = bd.eigh(mat + bd.eye(mat.shape[0]))
                freq1 = bd.sqrt(
                    bd.abs(freq2 - bd.ones(mat.shape[0]))) / 2 / np.pi
                i_near = find_nearest(get_value(freq1), self.eig_sigma,
                                      self.numeig)
                i_sort = bd.argsort(freq1[i_near])
                freq = freq1[i_near[i_sort]]
                evec = evecs[:, i_near[i_sort]]
            elif self.eig_solver == 'eigsh':
                (freq2,
                 evecs) = bd.eigsh(mat + bd.eye(mat.shape[0]),
                                   k=self.numeig,
                                   sigma=(self.eig_sigma * 2 * np.pi)**2 + 1)
                freq1 = bd.sqrt(
                    bd.abs(freq2 - bd.ones(self.numeig))) / 2 / np.pi
                i_sort = bd.argsort(freq1)
                freq = freq1[i_sort]
                evec = evecs[:, i_sort]
            else:
                raise ValueError("'eig_solver' can be 'eigh' or 'eigsh'")
            self.t_eig += time.time() - t_eig

            freqs.append(freq)
            self._eigvecs.append(evec)

        # Store the eigenfrequencies taking the standard reduced frequency
        # convention for the units (2pi a/c)
        self._freqs = bd.array(freqs)

        self._print("", flush=True)
        self._print(
            "%1.4fs total time for real part of frequencies, of which" %
            (time.time() - t_start))
        self._print("  %1.4fs for guided modes computation using"
                    " the gmode_compute='%s' method" %
                    (self.t_guided, self.gmode_compute.lower()))
        self._print("  %1.4fs for inverse matrix of Fourier-space "
                    "permittivity" % t_eps_inv)
        self._print(
            "  %1.4fs for matrix diagionalization using the '%s' solver" %
            (self.t_eig, self.eig_solver.lower()))

        if self.compute_im == True:
            t = time.time()
            self.run_im()
            self._print("%1.4fs for imaginary part computation" %
                        (time.time() - t))
        else:
            self._print(
                "Skipping imaginary part computation, use run_im() to"
                " run it, or compute_rad() to compute the radiative rates"
                " of selected eigenmodes")

    def run_im(self):
        """
        Compute the radiative rates associated to all the eigenmodes that were 
        computed during :meth:`GuidedModeExp.run`. Results are stored in 
        :attr:`GuidedModeExp.freqs_im`, :attr:`GuidedModeExp.rad_coup`, and 
        :attr:`GuidedModeExp.rad_gvec`.
        """
        if len(self.freqs) == 0:
            raise RuntimeError("Run the GME computation first!")

        freqs_i = []  # Imaginary part of frequencies

        # Coupling constants to lower- and upper-cladding radiative modes
        rad_coup = {'l_te': [], 'l_tm': [], 'u_te': [], 'u_tm': []}
        rad_gvec = {'l': [], 'u': []}

        for kind in range(len(self.freqs)):
            minds = np.arange(0, self.numeig)
            (freqs_im, rc, rv) = self.compute_rad(kind, minds)
            freqs_i.append(freqs_im)
            for clad in ['l', 'u']:
                rad_coup[clad + '_te'].append(rc[clad + '_te'])
                rad_coup[clad + '_tm'].append(rc[clad + '_tm'])
                rad_gvec[clad].append(rv[clad])

        self._freqs_im = bd.array(freqs_i)
        self._rad_coup = rad_coup
        self._rad_gvec = rad_gvec

    def compute_rad(self, kind: int, minds: list = [0]):
        """
        Compute the radiation losses of the eigenmodes after the dispersion
        has been computed.
        
        Parameters
        ----------
        kind : int
            Index of the k-point for the computation.
        minds : list, optional
            Indexes of which modes to be computed. Max value must be smaller 
            than `GuidedModeExp.numeig` set in :meth:`GuidedModeExp.run`.
        
        Returns
        -------
        freqs_im : np.ndarray
            Imaginary part of the frequencies of the eigenmodes computed by the 
            guided-mode expansion.
        rad_coup : dict 
            Coupling to TE and TM radiative modes in the lower/upper cladding.
        rad_gvec : dict
            Reciprocal lattice vectors in the lower/upper cladding 
            corresponding to ``rad_coup``.
        """
        if len(self.freqs) == 0:
            raise RuntimeError("Run the GME computation first!")
        if np.max(np.array(minds)) > self.numeig - 1:
            raise ValueError("Requested mode index out of range for the %d "
                             "stored eigenmodes" % self.numeig)

        # G + k vectors
        gkx = self.gvec[0, :] + self.kpoints[0, kind] + 1e-10
        gky = self.gvec[1, :] + self.kpoints[1, kind]
        gk = np.sqrt(np.square(gkx) + np.square(gky))

        # Unit vectors in the propagation direction; we add a tiny component
        # in the x-direction to avoid problems at gk = 0
        pkx = gkx / gk
        pky = gky / gk

        # Unit vectors in-plane orthogonal to the propagation direction
        qkx = gky / gk
        qky = -gkx / gk

        pp = np.outer(pkx, pkx) + np.outer(pky, pky)
        pq = np.outer(pkx, qkx) + np.outer(pky, qky)
        qq = np.outer(qkx, qkx) + np.outer(qky, qky)

        # Variables to store the results
        rad_tot = []
        rad_gvec = {'l': [], 'u': []}
        rad_coup = {'l_te': [], 'l_tm': [], 'u_te': [], 'u_tm': []}
        # Iterate over all the modes to be computed
        for im in minds:
            omr = 2 * np.pi * self.freqs[kind, im]
            evec = self.eigvecs[kind][:, im]

            # Reciprocal vedctors within the radiative cone for the claddings
            indmoder = [bd.argwhere(gk**2 <= \
                    self.phc.claddings[0].eps_avg*omr**2).ravel(),
                        bd.argwhere(gk**2 <= \
                    self.phc.claddings[1].eps_avg*omr**2).ravel()
                        ]
            gkr = [gk[indmode] for indmode in indmoder]

            # Coupling constants to TE/TM modes in lower and upper cladding
            rad_c = {
                'te': [
                    bd.zeros((indmode.size, ), dtype=bd.complex)
                    for indmode in indmoder
                ],
                'tm': [
                    bd.zeros((indmode.size, ), dtype=bd.complex)
                    for indmode in indmoder
                ]
            }

            # Compute leaky modes
            [Xs, Ys, chis] = [{'te': [], 'tm': []} for i in range(3)]
            for clad_ind in [0, 1]:
                for pol in ['te', 'tm']:
                    (X, Y, chi) = self._get_rad(gkr[clad_ind],
                                                omr,
                                                pol=pol,
                                                clad=clad_ind)
                    Xs[pol].append(X)
                    Ys[pol].append(Y)
                    chis[pol].append(chi)

            if self.gradients == 'exact':
                (e_a, d_a) = (self.eps_array, self.d_array)
            elif self.gradients == 'approx':
                (e_a, d_a) = (self.eps_array_val, self.d_array)

            # Iterate over the 'gmode_include' basis of the PhC mode
            count = 0
            for im1 in range(self.gmode_include[kind].size):
                mode1 = self.gmode_include[kind][im1]
                (indmode1, oms1, As1, Bs1, chis1) = \
                            self._get_guided(gk, kind, mode1)
                # Iterate over lower cladding (0) and upper cladding (1)
                for clad_ind in [0, 1]:
                    omr_arr = omr * bd.ones((indmoder[clad_ind].size, ))
                    # Radiation to TE-polarized states
                    if mode1 % 2 == 0:
                        # TE-TE coupling
                        rad = matrix_elements.mat_te_te(
                            e_a, d_a, self.eps_inv_mat, indmode1, oms1, As1,
                            Bs1, chis1, indmoder[clad_ind], omr_arr,
                            Ys['te'][clad_ind], Xs['te'][clad_ind],
                            chis['te'][clad_ind], qq)
                    else:
                        # TM-TE coupling
                        rad = matrix_elements.mat_tm_te(
                            e_a, d_a, self.eps_inv_mat, indmode1, oms1, As1,
                            Bs1, chis1, indmoder[clad_ind], omr_arr,
                            Ys['te'][clad_ind], Xs['te'][clad_ind],
                            chis['te'][clad_ind], pq)
                    # print(kind, im, indmode1.shape, self.modes_numg[kind][im1])
                    rad = rad * bd.conj(
                        evec[count:count +
                             self.modes_numg[kind][im1]][:, np.newaxis])
                    # We divide by 1j so that the coupling constants are w.r.t.
                    # TE and TM waves with E field that is in-phase
                    # (See equations in legume paper where the TE fields are
                    # proportional to i, we're just getting rid of that here)
                    rad_c['te'][clad_ind] += -1j * bd.sum(rad, axis=0)

                    # Radiation to TM-polarized states
                    if mode1 % 2 == 0:
                        # TE-TM coupling
                        rad = matrix_elements.mat_te_tm(
                            e_a, d_a, self.eps_inv_mat, indmode1, oms1, As1,
                            Bs1, chis1, indmoder[clad_ind], omr_arr,
                            Ys['tm'][clad_ind], Xs['tm'][clad_ind],
                            chis['tm'][clad_ind], bd.transpose(pq))
                    else:
                        # TM-TM coupling
                        rad = matrix_elements.mat_tm_tm(
                            e_a, d_a, self.eps_inv_mat, gk, indmode1, oms1,
                            As1, Bs1, chis1, indmoder[clad_ind], omr_arr,
                            Ys['tm'][clad_ind], Xs['tm'][clad_ind],
                            chis['tm'][clad_ind], pp)

                    # Multiply the overlap and the expansion coefficients
                    rad = rad * bd.conj(
                        evec[count:count +
                             self.modes_numg[kind][im1]][:, np.newaxis])
                    # Add everything up
                    rad_c['tm'][clad_ind] += bd.sum(rad, axis=0)
                count += self.modes_numg[kind][im1]

            # Density of states of leaky modes
            rad_dos = [self.phc.claddings[i].eps_avg/bd.sqrt(
                    self.phc.claddings[i].eps_avg*omr**2 - gkr[i]**2) / \
                    4 / np.pi for i in [0, 1]]

            # Store the reciprocal lattice vectors corresponding to the
            # radiation channels (diffraction orders)
            rad_gvec['l'].append(self.gvec[:, indmoder[0]])
            rad_gvec['u'].append(self.gvec[:, indmoder[1]])

            rad_t = 0  # variable suming up contributions from all the channels
            (c_l, c_u) = ({}, {})
            for pol in ['te', 'tm']:
                # Couplings normalized such that Im(omega^2/c^2) is equal to
                # sum(square(abs(c_l))) + sum(square(abs(c_u)))
                c_l[pol] = bd.sqrt(np.pi * rad_dos[0]) * rad_c[pol][0]
                c_u[pol] = bd.sqrt(np.pi * rad_dos[1]) * rad_c[pol][1]
                rad_t = rad_t + \
                    bd.sum(bd.square(bd.abs(c_l[pol]))) + \
                    bd.sum(bd.square(bd.abs(c_u[pol])))

                # Store the coupling constants (they are effectively in units
                # of angular frequency omega)
                rad_coup['l_' + pol].append(c_l[pol])
                rad_coup['u_' + pol].append(c_u[pol])

            rad_tot.append(bd.imag(bd.sqrt(omr**2 + 1j * rad_t)))

        # Compute radiation rate in units of frequency
        freqs_im = bd.array(rad_tot) / 2 / np.pi
        return (freqs_im, rad_coup, rad_gvec)

    def get_eps_xy(self, z: float, xgrid=None, ygrid=None, Nx=100, Ny=100):
        """
        Get the xy-plane permittivity of the PhC at a given z as computed from 
        an inverse Fourier transform with the GME reciprocal lattice vectors.
        
        Parameters
        ----------
        z : float
            Position of the xy-plane.
        xgrid : None, optional
            None or a 1D np.array defining a grid in x.
        ygrid : None, optional
            None or a 1D np.array defining a grid in y.
        Nx : int, optional
            If xgrid==None, a grid of Nx points in the elementary cell is 
            created.
        Ny : int, optional
            If ygrid==None, a grid of Ny points in the elementary cell is 
            created.
        
        Returns
        -------
        eps_r : np.ndarray
            The in-plane real-space permittivity.
        xgrid : np.ndarray
            The input or constructed grid in x.
        ygrid : np.ndarray
            The input or constructed grid in y.
        """

        # Make a grid in the x-y plane
        if xgrid is None or ygrid is None:
            (xgr, ygr) = self.phc.lattice.xy_grid(Nx=Nx, Ny=Ny)
            if xgrid is None:
                xgrid = xgr
            if ygrid is None:
                ygrid = ygr

        # Layer index where z lies
        lind = self._z_to_lind(z)

        ft_coeffs = np.hstack((self.T1[lind], self.T2[lind],
                               np.conj(self.T1[lind]), np.conj(self.T2[lind])))
        gvec = np.hstack((self.G1, self.G2, -self.G1, -self.G2))

        eps_r = ftinv(ft_coeffs, gvec, xgrid, ygrid)

        return (eps_r, xgrid, ygrid)

    def ft_field_xy(self, field, kind, mind, z):
        """
        Compute the 'H', 'D' or 'E' field Fourier components in the xy-plane at 
        position z.
        
        Parameters
        ----------
        field : {'H', 'D', 'E'}
            The field to be computed. 
        kind : int
            The field of the mode at `GuidedModeExp.kpoints[:, kind]` is 
            computed.
        mind : int
            The field of the `mind` mode at that kpoint is computed.
        z : float
            Position of the xy-plane.

        Note
        ----
        The function outputs 1D arrays with the same size as 
        `GuidedModeExp.gvec[0, :]` corresponding to the G-vectors in 
        that array.
        
        Returns
        -------
        fi_x : np.ndarray
            The Fourier transform of the x-component of the specified field. 
        fi_y : np.ndarray
            The Fourier transform of the y-component of the specified field. 
        fi_z : np.ndarray
            The Fourier transform of the z-component of the specified field. 

        """
        evec = self.eigvecs[kind][:, mind]
        omega = self.freqs[kind][mind] * 2 * np.pi
        k = self.kpoints[:, kind]

        # G + k vectors
        gkx = self.gvec[0, :] + k[0] + 1e-10
        gky = self.gvec[1, :] + k[1]
        gnorm = bd.sqrt(bd.square(gkx) + bd.square(gky))

        # Unit vectors in the propagation direction
        px = gkx / gnorm
        py = gky / gnorm

        # Unit vectors in-plane orthogonal to the propagation direction
        qx = py
        qy = -px

        lind = self._z_to_lind(z)

        if field.lower() == 'h':
            count = 0
            [Hx_ft, Hy_ft, Hz_ft
             ] = [bd.zeros(gnorm.shape, dtype=bd.complex) for i in range(3)]
            for im1 in range(self.gmode_include[kind].size):
                mode1 = self.gmode_include[kind][im1]
                (indmode, oms, As, Bs, chis) = \
                            self._get_guided(gnorm, kind, mode1)

                # TE-component
                if mode1 % 2 == 0:
                    # Do claddings separately
                    if lind == 0:
                        H = Bs[0, :] * bd.exp(
                            -1j * chis[0, :] *
                            (z - self.phc.claddings[0].z_max))
                        Hx = H * 1j * chis[0, :] * px[indmode]
                        Hy = H * 1j * chis[0, :] * py[indmode]
                        Hz = H * 1j * gnorm[indmode]
                    elif lind == self.eps_array.size - 1:
                        H = As[-1, :] * bd.exp(
                            1j * chis[-1, :] *
                            (z - self.phc.claddings[1].z_min))
                        Hx = -H * 1j * chis[-1, :] * px[indmode]
                        Hy = -H * 1j * chis[-1, :] * py[indmode]
                        Hz = H * 1j * gnorm[indmode]
                    else:
                        z_cent = (self.phc.layers[lind - 1].z_min +
                                  self.phc.layers[lind - 1].z_max) / 2
                        zp = bd.exp(1j * chis[lind, :] * (z - z_cent))
                        zn = bd.exp(-1j * chis[lind, :] * (z - z_cent))
                        Hxy = -1j * As[lind, :] * zp + 1j * Bs[lind, :] * zn
                        Hx = Hxy * chis[lind, :] * px[indmode]
                        Hy = Hxy * chis[lind, :] * py[indmode]
                        Hz = 1j*(As[lind, :]*zp + Bs[lind, :]*zn) *\
                                gnorm[indmode]

                # TM-component
                elif mode1 % 2 == 1:
                    Hz = bd.zeros(indmode.shape, dtype=bd.complex)
                    # Do claddings separately
                    if lind == 0:
                        H = Bs[0, :] * bd.exp(
                            -1j * chis[0, :] *
                            (z - self.phc.claddings[0].z_max))
                        Hx = H * qx[indmode]
                        Hy = H * qy[indmode]
                    elif lind == self.eps_array.size - 1:
                        H = As[-1, :] * bd.exp(
                            1j * chis[-1, :] *
                            (z - self.phc.claddings[1].z_min))
                        Hx = H * qx[indmode]
                        Hy = H * qy[indmode]
                    else:
                        z_cent = (self.phc.layers[lind - 1].z_min +
                                  self.phc.layers[lind - 1].z_max) / 2
                        zp = bd.exp(1j * chis[lind, :] * (z - z_cent))
                        zn = bd.exp(-1j * chis[lind, :] * (z - z_cent))
                        Hxy = As[lind, :] * zp + Bs[lind, :] * zn
                        Hx = Hxy * qx[indmode]
                        Hy = Hxy * qy[indmode]

                valsx = evec[count:count+self.modes_numg[kind][im1]]*\
                                    Hx/bd.sqrt(self.phc.lattice.ec_area)
                Hx_ft = Hx_ft + bd.extend(valsx, indmode, Hx_ft.shape)
                valsy = evec[count:count+self.modes_numg[kind][im1]]*\
                                    Hy/bd.sqrt(self.phc.lattice.ec_area)
                Hy_ft = Hy_ft + bd.extend(valsy, indmode, Hy_ft.shape)
                valsz = evec[count:count+self.modes_numg[kind][im1]]*\
                                    Hz/bd.sqrt(self.phc.lattice.ec_area)
                Hz_ft = Hz_ft + bd.extend(valsz, indmode, Hz_ft.shape)
                count += self.modes_numg[kind][im1]

            return (Hx_ft, Hy_ft, Hz_ft)

        elif field.lower() == 'd' or field.lower() == 'e':
            count = 0
            [Dx_ft, Dy_ft, Dz_ft
             ] = [bd.zeros(gnorm.shape, dtype=bd.complex) for i in range(3)]
            for im1 in range(self.gmode_include[kind].size):
                mode1 = self.gmode_include[kind][im1]
                (indmode, oms, As, Bs, chis) = \
                            self._get_guided(gnorm, kind, mode1)

                # TE-component
                if mode1 % 2 == 0:
                    Dz = bd.zeros(indmode.shape, dtype=bd.complex)
                    # Do claddings separately
                    if lind == 0:
                        D = 1j * Bs[0, :] * oms**2 / omega * \
                            self.eps_array[0] * bd.exp(-1j*chis[0, :] * \
                            (z-self.phc.claddings[0].z_max))
                        Dx = D * qx[indmode]
                        Dy = D * qy[indmode]
                    elif lind == self.eps_array.size - 1:
                        D = 1j * As[-1, :] * oms**2 / omega * \
                            self.eps_array[-1] * bd.exp(1j*chis[-1, :] * \
                            (z-self.phc.claddings[1].z_min))
                        Dx = D * qx[indmode]
                        Dy = D * qy[indmode]
                    else:
                        z_cent = (self.phc.layers[lind - 1].z_min +
                                  self.phc.layers[lind - 1].z_max) / 2
                        zp = bd.exp(1j * chis[lind, :] * (z - z_cent))
                        zn = bd.exp(-1j * chis[lind, :] * (z - z_cent))
                        Dxy = 1j*oms**2 / omega * \
                            self.eps_array[lind] * \
                            (As[lind, :]*zp + Bs[lind, :]*zn)
                        Dx = Dxy * qx[indmode]
                        Dy = Dxy * qy[indmode]

                # TM-component
                elif mode1 % 2 == 1:
                    if lind == 0:
                        D = 1j / omega * Bs[0,:] * \
                            bd.exp(-1j*chis[0,:] * \
                            (z-self.phc.claddings[0].z_max))
                        Dx = D * 1j * chis[0, :] * px[indmode]
                        Dy = D * 1j * chis[0, :] * py[indmode]
                        Dz = D * 1j * gnorm[indmode]
                    elif lind == self.eps_array.size - 1:
                        D = 1j / omega * As[-1,:] * \
                            bd.exp(1j*chis[-1, :] * \
                            (z-self.phc.claddings[1].z_min))
                        Dx = -D * 1j * chis[-1, :] * px[indmode]
                        Dy = -D * 1j * chis[-1, :] * py[indmode]
                        Dz = D * 1j * gnorm[indmode]
                    else:
                        z_cent = (self.phc.layers[lind - 1].z_min +
                                  self.phc.layers[lind - 1].z_max) / 2
                        zp = bd.exp(1j * chis[lind, :] * (z - z_cent))
                        zn = bd.exp(-1j * chis[lind, :] * (z - z_cent))
                        Dxy = 1 / omega * chis[lind, :] * \
                            (As[lind, :]*zp - Bs[lind, :]*zn)
                        Dx = Dxy * px[indmode]
                        Dy = Dxy * py[indmode]
                        Dz = -1 / omega * gnorm[indmode] * \
                            (As[lind, :]*zp + Bs[lind, :]*zn)

                valsx = evec[count:count+self.modes_numg[kind][im1]]*\
                                    Dx/bd.sqrt(self.phc.lattice.ec_area)
                Dx_ft = Dx_ft + bd.extend(valsx, indmode, Dx_ft.shape)
                valsy = evec[count:count+self.modes_numg[kind][im1]]*\
                                    Dy/bd.sqrt(self.phc.lattice.ec_area)
                Dy_ft = Dy_ft + bd.extend(valsy, indmode, Dy_ft.shape)
                valsz = evec[count:count+self.modes_numg[kind][im1]]*\
                                    Dz/bd.sqrt(self.phc.lattice.ec_area)
                Dz_ft = Dz_ft + bd.extend(valsz, indmode, Dz_ft.shape)
                count += self.modes_numg[kind][im1]

            if field.lower() == 'd':
                return (Dx_ft, Dy_ft, Dz_ft)
            else:
                # Get E-field by convolving FT(1/eps) with FT(D)
                Ex_ft = bd.dot(self.eps_inv_mat[lind], Dx_ft)
                Ey_ft = bd.dot(self.eps_inv_mat[lind], Dy_ft)
                Ez_ft = bd.dot(self.eps_inv_mat[lind], Dz_ft)
                return (Ex_ft, Ey_ft, Ez_ft)

    def get_field_xy(self,
                     field,
                     kind,
                     mind,
                     z,
                     xgrid=None,
                     ygrid=None,
                     component='xyz',
                     Nx=100,
                     Ny=100):
        """
        Compute the 'H', 'D' or 'E' field components in the xy-plane at 
        position z.
        
        Parameters
        ----------
        field : {'H', 'D', 'E'}
            The field to be computed. 
        kind : int
            The field of the mode at `GuidedModeExp.kpoints[:, kind]` is 
            computed.
        mind : int
            The field of the `mind` mode at that kpoint is computed.
        z : float
            Position of the xy-plane.
        xgrid : None, optional
            None or a 1D np.array defining a grid in x.
        ygrid : None, optional
            None or a 1D np.array defining a grid in y.
        component : str, optional
            A string containing 'x', 'y', and/or 'z'
        Nx : int, optional
            If xgrid==None, a grid of Nx points in the elementary cell is 
            created.
        Ny : int, optional
            If ygrid==None, a grid of Ny points in the elementary cell is 
            created.
        
        Returns
        -------
        fi : dict
            A dictionary with the requested components, 'x', 'y', and/or 'z'.
        xgrid : np.ndarray
            The input or constructed grid in x.
        ygrid : np.ndarray
            The input or constructed grid in y.
        """

        # Make a grid in the x-y plane
        if xgrid is None or ygrid is None:
            (xgr, ygr) = self.phc.lattice.xy_grid(Nx=Nx, Ny=Ny)
            if xgrid is None:
                xgrid = xgr
            if ygrid is None:
                ygrid = ygr

        # Get the field fourier components
        ft, fi = {}, {}
        (ft['x'], ft['y'], ft['z']) = self.ft_field_xy(field, kind, mind, z)

        for comp in component:
            if comp in ft.keys():
                if not (comp in fi.keys()):
                    fi[comp] = ftinv(ft[comp], self.gvec, xgrid, ygrid)
            else:
                raise ValueError("'component' can be any combination of "
                                 "'x', 'y', and 'z' only.")

        return (fi, xgrid, ygrid)

    def get_field_xz(self,
                     field,
                     kind,
                     mind,
                     y,
                     xgrid=None,
                     zgrid=None,
                     component='xyz',
                     Nx=100,
                     Nz=100,
                     dist=1.):
        """
        Compute the 'H', 'D' or 'E' field components in the xz-plane at 
        position y.
        
        Parameters
        ----------
        field : {'H', 'D', 'E'}
            The field to be computed. 
        kind : int
            The field of the mode at `GuidedModeExp.kpoints[:, kind]` is 
            computed.
        mind : int
            The field of the `mind` mode at that kpoint is computed.
        y : float
            Position of the xz-plane.
        xgrid : None, optional
            None or a 1D np.array defining a grid in x.
        zgrid : None, optional
            None or a 1D np.array defining a grid in z.
        component : str, optional
            A string containing 'x', 'y', and/or 'z'
        Nx : int, optional
            If xgrid==None, a grid of Nx points in the elementary cell is 
            created.
        Nz : int, optional
            If zgrid==None, a grid of Nz points in the elementary cell is 
            created.
        
        Returns
        -------
        fi : dict
            A dictionary with the requested components, 'x', 'y', and/or 'z'.
        xgrid : np.ndarray
            The input or constructed grid in x.
        zgrid : np.ndarray
            The input or constructed grid in z.
        """

        if xgrid is None:
            xgrid = self.phc.lattice.xy_grid(Nx=Nx, Ny=2)[0]
        ygrid = np.array([y])
        if zgrid is None:
            zgrid = self.phc.z_grid(Nz=Nz, dist=dist)

        # Get the field components
        ft = {'x': [], 'y': [], 'z': []}
        fi = {}
        for i, z in enumerate(zgrid):
            (fx, fy, fz) = self.ft_field_xy(field, kind, mind, z)
            ft['x'].append(fx)
            ft['y'].append(fy)
            ft['z'].append(fz)

        for comp in component:
            if comp in ft.keys():
                if not (comp in fi.keys()):
                    fi[comp] = []
                    for i, z in enumerate(zgrid):
                        fi[comp].append(
                            ftinv(ft[comp][i], self.gvec, xgrid,
                                  ygrid).ravel())
                    fi[comp] = bd.array(fi[comp])
            else:
                raise ValueError("'component' can be any combination of "
                                 "'x', 'y', and 'z' only.")

        return (fi, xgrid, zgrid)

    def get_field_yz(self,
                     field,
                     kind,
                     mind,
                     x,
                     ygrid=None,
                     zgrid=None,
                     component='xyz',
                     Ny=100,
                     Nz=100,
                     dist=1.):
        """
        Compute the 'H', 'D' or 'E' field components in the yz-plane at 
        position x.
        
        Parameters
        ----------
        field : {'H', 'D', 'E'}
            The field to be computed. 
        kind : int
            The field of the mode at `GuidedModeExp.kpoints[:, kind]` is 
            computed.
        mind : int
            The field of the `mind` mode at that kpoint is computed.
        x : float
            Position of the yz-plane.
        ygrid : None, optional
            None or a 1D np.array defining a grid in y.
        zgrid : None, optional
            None or a 1D np.array defining a grid in z.
        component : str, optional
            A string containing 'x', 'y', and/or 'z'
        Ny : int, optional
            If xgrid==None, a grid of Ny points in the elementary cell is 
            created.
        Nz : int, optional
            If ygrid==None, a grid of Nz points in the elementary cell is 
            created.
        
        Returns
        -------
        fi : dict
            A dictionary with the requested components, 'x', 'y', and/or 'z'.
        ygrid : np.ndarray
            The input or constructed grid in y.
        zgrid : np.ndarray
            The input or constructed grid in z.
        """
        xgrid = np.array([x])
        if ygrid is None:
            ygrid = self.phc.lattice.xy_grid(Nx=2, Ny=Ny)[1]
        if zgrid is None:
            zgrid = self.phc.z_grid(Nz=Nz, dist=dist)

        # Get the field components
        ft = {'x': [], 'y': [], 'z': []}
        fi = {}
        for i, z in enumerate(zgrid):
            (fx, fy, fz) = self.ft_field_xy(field, kind, mind, z)
            ft['x'].append(fx)
            ft['y'].append(fy)
            ft['z'].append(fz)

        for comp in component:
            if comp in ft.keys():
                if not (comp in fi.keys()):
                    fi[comp] = []
                    for i, z in enumerate(zgrid):
                        fi[comp].append(
                            ftinv(ft[comp][i], self.gvec, xgrid,
                                  ygrid).ravel())
                    fi[comp] = bd.array(fi[comp])
            else:
                raise ValueError("'component' can be any combination of "
                                 "'x', 'y', and 'z' only.")

        return (fi, ygrid, zgrid)
