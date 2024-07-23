import numpy as np
from scipy.linalg import block_diag
from scipy.sparse import coo_matrix as coo  # Older versions of Scipy have coo_matrix attribute only

import time
from itertools import zip_longest
from typing import Optional

from .slab_modes import guided_modes, rad_modes
from . import matrix_elements
from legume.backend import backend as bd
from legume.print_backend import print_backend as prbd
from legume.utils import get_value, ftinv, find_nearest, z_to_lind
from legume.print_utils import verbose_print, load_bar


class GuidedModeExp(object):
    """
    Main simulation class of the guided-mode expansion.
    """

    def __init__(self, phc, gmax=3., truncate_g='abs'):
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
        # An integer gmax can cause trouble with symmetry separation wrt vertical plane

        self.gmax = gmax
        self.truncate_g = truncate_g

        # Number of layers in the PhC
        self.N_layers = len(phc.layers)

        # Parameters below are defined when self.run() is called
        # Number of G points included for every mode, will be defined after run
        self.modes_numg = []
        #Index of G points included for every mode
        self.ind_modes = []
        # Total number of basis vectors (equal to np.sum(self.modes_numg))
        self.N_basis = []
        # Indexes of guided modes which are actually included in the computation
        # (in case gmode_inds includes modes that are above the gmax cutoff)
        self.gmode_include = []

        # Initialize all the attributes defined as properties below
        self._symm = []
        self._freqs = []
        self._freqs_im = []
        self._unbalance_sp = []
        self._eigvecs = []
        self._rad_coup = {}
        self._rad_gvec = {}
        self._kpoints = []
        self._gvec = []

        # Define direction angles in degrees of high symmetry lines for each lattice
        self._square_an = np.arange(-360, 405, 45)
        self._hex_an = np.arange(-360, 390, 30)
        self._rec_an = np.arange(-360, 450, 90)

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
    def kz_symms(self):
        """Symmetry of the eigenmodes computed by the 
        guided-mode expansion  w.r.t. the vertical kz plane.
        """
        return self._kz_symms

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
    def unbalance_sp(self):
        """Unbalance between the two addends in the summation
            of the imaginary part of the energy. The two addends
            corresponds to s (te) coupling component, and 
            p (tm) component. 

            When ``unbalance_sp=``: 

            * 1 -> all coupling comes from s (te) wave
            * 0 -> all coupling comes from p (tm) wave
            * 0.5 -> half coupling from s (te) waves and half from p (tm) waves
            
            Note
            ----
            If ``freqs_im=0`` then ``unbalance_sp = 0.5``.
        """
        return self._unbalance_sp

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
        """Reciprocal lattice vectors corresponding to the radiation emission 
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

        # Add dimension to indexes arrays
        #inds1 = inds1[np.newaxis, :]
        #inds2 = inds2[np.newaxis, :]
        gvec = self.phc.lattice.b1[:, np.newaxis].dot(inds1[np.newaxis, :]) + \
                self.phc.lattice.b2[:, np.newaxis].dot(inds2[np.newaxis, :])
        gnorm = np.sqrt(gvec[0, :]**2 + gvec[1, :]**2)

        # Avoid to cut with gmax equal to one of the |G|
        if self.gmax * 2 * np.pi in gnorm:
            gmax = self.gmax
            while gmax * 2 * np.pi in gnorm:
                gmax += 1e-10

            gvec = gvec[:, gnorm <= 2 * np.pi * gmax]
            inds1 = inds1[gnorm <= 2 * np.pi * gmax]
            inds2 = inds2[gnorm <= 2 * np.pi * gmax]
            print(
                f"Warning: gmax={self.gmax} exactly equal to one of the g-vectors modulus"
                f", reciprocal lattice truncated with gmax={gmax}"
                f" to avoid problems."
                f"\nPlane waves used in the expansion = {np.shape(gvec)[1]}.")
        else:
            gvec = gvec[:, gnorm <= 2 * np.pi * self.gmax]
            inds1 = inds1[gnorm <= 2 * np.pi * self.gmax]
            inds2 = inds2[gnorm <= 2 * np.pi * self.gmax]

        # Save the reciprocal lattice vectors
        self._gvec = gvec

        self.n1g = bd.max(inds1)
        self.n2g = bd.max(inds2)
        self.inds1 = inds1
        self.inds2 = inds2

    def _ind_g(self, gvecs, g_x, g_y, thr=1e-7):
        """
        Find the index of the vector with components (g_x,g_y) in gvecs
        within a certain thershold given by 'thr' 
        """
        ind = bd.where((bd.abs((gvecs[0]) - g_x) < thr)
                       & (bd.abs((gvecs[1]) - g_y) < thr))
        if len(ind[0]) == 0:
            raise ValueError(f"The vector ({g_x},{g_y}) reflected by the "
                             "symmetry wrt the vertical plane"
                             " does not match any g-vector.")

        return ind

    def _construct_sym_mat(self, theta):
        """
        Construct the matrix which reflects a g-vector w.r.t. a vertical
        plane at angle theta (in degrees)
        """
        mat = bd.array([[
            bd.cos(2 * theta * np.pi / 180),
            bd.sin(2 * theta * np.pi / 180)
        ], [bd.sin(2 * theta * np.pi / 180),
            -bd.cos(2 * theta * np.pi / 180)]])
        return mat

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

    def _compute_guided(self, g_array):
        """
        Compute the guided modes using the slab_modes module, reshape the 
        results appropriately and store
        """

        # Expand boundaries a bit to make sure we get all the modes
        # Note that the 'exact' computation still uses interpolation,
        # but the grid is defined by the actual gk values
        g_array -= self.delta_gabs
        g_array[-1] += self.delta_gabs
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

        # x and y components of delta_G = G1-G2,
        #the dimension of ggrdix and ggrdiy is [Ng^2]
        ggridx = (self.gvec[0, :][np.newaxis, :] -
                  self.gvec[0, :][:, np.newaxis]).ravel()
        ggridy = (self.gvec[1, :][np.newaxis, :] -
                  self.gvec[1, :][:, np.newaxis]).ravel()

        # The dimension of G1 is [2,Ng^2]
        self.G1 = bd.vstack((ggridx, ggridy))

        # indexes corresponding to ggdrix,ggridy
        indgridx = (self.inds1[np.newaxis, :] -
                    self.inds1[:, np.newaxis]).ravel().astype(int)
        indgridy = (self.inds2[np.newaxis, :] -
                    self.inds2[:, np.newaxis]).ravel().astype(int)

        # We find the list of unique (indgridx,indgridy)
        # This is the bottleneck in terms of timing, we should find
        # an analytical expression for the unique elements to speedup
        indgrid = bd.array([indgridx, indgridy])
        unique, ind_unique = bd.unique(indgrid, axis=1, return_inverse=True)
        """
        From numpy 2.0, 'ind_unique' is returned as a two-dimensional array, with older
        numpy it was just a one-dimensional array containing the indexes to reconstruct
        the original array. We use squeeze() to remove the dimension of length 1 in case
        of numpy =>1.
        """
        ind_unique = ind_unique.squeeze()
        num_unique = np.shape(unique)[1]

        # Unique g-vectors for calculting f-transform
        gvec_unique = self.phc.lattice.b1[:, np.newaxis].dot(unique[0][np.newaxis, :]) + \
                        self.phc.lattice.b2[:, np.newaxis].dot(unique[1][np.newaxis, :])

        # T1 stores FT of all possible delta_G
        self.T1 = []

        # T1_unique and G1_unique stores unique delta_G and their FT, used for invft()
        self.T1_unique = []
        self.G1_unique = gvec_unique

        layers = [self.phc.claddings[0]] + self.phc.layers + \
                                [self.phc.claddings[1]]

        for layer in layers:
            """
            In the old code we calculated for all delta_G, but most of
            them were redundant. We aviod the redundancies by
            calculating the FT only for the unique delta_G.
            eps_ft = layer.compute_ft(np.vstack((ggridx, ggridy))) 
            """
            """
            here we calculate only for unique delta_G with gx> = 0,
            and then we just calculate conjugates for -delta_G elements.
            In gvec_unique, the first half of the array has negative gx,
            in the "middle" of the array there is (0,0), and in
            the second half the are all gx>0 terms with a reversed
            order, e.g. something like:
            [[...,  0,  0,  0,  0,  0, ...],
            [...,  -2, -1,  0,  1,  2, ...]]

            To explicity calulate all unique delta_G
            eps_ft_uniq = layer.compute_ft((gvec_unique))
            """
            eps_ft_pos = layer.compute_ft(
                (gvec_unique[:, 0:(num_unique - 1) // 2 + 1]))
            # The FT at -delta_G is the complex conjugate of FT at delta_G
            eps_ft_uniq = bd.concatenate(
                (eps_ft_pos, bd.conj(eps_ft_pos[-2::-1])))

            self.T1_unique.append(eps_ft_uniq)

            eps_ft = eps_ft_uniq[ind_unique]
            self.T1.append(eps_ft)

    def _construct_mat(self, kind):
        """
        Construct the Hermitian matrix for diagonalization for a given k
        """
        # G + k vectors
        gkx = self.gvec[0, :] + self.kpoints[0, kind] + self.delta_gx
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
        ind_modes = []

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
                f"No guided modes were found for k-index {kind}. "
                "One possibility is "
                "that the effective permittivity of all layers is smaller than "
                "that of at least one cladding. Reconsider your structure, or "
                "try changing 'eps_eff' from 'average' to 'background' in "
                "the options to GuidedModeExp.run().")
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
            ind_modes.append(indmode1)
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
        #Store a list of g-points indexes used for wach mode index
        self.ind_modes.append(ind_modes)

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
        #print(np.round(bd.triu(mat, 1) + bd.transpose(bd.conj(bd.triu(mat, 1))) + bd.real(bd.diag(bd.diag(mat))),3))
        #print(f'Dimension = {np.shape(np.round(bd.triu(mat, 1) + bd.transpose(bd.conj(bd.triu(mat, 1))) + bd.real(bd.diag(bd.diag(mat))),3))}')
        return bd.triu(mat, 1) + bd.transpose(bd.conj(bd.triu(mat, 1))) + \
                bd.real(bd.diag(bd.diag(mat)))

    def compute_eps_inv(self, only_gmodes=False):
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
                    if bd.sum(bd.abs(T1[1:])) < 1e-10 or only_gmodes:
                        self.eps_inv_mat.append(
                            bd.eye(T1.size, T1.size) / T1[0])
                        self.hom_layer.append(True)
                    else:
                        eps_mat = bd.toeplitz_block(self.n1g, T1, self.T2[it])
                        self.eps_inv_mat.append(bd.inv(eps_mat))
                        self.hom_layer.append(False)
            elif self.truncate_g == 'abs':

                self.eps_mat = []
                self.hom_layer = []

                layers = [self.phc.claddings[0]] + self.phc.layers + \
                                [self.phc.claddings[1]]
                for it, T1 in enumerate(self.T1):
                    # We are iterating of layers

                    mod_G1 = bd.norm(self.G1, axis=0)
                    # Calculate the eps only if there are shapes in layer
                    # and eps is not trivially diagonal

                    index_G0 = np.argmin(mod_G1)
                    eps_ft = bd.reshape(
                        T1, (self.gvec[0, :].size, self.gvec[0, :].size))

                    if bd.sum(bd.abs(
                            T1[mod_G1 > 1e-10])) < 1e-10 or only_gmodes:
                        # Here eps is diagonal
                        self.hom_layer.append(True)
                        self.eps_mat.append(
                            bd.eye(np.shape(eps_ft)[0],
                                   np.shape(eps_ft)[1]) * T1[index_G0])
                        self.eps_inv_mat.append(
                            bd.eye(np.shape(eps_ft)[0],
                                   np.shape(eps_ft)[1]) / T1[index_G0])
                    else:
                        # Here explicitly calculate eps

                        self.eps_mat.append(eps_ft)
                        self.eps_inv_mat.append(bd.inv(eps_ft))
                        self.hom_layer.append(False)

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
                        verbose: bool = True,
                        kz_symmetry: Optional[str] = None,
                        symm_thr: float = 1e-8,
                        delta_gx: float = 1e-15,
                        delta_gabs: float = 1e-15,
                        use_sparse: bool = False,
                        only_gmodes: bool = False):
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
                Print information at intermediate steps. Default is True.
            kz_symmetry : string, optional
                Symmetry seleced with respect to the vertical kz plane of incidence,
                it can be 'both', 'odd', 'even' or None.
                With None there is no symmetry sepeartion, and :attr:`GuidedModeExp.kz_symms`
                is an empty list.
                If 'both', both even and odd modes are saved in
                attr:`GuidedModeExp.freqs` and their parity wrt the veritcal
                plane are saved in :attr:`GuidedModeExp.kz_symms`.
                With 'even' ('odd') only 'even' ('odd') modes are calculated and 
                stored.
                Default is None.
            symm_thr : float, optional
                Threshold for out-of-diagonal terms in odd/even separated
                Hamiltonian.
                Default is 1e-8.
            delta_gx: float, optional,
                little component added to the x-component of vectors
                g = k + G to avoid problems at g = 0.
                Default is 1e-15.
            delta_gabs:  float, optional,
                small shift of the absolute value of g = k + G
                , kept for backwards compatibility.
                Default is 1e-15.
            use_sparse: boolean, optional
                If True, use sparse matrices for separating
                even and odd modes w.r.t. the vertical plane of symmetry.
                Default is False.
            only_gmodes: boolean, optional
                Should only the guided modes computed withouth the 
                coupling of the PhC periodic patterning. 
                Default is False.
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
            'verbose': verbose,
            'kz_symmetry': kz_symmetry,
            'symm_thr': symm_thr,
            'delta_gx': delta_gx,
            'delta_gabs': delta_gabs,
            'use_sparse': use_sparse,
            'only_gmodes': only_gmodes
        }

        # Also store the options as separate attributes
        for (option, value) in self._run_options.items():
            # Make sure 'gmode_inds' is a numpy array
            if option.lower() == 'gmode_inds':
                value = np.array(value)
            # Set all the options as class attributes
            setattr(self, option, value)

    def run(self,
            kpoints: np.ndarray = np.array([[0], [0]]),
            angles: np.ndarray = np.array([]),
            **kwargs):
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
            If kz_symmetry = 'odd', 'even' or 'both' calculates symmetries for each k,
            and imaginary part of frequencies of both polarizations
            if kz_symmetry = odd' or 'even' calculates imaginary part of 
            'odd' or 'even' modes.
        
        Parameters
        ----------
        kpoints : np.ndarray, optional
            numpy array of shape (2, :) with the [kx, ky] coordinates of the 
            k-vectors over which the simulation is run.
        angles : np.ndarray, optional
            This is needed only in case symmetry is differnet from None.
            Numpy array with direction angle in kx-ky plane of kpoints.

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

        #Check if angles are provided in case is not None
        if self.kz_symmetry is None:
            pass
        elif self.kz_symmetry.lower() in {'odd', 'even', 'both'}:
            if bd.shape(angles)[0] == 0:
                raise ValueError(
                    "path['angles'] must be passed to GuidedModeExp.run()"
                    " if kz_symmetry is either 'odd', 'even' or 'both'.")
        else:
            raise ValueError(
                "'kz_symmetry' can be None, 'odd', 'even' or 'both' ")

        if self.kz_symmetry:
            #     if self.truncate_g == 'tbt':
            #         raise ValueError(
            #             "'truncate_g' must be 'abs' to separate odd and even modes"
            #             " w.r.t. a vertical plane of symmetry.")
            refl_mat = self._calculate_refl_mat(angles)

        if self.truncate_g == 'tbt':
            if self.only_gmodes:
                raise ValueError(
                    "only_gmodes can be true only with 'abs' truncation.")

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
        eps_array = bd.array(list(
            getattr(layer, layer_eps) for layer in [self.phc.claddings[0]] +
            self.phc.layers + [self.phc.claddings[1]]),
                             dtype=object).ravel()
        eps_array = bd.array(eps_array, dtype=float)
        # A separate array where the values are converted from ArrayBox to numpy
        # array, if using the 'autograd' backend.
        eps_array_val = np.array(list(
            float(get_value(getattr(layer, layer_eps)))
            for layer in [self.phc.claddings[0]] + self.phc.layers +
            [self.phc.claddings[1]]),
                                 dtype=object).ravel()
        eps_array_val = bd.array(eps_array_val, dtype=bd.float)
        # Store an array of thickness of every layer (not including claddings)
        d_array = bd.array(list(layer.d for layer in \
            self.phc.layers), dtype=object).ravel()
        d_array = bd.array(d_array, dtype=float)
        # A separate array where the values are converted from ArrayBox to numpy
        # array, if using the 'autograd' backend.
        d_array_val = np.array(list(get_value(layer.d) for layer in \
            self.phc.layers), dtype=object).ravel()
        d_array_val = bd.array(d_array_val, dtype=bd.float)

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
        self.t_symmetry = 0  # For timing of symmetry matrix construction
        self.t_creat_mat = 0
        # Compute inverse matrix of FT of permittivity
        t = time.time()
        self.compute_eps_inv(self.only_gmodes)

        t_eps_inv = time.time() - t

        # Loop over all k-points, construct the matrix, diagonalize, and compute
        # radiative losses for the modes requested by kinds_rad and minds_rad
        t_rad = 0

        freqs = []
        freqs_im = []
        kz_symms = []
        self._eigvecs = []
        self.even_counts = []
        self.odd_counts = []

        num_k = kpoints.shape[1]  # Number of wavevectors
        t_loop_k = time.time()  # Fro printing in progress
        for ik, k in enumerate(kpoints.T):

            prbd.update_prog(ik, num_k, self.verbose, "Running gme k-points:")
            t_create = time.time()
            mat = self._construct_mat(kind=ik)

            # The guided modes are calculate inside _construct_mat, later we have to subtract that time
            self.t_creat_mat += time.time() - t_create

            if self.numeig > mat.shape[0]:
                raise ValueError(
                    "Requested number of eigenvalues 'numeig' "
                    "larger than total size of basis set. Reduce 'numeig' or "
                    "increase 'gmax'")

            t_eig = time.time()

            if self.eig_solver == 'eigh':

                # Separates odd and even blocks of Hamiltonian
                if self.kz_symmetry:
                    t_sym = time.time()
                    symm_mat = refl_mat[str(angles[ik])]

                    if self.use_sparse == True:
                        mat_even, mat_odd, v_sigma_perm = self._separate_hamiltonian_sparse(
                            mat, symm_mat, ik)
                    elif self.use_sparse == False:
                        mat_even, mat_odd, v_sigma_perm = self._separate_hamiltonian_dense(
                            mat, symm_mat, ik)

                    self.t_symmetry += time.time() - t_sym

                # Diagonalise matrix
                if self.kz_symmetry is None:
                    # NB: we shift the matrix by np.eye to avoid problems at the zero-
                    # frequency mode at Gamma
                    (freq2, evecs) = bd.eigh(mat + bd.eye(mat.shape[0]))
                    freq1 = bd.sqrt(
                        bd.abs(freq2 - bd.ones(mat.shape[0]))) / 2 / np.pi

                    i_near = find_nearest(get_value(freq1), self.eig_sigma,
                                          self.numeig)
                    i_sort = bd.argsort(freq1[i_near])
                    freq = freq1[i_near[i_sort]]
                    evec = evecs[:, i_near[i_sort]]

                elif self.kz_symmetry.lower() == 'both':
                    (freq2_odd,
                     evecs_odd) = bd.eigh(mat_odd + bd.eye(mat_odd.shape[0]))
                    freq1_odd = bd.sqrt(
                        bd.abs(freq2_odd -
                               bd.ones(mat_odd.shape[0]))) / 2 / np.pi
                    zeros_arr = bd.zeros(
                        (self.even_counts[ik], np.shape(evecs_odd)[1]))
                    evecs_odd = bd.concatenate((zeros_arr, evecs_odd))

                    (freq2_even,
                     evecs_even) = bd.eigh(mat_even +
                                           bd.eye(mat_even.shape[0]))
                    freq1_even = bd.sqrt(
                        bd.abs(freq2_even -
                               bd.ones(mat_even.shape[0]))) / 2 / np.pi
                    zeros_arr = bd.zeros(
                        (self.odd_counts[ik], np.shape(evecs_even)[1]))
                    evecs_even = bd.concatenate((evecs_even, zeros_arr))

                    symm1 = bd.concatenate((np.full(self.even_counts[ik],
                                                    1,
                                                    dtype=int),
                                            np.full(self.even_counts[ik],
                                                    -1,
                                                    dtype=int)))
                    freq1 = bd.concatenate((freq1_even, freq1_odd))
                    evecs = bd.concatenate((evecs_even, evecs_odd), axis=1)
                    i_near = find_nearest(get_value(freq1), self.eig_sigma,
                                          self.numeig)
                    i_sort = bd.argsort(freq1[i_near])
                    freq = freq1[i_near[i_sort]]
                    symm = symm1[i_near[i_sort]]
                    evec = evecs[:, i_near[i_sort]]
                    #Rewrite eigenvector in original basis
                    if self.use_sparse == True:
                        evec = bd.spdot(v_sigma_perm, evec)
                    elif self.use_sparse == False:
                        evec = bd.matmul(v_sigma_perm, evec)

                elif self.kz_symmetry.lower() == 'odd':
                    if self.numeig > mat_odd.shape[0]:
                        raise ValueError(
                            "Requested number of odd eigenvalues 'numeig' "
                            "larger than total size of basis set. Reduce 'numeig' or "
                            "increase 'gmax'")
                    (freq2,
                     evecs) = bd.eigh(mat_odd + bd.eye(mat_odd.shape[0]))
                    freq1 = bd.sqrt(
                        bd.abs(freq2 - bd.ones(mat_odd.shape[0]))) / 2 / np.pi

                    i_near = find_nearest(get_value(freq1), self.eig_sigma,
                                          self.numeig)
                    i_sort = bd.argsort(freq1[i_near])

                    freq = freq1[i_near[i_sort]]
                    evec = evecs[:, i_near[i_sort]]
                    #Rewrite eigenvector in original basis
                    zeros_arr = bd.zeros(
                        (self.even_counts[ik], np.shape(evec)[1]))
                    evec = bd.concatenate((zeros_arr, evec))
                    if self.use_sparse == True:
                        evec = bd.spdot(v_sigma_perm, evec)
                    elif self.use_sparse == False:
                        evec = bd.matmul(v_sigma_perm, evec)
                elif self.kz_symmetry.lower() == 'even':
                    if self.numeig > mat_even.shape[0]:
                        raise ValueError(
                            "Requested number of even eigenvalues 'numeig' "
                            "larger than total size of basis set. Reduce 'numeig' or "
                            "increase 'gmax'")
                    (freq2,
                     evecs) = bd.eigh(mat_even + bd.eye(mat_even.shape[0]))
                    freq1 = bd.sqrt(
                        bd.abs(freq2 - bd.ones(mat_even.shape[0]))) / 2 / np.pi

                    i_near = find_nearest(get_value(freq1), self.eig_sigma,
                                          self.numeig)
                    i_sort = bd.argsort(freq1[i_near])

                    freq = freq1[i_near[i_sort]]
                    evec = evecs[:, i_near[i_sort]]
                    #Rewrite eigenvector in original basis
                    zeros_arr = bd.zeros(
                        (self.odd_counts[ik], np.shape(evec)[1]))
                    evec = bd.concatenate((evec, zeros_arr))
                    if self.use_sparse == True:
                        evec = bd.spdot(v_sigma_perm, evec)
                    elif self.use_sparse == False:
                        evec = bd.matmul(v_sigma_perm, evec)
            elif self.eig_solver == 'eigsh':
                if self.kz_symmetry:
                    raise ValueError(
                        "odd/even separation implemented with 'eigh' solver only."
                    )
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
            if self.kz_symmetry:
                if self.kz_symmetry.lower() == 'both':
                    kz_symms.append(symm)
        if self.kz_symmetry:
            if self.kz_symmetry.lower() == 'odd':
                kz_symms = -np.ones_like(np.array(freqs))
            elif self.kz_symmetry.lower() == 'even':
                kz_symms = np.ones_like(np.array(freqs))

        # Store the eigenfrequencies taking the standard reduced frequency
        # convention for the units (2pi a/c)
        self._freqs = bd.array(freqs)
        self._kz_symms = bd.array(kz_symms)
        # Guided modes are calculated inside _construct_mat()
        self.t_creat_mat = self.t_creat_mat - self.t_guided
        self.t_eps_inv = t_eps_inv

        total_time = time.time() - t_start
        self.total_time = total_time
        prbd.GME_report(self)

        if self.compute_im == True:
            self.run_im()
        else:
            verbose_print(
                "Skipping imaginary part computation, use run_im() to"
                " run it, or compute_rad() to compute the radiative rates"
                " of selected eigenmodes", self.verbose)

    def run_im(self):
        """
        Compute the radiative rates associated to all the eigenmodes that were 
        computed during :meth:`GuidedModeExp.run`. Results are stored in 
        :attr:`GuidedModeExp.freqs_im`, :attr:`GuidedModeExp.rad_coup`, 
        :attr:`GuidedModeExp.rad_gvec` and :attr:`GuidedModeExp.unbalance_sp`.
        """
        t = time.time()
        freqs_i = []  # Imaginary part of frequencies
        freqs_i_te = []  # Imaginary part of frequencies
        freqs_i_tm = []  # Imaginary part of frequencies

        # Coupling constants to lower- and upper-cladding radiative modes
        rad_coup = {'l_te': [], 'l_tm': [], 'u_te': [], 'u_tm': []}
        rad_gvec = {'l': [], 'u': []}

        minds = np.arange(0, self.numeig)

        if len(self.freqs) == 0:
            raise RuntimeError("Run the GME computation first!")
        num_k = np.shape(self.freqs)[0]
        for kind in range(len(self.freqs)):
            prbd.update_prog(kind, num_k, self.verbose,
                             "Running GME losses k-point:")
            (freqs_im, freqs_im_te, freqs_im_tm, rc,
             rv) = self.compute_rad_sp(kind, minds)
            freqs_i.append(freqs_im)
            freqs_i_te.append(
                freqs_im_te
            )  # We won't save these, but just the unbalance between te and tm
            freqs_i_tm.append(freqs_im_tm)
            for clad in ['l', 'u']:
                rad_coup[clad + '_te'].append(rc[clad + '_te'])
                rad_coup[clad + '_tm'].append(rc[clad + '_tm'])
                rad_gvec[clad].append(rv[clad])

        #calculate unbalance between s and p coupling
        unbalance_i = np.array(freqs_i_te) / (np.array(freqs_i_te) +
                                              np.array(freqs_i_tm) + 1e-15)
        # unbalance 0.5 where there are no losses
        unbalance_i[np.abs(np.array(freqs_i)) < 1e-15] = 0.5

        self._unbalance_sp = bd.array(unbalance_i)
        self._freqs_im = bd.array(freqs_i)
        self._rad_coup = rad_coup
        self._rad_gvec = rad_gvec
        self.t_imag = time.time() - t

        prbd.GME_im_report(self)

    def _separate_hamiltonian_dense(self, mat, symm_mat, ik):
        """
        Separates the Hamiltonian matrix into 
        even and odd block w.r.t. the vertical plane of 
        symmetry chosen. This function uses numpy dense
        matrices.
        
        Parameters
        ----------
        mat : np.array 
            Hamiltonian to separate
        symm_mat : np.array
            2X2 reflection matrix
        ik : int
            Index of k-vector


        Returns
        -------
        mat_even: np.array
            Array corresponding to even block of Hamiltonian
        mat_odd: np.array
            Array corresponding to odd block of Hamiltonian
        v_sigma_perm : np.array
            Array which performs the change of basis and
            a permutation that orders all even modes 
            before odd modes
        """

        blocks = []  #
        blocks_w = []  #eigenvectors

        for ind_alpha, alpha in enumerate(self.gmode_inds):
            dim = self.modes_numg[ik][ind_alpha]

            #Block of symmetry operator corresponding to (k,ind_alpha)

            block = bd.zeros((dim, dim))
            block_w = bd.zeros(dim)

            #Array with indexes used for a given guided mode @ k
            ind = self.ind_modes[ik][ind_alpha]
            #Array with G vectors used for a given guided mode @ k
            gvec_used = bd.array([self.gvec[0][ind], self.gvec[1][ind]])
            #Array that will store reflected g-vectors by symmetry operator
            gvec_used_ex = np.zeros(np.shape(gvec_used))
            #Loop over used g-vectors
            for j in range(bd.shape(gvec_used)[1]):
                #Calculate the symmetry-reflected g-vector
                g_ex = bd.matmul(symm_mat, [gvec_used[0][j], gvec_used[1][j]])

                #Find index of reflected G-vector
                index_exc = self._ind_g(gvec_used, g_ex[0], g_ex[1])

                if j == index_exc[0][0]:
                    # G vector reflected on itself
                    block[j, j] = 1
                    block_w[j] = 1
                else:
                    # G vector reflect on another G' vector
                    block[j, j] = -1 / np.sqrt(2)
                    block[j, index_exc] = 1 / bd.sqrt(2)
                    block[index_exc, j] = 1 / bd.sqrt(2)
                    block[index_exc, index_exc] = 1 / bd.sqrt(2)
                    block_w[j] = -1
                    block_w[index_exc] = 1

            #N.B. TE guided modes are odd, TM guided modes are even w.r.t. vertical symmetry plane
            if np.remainder(alpha, 2) == 0:
                blocks.append(block)
                blocks_w.append(-block_w)
            elif np.remainder(alpha, 2) != 0:
                blocks.append(block)
                blocks_w.append(block_w)

        # This is without sparse matrix
        v_sigma = block_diag(*blocks)
        sigma_diag = bd.hstack(blocks_w)

        indexes_sigma = []
        even_count = 0
        odd_count = 0

        # Indexing for ordering all even modes before odd modes
        for ind_s, s_eig in enumerate(sigma_diag):
            if s_eig == 1:
                even_count += 1
                indexes_sigma.insert(0, ind_s)
            if s_eig == -1:
                odd_count += 1
                indexes_sigma.append(ind_s)

        v_sigma_perm = v_sigma[:, indexes_sigma]

        separ_mat = bd.matmul(v_sigma_perm.T, bd.matmul(mat, v_sigma_perm))

        mat_even = separ_mat[0:even_count, 0:even_count]
        mat_odd = separ_mat[even_count:, even_count:]

        out_diag_1 = separ_mat[even_count:, 0:even_count]
        out_diag_2 = separ_mat[0:even_count, even_count:]
        """
        Check that Hamiltonian is completely separated in odd and even blocks
        only if they are both not empty, otherwise it means
        that there are only completely odd or even blocks. In that
        case, there are not out-of-diagonal terms.
        """
        if bd.size(mat_odd) != 0 and bd.size(mat_even) != 0:
            max_out_1 = bd.max(np.abs(out_diag_1))
            max_out_2 = bd.max(np.abs(out_diag_2))
        else:
            max_out_1 = 0
            max_out_2 = 0
            raise ValueError("Only purely odd or even modes,"
                             " we need to implement this possibility, add"
                             " a guided mode with different polarisation.")

        if bd.max((max_out_1, max_out_2)) > self.symm_thr:
            raise ValueError(
                "Something went wrong with (odd/even) separation"
                f" of the Hamiltonian. Max out of diagonal value = {bd.max((max_out_1,max_out_2))}"
                " One possibility is that the basis"
                " of the lattice breaks the symmetry w.r.t. the vertical plane."
                " Otherwise, try to increase 'symm_thr' (default value = 1e-8)."
            )

        self.odd_counts.append(odd_count)
        self.even_counts.append(even_count)

        return mat_even, mat_odd, v_sigma_perm

    def _separate_hamiltonian_sparse(self, mat, symm_mat, ik):
        """
        Separates the Hamiltonian matrix into 
        even and odd block w.r.t. the vertical plane of 
        symmetry chosen. This function uses sparse
        matrices, since the change of basis matrix is
        mostly filled with 0.
        
        Parameters
        ----------
        mat : np.array 
            Hamiltonian to separate
        symm_mat : np.array
            2X2 reflection matrix
        ik : int
            Index of k-vector


        Returns
        -------
        mat_even: np.array
            Array corresponding to even block of Hamiltonian
        mat_odd: np.array
            Array corresponding to odd block of Hamiltonian
        v_sigma_perm : np.array
            Array which performs the change of basis and
            a permutation that orders all even modes 
            before odd modes
        """

        data_blocks = []
        row_blocks = []
        col_blocks = []

        blocks_w = []

        #dim_final = np.shape(mat)[0]
        dim_final = 0

        for ind_alpha, alpha in enumerate(self.gmode_inds):
            dim = self.modes_numg[ik][ind_alpha]
            # These are for creating sparse matrix
            # check_g to avoid double counting on reflected g
            block_w = bd.zeros(dim)
            check_g = bd.full((dim), False)
            data_block = []
            row_block = []
            col_block = []
            #Array with indexes used for a given guided mode @ k
            ind = self.ind_modes[ik][ind_alpha]
            #Array with G vectors used for a given guided mode @ k
            gvec_used = bd.array([self.gvec[0][ind], self.gvec[1][ind]])
            #Array that will store reflected g-vectors by symmetry operator
            gvec_used_ex = np.zeros(np.shape(gvec_used))
            #Loop over used g-vectors
            for j in range(bd.shape(gvec_used)[1]):
                #Calculate the symmetry-reflected g-vector
                #We could implement specific cases for speed-up code, e.g. theta=0
                t_mat_refl = time.time()
                g_ex = bd.matmul(symm_mat, [gvec_used[0][j], gvec_used[1][j]])
                #Find index of reflected G-vector
                index_exc = self._ind_g(gvec_used, g_ex[0], g_ex[1])
                if j == index_exc[0][0]:
                    # G vector reflect on itself
                    block_w[j] = 1
                    data_block.append(1)
                    row_block.append(j + dim_final)
                    col_block.append(j + dim_final)
                else:
                    # G vector reflect on another G' vector
                    # we avoid double counting

                    if check_g[j] == False:
                        block_w[j] = -1
                        block_w[index_exc] = 1
                        row = np.array([
                            j, j, index_exc[0][0], index_exc[0][0]
                        ]) + dim_final
                        col = np.array([
                            j, index_exc[0][0], j, index_exc[0][0]
                        ]) + dim_final
                        data_block.extend([
                            -1 / bd.sqrt(2), 1 / bd.sqrt(2), 1 / bd.sqrt(2),
                            1 / bd.sqrt(2)
                        ])
                        row_block.extend(row)
                        col_block.extend(col)
                        check_g[index_exc] = True
                        check_g[j] = True

            dim_final += dim

            data_blocks.append(bd.array(data_block))
            row_blocks.append(bd.array(row_block))
            col_blocks.append(bd.array(col_block))

            #N.B. TE guided modes are odd, TM guided modes are even w.r.t. vertical symmetry plane
            if np.remainder(alpha, 2) == 0:
                blocks_w.append(-block_w)
            elif np.remainder(alpha, 2) != 0:
                blocks_w.append(block_w)

        # Change of basis matrix
        data_blocks = bd.hstack(data_blocks)
        col_blocks = bd.hstack(col_blocks)
        row_blocks = bd.hstack(row_blocks)
        v_sigma_coo = coo((data_blocks, (row_blocks, col_blocks)),
                          shape=(dim_final, dim_final))

        sigma_diag = bd.hstack(blocks_w)

        indexes_sigma = []
        even_count = 0
        odd_count = 0

        # Indexing for ordering all even modes before odd modes
        for ind_s, s_eig in enumerate(sigma_diag):
            if s_eig == 1:
                even_count += 1
                indexes_sigma.insert(0, ind_s)
            if s_eig == -1:
                odd_count += 1
                indexes_sigma.append(ind_s)

        data_P = np.ones(len(indexes_sigma))
        col_P = np.arange(len(indexes_sigma))
        row_P = np.array(indexes_sigma)
        P = coo((data_P, (row_P, col_P)), shape=(dim_final, dim_final))

        v_sigma_perm = v_sigma_coo.dot(P)

        separ_mat_sparse = bd.spdot(v_sigma_perm.T, mat.T)
        separ_mat_sparse = bd.spdot(v_sigma_perm.T, separ_mat_sparse.T)

        mat_even = separ_mat_sparse[0:even_count, 0:even_count]
        mat_odd = separ_mat_sparse[even_count:, even_count:]

        out_diag_1 = separ_mat_sparse[even_count:, 0:even_count]
        out_diag_2 = separ_mat_sparse[0:even_count, even_count:]
        if bd.size(mat_odd) != 0 and bd.size(mat_even) != 0:
            max_out_1 = bd.max(np.abs(out_diag_1))
            max_out_2 = bd.max(np.abs(out_diag_2))
        else:
            max_out_1 = 0
            max_out_2 = 0
            raise ValueError("Only purely odd or even modes,"
                             " we need to implement this possibility, add"
                             " a guided mode with different polarisation.")

        if bd.max((max_out_1, max_out_2)) > self.symm_thr:
            raise ValueError(
                "Something went wrong with (odd/even) separation"
                f" of the Hamiltonian. Max out of diagonal value = {bd.max((max_out_1,max_out_2))}"
                " One possibility is that the basis"
                " of the lattice breaks the symmetry w.r.t. the vertical plane."
                " Otherwise, try to increase 'symm_thr' (default value = 1e-8)."
            )

        self.odd_counts.append(odd_count)
        self.even_counts.append(even_count)

        return mat_even, mat_odd, v_sigma_perm

    def _calculate_refl_mat(self, angles):
        """
        Calculate the reflection matrices used for 
        the symmetry separation with respect to
        the vertical (kz) plane of symmetry, where
        k is the in-plane wavevector and z is vertical direction.
        Before calculating the reflection matrices, 
        we check that kpoints are in high symmetry lines of the lattice.

        Returns
        -------

        refl_mat : dict
            Dictionary containing the reflection matrix
            for each angle theta of the wavevector k in
            the xy plane.
        """

        if self.phc.lattice.type == "square":
            for ang in angles:
                if (bd.round(ang, 8) in self._square_an) == False:
                    raise ValueError(
                        "Some kpoints are not along a high-symmetry line"
                        " of square lattice")
            #Dictionary with all reflection matrices used
            refl_mat = {}
            # Loop over unique angles
            for ang in set(angles):
                re_mat = self._construct_sym_mat(ang)
                refl_mat.update({str(ang): re_mat})

        elif self.phc.lattice.type == "hexagonal":
            for ang in angles:
                if (bd.round(ang, 8) in self._hex_an) == False:
                    raise ValueError(
                        "Some kpoints are not along a high-symmetry line"
                        " of hexagonal lattice")
            #Dictionary with all reflection matrices used
            refl_mat = {}
            # Loop over unique angles
            for ang in set(angles):
                re_mat = self._construct_sym_mat(ang)
                refl_mat.update({str(ang): re_mat})

        elif self.phc.lattice.type == "rectangular":
            for ang in angles:
                if (bd.round(ang, 8) in self._rec_an) == False:
                    raise ValueError(
                        "Some kpoints are not along a high-symmetry line"
                        " of rectangular lattice")
            #Dictionary with all reflection matrices used
            refl_mat = {}
            # Loop over unique angles
            for ang in set(angles):
                re_mat = self._construct_sym_mat(ang)
                refl_mat.update({str(ang): re_mat})
        else:
            raise ValueError(
                "Symmetry separation w.r.t. vertical kz plane is implemented"
                " for 'square', 'hexagonal' and rectangular lattices only.")

        return refl_mat

    def _compute_rad_components(self, kind: int, minds: list = [0]):
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
            Coupling to te (s) and tm (p) radiative modes in the lower/upper cladding.
        rad_gvec : dict
            Reciprocal lattice vectors in the lower/upper cladding 
            corresponding to ``rad_coup``.

        """

        freqs = self.freqs
        eigvecs = self.eigvecs

        if len(freqs) == 0:
            raise RuntimeError("Run the GME computation first!")
        if np.max(np.array(minds)) > self.numeig - 1:
            raise ValueError("Requested mode index out of range for the %d "
                             "stored eigenmodes" % self.numeig)

        # G + k vectors
        gkx = self.gvec[0, :] + self.kpoints[0, kind] + self.delta_gx
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
        rad_tot_te = []
        rad_tot_tm = []
        rad_gvec = {'l': [], 'u': []}
        rad_coup = {'l_te': [], 'l_tm': [], 'u_te': [], 'u_tm': []}
        # Iterate over all the modes to be computed
        for im in minds:
            omr = 2 * np.pi * freqs[kind, im]
            evec = eigvecs[kind][:, im]
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
                    #print(f"kind={kind}, {im}, {indmode1.shape},{np.shape(evec)}, {self.modes_numg[kind][im1]}")
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

                # Store the coupling constants (they are effectively in units
                # of angular frequency omega)
                rad_coup['l_' + pol].append(c_l[pol])
                rad_coup['u_' + pol].append(c_u[pol])

            # radiative coupling to s (te) wave
            rad_te = bd.sum(bd.square(bd.abs(c_l["te"]))) + \
                    bd.sum(bd.square(bd.abs(c_u["te"])))
            # radiative coupling to p (tm) wave
            rad_tm = bd.sum(bd.square(bd.abs(c_l["tm"]))) + \
                    bd.sum(bd.square(bd.abs(c_u["tm"])))
            rad_t = rad_te + rad_tm

            rad_tot.append(bd.imag(bd.sqrt(omr**2 + 1j * rad_t)))
            rad_tot_te.append(bd.imag(bd.sqrt(omr**2 + 1j * rad_te)))
            rad_tot_tm.append(bd.imag(bd.sqrt(omr**2 + 1j * rad_tm)))

        return (rad_tot, rad_tot_te, rad_tot_tm, rad_coup, rad_gvec)

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
            Coupling to te (s) and tm (p) radiative modes in the lower/upper cladding.
        rad_gvec : dict
            Reciprocal lattice vectors in the lower/upper cladding 
            corresponding to ``rad_coup``.
        """

        rad_tot, _, _, rad_coup, rad_gvec = self._compute_rad_components(
            kind, minds)

        # Compute radiation rate in units of frequency
        freqs_im = bd.array(rad_tot) / 2 / np.pi

        return (freqs_im, rad_coup, rad_gvec)

    def compute_rad_sp(self, kind: int, minds: list = [0]):
        """
        Compute the radiation losses of the eigenmodes after the dispersion
        has been computed. Unlike :meth:`GuidedModeExp.compute_rad`, this
        method separates the contribution form coupling to te-(s-) and
        tm-(p-)polarized radiative modes.
        
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
        freqs_im_te : np.ndarray
            Imaginary part of the frequencies of the eigenmodes computed by the 
            guided-mode expansion due to coupling to te (s) radiative mode
        freqs_im_tm : np.ndarray
            Imaginary part of the frequencies of the eigenmodes computed by the 
            guided-mode expansion due to coupling to tm (p) radiative mode
        rad_coup : dict 
            Coupling to te (s) and tm (p) radiative modes in the lower/upper cladding.
        rad_gvec : dict
            Reciprocal lattice vectors in the lower/upper cladding 
            corresponding to ``rad_coup``.
        """

        rad_tot, rad_tot_te, rad_tot_tm, rad_coup, rad_gvec = self._compute_rad_components(
            kind, minds)

        # Compute radiation rate in units of frequency
        freqs_im = bd.array(rad_tot) / 2 / np.pi
        freqs_im_te = bd.array(rad_tot_te) / 2 / np.pi
        freqs_im_tm = bd.array(rad_tot_tm) / 2 / np.pi

        return (freqs_im, freqs_im_te, freqs_im_tm, rad_coup, rad_gvec)

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
        lind = z_to_lind(self.phc, z)

        if self.truncate_g == "tbt":
            ft_coeffs = np.hstack(
                (self.T1[lind], self.T2[lind], np.conj(self.T1[lind]),
                 np.conj(self.T2[lind])))
            gvec = np.hstack((self.G1, self.G2, -self.G1, -self.G2))
            eps_r = ftinv(ft_coeffs, gvec, xgrid, ygrid)

        elif self.truncate_g == "abs":
            ft_coeffs = self.T1_unique[lind]
            gvec = self.G1_unique
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
        gkx = self.gvec[0, :] + k[0] + self.delta_gx
        gky = self.gvec[1, :] + k[1]
        gnorm = bd.sqrt(bd.square(gkx) + bd.square(gky))

        # Unit vectors in the propagation direction
        px = gkx / gnorm
        py = gky / gnorm

        # Unit vectors in-plane orthogonal to the propagation direction
        qx = py
        qy = -px

        lind = z_to_lind(self.phc, z)

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

        # Get the field Fourier components
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
