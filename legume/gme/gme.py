import numpy as np

import time
from itertools import zip_longest

from .slab_modes import guided_modes, rad_modes
from . import matrix_elements
from legume.backend import backend as bd
from legume.utils import get_value, ftinv

class GuidedModeExp(object):
    '''
    Main simulation class of the guided-mode expansion
    '''
    def __init__(self, phc, gmax=3):
        # Object of class Phc which will be simulated
        self.phc = phc
        # Number of layers
        self.N_layers = len(phc.layers)
        # Maximum reciprocal lattice wave-vector length in units of 2pi/a
        self.gmax = gmax

        # Number of G points included for every mode, will be defined after run
        self.modes_numg = []
        # Total number of basis vectors (equal to np.sum(self.modes_numg))
        self.N_basis = []
        # Indexes of guided modes which are actually included in the computation
        # (in case gmode_inds includes modes that are above the gmax cutoff)
        self.gmode_include = []

        # Eigenfrequencies and eigenvectors
        self.freqs = []
        self.freqs_im = []
        self.eigvecs = []

        # Initialize the reciprocal lattice vectors and compute the FT of all
        # the layers of the PhC
        self._init_reciprocal()
        self.compute_ft()

    def __repr__(self):
        rep = 'GuidedModeExp(\n'
        rep += 'gmax = ' + repr(self.gmax) + '\n'
        rep += 'modes_numg = ' + repr(self.modes_numg) + '\n'
        rep += 'N_basis = ' + repr(self.N_basis) + '\n'
        rep += repr(self.phc) + '\n)'
        return rep

    def _init_reciprocal(self):
        '''
        Initialize reciprocal lattice vectors based on self.phc and self.gmax
        '''
        n1max = np.int_((2*np.pi*self.gmax)/np.linalg.norm(self.phc.lattice.b1))
        n2max = np.int_((2*np.pi*self.gmax)/np.linalg.norm(self.phc.lattice.b2))

        # This constructs the reciprocal lattice in a way that is suitable
        # for Toeplitz-Block-Toeplitz inversion of the permittivity in the main
        # code. However, one caveat is that the hexagonal lattice symmetry is 
        # not preserved. For that, the option to construct a hexagonal mesh in 
        # reciprocal space could is needed.
        inds1 = np.tile(np.arange(-n1max, n1max + 1), (2*n2max + 1, 1))  \
                         .reshape((2*n2max + 1)*(2*n1max + 1), order='F')
        inds2 = np.tile(np.arange(-n2max, n2max + 1), 2*n1max + 1)

        gvec = self.phc.lattice.b1[:, np.newaxis].dot(inds1[np.newaxis, :]) + \
                self.phc.lattice.b2[:, np.newaxis].dot(inds2[np.newaxis, :])

        # Save the reciprocal lattice vectors
        self.gvec = gvec

        # Save the number of vectors along the b1 and the b2 directions 
        # Note: gvec.shape[1] = n1g*n2g
        self.n1g = 2*n1max + 1
        self.n2g = 2*n2max + 1

    def _run_options(self, options):
        default_options = {
            # Define whether the guided modes are computed in the beginning
            # and then interpolated ('interp'), or whether they are computed
            # exactly at every k-step ('exact')
            'gmode_compute': 'exact',

            # Indexes of modes to be included in the expansion
            'gmode_inds'   : [0],

            # Number of points over which guided modes are computed
            'gmode_npts'   : 1000,

            # Step in frequency in the search for guided mode solutions
            'gmode_step'   : 1e-2,

            # Tolerance in the minimum and maximum omega value when looking for 
            # the guided-mode solutions
            'gmode_tol'    : 1e-10,

            # Number of eigen-frequencies to be stored (starting from lowest)
            'numeig'       : 10,

            # Should the imaginary parts of the frequencies also be computed
            'compute_im'   : True,

            # Using the 'average' or the 'background' permittivity of the layers
            # in the guided mode computation
            'eps_eff'      : 'average',

            # Print information at intermmediate steps
            'verbose'      : True
            }

        if 'gmode_compute' in options.keys():
            if options['gmode_compute'] == 'exact':
                if 'gmode_npts' in options.keys():
                    print("Warning: ignoring 'gmode_npts' supplied in options "
                            "when using 'gmode_compute' = 'exact'")
            elif options['gmode_compute'] != 'interp':
                raise ValueError("options['gmode_compute'] can be one of"
                            "'interp' or 'exact'")

        for key in default_options.keys():
            if key not in options.keys():
                options[key] = default_options[key]

        for key in options.keys():
            if key not in default_options.keys():
                raise ValueError("Unknown run() argument '%s'" % key)

        # Store the dictionary of all options
        self.run_options = options

        # Also store the options as separate attributes
        for (option, value) in options.items():
            # Make sure 'gmode_inds' is a numpy array
            if option.lower() == 'gmode_inds':
                value = np.array(value)
            # Set all the options as class attributes
            setattr(self, option, value)

    def _get_guided(self, gk, kind, mode):
        '''
        Get all the guided mode parameters over 'gk' for mode number 'mode'
        Variable 'indmode' stores the indexes of 'gk' over which a guided
        mode solution was found
        '''

        def interp_coeff(coeffs, il, ic, indmode, gs):
            '''
            Interpolate the A/B coefficient (ic = 0/1) in layer number il
            '''
            param_list = [coeffs[i][il, ic, 0] for i in range(len(coeffs))]
            c_interp = np.interp(gk[indmode], gs, np.array(param_list))
            # cfun = interp1d(gs, np.array(param_list), kind='cubic')
            # c_interp = cfun(gk[indmode])
            return c_interp.ravel()

        def interp_guided(im, ik, omegas, coeffs):
            '''
            Interpolate all the relevant guided mode parameters over gk
            '''
            gs = self.g_array[ik][-len(omegas[ik][im]):]
            indmode = np.argwhere(gk > gs[0]-1e-10).ravel()
            oms = np.interp(gk[indmode], gs, omegas[ik][im])

            As, Bs, chis = (np.zeros((self.N_layers + 2, 
                    indmode.size), dtype=np.complex128) for i in range(3))

            chis = self._get_chi(gk[indmode], oms, self.eps_array_val)
            for il in range(self.N_layers + 2):
                As[il, :] = interp_coeff(coeffs[ik][im], il, 0, indmode, gs)
                Bs[il, :] = interp_coeff(coeffs[ik][im], il, 1, indmode, gs)
            return (indmode, oms, As, Bs, chis)
        
        ik = 0 if self.gmode_compute.lower() == 'interp' else kind

        if mode%2 == 0:
            (indmode, oms, As, Bs, chis) = interp_guided(
                        mode//2, ik, self.omegas_te, self.coeffs_te)
        else:
            (indmode, oms, As, Bs, chis) = interp_guided(
                        mode//2, ik, self.omegas_tm, self.coeffs_tm)
        return (indmode, oms, As, Bs, chis)

    def _get_chi(self, gs, oms, eps_array):
        '''
        Function to get the z-direction wave-vectors chi for in-plane wave
        vectors gk and frequencies oms (can be either a single number or an 
        array of the same shape as gk)
        '''
        chis = []
        for il in range(self.N_layers + 2):
            sqarg = bd.array(eps_array[il]*oms**2 - gs**2, dtype=bd.complex)
            chi = bd.where(bd.real(sqarg)>=0, bd.sqrt(sqarg), 
                                1j*bd.sqrt(-sqarg))
            chis.append(chi)
        return bd.array(chis, dtype=bd.complex)

    def _get_rad(self, gkr, omr, pol, clad):
        '''
        Get all the radiative mode parameters over 'gkr' at frequency 'omr' with
        polarization 'pol' and out-going in cladding 'clad'
        '''
        chis = self._get_chi(gkr, omr, self.eps_array)       
        (Xs, Ys) = rad_modes(omr, gkr, self.eps_array, 
                                    self.d_array, pol, clad)
        
        return (Xs, Ys, chis)

    def _z_to_lind(self, z):
        '''
        Get a layer index corresponding to a position z. Claddings are included 
        as first and last layer
        '''

        z_max = self.phc.claddings[0].z_max
        lind = 0 # Index denoting which layer (including claddings) z is in 
        while z > z_max and lind<self.N_layers:
            lind+=1
            z_max = self.phc.layers[lind-1].z_max
        if z > z_max and lind==self.N_layers: lind += 1

        return lind

    def compute_guided(self, g_array):
        '''
        Compute the guided modes using the slab_modes module, reshape the 
        results appropriately and store
        '''
        self.g_array.append(g_array)
        self.gmode_te = self.gmode_inds[np.remainder(self.gmode_inds, 2) == 0]
        self.gmode_tm = self.gmode_inds[np.remainder(self.gmode_inds, 2) != 0]
        reshape_list = lambda x: [list(filter(lambda y: y is not None, i)) \
                        for i in zip_longest(*x)]

        if self.gmode_te.size > 0:
            (omegas_te, coeffs_te) = guided_modes(g_array,
                    self.eps_array_val, self.d_array_val, 
                    step=self.gmode_step, n_modes=1 + np.amax(self.gmode_te)//2, 
                    tol=self.gmode_tol, pol='TE')
            omte = reshape_list(omegas_te)
            self.omegas_te.append(reshape_list(omegas_te))
            self.coeffs_te.append(reshape_list(coeffs_te))
            
        if self.gmode_tm.size > 0:
            (omegas_tm, coeffs_tm) = guided_modes(g_array, 
                    self.eps_array_val, self.d_array_val, 
                    step=self.gmode_step, n_modes=1 + np.amax(self.gmode_tm)//2, 
                    tol=self.gmode_tol, pol='TM')
            self.omegas_tm.append(reshape_list(omegas_tm))
            self.coeffs_tm.append(reshape_list(coeffs_tm))     

    def compute_ft(self):
        '''
        Compute the unique FT coefficients of the permittivity, eps(g-g') for
        every layer in the PhC.
        '''
        (n1max, n2max) = (self.n1g, self.n2g)
        G1 = - self.gvec + self.gvec[:, [0]]
        G2 = np.zeros((2, n1max*n2max))

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
            self.T1.append(T1)
            self.T2.append(T2)

        # Store the g-vectors to which T1 and T2 correspond
        self.G1 = G1
        self.G2 = G2

    def get_eps_xy(self, z, Nx=100, Ny=100):
        '''
        Get the permittivity of the PhC cross-sections as computed from an 
        inverse Fourier transform with the GME reciprocal lattice vectors.
        '''
        (xgrid, ygrid) = self.phc.lattice.xy_grid(Nx=Nx, Ny=Ny)
        lind = self._z_to_lind(z)

        ft_coeffs = np.hstack((self.T1[lind], self.T2[lind], 
                            np.conj(self.T1[lind]), np.conj(self.T2[lind])))
        gvec = np.hstack((self.G1, self.G2, 
                                -self.G1, -self.G2))

        eps_r = ftinv(ft_coeffs, gvec, xgrid, ygrid)

        return (eps_r, xgrid, ygrid)

    def run(self, kpoints=np.array([[0], [0]]), **kwargs):
        ''' 
        Run the simulation. Basically:

        - Compute the guided modes over a grid of g-points
        - Compute the inverse matrix of the FT of the permittivity eps(G - G')
            in every phc layer, with G, G' reciprocal lattice vectors
        - Iterate over the k points:
            - compute the Hermitian matrix for diagonalization 
            - compute the real eigenvalues and corresponding eigenvectors
        '''
        t_start = time.time()
        
        def print_vb(*args):
            if self.verbose==True: print(*args)

        # Parse the input arguments
        self._run_options(kwargs)
        # Bloch momenta over which band structure is simulated 
        self.kpoints = kpoints

        # Array of effective permittivity of every layer (including claddings)
        if self.eps_eff=='average':
            layer_eps = 'eps_avg'
        elif self.eps_eff=='background':
            layer_eps = 'eps_b'
        else:
            raise ValueError("'eps_eff' can be 'average' or 'background'")
        
        # Store an array of the effective permittivity for every layer
        #(including claddings)
        eps_array = bd.array(list(getattr(layer, layer_eps)
            for layer in [self.phc.claddings[0]] + self.phc.layers + 
            [self.phc.claddings[1]]), dtype=np.float).ravel()
        # A separate array where the values are converted from ArrayBox to numpy
        # array, if using the 'autograd' backend.
        eps_array_val = np.array(list(get_value(getattr(layer, layer_eps))
            for layer in [self.phc.claddings[0]] + self.phc.layers + 
            [self.phc.claddings[1]]), dtype=np.float).ravel()
        
        # Store an array of thickness of every layer (not including claddings)
        d_array = bd.array(list(layer.d for layer in \
            self.phc.layers), dtype=np.float).ravel()
        # A separate array where the values are converted from ArrayBox to numpy
        # array, if using the 'autograd' backend.
        d_array_val = np.array(list(get_value(layer.d) for layer in \
            self.phc.layers), dtype=np.float).ravel()

        (self.eps_array_val, self.eps_array, self.d_array_val, self.d_array) = \
                                (eps_array_val, eps_array, d_array_val, d_array)

        # Initialize attributes for guided-mode storing
        self.g_array = []
        '''
        List dimensions in order: 
            - k-point index (length 1 if gmode_compute=='interp')
            - guided mode index
            - g_array index
        '''
        self.omegas_te = []
        self.coeffs_te = []
        self.omegas_tm = []
        self.coeffs_tm = []

        # Pre-compute guided modes if the 'interp' option is used
        if self.gmode_compute.lower() == 'interp':
            t = time.time()
            kmax = np.amax(np.sqrt(np.square(kpoints[0, :]) +
                                np.square(kpoints[1, :])))
            Gmax = np.amax(np.sqrt(np.square(self.gvec[0, :]) +
                                np.square(self.gvec[1, :])))

            # Array of g-points over which the guided modes will be computed
            g_array = np.linspace(0, Gmax + kmax, self.gmode_npts)
            self.compute_guided(g_array)
            self.t_guided = time.time()-t

            print_vb("%1.4f seconds for guided mode computation"% 
                            (time.time()-t))
        else:
            print_vb("Using the 'exact' method of guided mode computation")
            self.t_guided = 0

        # Compute inverse matrix of FT of permittivity
        t = time.time()
        self.compute_ft()   # Just in case something changed after __init__()
        self.compute_eps_inv()
        print_vb("%1.4f seconds for inverse matrix of Fourier-space "
            "permittivity"% (time.time()-t))

        # Loop over all k-points, construct the matrix, diagonalize, and compute
        # radiative losses for the modes requested by kinds_rad and minds_rad
        t_rad = 0
        freqs = []
        freqs_im = []
        eigvecs = []
        for ik, k in enumerate(kpoints.T):
            print_vb("Running k-point %d of %d" % (ik+1, kpoints.shape[1]))
            mat = self.construct_mat(kind=ik)
            if self.numeig > mat.shape[0]:
                raise ValueError("Requested number of eigenvalues 'numeig' "
                    "larger than total size of basis set. Reduce 'numeig' or "
                    "increase 'gmax'. ")

            # Diagonalize using numpy.linalg.eigh() for now; should maybe switch 
            # to scipy.sparse.linalg.eigsh() in the future
            # NB: we shift the matrix by np.eye to avoid problems at the zero-
            # frequency mode at Gamma
            (freq2, evec) = bd.eigh(mat + bd.eye(mat.shape[0]))
            freq = bd.sort(bd.sqrt(bd.abs(freq2[:self.numeig]
                        - bd.ones(self.numeig))))
            freqs.append(freq)
            eigvecs.append(evec[:, :self.numeig])

        # Store the eigenfrequencies taking the standard reduced frequency 
        # convention for the units (2pi a/c)
        self.freqs = bd.array(freqs)/2/np.pi
        self.eigvecs = eigvecs

        print_vb("%1.4f seconds total time to run"% (time.time()-t_start))
        if self.gmode_compute.lower() == 'exact':
            print_vb("%1.4f seconds of that was guided modes computation"
                    % self.t_guided)

        if self.compute_im:
            print_vb("Computing imaginary part of the freqeuncies")
            t = time.time()
            self.run_im()
            print_vb("%1.4f seconds to compute radiative rates"
                    % (time.time()-t))

    def compute_eps_inv(self):
        '''
        Construct the inverse FT matrices for the permittivity in each layer
        '''

        # List of inverse permittivity matrices for every layer
        self.eps_inv_mat = []

        for it, T1 in enumerate(self.T1):
            # For now we just use the numpy inversion. Later on we could 
            # implement the Toeplitz-Block-Toeplitz inversion (faster)
            eps_mat = bd.toeplitz_block(self.n1g, T1, self.T2[it])
            self.eps_inv_mat.append(bd.inv(eps_mat))

    def construct_mat(self, kind):
        '''
        Construct the Hermitian matrix for diagonalization for a given k
        '''

        # G + k vectors
        gkx = self.gvec[0, :] + self.kpoints[0, kind] + 1e-10
        gky = self.gvec[1, :] + self.kpoints[1, kind]
        gk = np.sqrt(np.square(gkx) + np.square(gky))

        # Compute the guided modes over gk if using the 'exact' method
        if self.gmode_compute.lower() == 'exact':
            t = time.time()
            g_array = np.sort(gk)
            # Expand boundaries a bit to make sure we get all the modes
            # Note that the 'exact' computation still uses interpolation, 
            # but the grid is defined by the actual gk values
            g_array[0] -= 1e-6
            g_array[-1] += 1e-6
            self.compute_guided(g_array)
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
            raise RuntimeError("No guided modes were found for k-index %d. "
                "One possibility is "
                "that the effective permittivity of all layers is smaller than "
                "that of at least one cladding. Reconsider your structure, or "
                "try changing 'eps_eff' from 'average' to 'background' in "
                "the options to GuidedModeExp.run()." % kind)
        else:
            self.gmode_include.append(np.array(gmode_include))

        # We now construct the matrix block by block
        mat_blocks = [[] for i in range(self.gmode_include[-1].size)]

        for im1 in range(self.gmode_include[-1].size):
            mode1 = self.gmode_include[-1][im1]
            (indmode1, oms1, As1, Bs1, chis1) = \
                        self._get_guided(gk, kind, mode1)
            modes_numg.append(indmode1.size)

            if len(modes_numg) > 1:
                mat_blocks[im1].append(bd.zeros((modes_numg[-1], 
                    bd.sum(modes_numg[:-1]))))

            for im2 in range(im1, self.gmode_include[-1].size):
                mode2 = self.gmode_include[-1][im2]
                (indmode2, oms2, As2, Bs2, chis2) = \
                            self._get_guided(gk, kind, mode2)

                if mode1%2 + mode2%2 == 0:
                    mat_block = matrix_elements.mat_te_te(
                                    self.eps_array_val, self.d_array_val, 
                                    self.eps_inv_mat, indmode1, oms1,
                                    As1, Bs1, chis1, indmode2, oms2, As2, Bs2, 
                                    chis2, qq)
                elif mode1%2 + mode2%2 == 2:
                    mat_block = matrix_elements.mat_tm_tm(
                                    self.eps_array_val, self.d_array_val, 
                                    self.eps_inv_mat, gk, indmode1, oms1,
                                    As1, Bs1, chis1, indmode2, oms2, As2, Bs2, 
                                    chis2, pp)
                elif mode1%2==0 and mode2%2==1:
                    mat_block = matrix_elements.mat_te_tm(
                                    self.eps_array_val, self.d_array_val, 
                                    self.eps_inv_mat, indmode1, oms1,
                                    As1, Bs1, chis1, indmode2, oms2, As2, Bs2, 
                                    chis2, pq.transpose(), 1j)
                elif mode1%2==1 and mode2%2==0:
                    # Note: TM-TE is just hermitian conjugate of TE-TM
                    # with switched indexes 1 <-> 2
                    mat_block = matrix_elements.mat_te_tm(
                                    self.eps_array_val, self.d_array_val, 
                                    self.eps_inv_mat, indmode2, oms2,
                                    As2, Bs2, chis2, indmode1, oms1, As1, Bs1, 
                                    chis1, pq, -1j) 
                    mat_block = bd.conj(bd.transpose(mat_block))

                mat_blocks[im1].append(mat_block)

        # Store how many modes total were included in the matrix
        self.N_basis.append(np.sum(modes_numg))
        # Store a list of how many g-points were used for each mode index
        self.modes_numg.append(modes_numg) 

        # Stack all the blocks together
        mat_rows = [bd.hstack(mb) for mb in mat_blocks]
        mat = bd.vstack(mat_rows)

        '''
        If the matrix is within numerical precision to real symmetric, 
        make it explicitly so. This will speed up the diagonalization and will
        often be the case, specifically when there is in-plane inversion
        symmetry in the PhC elementary cell
        '''
        if bd.amax(bd.abs(bd.imag(mat))) < 1e-10*bd.amax(bd.abs(bd.real(mat))):
            mat = bd.real(mat)

        '''
        Make the matrix Hermitian (note that only upper part of the blocks, i.e.
        (im2 >= im1) was computed
        '''
        mat = bd.triu(mat) + bd.transpose(bd.conj(bd.triu(mat, 1)))
        self.mat = mat  

        return mat

    def run_im(self):
        '''
        Compute the radiative rates associated to all the freqeuncies that were 
        computed during self.run()
        '''
        if len(self.freqs)==0:
            raise RuntimeError("Run the GME computation first!")

        freqs_i = [] # Imaginary part of frequencies
        cl = []      # Coupling constants to lower-cladding radiative modes
        cu = []      # Coupling constants to upper-cladding radiative modes
        for kind in range(len(self.freqs)):
            minds = np.arange(0, self.numeig)
            (freqs_im, coup_l, coup_u) = self.compute_rad(kind, minds)
            freqs_i.append(freqs_im)
            cl.append(coup_l)
            cu.append(coup_u)

        self.freqs_im = bd.array(freqs_i)
        self.coup_l = cl
        self.coup_u = cu

    def compute_rad(self, kind, minds=[0]):
        '''
        Compute the radiation losses of the eigenmodes after the dispersion
        has been computed.
        Input
            kind            : index of the k-point for the computation
            minds           : indexes of which modes to be computed 
                              (max value must be smaller than self.numeig)
        Output
            freqs_im        : imaginary part of the frequencies 
        '''
        if len(self.freqs)==0:
            raise RuntimeError("Run the GME computation first!")
        if np.max(np.array(minds)) > self.numeig - 1:
            raise ValueError("Requested mode index out of range for the %d "
                "stored eigenmodes" % self.numeig)
        
        # G + k vectors
        gkx = self.gvec[0, :] + self.kpoints[0, kind] + 1e-10
        gky = self.gvec[1, :] + self.kpoints[1, kind]
        gk = np.sqrt(np.square(gkx) + np.square(gky))

        # Iterate over all the modes to be computed
        rad_tot = []
        (coup_l, coup_u) = ([], [])
        for im in minds:
            omr = 2*np.pi*self.freqs[kind, im]
            evec = self.eigvecs[kind][:, im]

            # Reciprocal vedctors within the radiative cone for the claddings
            indmoder = [bd.argwhere(gk**2 <= \
                    self.phc.claddings[0].eps_avg*omr**2).ravel(), 
                        bd.argwhere(gk**2 <= \
                    self.phc.claddings[1].eps_avg*omr**2).ravel()
                        ]

            gkr = [gk[indmode] for indmode in indmoder]
            rad_coup = {'te': [bd.zeros((indmode.size, ), dtype=bd.complex)
                            for indmode in indmoder],
                        'tm': [bd.zeros((indmode.size, ), dtype=bd.complex) 
                            for indmode in indmoder]}

            [Xs, Ys, chis] = [{'te': [], 'tm': []} for i in range(3)]
            for clad_ind in [0, 1]:
                for pol in ['te', 'tm']:
                    (X, Y, chi) = self._get_rad(gkr[clad_ind], omr, 
                            pol=pol, clad=clad_ind)
                    Xs[pol].append(X)
                    Ys[pol].append(Y)
                    chis[pol].append(chi)

            # Iterate over the 'gmode_include' basis of the PhC mode
            count = 0
            for im1 in range(self.gmode_include[kind].size):
                mode1 = self.gmode_include[kind][im1]
                (indmode1, oms1, As1, Bs1, chis1) = \
                            self._get_guided(gk, kind, mode1)
                # Iterate over lower cladding (0) and upper cladding (1)
                for clad_ind in [0, 1]:         
                    # Radiation to TE-polarized states
                    if mode1%2 == 0:
                        qq = (np.outer(gkx[indmode1], gkx[indmoder[clad_ind]])
                            + np.outer(gky[indmode1], gky[indmoder[clad_ind]]))\
                            / np.outer(gk[indmode1], gk[indmoder[clad_ind]])
                        rad = matrix_elements.rad_te_te(
                            self.eps_array_val, self.d_array, 
                            self.eps_inv_mat, indmode1, oms1, As1, Bs1, chis1, 
                            indmoder[clad_ind], omr, Xs['te'][clad_ind], 
                            Ys['te'][clad_ind], chis['te'][clad_ind], qq)
                    else:
                        pq = (np.outer(gkx[indmode1], gky[indmoder[clad_ind]])
                            - np.outer(gky[indmode1], gkx[indmoder[clad_ind]]))\
                            / np.outer(gk[indmode1], gk[indmoder[clad_ind]])
                        rad = matrix_elements.rad_tm_te(
                            self.eps_array_val, self.d_array, 
                            self.eps_inv_mat, indmode1, oms1, As1, Bs1, chis1, 
                            indmoder[clad_ind], omr, Xs['te'][clad_ind], 
                            Ys['te'][clad_ind], chis['te'][clad_ind], pq)

                    rad = rad*bd.conj(evec[count:
                        count+self.modes_numg[kind][im1]][:, np.newaxis])
                    rad_coup['te'][clad_ind] += bd.sum(rad, axis=0)

                    # Radiation to TM-polarized states
                    if mode1%2 == 0:
                        qp = (np.outer(gky[indmode1], gkx[indmoder[clad_ind]])
                            - np.outer(gkx[indmode1], gky[indmoder[clad_ind]]))\
                            / np.outer(gk[indmode1], gk[indmoder[clad_ind]])
                        rad = matrix_elements.rad_te_tm(
                            self.eps_array_val, self.d_array, 
                            self.eps_inv_mat, indmode1, oms1, As1, Bs1, chis1, 
                            indmoder[clad_ind], omr, Xs['tm'][clad_ind], 
                            Ys['tm'][clad_ind], chis['tm'][clad_ind], qp)
                    else:
                        pp = (np.outer(gkx[indmode1], gkx[indmoder[clad_ind]])
                            + np.outer(gky[indmode1], gky[indmoder[clad_ind]]))\
                            / np.outer(gk[indmode1], gk[indmoder[clad_ind]])
                        rad = matrix_elements.rad_tm_tm(
                            self.eps_array_val, self.d_array, 
                            self.eps_inv_mat, gk, indmode1, oms1, As1, Bs1, 
                            chis1, indmoder[clad_ind], omr, Xs['tm'][clad_ind], 
                            Ys['tm'][clad_ind], chis['tm'][clad_ind], pp)
                    rad = rad*bd.conj(evec[count:
                        count+self.modes_numg[kind][im1]][:, np.newaxis])
                    rad_coup['tm'][clad_ind] += bd.sum(rad, axis=0)
                count += self.modes_numg[kind][im1]
            rad_dos = [self.phc.claddings[i].eps_avg/bd.sqrt(
                    self.phc.claddings[i].eps_avg*omr**2 - gkr[i]**2) / \
                    4 / np.pi for i in [0, 1]]
            rad_t = 0 # will sum up contributions from all the channels
            (c_l, c_u) = ({}, {})
            for pol in ['te', 'tm']:
                c_l[pol] = rad_coup[pol][0]
                c_u[pol] = rad_coup[pol][1]
                rad_t = rad_t + \
                    np.pi*bd.sum(bd.square(bd.abs(c_l[pol]))*rad_dos[0]) + \
                    np.pi*bd.sum(bd.square(bd.abs(c_u[pol]))*rad_dos[1])
            rad_tot.append(bd.imag(bd.sqrt(omr**2 + 1j*rad_t)))

            # NB: think about the normalization of the couplings!
            coup_l.append(c_l)
            coup_u.append(c_u)
        # Compute radiation rate in units of frequency  
        freqs_im = bd.array(rad_tot)/2/np.pi
        return (freqs_im, coup_l, coup_u)

    def ft_field_xy(self, field, kind, mind, z):
        '''
        Compute the 'H', 'D' or 'E' field FT in the xy-plane at position z
        '''
        evec = self.eigvecs[kind][:, mind]
        omega = self.freqs[kind][mind]*2*np.pi
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

        if field.lower()=='h':
            count = 0
            [Hx_ft, Hy_ft, Hz_ft] = [bd.zeros(gnorm.shape, dtype=np.complex128)
                                     for i in range(3)]
            for im1 in range(self.gmode_include[kind].size):
                mode1 = self.gmode_include[kind][im1]
                (indmode, oms, As, Bs, chis) = \
                            self._get_guided(gnorm, kind, mode1)

                # TE-component
                if mode1%2==0:
                    # Do claddings separately
                    if lind==0:
                        H = Bs[0, :] * bd.exp(-1j*chis[0, :]
                                            *(z-self.phc.claddings[0].z_max))
                        Hx = H * 1j*chis[0, :] * px[indmode]
                        Hy = H * 1j*chis[0, :] * py[indmode]
                        Hz = H * 1j*gnorm[indmode]
                    elif lind==self.eps_array.size-1:
                        H = As[-1, :] * bd.exp(1j*chis[-1, :]
                                            *(z-self.phc.claddings[1].z_min))
                        Hx = -H * 1j*chis[-1, :] * px[indmode]
                        Hy = -H * 1j*chis[-1, :] * py[indmode]
                        Hz = H * 1j*gnorm[indmode]
                    else:
                        z_cent = (self.phc.layers[lind-1].z_min + 
                                    self.phc.layers[lind-1].z_max) / 2
                        zp = bd.exp(1j*chis[lind, :]*(z-z_cent))
                        zn = bd.exp(-1j*chis[lind, :]*(z-z_cent))
                        Hxy = -1j*As[lind, :]*zp + 1j*Bs[lind, :]*zn
                        Hx = Hxy * chis[lind, :] * px[indmode]
                        Hy = Hxy * chis[lind, :] * py[indmode]
                        Hz = 1j*(As[lind, :]*zp + Bs[lind, :]*zn) *\
                                gnorm[indmode]

                # TM-component
                elif mode1%2==1:
                    Hz = bd.zeros(indmode.shape)
                    # Do claddings separately
                    if lind==0:
                        H = Bs[0, :] * bd.exp(-1j*chis[0, :]
                                            *(z-self.phc.claddings[0].z_max))
                        Hx = H * qx[indmode]
                        Hy = H * qy[indmode]
                    elif lind==self.eps_array.size-1:
                        H = As[-1, :] * bd.exp(1j*chis[-1, :]
                                            *(z-self.phc.claddings[1].z_min))
                        Hx = H * qx[indmode]
                        Hy = H * qy[indmode]
                    else:
                        z_cent = (self.phc.layers[lind-1].z_min + 
                                    self.phc.layers[lind-1].z_max) / 2
                        zp = bd.exp(1j*chis[lind, :]*(z-z_cent))
                        zn = bd.exp(-1j*chis[lind, :]*(z-z_cent))
                        Hxy = As[lind, :]*zp + Bs[lind, :]*zn
                        Hx = Hxy * qx[indmode]
                        Hy = Hxy * qy[indmode]

                Hx_ft[indmode] += evec[count:count+self.modes_numg[kind][im1]]*Hx
                Hy_ft[indmode] += evec[count:count+self.modes_numg[kind][im1]]*Hy
                Hz_ft[indmode] += evec[count:count+self.modes_numg[kind][im1]]*Hz
                count += self.modes_numg[kind][im1]

            return (Hx_ft, Hy_ft, Hz_ft)

        elif field.lower()=='d' or field.lower()=='e':
            count = 0
            [Dx_ft, Dy_ft, Dz_ft] = [bd.zeros(gnorm.shape, dtype=np.complex128)
                                     for i in range(3)]
            for im1 in range(self.gmode_include[kind].size):
                mode1 = self.gmode_include[kind][im1]
                (indmode, oms, As, Bs, chis) = \
                            self._get_guided(gnorm, kind, mode1)

                # TE-component
                if mode1%2==0:
                    Dz = bd.zeros(indmode.shape)
                    # Do claddings separately
                    if lind==0:
                        D = 1j * Bs[0, :] * oms**2 / omega * \
                            self.eps_array[0] * bd.exp(-1j*chis[0, :] * \
                            (z-self.phc.claddings[0].z_max))
                        Dx = D * qx[indmode]
                        Dy = D * qy[indmode]
                    elif lind==self.eps_array.size-1:
                        D = 1j * As[-1, :] * oms**2 / omega * \
                            self.eps_array[-1] * bd.exp(1j*chis[-1, :] * \
                            (z-self.phc.claddings[1].z_min))
                        Dx = D * qx[indmode]
                        Dy = D * qy[indmode]
                    else:
                        z_cent = (self.phc.layers[lind-1].z_min + 
                                    self.phc.layers[lind-1].z_max) / 2
                        zp = bd.exp(1j*chis[lind, :]*(z-z_cent))
                        zn = bd.exp(-1j*chis[lind, :]*(z-z_cent))
                        Dxy = 1j*oms**2 / omega * \
                            self.eps_array[lind] * \
                            (As[lind, :]*zp + Bs[lind, :]*zn)
                        Dx = Dxy * qx[indmode]
                        Dy = Dxy * qy[indmode]

                # TM-component
                elif mode1%2==1:
                    if lind==0:
                        D = 1j / omega * Bs[0,:] * \
                            bd.exp(-1j*chis[0,:] * \
                            (z-self.phc.claddings[0].z_max))
                        Dx = D * 1j*chis[0,:] * px[indmode]
                        Dy = D * 1j*chis[0,:] * py[indmode]
                        Dz = D * 1j*gnorm[indmode]
                    elif lind==self.eps_array.size-1:
                        D = 1j / omega * As[-1,:] * \
                            bd.exp(1j*chis[-1, :] * \
                            (z-self.phc.claddings[1].z_min))
                        Dx = -D * 1j*chis[-1, :] * px[indmode]
                        Dy = -D * 1j*chis[-1, :] * py[indmode]
                        Dz = D * 1j*gnorm[indmode]
                    else:
                        z_cent = (self.phc.layers[lind-1].z_min + 
                                    self.phc.layers[lind-1].z_max) / 2
                        zp = bd.exp(1j*chis[lind, :]*(z-z_cent))
                        zn = bd.exp(-1j*chis[lind, :]*(z-z_cent))
                        Dxy = 1 / omega * chis[lind, :] * \
                            (As[lind, :]*zp - Bs[lind, :]*zn)
                        Dx = Dxy * px[indmode]
                        Dy = Dxy * py[indmode]
                        Dz = -1 / omega * gnorm[indmode] * \
                            (As[lind, :]*zp + Bs[lind, :]*zn)

                Dx_ft[indmode] += evec[count:count+self.modes_numg[kind][im1]]*Dx
                Dy_ft[indmode] += evec[count:count+self.modes_numg[kind][im1]]*Dy
                Dz_ft[indmode] += evec[count:count+self.modes_numg[kind][im1]]*Dz
                count += self.modes_numg[kind][im1]

            if field.lower()=='d':
                return (Dx_ft, Dy_ft, Dz_ft)
            else:
                # Get E-field by convolving FT(1/eps) with FT(D)
                Ex_ft = self.eps_inv_mat[lind].dot(Dx_ft)
                Ey_ft = self.eps_inv_mat[lind].dot(Dy_ft)
                Ez_ft = self.eps_inv_mat[lind].dot(Dz_ft)
                return (Ex_ft, Ey_ft, Ez_ft)

    def get_field_xy(self, field, kind, mind, z,
                    component='xyz', Nx=100, Ny=100):
        '''
        Get the 'field' ('H' or 'D') at an xy-plane at position z for mode 
        number mind at k-vector kind.
        Returns a dictionary with the ['x'], ['y'], and ['z'] components of the 
        corresponding field; only the ones requested in 'comp' are computed 
        '''

        # Make a grid in the x-y plane
        (xgrid, ygrid) = self.phc.lattice.xy_grid(Nx=Nx, Ny=Ny)

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

    def get_field_xz(self, field, kind, mind, y,
                    component='xyz', Nx=100, Nz=100, dist=1.):
        '''
        Hacked version for getting the field in an xz plane by stitching 
        together xy "planes" for various z slices
        '''
        xgrid = self.phc.lattice.xy_grid(Nx=Nx, Ny=2)[0]
        ygrid = np.array([y])
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
                        fi[comp].append(ftinv(ft[comp][i], 
                                            self.gvec, xgrid, ygrid).ravel())
                    fi[comp] = bd.array(fi[comp])
            else:
                raise ValueError("'component' can be any combination of "
                    "'x', 'y', and 'z' only.")

        return (fi, xgrid, zgrid)

    def get_field_yz(self, field, kind, mind, x,
                    component='xyz', Ny=100, Nz=100, dist=1.):
        '''
        Hacked version for getting the field in a yz plane by stitching 
        together xy "planes" for various z slices
        '''
        xgrid = np.array([x])
        ygrid = self.phc.lattice.xy_grid(Nx=2, Ny=Ny)[1]        
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
                        fi[comp].append(ftinv(ft[comp][i], 
                                            self.gvec, xgrid, ygrid).ravel())
                    fi[comp] = bd.array(fi[comp])
            else:
                raise ValueError("'component' can be any combination of "
                    "'x', 'y', and 'z' only.")

        return (fi, ygrid, zgrid)