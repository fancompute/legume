import numpy as np
import matplotlib.pyplot as plt
from legume.utils import ftinv
import legume.viz as viz
from legume.backend import backend as bd

class PlaneWaveExp(object):
    '''
    Main simulation class of the guided-mode expansion
    '''
    def __init__(self, layer, gmax=3, eps_eff=None):
        # Object of class Layer which will be simulated
        self.layer = layer
        # Maximum reciprocal lattice wave-vector length in units of 2pi/a
        self.gmax = gmax

        if not eps_eff:
            eps_eff = layer.eps_b

        self.eps_eff = eps_eff

        # Initialize the reciprocal lattice vectors and compute the eps FT
        self._init_reciprocal()
        self.compute_ft()

    def _init_reciprocal(self):
        '''
        Initialize reciprocal lattice vectors based on self.layer and self.gmax
        '''
        n1max = np.int_((2*np.pi*self.gmax)/np.linalg.norm(self.layer.lattice.b1))
        n2max = np.int_((2*np.pi*self.gmax)/np.linalg.norm(self.layer.lattice.b2))

        # This constructs the reciprocal lattice in a way that is suitable
        # for Toeplitz-Block-Toeplitz inversion of the permittivity in the main
        # code. However, one caveat is that the hexagonal lattice symmetry is 
        # not preserved. For that, the option to construct a hexagonal mesh in 
        # reciprocal space could is needed.
        inds1 = np.tile(np.arange(-n1max, n1max + 1), (2*n2max + 1, 1))  \
                         .reshape((2*n2max + 1)*(2*n1max + 1), order='F')
        inds2 = np.tile(np.arange(-n2max, n2max + 1), 2*n1max + 1)

        gvec = self.layer.lattice.b1[:, np.newaxis].dot(inds1[np.newaxis, :]) + \
                self.layer.lattice.b2[:, np.newaxis].dot(inds2[np.newaxis, :])

        # Save the reciprocal lattice vectors
        self.gvec = gvec

        # Save the number of vectors along the b1 and the b2 directions 
        # Note: gvec.shape[1] = n1g*n2g
        self.n1g = 2*n1max + 1
        self.n2g = 2*n2max + 1

    def compute_ft(self):
        '''
        Compute the unique FT coefficients of the permittivity, eps(g-g')
        '''
        (n1max, n2max) = (self.n1g, self.n2g)
        G1 = - self.gvec + self.gvec[:, [0]]
        G2 = np.zeros((2, n1max*n2max))

        for ind1 in range(n1max):
            G2[:, ind1*n2max:(ind1+1)*n2max] = - self.gvec[:, [ind1*n2max]] + \
                            self.gvec[:, range(n2max)]

        # Compute and store T1 and T2
        self.T1 = self.layer.compute_ft(G1)
        self.T2 = self.layer.compute_ft(G2)

        # Store the g-vectors to which T1 and T2 correspond
        self.G1 = G1
        self.G2 = G2

    def get_eps_xy(self, Nx=100, Ny=100, z=0):
        '''
        Plot the permittivity of the layer as computed from an 
        inverse Fourier transform with the GME reciprocal lattice vectors.
        z is technically unused, but useful for viz.structure_ft
        '''
        (xgrid, ygrid) = self.layer.lattice.xy_grid(Nx=Nx, Ny=Ny)

        ft_coeffs = np.hstack((self.T1, self.T2, 
                            np.conj(self.T1), np.conj(self.T2)))
        gvec = np.hstack((self.G1, self.G2, -self.G1, -self.G2))

        eps_r = ftinv(ft_coeffs, gvec, xgrid, ygrid)
        return (eps_r, xgrid, ygrid)

    def run(self, kpoints=np.array([[0], [0]]), pol='te', numeig=10):
        ''' 
        Run the simulation. Input:
            - kpoints, [2xNk] numpy array over which band structure is simulated
            - pol, polarization of the computation (TE/TM)
            - numeig, number of eigenmodes to be stored
        '''
         
        self.kpoints = kpoints
        self.pol = pol.lower()
        # Change this if switching to a solver that allows for variable numeig
        self.numeig = numeig

        self.compute_ft()
        self.compute_eps_inv()

        freqs = []
        self.eigvecs = []
        for ik, k in enumerate(kpoints.T):
            # Construct the matrix for diagonalization
            if self.pol == 'te':
                mat = bd.dot(bd.transpose(k[:, bd.newaxis] + self.gvec), 
                                (k[:, bd.newaxis] + self.gvec))
                mat = mat * self.eps_inv_mat 
                
            elif self.pol == 'tm':
                Gk = bd.sqrt(bd.square(k[0] + self.gvec[0, :]) + \
                        bd.square(k[1] + self.gvec[1, :]))
                mat = bd.outer(Gk, Gk)
                mat = mat * self.eps_inv_mat
            else:
                raise ValueError("Polarization should be 'TE' or 'TM'")

            # Diagonalize using numpy.linalg.eigh() for now; should maybe switch 
            # to scipy.sparse.linalg.eigsh() in the future
            # NB: we shift the matrix by np.eye to avoid problems at the zero-
            # frequency mode at Gamma
            (freq2, evecs) = bd.eigh(mat + bd.eye(mat.shape[0]))
            freq1 = bd.sqrt(bd.abs(freq2 - bd.ones(mat.shape[0])))/2/np.pi
            i_sort = bd.argsort(freq1)[0:self.numeig]
            freq = freq1[i_sort]
            evec = evecs[:, i_sort]
            freqs.append(freq)
            self.eigvecs.append(evec)

        # Store the eigenfrequencies taking the standard reduced frequency 
        # convention for the units (2pi a/c)    
        self.freqs = bd.array(freqs)
        self.mat = mat

    def compute_eps_inv(self):
        '''
        Construct the inverse FT matrix of the permittivity
        '''

        # For now we just use the numpy inversion. Later on we could 
        # implement the Toeplitz-Block-Toeplitz inversion (faster)
        eps_mat = bd.toeplitz_block(self.n1g, self.T1, self.T2)
        self.eps_inv_mat = bd.inv(eps_mat)

    def ft_field_xy(self, field, kind, mind):
        '''
        Compute the 'H', 'D' or 'E' field FT in the xy-plane at position z
        Nothing really depends on z but we keep it here for compatibility with
        the GuidedModeExp methods and legume.viz.field
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

        if field.lower()=='h':
            if self.pol == 'te':
                Hx_ft = bd.zeros(gnorm.shape)
                Hy_ft = bd.zeros(gnorm.shape)
                Hz_ft = evec

            elif self.pol == 'tm':
                Hx_ft = evec * qx
                Hy_ft = evec * qy
                Hz_ft = bd.zeros(gnorm.shape)

            return (Hx_ft, Hy_ft, Hz_ft)

        elif field.lower()=='d' or field.lower()=='e':
            if self.pol == 'te':
                Dx_ft = 1j / omega * evec * qx
                Dy_ft = 1j / omega * evec * qy
                Dz_ft = bd.zeros(gnorm.shape)

            elif self.pol == 'tm':
                Dx_ft = bd.zeros(gnorm.shape)
                Dy_ft = bd.zeros(gnorm.shape)
                Dz_ft = 1j / omega * evec

            if field.lower()=='d':
                return (Dx_ft, Dy_ft, Dz_ft)
            else:
                # Get E-field by convolving FT(1/eps) with FT(D)
                Ex_ft = bd.dot(self.eps_inv_mat, Dx_ft)
                Ey_ft = bd.dot(self.eps_inv_mat, Dy_ft)
                Ez_ft = bd.dot(self.eps_inv_mat, Dz_ft)
                return (Ex_ft, Ey_ft, Ez_ft)


    def get_field_xy(self, field, kind, mind, z=0,
                    component='xyz', Nx=100, Ny=100):
        '''
        Get the 'field' ('H', 'D', or 'E') at an xy-plane at position z for mode 
        number 'mind' at k-vector 'kind'.
        Returns a dictionary with the ['x'], ['y'], and ['z'] components of the 
        corresponding field; only the ones requested in 'comp' are computed 
        '''

        # Make a grid in the x-y plane
        (xgrid, ygrid) = self.layer.lattice.xy_grid(Nx=Nx, Ny=Ny)

        # Get the field fourier components
        ft, fi = {}, {}
        (ft['x'], ft['y'], ft['z']) = self.ft_field_xy(field, kind, mind)

        for comp in component:
            if comp in ft.keys():
                if not (comp in fi.keys()):
                    fi[comp] = ftinv(ft[comp], self.gvec, xgrid, ygrid)
            else:
                raise ValueError("'component' can be any combination of "
                    "'x', 'y', and 'z' only.")

        return (fi, xgrid, ygrid)
        