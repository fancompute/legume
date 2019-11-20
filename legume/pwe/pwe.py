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
        G1 = self.gvec - self.gvec[:, [0]]
        G2 = np.zeros((2, n1max*n2max))

        for ind1 in range(n1max):
            G2[:, ind1*n2max:(ind1+1)*n2max] = self.gvec[:, [ind1*n2max]] - \
                            self.gvec[:, range(n2max)]

        T1 = bd.zeros(self.gvec.shape[1])
        T2 = bd.zeros(self.gvec.shape[1])
        eps_avg = self.eps_eff
        
        for shape in self.layer.shapes:
            # Note: compute_ft() returns the FT of a function that is one 
            # inside the shape and zero outside
            T1 = T1 + (shape.eps - self.eps_eff)*shape.compute_ft(G1)
            T2 = T2 + (shape.eps - self.eps_eff)*shape.compute_ft(G2)
            eps_avg = eps_avg + (shape.eps - self.eps_eff) * shape.area / \
                            self.layer.lattice.ec_area

        # Apply some final coefficients
        # Note the hacky way to set the zero element so as to work with
        # 'autograd' backend
        ind0 = bd.arange(T1.size) < 1  
        T1 = T1 / self.layer.lattice.ec_area
        T1 = T1*(1-ind0) + eps_avg*ind0
        T2 = T2 / self.layer.lattice.ec_area
        T2 = T2*(1-ind0) + eps_avg*ind0

        # Store T1 and T2
        self.T1 = T1
        self.T2 = T2

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
                mat = mat * self.eps_inv_mat #/ bd.outer(kgnorm[:, bd.newaxis], 
                                              #      kgnorm[bd.newaxis, :])
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
        eps_mat = bd.toeplitz_block(self.n2g, self.T1, self.T2)
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

                Dx_ft[indmode] += evec[count:count+self.modes_numg[kind][im1]]*\
                                    Dx/bd.sqrt(self.phc.lattice.ec_area)
                Dy_ft[indmode] += evec[count:count+self.modes_numg[kind][im1]]*\
                                    Dy/bd.sqrt(self.phc.lattice.ec_area)
                Dz_ft[indmode] += evec[count:count+self.modes_numg[kind][im1]]*\
                                    Dz/bd.sqrt(self.phc.lattice.ec_area)
                count += self.modes_numg[kind][im1]

            if field.lower()=='d':
                return (Dx_ft, Dy_ft, Dz_ft)
            else:
                # Get E-field by convolving FT(1/eps) with FT(D)
                Ex_ft = self.eps_inv_mat[lind].dot(Dx_ft)
                Ey_ft = self.eps_inv_mat[lind].dot(Dy_ft)
                Ez_ft = self.eps_inv_mat[lind].dot(Dz_ft)
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
        