'''
Various utilities used in the main code.
NOTE: there should be no autograd functions here, only plain numpy/scipy
'''

import numpy as np
from scipy.linalg import toeplitz
from scipy.optimize import brentq

def ftinv(ft_coeff, gvec, xgrid, ygrid):
    ''' 
    Returns the discrete inverse Fourier transform over a real-space mesh 
    defined by 'xgrid', 'ygrid', computed given a number of FT coefficients 
    'ft_coeff' defined over a set of reciprocal vectors 'gvec'.
    This could be sped up through an fft function but written like this it is 
    more general as we don't have to deal with grid and lattice issues.
    '''
    (xmesh, ymesh) = np.meshgrid(xgrid, ygrid)
    ftinv = np.zeros(xmesh.shape, dtype=np.complex128)

    # Take only the unique components
    (g_unique, ind_unique) = np.unique(gvec, return_index=True, axis=1)

    for indg in ind_unique:
        ftinv += ft_coeff[indg]*np.exp(1j*gvec[0, indg]*xmesh + \
                            1j*gvec[1, indg]*ymesh)

    # # Do the x- and y-transforms separately 
    # # I wrote this but then realized it doesn't improve anything
    # (gx_u, indx) = np.unique(gvec[0, :], return_inverse=True)
    # for ix, gx in enumerate(gx_u):
    #   ind_match = np.where(indx==ix)[0]
    #   (gy_u, indy) = np.unique(gvec[1, ind_match], return_index=True)
    #   term = np.zeros(xmesh.shape, dtype=np.complex128)
    #   for iy, gy in enumerate(gy_u):
    #       # print(ft_coeff[indx[indy[iy]]])
    #       term += ft_coeff[ind_match[indy[iy]]]*np.exp(-1j*gy*ymesh)
    #   ftinv += term*np.exp(-1j*gx*xmesh)

    # # Can also be defined through a DFT matrix but it doesn't seem faster and 
    # # it's *very* memory intensive.
    # exp_matrix = xmesh.reshape((-1, 1)).dot(g_unique[[0], :]) + \
    #               ymesh.reshape((-1, 1)).dot(g_unique[[1], :])

    # dft_matrix = np.exp(1j*exp_matrix)
    # ftinv = dft_matrix.dot(ft_coeff[ind_unique]).reshape(xmesh.shape)
    # print(ftinv)
    return ftinv

def ft2square(lattice, ft_coeff, gvec):
    '''
    Make a square array of Fourier components given a number of them defined 
    over a set of reciprocal vectors gvec.
    NB: function hasn't really been tested, just storing some code.
    '''
    if lattice.type not in ['hexagonal', 'square']:
        raise NotImplementedError("ft2square probably only works for" \
                 "a lattice initialized as 'square' or 'hexagonal'")

    dgx = np.abs(lattice.b1[0])
    dgy = np.abs(lattice.b2[1])
    nx = np.int_(np.abs(np.max(gvec[0, :])/dgx))
    ny = np.int_(np.abs(np.max(gvec[1, :])/dgy))
    nxtot = 2*nx + 1
    nytot = 2*ny + 1
    eps_ft = np.zeros((nxtot, nytot), dtype=np.complex128)
    gx_grid = np.arange(-nx, nx)*dgx
    gy_grid = np.arange(-ny, ny)*dgy

    for jG in range(gvec.shape[1]):
        nG = np.int_(gvec[:, jG]/[dgx, dgy])
        eps_ft[nx + nG1[0], ny + nG1[1]] = ft_coeff[jG]

    return (eps_ft, gx_grid, gy_grid)

def grad_num(fn, arg, step_size=1e-7):
    ''' Numerically differentiate `fn` w.r.t. its argument `arg` 
    `arg` can be a numpy array of arbitrary shape
    `step_size` can be a number or an array of the same shape as `arg` '''

    N = arg.size
    shape = arg.shape
    gradient = np.zeros((N,))
    f_old = fn(arg)

    if type(step_size) == float:
        step = step_size*np.ones((N))
    else:
        step = step_size.ravel()

    for i in range(N):
        arg_new = arg.flatten()
        arg_new[i] += step[i]
        f_new_i = fn(arg_new.reshape(shape))
        gradient[i] = (f_new_i - f_old) / step[i]

    return gradient.reshape(shape)

def toeplitz_block(n, T1, T2):
    '''
    Constructs a Hermitian Toeplitz-block-Toeplitz matrix with n blocks and 
    T1 in the first row and T2 in the first column of every block in the first
    row of blocks 
    '''
    ntot = T1.shape[0]
    p = int(ntot/n) # Linear size of each block
    Tmat = np.zeros((ntot, ntot), dtype=T1.dtype)
    for ind1 in range(n):
        for ind2 in range(ind1, n):
            toep1 = T1[(ind2-ind1)*p:(ind2-ind1+1)*p]
            toep2 = T2[(ind2-ind1)*p:(ind2-ind1+1)*p]
            Tmat[ind1*p:(ind1+1)*p, ind2*p:(ind2+1)*p] = \
                    toeplitz(toep2, toep1)

    return np.triu(Tmat) + np.conj(np.transpose(np.triu(Tmat,1)))

    return np.triu(Tmat) + np.conj(np.transpose(np.triu(Tmat,1)))
def toeplitz_block2(T1, n1g,n2g): ## P for y direction, Q for x direction; 
    ## n1g for y, n2g for x
    Af = np.zeros([2*n1g-1,2*n2g-1],dtype=np.complex)
    Af[n1g-1:,n2g-1:] = T1.reshape([n1g,n2g])
    for i,ii in enumerate(Af): ## to account for some different convention
        for j, jj in enumerate(ii):
            if (i+j) %2 ==1:
                Af[i,j] *= -1
    Af[n1g-1:,:n2g] = np.flip(Af[n1g-1:,n2g-1:],axis=1)
    Af[:n1g,:] = np.flip(Af[n1g-1:,:],axis=0)
    P = n1g//2; Q = n2g//2;
    print('Af=','np.'+repr(Af))
    print(Af.shape,P,Q)
    econv = convmat2D(Af,P,Q)
    # return np.linalg.inv(econv)
    return econv
    # return Af
def convmat2D(Af, P,Q):
    N = Af.shape;
    NH = (2*P+1) * (2*Q+1) ;
    p = list(range(-P, P + 1)); #array of size 2Q+1
    q = list(range(-Q, Q + 1));

    ## do fft
    # Af = (1 / np.prod(N)) * np.fft.fftshift(np.fft.fft2(A));
    # natural question is to ask what does Af consist of..., what is the normalization for?

    # central indices marking the (0,0) order
    p0 = int((N[0] / 2)); #Af grid is Nx, Ny
    q0 = int((N[1] / 2)); #no +1 offset or anything needed because the array is orders from -P to P

    C = np.zeros((NH, NH))
    C = C.astype(complex);
    for qrow in range(2*Q+1): #remember indices in the arrary are only POSITIVE
        for prow in range(2*P+1): #outer sum
            # first term locates z plane, 2nd locates y column, prow locates x
            row = (prow) * (2*Q+1) + qrow; #natural indexing
            for qcol in range(2*Q+1): #inner sum
                for pcol in range(2*P+1):
                    col = (pcol) * (2*Q+1) + qcol; #natural indexing
                    pfft = p[prow] - p[pcol]; #get index in Af; #index may be negative.
                    qfft = q[qrow] - q[qcol];
                    C[row, col] = Af[p0 + pfft, q0 + qfft]; #index may be negative.
    return C;
def get_value(x):
    '''
    This is for when using the 'autograd' backend and you want to detach an 
    ArrayBox and just convert it to a numpy array.
    '''
    if str(type(x)) == "<class 'autograd.numpy.numpy_boxes.ArrayBox'>":
        return x._value
    else:
        return x

def fsolve(f, lb, ub, *args):
    '''
    Solve for scalar f(x, *args) = 0 w.r.t. scalar x within lb < x < ub
    '''
    args_value = tuple([get_value(arg) for arg in args])
    return brentq(f, lb, ub, args=args_value)

def find_nearest(array, value, N):
    '''
    Find the indexes of the N elements in an array nearest to a given value
    (Not the most efficient way but this is not a coding interview...)
    ''' 
    idx = np.abs(array - value).argsort()
    return idx[:N]

def RedhefferStar(SA,SB): #SA and SB are both 2x2 matrices;
    assert type(SA) == np.ndarray, 'not np.matrix'
    assert type(SB) == np.ndarray, 'not np.matrix'

    I = 1;
    # once we break every thing like this, we should still have matrices
    SA_11 = SA[0, 0]; SA_12 = SA[0, 1]; SA_21 = SA[1, 0]; SA_22 = SA[1, 1];
    SB_11 = SB[0, 0]; SB_12 = SB[0, 1]; SB_21 = SB[1, 0]; SB_22 = SB[1, 1];

    D = 1.0/(I-SB_11*SA_22);
    F = 1.0/(I-SA_22*SB_11);

    SAB_11 = SA_11 + SA_12*D*SB_11*SA_21;
    SAB_12 = SA_12*D*SB_12;
    SAB_21 = SB_21*F*SA_21;
    SAB_22 = SB_22 + SB_21*F*SA_22*SB_12;

    SAB = np.array([[SAB_11, SAB_12],[SAB_21, SAB_22]])
    return SAB

def generate_gds(phc, filename, unit=1e-6, tolerance=0.01):
    """Takes a photonic crystal object and generates a GDS file with layers
    """
    import gdspy

    polygon_based_shapes = [legume.phc.shapes.Poly, legume.phc.shapes.Square, legume.phc.shapes.Hexagon]

    gdspy.current_library = gdspy.GdsLibrary()
    cell = gdspy.Cell('CELL')

    # TODO: Can also add a `datatype`, ranging from 0-255, to each shape for use
    # by whatever program ends up reading the GDS

    for i, layer in enumerate(phc.layers):
        for shape in layer.shapes:
            if type(shape) in polygon_based_shapes:
                points = [(x, y) for (x,y) in zip(shape.x_edges[:-1], shape.y_edges[:-1])]
                poly = gdspy.Polygon(points, layer=i, datatype=1)
                cell.add(poly)
            elif type(shape) == legume.phc.shapes.Circle:
                circle = gdspy.Round((shape.x_cent, shape.y_cent), shape.r, layer=i, datatype=1, tolerance=tolerance)
                cell.add(circle)
            else:
                raise RuntimeError("Unknown shape type, %s, found in layer %d of phc" % (type(shape), i))

    gdspy.write_gds(filename, unit=unit)

def generate_gds_raster(lattice, raster, filename, unit=1e-6, tolerance=0.01, level=0.5, cell_bound=True, levels=0.5):
    """Traces the rasterization of a "freeform" layer and generates a single-layer GDS file
    """
    import skimage
    import gdspy

    contours = skimage.measure.find_contours(raster, levels)
    polygons = []

    T = np.hstack((lattice.a1[:, np.newaxis], lattice.a2[:, np.newaxis]))

    for contour in contours:
        #TODO(ian): make sure that this coord transform is correct
        #TODO(ian): generalize the 0.5 boundary
        coords = T @ (contour/(np.array(raster.shape)[np.newaxis,:]-1) - 0.5).T

        points = [(x, y) for (x, y) in zip(coords[0,:], coords[1,:])]
        poly = gdspy.Polygon(points, layer=0, datatype=0)
        polygons.append(poly)

    gdspy.current_library = gdspy.GdsLibrary()
    cell = gdspy.Cell('CELL')
    cell.add(polygons)

    # TODO(ian): Need to do a boolean operation here
    if cell_bound:
        bounds = T @ np.array([[-0.5, -0.5, +0.5, +0.5],[-0.5, +0.5, +0.5, -0.5]])
        points = [(x, y) for (x, y) in zip(bounds[0,:], bounds[1,:])]
        boundary = gdspy.Polygon(points, layer=0, datatype=1)

    cell.add(boundary)

    gdspy.write_gds(filename, unit=unit)
