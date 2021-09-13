import numpy as np
import matplotlib.path as mpltPath

from legume.backend import backend as bd


class Shape(object):
    """Geometric shape base class
    """
    def __init__(self, eps=1.):
        """Create a shape
        """
        self.eps = eps
        self.area = bd.real(self.compute_ft(bd.array([[0.], [0.]])))

    def __repr__(self):
        return "Shape"

    def _parse_ft_gvec(self, gvec):
        if type(gvec) == list:
            gvec = np.array(gvec)
        elif type(gvec) != np.ndarray:
            raise (TypeError, "Input gvec must be either of type list or \
                                    np.ndarray")

        if len(gvec.shape) != 2:
            raise (ValueError, "Input gvec must be a 2D array")

        elif gvec.shape[0] != 2:
            if gvec.shape[1] == 2:
                gvec = gvec.T
            else:
                raise (ValueError, "Input gvec must have length 2 \
                                along one axis")

        return (gvec[0, :], gvec[1, :])

    def compute_ft(self, gvec):
        """Compute Fourier transform of a 2D shape function

        The shape function is assumed to take a value of 1 inside
        the shape and a value of 0 outside the shape.

        Parameters
        ----------
        gvec : np.ndarray of shape (2, Ng)
            g-vectors at which the Fourier transform is evaluated
        """
        raise NotImplementedError("compute_ft() needs to be implemented by"
                                  "Shape subclasses")

    def is_inside(self, x, y):
        """Elementwise indicator function for the shape

        x and y are arrays of the same shape. This function returns an array of
        the same shape, where every element is equal to 1 if the corresponding
        (x, y) point is inside the shape, and 0 if (x, y) outside.
        """
        raise NotImplementedError("is_inside() needs to be implemented by"
                                  "Shape subclasses")


class Circle(Shape):
    """Circle shape
    """
    def __init__(self, eps=1., x_cent=0., y_cent=0., r=0.):
        """Create a circle shape

        Parameters
        ----------
        eps : float
            Permittivity value
        x_cent : float
            x-coordinate of circle center
        y_cent : float
            y-coordinate of circle center
        r : float
            radius of circle
        """
        self.x_cent = x_cent
        self.y_cent = y_cent
        self.r = r
        super().__init__(eps=eps)

    def __repr__(self):
        return "Circle(eps = %.2f, x = %.4f, y = %.4f, r = %.4f)" % \
               (self.eps, self.x_cent, self.y_cent, self.r)

    def compute_ft(self, gvec):
        (gx, gy) = self._parse_ft_gvec(gvec)

        gabs = np.sqrt(np.abs(np.square(gx)) + np.abs(np.square(gy)))
        gabs += 1e-10  # To avoid numerical instability at zero

        ft = bd.exp(-1j*gx*self.x_cent - 1j*gy*self.y_cent)*self.r* \
                            2*np.pi/gabs*bd.bessel1(gabs*self.r)

        return ft

    def is_inside(self, x, y):
        return (np.square(x - self.x_cent) + np.square(y - self.y_cent) <=
                np.square(self.r))


class Poly(Shape):
    """Polygon shape
    """
    def __init__(self, eps=1., x_edges=[0.], y_edges=[0.]):
        """Create a polygon shape

        Parameters
        ----------
        eps : float
            Permittivity value
        x_edges : List or np.ndarray
            x-coordinates of polygon vertices
        y_edges : List or np.ndarray
            y-coordinates of polygon vertices

        Note
        ----
        The polygon vertices must be supplied in counter-clockwise order.
        """

        # Make extra sure that the last point of the polygon is the same as the
        # first point
        self.x_edges = bd.hstack((bd.array(x_edges), x_edges[0]))
        self.y_edges = bd.hstack((bd.array(y_edges), y_edges[0]))
        super().__init__(eps)

        if self.compute_ft([[0.], [0.]]) < 0:
            raise ValueError("The edges defined by x_edges and y_edges must be"
                             " specified in counter-clockwise order")

    def __repr__(self):
        return "Poly(eps = %.2f, x_edges = %s, y_edges = %s)" % \
               (self.eps, self.x_edges, self.y_edges)

    def compute_ft(self, gvec):
        """Compute Fourier transform of the polygon

        The polygon is assumed to take a value of 1 inside and a value of 0 
        outside.

        The Fourier transform calculation follows that of Lee, IEEE TAP (1984).

        Parameters
        ----------
        gvec : np.ndarray of shape (2, Ng)
            g-vectors at which FT is evaluated
        """
        (gx, gy) = self._parse_ft_gvec(gvec)

        (xj, yj) = self.x_edges, self.y_edges
        npts = xj.shape[0]
        ng = gx.shape[0]
        # Note: the paper uses +1j*g*x convention for FT while we use
        # -1j*g*x everywhere in legume
        gx = -gx[:, bd.newaxis]
        gy = -gy[:, bd.newaxis]
        xj = xj[bd.newaxis, :]
        yj = yj[bd.newaxis, :]

        ft = bd.zeros((ng), dtype=bd.complex)

        aj = (bd.roll(xj, -1, axis=1) - xj + 1e-10) / \
                (bd.roll(yj, -1, axis=1) - yj + 1e-20)
        bj = xj - aj * yj

        # We first handle the Gx = 0 case
        ind_gx0 = np.abs(gx[:, 0]) < 1e-10
        ind_gx = ~ind_gx0
        if np.sum(ind_gx0) > 0:
            # And first the Gy = 0 case
            ind_gy0 = np.abs(gy[:, 0]) < 1e-10
            if np.sum(ind_gy0 * ind_gx0) > 0:
                ft = ind_gx0*ind_gy0*bd.sum(xj * bd.roll(yj, -1, axis=1)-\
                                yj * bd.roll(xj, -1, axis=1))/2
                # Remove the Gx = 0, Gy = 0 component
                ind_gx0[ind_gy0] = False

            # Compute the remaining Gx = 0 components
            a2j = 1 / aj
            b2j = yj - a2j * xj
            bgtemp = gy * b2j
            agtemp1 = bd.dot(gx, xj) + bd.dot(gy, a2j * xj)
            agtemp2 = bd.dot(gx, bd.roll(xj, -1, axis=1)) + \
                    bd.dot(gy, a2j * bd.roll(xj, -1, axis=1))
            denom = gy * (gx + bd.dot(gy, a2j))
            ftemp = bd.sum(bd.exp(1j*bgtemp) * (bd.exp(1j*agtemp2) - \
                    bd.exp(1j*agtemp1)) * \
                    denom / (bd.square(denom) + 1e-50) , axis=1)
            ft = bd.where(ind_gx0, ftemp, ft)

        # Finally compute the general case for Gx != 0
        if np.sum(ind_gx) > 0:
            bgtemp = bd.dot(gx, bj)
            agtemp1 = bd.dot(gy, yj) + bd.dot(gx, aj * yj)
            agtemp2 = bd.dot(gy, bd.roll(yj, -1, axis=1)) + \
                        bd.dot(gx, aj * bd.roll(yj, -1, axis=1))
            denom = gx * (gy + bd.dot(gx, aj))
            ftemp = -bd.sum(bd.exp(1j*bgtemp) * (bd.exp(1j * agtemp2) - \
                    bd.exp(1j * agtemp1)) * \
                    denom / (bd.square(denom) + 1e-50) , axis=1)
            ft = bd.where(ind_gx, ftemp, ft)

        return ft

    def is_inside(self, x, y):
        vert = np.vstack((np.array(self.x_edges), np.array(self.y_edges)))
        path = mpltPath.Path(vert.T)
        points = np.vstack((np.array(x).ravel(), np.array(y).ravel()))
        test = path.contains_points(points.T)
        return test.reshape((x.shape))

    def rotate(self, angle):
        """Rotate the polygon around its center of mass by `angle` radians
        """

        rotmat = bd.array([[bd.cos(angle), -bd.sin(angle)], \
                            [bd.sin(angle), bd.cos(angle)]])
        (xj, yj) = (bd.array(self.x_edges), bd.array(self.y_edges))
        com_x = bd.sum((xj + bd.roll(xj, -1)) * (xj * bd.roll(yj, -1) - \
                    bd.roll(xj, -1) * yj))/6/self.area
        com_y = bd.sum((yj + bd.roll(yj, -1)) * (xj * bd.roll(yj, -1) - \
                    bd.roll(xj, -1) * yj))/6/self.area
        new_coords = bd.dot(rotmat, bd.vstack((xj - com_x, yj - com_y)))

        self.x_edges = new_coords[0, :] + com_x
        self.y_edges = new_coords[1, :] + com_y

        return self


class Square(Poly):
    """Square shape
    """
    def __init__(self, eps=1, x_cent=0, y_cent=0, a=0):
        """Create a square shape

        Parameters
        ----------
        eps : float
            Permittivity value
        x_cent : float
            x-coordinate of square center
        y_cent : float
            y-coordinate of square center
        a : float
            square edge length
        """
        self.x_cent = x_cent
        self.y_cent = y_cent
        self.a = a
        x_edges = x_cent + bd.array([-a / 2, a / 2, a / 2, -a / 2])
        y_edges = y_cent + bd.array([-a / 2, -a / 2, a / 2, a / 2])
        super().__init__(eps, x_edges, y_edges)

    def __repr__(self):
        return "Square(eps = %.2f, x_cent = %.4f, y_cent = %.4f, a = %.4f)" % \
               (self.eps, self.x_cent, self.y_cent, self.a)


class Hexagon(Poly):
    """Hexagon shape
    """
    def __init__(self, eps=1, x_cent=0, y_cent=0, a=0):
        """Create a hexagon shape

        Parameters
        ----------
        eps : float
            Permittivity value
        x_cent : float
            x-coordinate of hexagon center
        y_cent : float
            y-coordinate of hexagon center
        a : float
            hexagon edge length
        """
        self.x_cent = x_cent
        self.y_cent = y_cent
        self.a = a
        x_edges = x_cent + bd.array([a, a / 2, -a / 2, -a, -a / 2, a / 2, a])
        y_edges = y_cent + bd.array([0, np.sqrt(3)/2*a, np.sqrt(3)/2*a, 0, \
                    -np.sqrt(3)/2*a, -np.sqrt(3)/2*a, 0])
        super().__init__(eps, x_edges, y_edges)

    def __repr__(self):
        return "Hexagon(eps = %.2f, x_cent = %.4f, y_cent = %.4f, a = %.4f)" % \
               (self.eps, self.x_cent, self.y_cent, self.a)


class FourierShape(Poly):
    """Fourier coefficinets of the polar coordinates
    """
    def __init__(self,
                 eps=1,
                 x_cent=0,
                 y_cent=0,
                 f_as=np.array([0.]),
                 f_bs=np.array([]),
                 npts=100):
        """Create a shape defined by its Fourier coefficients in polar 
        coordinates.

        Parameters
        ----------
        eps : float
            Permittivity value
        x_cent : float
            x-coordinate of shape center
        y_cent : float
            y-coordinate of shape center
        f_as : Numpy array
            Fourier coefficients an (see Note)
        f_bs : Numpy array
            Fourier coefficients bn (see Note)
        npts : int
            Number of points in the polygonal discretization

        Note
        ----
        We use the discrete Fourier expansion 
        ``R(phi) = a0/2 + sum(an*cos(n*phi)) + sum(bn*sin(n*phi))``
        The coefficients ``f_as`` are an array containing ``[a0, a1, ...]``,  
        while ``f_bs`` define ``[b1, b2, ...]``.

        Note
        ----
        This is a subclass of Poly because we discretize the shape into 
        a polygon and use that to compute the fourier transform for the 
        mode expansions. For intricate shapes, increase ``npts``
        to make the discretization smoother. 

        """
        self.x_cent = x_cent
        self.y_cent = y_cent
        self.npts = npts

        phis = bd.linspace(0, 2 * np.pi, npts + 1)

        R_phi = f_as[0] / 2 * bd.ones(phis.shape)
        for (n, an) in enumerate(f_as[1:]):
            R_phi = R_phi + an * bd.cos((n + 1) * phis)

        for (n, bn) in enumerate(f_bs):
            R_phi = R_phi + bn * bd.sin((n + 1) * phis)

        if np.any(R_phi < 0):
            raise ValueError("Coefficients of FourierShape should be such "
                             "that R(phi) is non-negative for all phi.")

        x_edges = R_phi * bd.cos(phis)
        y_edges = R_phi * bd.sin(phis)

        super().__init__(eps, x_edges, y_edges)

    def __repr__(self):
        return "FourierShape(eps = %.2f, x_cent = %.4f, y_cent = %.4f"\
                ", npts = %d)" % \
               (self.eps, self.x_cent, self.y_cent, self.npts)
