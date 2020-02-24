import numpy as np
import matplotlib.path as mpltPath

from legume.backend import backend as bd

class Shape(object):
    """ 
    Parent class for shapes.
    """
    def __init__(self, eps=1.):
        
        self.eps = eps
        self.area = bd.real(self.compute_ft(bd.array([[0.], [0.]])))

    def __repr__(self):
        return "Shape"

    def _parse_ft_gvec(self, gvec):
        if type(gvec) == list:
            gvec = np.array(gvec)
        elif type(gvec) != np.ndarray:
            raise(TypeError, "Input gvec must be either of type list or \
                                    np.ndarray")

        if len(gvec.shape) != 2:
            raise(ValueError, "Input gvec must be a 2D array")

        elif gvec.shape[0] != 2:
            if gvec.shape[1] == 2:
                gvec = gvec.T
            else:
                raise(ValueError, "Input gvec must have length 2 \
                                along one axis")

        return (gvec[0, :], gvec[1, :])

    def compute_ft(self, gvec):
        """
        FT of a 2D function equal to 1 inside the shape and 0 outside.
        """
        raise NotImplementedError("compute_ft() needs to be implemented by"
            "Shape subclasses")

    def is_inside(self, x, y):
        """Vectorized check if points are inside the shape.
        
        Parameters
        ----------
        x : np.ndarray
            x-coordinates of points to test.
        y : np.ndarray
            y-coordinates of points to test.

        Note
        ----
        `x` and `y` must have the same shape.

        Returns
        -------
        np.ndarray
            An array of the same shape as `x` and `y` with elements equal to 1 
            if the corresponding point (x, y) is inside the shape, and 0 
            otherwise.
        """
        raise NotImplementedError("is_inside() needs to be implemented by"
            "Shape subclasses")

class Circle(Shape):
    """
    Subclass for a circular shape.
    """
    def __init__(self, eps=1., x_cent=0., y_cent=0., r=0.):
        """Initialize a circle.
        
        Parameters
        ----------
        eps : float, optional
            Permittivity inside the circle.
        x_cent : float, optional
            x-coordinate of the center.
        y_cent : float, optional
            y-coordinate of the center.
        r : float, optional
            Radius.
        """
        self.x_cent = x_cent
        self.y_cent = y_cent
        self.r = r
        super().__init__(eps=eps)

    def __repr__(self):
        return "Circle(eps = %.2f, x = %.4f, y = %.4f, r = %.4f)" % \
               (self.eps, self.x_cent, self.y_cent, self.r)

    def compute_ft(self, gvec):
        """
        Fourier transform of a 2D function equal to 1 inside the circle and 0 
        outside.
        
        Parameters
        ----------
        gvec : np.ndarray
            An array of shape (2, Ng) of G-points in the xy-plane over which 
            the FT is computed.
        
        Returns
        -------
        ft : np.ndarray
            An array of shape (Ng, ) with the corresponding Fourier components.
        """
        (gx, gy) = self._parse_ft_gvec(gvec)

        gabs = np.sqrt(np.abs(np.square(gx)) + np.abs(np.square(gy)))
        gabs += 1e-10 # To avoid numerical instability at zero

        ft = bd.exp(-1j*gx*self.x_cent - 1j*gy*self.y_cent)*self.r* \
                            2*np.pi/gabs*bd.bessel1(gabs*self.r)

        return ft

    def is_inside(self, x, y):

        return (np.square(x - self.x_cent) + np.square(y - self.y_cent)
                            <= np.square(self.r))

class Poly(Shape):
    """
    Subclass for a polygonal shape.
    """
    def __init__(self, eps=1., x_edges=[0.], y_edges=[0.]):
        """Initialize a polygon.
        
        Parameters
        ----------
        eps : float, optional
            Permittivity inside the polygon.
        x_edges : list or np.ndarray, optional
            A 1D array containing the x-coordinates of the polygon vertices.
        y_edges : list or np.ndarray, optional
            A 1D array containing the y-coordinates of the polygon vertices.

        Note
        ----
        x_edges and y_edges must have the same size. The vertices of the polygon
        must be defined in a counter-clockwise direction. This is checked for 
        when initializing a Poly object.
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
        """
        Fourier transform of a 2D function equal to 1 inside the polygon and 0 
        outside. Computed as per Lee, IEEE TAP (1984).
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

        ft = bd.zeros((ng), dtype=bd.complex);

        aj = (bd.roll(xj, -1, axis=1) - xj + 1e-10) / \
                (bd.roll(yj, -1, axis=1) - yj + 1e-20)
        bj = xj - aj * yj

        # We first handle the Gx = 0 case
        ind_gx0 = np.abs(gx[:, 0]) < 1e-10
        ind_gx = ~ind_gx0
        if np.sum(ind_gx0) > 0:
            # And first the Gy = 0 case
            ind_gy0 = np.abs(gy[:, 0]) < 1e-10
            if np.sum(ind_gy0*ind_gx0) > 0:
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
        """
        Rotate a polygon around its center of mass by `angle` radians.
        """

        rotmat = bd.array([[bd.cos(angle), -bd.sin(angle)], \
                            [bd.sin(angle), bd.cos(angle)]])
        (xj, yj) = (bd.array(self.x_edges), bd.array(self.y_edges))
        com_x = bd.sum((xj + bd.roll(xj, -1)) * (xj * bd.roll(yj, -1) - \
                    bd.roll(xj, -1) * yj))/6/self.area
        com_y = bd.sum((yj + bd.roll(yj, -1)) * (xj * bd.roll(yj, -1) - \
                    bd.roll(xj, -1) * yj))/6/self.area
        new_coords = bd.dot(rotmat, bd.vstack((xj-com_x, yj-com_y)))

        self.x_edges = new_coords[0, :] + com_x
        self.y_edges = new_coords[1, :] + com_y

        return self

class Square(Poly):
    """
    Sublass for a square shape
    """
    def __init__(self, eps=1, x_cent=0, y_cent=0, a=0):
        """Initialize a square.
        
        Parameters
        ----------
        eps : float, optional
            Permittivity inside the square.
        x_cent : float, optional
            x-coordinate of the center.
        y_cent : float, optional
            y-coordinate of the center.
        a : float, optional
            Side-length.
        """
        self.x_cent = x_cent
        self.y_cent = y_cent
        self.a = a
        x_edges = x_cent + bd.array([-a/2, a/2, a/2, -a/2])
        y_edges = y_cent + bd.array([-a/2, -a/2, a/2, a/2])
        super().__init__(eps, x_edges, y_edges)

    def __repr__(self):
        return "Square(eps = %.2f, x_cent = %.4f, y_cent = %.4f, a = %.4f)" % \
               (self.eps, self.x_cent, self.y_cent, self.a)

class Hexagon(Poly):
    """
    Subclass for a hexagon shape
    """
    def __init__(self, eps=1, x_cent=0, y_cent=0, a=0):
        """Initialize a hexagon.
        
        Parameters
        ----------
        eps : float, optional
            Permittivity inside the square.
        x_cent : float, optional
            x-coordinate of the center.
        y_cent : float, optional
            y-coordinate of the center.
        a : float, optional
            Side-length.
        """
        self.x_cent = x_cent
        self.y_cent = y_cent
        self.a = a
        x_edges = x_cent + bd.array([a, a/2, -a/2, -a, -a/2, a/2, a])
        y_edges = y_cent + bd.array([0, np.sqrt(3)/2*a, np.sqrt(3)/2*a, 0, \
                    -np.sqrt(3)/2*a, -np.sqrt(3)/2*a, 0])
        super().__init__(eps, x_edges, y_edges) 

    def __repr__(self):
        return "Hexagon(eps = %.2f, x_cent = %.4f, y_cent = %.4f, a = %.4f)" % \
               (self.eps, self.x_cent, self.y_cent, self.a)
