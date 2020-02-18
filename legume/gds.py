import logging
logger = logging.getLogger(__name__)

import numpy as np

from . import Circle, Poly, Square, Hexagon

try:
    import gdspy
except ImportError:
    logger.debug('Unable to import gdspy. Export to GDS will be unavailable')

try:
    import skimage
except ImportError:
    logger.debug('Unable to import skimage. Rasterization to GDS will be unavailable')


def generate_gds(phc, filename, unit=1e-6, tolerance=0.01):
    """Export a GDS file for all layers of a photonic crystal

    Note
    ----
    The ``gdspy`` package is required for exporting GDS files.

    Parameters
    ----------
    phc : PhotCryst
    filename : str
    unit : float, optional
        The GDS spatial units. 
        Default is 1e-6, or 1 um.
    tolerance : float , optional
        The GDS tolerance parameter. 
        Default is 0.01.
    """

    gdspy.current_library = gdspy.GdsLibrary()
    cell = gdspy.Cell('CELL')

    # TODO: Can also add a `datatype`, ranging from 0-255, to each shape for use
    # by whatever program ends up reading the GDS

    for i, layer in enumerate(phc.layers):
        for shape in layer.shapes:
            if type(shape) in [Poly, Square, Hexagon]:
                points = [(x, y) for (x,y) in zip(shape.x_edges[:-1], shape.y_edges[:-1])]
                poly = gdspy.Polygon(points, layer=i, datatype=1)
                cell.add(poly)
            elif type(shape) == Circle:
                circle = gdspy.Round((shape.x_cent, shape.y_cent), shape.r, layer=i, datatype=1, tolerance=tolerance)
                cell.add(circle)
            else:
                raise RuntimeError("Unknown shape type, %s, found in layer %d of phc" % (type(shape), i))

    gdspy.write_gds(filename, unit=unit)


def generate_gds_raster(lattice, raster, filename, unit=1e-6, tolerance=0.01, cell_bound=True, levels=[0.5]):
    """Traces a rasterization projected onto a lattice to generate a GDS file

    Note
    ----
    The ``gdspy`` and ``scikit-image`` packages are required for exporting rasterized GDS files.

    Parameters
    ----------
    lattice : Lattice
    raster : 2D array of floats
    filename : str
    unit : float, optional
        The GDS spatial units. 
        Default is 1e-6, or 1 um.
    tolerance : float , optional
        The GDS tolerance parameter. 
        Default is 0.01.
    cell_bound : bool, optional
        Whether a Polygon should be added to define the unit cell boundary.
        Default is True.
    levels : List[float], optional
        List of the raster levels to trace with ``skimage.measure.find_contours``.
        Default is [0.5].
    """

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
