"""core classes and functions"""

from .grid3d import RegularGrid3D
from .griddata import GridData


def create_regulargrid3d_from_griddata(
    griddata: GridData,
    gridres: float,
) -> RegularGrid3D:
    """create RegularGrid3D from GridData

    Args:
        griddata (GridData): GridData object
        gridres (float): grid resolution
    """
    return RegularGrid3D(
        gridres=gridres,
        xmin=griddata.specs.xmin,
        xmax=griddata.specs.xmax,
        ymin=griddata.specs.ymin,
        ymax=griddata.specs.ymax,
        zmin=griddata.specs.zmin,
        zmax=griddata.specs.zmax,
    )
