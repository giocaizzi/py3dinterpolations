"""grid3d object"""

import numpy as np

from py3dinterpolations.core.griddata import GridData


class GridAxis:
    def __init__(self, name: str, min: float, max: float, res: float):
        self._name = name
        self._min = min
        self._max = max
        self._res = res
        self._grid = np.arange(self._min, self._max, self._res)

    @property
    def name(self) -> str:
        return self._name

    @property
    def min(self) -> float:
        return self._min

    @property
    def max(self) -> float:
        return self._max

    @property
    def res(self) -> float:
        return self._res

    @property
    def grid(self) -> np.ndarray:
        return self._grid


class Grid3D:
    """base class for 3D grids

    This is the base class for 3D grids.
    It contains the basic attributes and methods.

    Implemented child grid types:
        - RegularGrid3D : regular 3D grid with equal spacing in X, Y, Z

    Args:
        x_min (float): minimum value in X
        x_max (float): maximum value in X
        x_res (float): resolution in X
        y_min (float): minimum value in Y
        y_max (float): maximum value in Y
        y_res (float): resolution in Y
        z_min (float): minimum value in Z
        z_max (float): maximum value in Z
        z_res (float): resolution in Z

    Attributes:
        X (GridAxis): X axis
        Y (GridAxis): Y axis
        Z (GridAxis): Z axis
        grid (dict): dictionary with `X`, `Y`, `Z` grid arrays
        normalized_grid (dict): dictionary with normalized `X`, `Y`, `Z` grid arrays
        mesh (dict): dictionary with `X`, `Y`, `Z` mesh arrays
        gridres (dict): grid resolution
        results (dict): dictionary with `interpolated`, `variance` results

    Properties:
        gridres (dict): grid resolution
        n_cells (int): number of cells
    """

    X: GridAxis = None
    Y: GridAxis = None
    Z: GridAxis = None

    # associated mesh
    mesh: dict = {}

    # results
    _results: dict = {}

    def __init__(
        self,
        x_min: float,
        x_max: float,
        x_res: float,
        y_min: float,
        y_max: float,
        y_res: float,
        z_min: float,
        z_max: float,
        z_res: float,
    ):
        """initialize grid"""
        self._X = GridAxis("X", x_min, x_max, x_res)
        self._Y = GridAxis("Y", y_min, y_max, y_res)
        self._Z = GridAxis("Z", z_min, z_max, z_res)

    @property
    def X(self):
        return self._X

    @property
    def Y(self):
        return self._Y

    @property
    def Z(self):
        return self._Z

    def get_axis(self, axis: str) -> GridAxis:
        """get grid axis

        Args:
            axis (str): axis name, one of `X`, `Y`, `Z`

        Returns:
            GridAxis: grid axis

        Raises:
            ValueError: invalid axis
        """
        if axis == "X":
            return self.X
        elif axis == "Y":
            return self.Y
        elif axis == "Z":
            return self.Z
        else:
            raise ValueError("Invalid axis.")

    @property
    def results(self) -> dict:
        return self._results

    @results.setter
    def results(self, results: dict) -> None:
        if list(results.keys()) == ["interpolated", "variance"]:
            self._results = results
        else:
            raise NotImplementedError("Results type not implemented.")

    @property
    def grid(self) -> dict:
        return {
            "X": self.X.grid,
            "Y": self.Y.grid,
            "Z": self.Z.grid,
        }

    @property
    def normalized_grid(self) -> dict:
        normalized_grid = {}
        for axis in ["X", "Y", "Z"]:
            normalized_grid[axis] = (self.grid[axis] - self.grid[axis].min()) / (
                self.grid[axis].max() - self.grid[axis].min()
            )
        return normalized_grid

    @property
    def gridres(self) -> dict:
        if self.X.res == self.Y.res == self.Z.res:
            return self.X.res
        else:
            return {
                "X": self.X.res,
                "Y": self.Y.res,
                "Z": self.Z.res,
            }

    @property
    def mesh(self) -> dict:
        mesh_array = {}
        mesh_array["X"], mesh_array["Y"], mesh_array["Z"] = np.meshgrid(
            self.grid["X"], self.grid["Y"], self.grid["Z"], indexing="xy"
        )
        return mesh_array


class RegularGrid3D(Grid3D):
    """class for regular 3D grids

    Class for a regular 3D grid
    with equal spacing in X, Y, Z.

    Args:
        x_min (float): minimum value in X
        x_max (float): maximum value in X
        y_min (float): minimum value in Y
        y_max (float): maximum value in Y
        z_min (float): minimum value in Z
        z_max (float): maximum value in Z
        gridres (float): grid resolution, equal in X, Y, Z
    """

    def __init__(
        self,
        x_min: float,
        x_max: float,
        y_min: float,
        y_max: float,
        z_min: float,
        z_max: float,
        gridres: float,
    ):
        """initialize regular grid"""
        kwargs = {
            "x_min": x_min,
            "x_max": x_max,
            "x_res": gridres,
            "y_min": y_min,
            "y_max": y_max,
            "y_res": gridres,
            "z_min": z_min,
            "z_max": z_max,
            "z_res": gridres,
        }
        # initialize base class
        super().__init__(**kwargs)


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
        x_min=griddata.specs.xmin,
        x_max=griddata.specs.xmax,
        y_min=griddata.specs.ymin,
        y_max=griddata.specs.ymax,
        z_min=griddata.specs.zmin,
        z_max=griddata.specs.zmax,
        gridres=gridres,
    )
