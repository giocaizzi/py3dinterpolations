"""grid3d object"""

import numpy as np
from typing import Union

from py3dinterpolations.core.griddata import GridData


class GridAxis:
    def __init__(self, name: str, min: float, max: float, res: float):
        self._name = name
        self._min = min
        self._max = max
        self._res = res
        self._grid = np.arange(self._min, self._max, self._res)

    @property
    def name(self):
        return self._name

    @property
    def min(self):
        return self._min

    @property
    def max(self):
        return self._max

    @property
    def res(self):
        return self._res

    @property
    def grid(self):
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
        mesh (dict): dictionary with `X`, `Y`, `Z` mesh arrays
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

    @property
    def results(self) -> dict:
        """get results"""
        return self._results

    @results.setter
    def results(self, results: dict) -> None:
        """set results"""
        if list(results.keys()) == ["interpolated", "variance"]:
            self._results = results
        else:
            raise NotImplementedError("Results type not implemented.")

    @property
    def grid(self) -> dict:
        """get grid"""
        return {
            "X": self.X.grid,
            "Y": self.Y.grid,
            "Z": self.Z.grid,
        }

    @property
    def mesh(self) -> dict:
        """get mesh"""
        mesh_array = {}
        mesh_array["X"], mesh_array["Y"], mesh_array["Z"] = np.meshgrid(
            self.grid["X"], self.grid["Y"], self.grid["Z"], indexing="xy"
        )
        return mesh_array

    @property
    def normalized_grid(self) -> dict:
        """get normalized grid

        Returns:
            dict: dictionary with normalized `X`, `Y`, `Z` grid arrays
        """
        normalized_grid = {}
        for axis in ["X", "Y", "Z"]:
            normalized_grid[axis] = (self.grid[axis] - self.grid[axis].min()) / (
                self.grid[axis].max() - self.grid[axis].min()
            )
        return normalized_grid


class Grid3DAxisProperties:
    """class for accessing a properties of a GridAxis from a Grid3D object"""

    def __init__(self, grid3d: Grid3D, target_property: str):
        self._grid3d = grid3d
        self._target_property = target_property

    def __getitem__(self, key):
        return getattr(self._grid3d, self._target_property)[key]

    @property
    def propriety(self):
        values = list(self._resolutions.values())
        if self._grid3d._target_property == {}:
            return self._resolutions
        elif all(value == values[0] for value in values):
            return values[0]
        else:
            return self._resolutions

    @propriety.setter
    def propriety(self, propriety: dict):
        self._gridres = propriety


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
        x_min=griddata.X.min,
        x_max=griddata.X.max,
        y_min=griddata.Y.min,
        y_max=griddata.Y.max,
        z_min=griddata.Z.min,
        z_max=griddata.Z.max,
        gridres=gridres,
    )
