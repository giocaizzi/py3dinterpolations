"""grid3d object"""

import numpy as np
from typing import Union


class Grid3D:
    """base class for 3D grids

    This is the base class for 3D grids.
    It contains the basic attributes and methods.

    Implemented child grid types:
        - RegularGrid3D : regular 3D grid with equal spacing in X, Y, Z

    Args:
        xmin (float): minimum x value
        xmax (float): maximum x value
        ymin (float): minimum y value
        ymax (float): maximum y value
        zmin (float): minimum z value
        zmax (float): maximum z value

    Attributes:
        grid (dict): dictionary with `X`, `Y`, `Z` grid arrays
        mesh (dict): dictionary with `X`, `Y`, `Z` mesh arrays

    Properties:
        bbox (tuple): bounding box (xmin, xmax, ymin, ymax, zmin, zmax)
        gridres (dict): grid resolution
        n_cells (int): number of cells
    """

    # grid boundaries
    _xmin: Union[float, None] = None
    _xmax: Union[float, None] = None
    _ymin: Union[float, None] = None
    _ymax: Union[float, None] = None
    _zmin: Union[float, None] = None
    _zmax: Union[float, None] = None

    # resoluton
    _gridres: dict = {}

    # grid and mesh
    grid: dict = {}
    mesh: dict = {}

    # results
    _results: dict = {}

    def __init__(
        self,
        xmin: float,
        xmax: float,
        ymin: float,
        ymax: float,
        zmin: float,
        zmax: float,
    ):
        """initialize grid"""
        self._xmin = xmin
        self._xmax = xmax
        self._ymin = ymin
        self._ymax = ymax
        self._zmin = zmin
        self._zmax = zmax

    @property
    def xmin(self) -> float:
        """get xmin"""
        return self._xmin

    @property
    def xmax(self) -> float:
        """get xmax"""
        return self._xmax

    @property
    def ymin(self) -> float:
        """get ymin"""
        return self._ymin

    @property
    def ymax(self) -> float:
        """get ymax"""
        return self._ymax

    @property
    def zmin(self) -> float:
        """get zmin"""
        return self._zmin

    @property
    def zmax(self) -> float:
        """get zmax"""
        return self._zmax

    @property
    def bbox(self) -> tuple:
        """get bounding box

        Returns:
            tuple: bounding box (xmin, xmax, ymin, ymax, zmin, zmax)
        """
        return (self._xmin, self._xmax, self._ymin, self._ymax, self._zmin, self._zmax)

    @property
    def gridres(self) -> dict:
        """get grid resolution"""
        return self._gridres

    @gridres.setter
    def gridres(self, gridres: dict) -> None:
        """set grid resolution"""
        self._gridres = gridres

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

    def _set_baseclass_attributes(self) -> None:
        """set base class attributes

        This method is called by the child classes
        to set the base class attributes,  i.e.
        the grid, mesh, normalized grid.
        """
        self._set_grid()
        self._set_mesh()

    def _set_grid(self) -> None:
        """set 3d grid

        Create the grid arrays, stored in a dictionary.
        """
        # grid arrays
        if self.gridres == {}:
            raise AttributeError("Grid resolution is empty.")
        else:
            self.grid = {
                "X": np.arange(self._xmin, self._xmax, self.gridres["X"]),
                "Y": np.arange(self._ymin, self._ymax, self.gridres["Y"]),
                "Z": np.arange(self._zmin, self._zmax, self.gridres["Z"]),
            }

    def _set_mesh(self) -> None:
        """set 3d mesh

        Create the mesh arrays, stored in a dictionary.
        """
        if self.grid == {}:
            raise AttributeError("Grid is empty.")
        else:
            self.mesh["X"], self.mesh["Y"], self.mesh["Z"] = np.meshgrid(
                self.grid["X"], self.grid["Y"], self.grid["Z"], indexing="xy"
            )

    @property
    def normalized_grid(self) -> dict:
        """get normalized grid

        Returns:
            dict: dictionary with normalized `X`, `Y`, `Z` grid arrays

        Raises:
            AttributeError: if grid is not set
        """
        normalized_grid = {}
        if self.grid == {}:
            raise AttributeError("Grid is empty.")
        else:
            # normalize grid
            for axis in ["X", "Y", "Z"]:
                normalized_grid[axis] = (self.grid[axis] - self.grid[axis].min()) / (
                    self.grid[axis].max() - self.grid[axis].min()
                )
            return normalized_grid

    @property
    def relative_grid_size(self) -> dict:
        """get grid relative size

        Returns:
            dict: dictionary with grid relative sizes

        Raises:
            AttributeError: if grid is not set
        """
        grid_relative_size = {}
        if self.grid == {}:
            raise AttributeError("Grid is empty.")
        else:
            grid_relative_size = {
                "X": len(self.grid["X"]),
                "Y": len(self.grid["Y"]),
                "Z": len(self.grid["Z"]),
            }
            return grid_relative_size

    @property
    def n_cells(self) -> int:
        """get number of cells on xyz

        Returns:
            int: number of cells

        Raises:
            AttributeError: if grid is not set
        """
        if self.grid == {}:
            raise AttributeError("Grid is empty.")
        else:
            return (
                self.grid_relative_size["X"]
                * self.grid_relative_size["Y"]
                * self.grid_relative_size["Z"]
            )


class RegularGrid3D(Grid3D):
    """class for regular 3D grids

    Class for a regular 3D grid
    with equal spacing in X, Y, Z.

    Args:
        gridres (float): grid resolution, equal in X, Y, Z
        \\**kwargs: keyword arguments for Grid3D base class
    """

    def __init__(
        self,
        gridres: float = 1.0,
        **kwargs,
    ):
        """initialize regular grid"""

        # initialize base class
        super().__init__(**kwargs)

        # grid resolution is fixed
        self.gridres = {
            "X": gridres,
            "Y": gridres,
            "Z": gridres,
        }
        self._set_baseclass_attributes()
