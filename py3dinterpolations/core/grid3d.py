"""3D grid definitions for interpolation."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import cached_property

import numpy as np
from shapely.geometry.base import BaseGeometry

from .griddata import GridData
from .types import Axis, GridResolution, InterpolationResult


@dataclass(frozen=True)
class GridAxis:
    """A single axis of a 3D grid.

    Args:
        name: Axis identifier.
        min: Minimum value.
        max: Maximum value.
        res: Resolution (step size).
    """

    name: Axis
    min: float
    max: float
    res: float

    @property
    def grid(self) -> np.ndarray:
        """1D array of evenly spaced values along this axis."""
        return np.arange(self.min, self.max, self.res)

    def __repr__(self) -> str:
        return f"GridAxis({self.name.value}: [{self.min}, {self.max}), res={self.res})"


class Grid3D(ABC):
    """Abstract base class for 3D interpolation grids.

    Args:
        x: X axis definition.
        y: Y axis definition.
        z: Z axis definition.
    """

    def __init__(self, x: GridAxis, y: GridAxis, z: GridAxis):
        self._x = x
        self._y = y
        self._z = z
        self._result: InterpolationResult | None = None

    @property
    def X(self) -> GridAxis:
        return self._x

    @property
    def Y(self) -> GridAxis:
        return self._y

    @property
    def Z(self) -> GridAxis:
        return self._z

    def get_axis(self, axis: str | Axis) -> GridAxis:
        """Get a grid axis by name.

        Args:
            axis: Axis name ("X", "Y", "Z") or Axis enum.

        Raises:
            ValueError: If axis name is invalid.
        """
        axis = Axis(axis)
        match axis:
            case Axis.X:
                return self._x
            case Axis.Y:
                return self._y
            case Axis.Z:
                return self._z

    @property
    def result(self) -> InterpolationResult | None:
        return self._result

    @result.setter
    def result(self, value: InterpolationResult) -> None:
        self._result = value

    @cached_property
    def grid(self) -> dict[str, np.ndarray]:
        """1D grid arrays per axis."""
        return {
            "X": self.X.grid,
            "Y": self.Y.grid,
            "Z": self.Z.grid,
        }

    @cached_property
    def normalized_grid(self) -> dict[str, np.ndarray]:
        """Min-max normalized 1D grid arrays per axis."""
        result = {}
        for axis_name, arr in self.grid.items():
            arr_min, arr_max = arr.min(), arr.max()
            if arr_max - arr_min == 0:
                result[axis_name] = np.zeros_like(arr)
            else:
                result[axis_name] = (arr - arr_min) / (arr_max - arr_min)
        return result

    @property
    def gridres(self) -> float | dict[str, float]:
        """Grid resolution; scalar if uniform, dict otherwise."""
        if self.X.res == self.Y.res == self.Z.res:
            return self.X.res
        return {"X": self.X.res, "Y": self.Y.res, "Z": self.Z.res}

    @cached_property
    def mesh(self) -> dict[str, np.ndarray]:
        """3D meshgrid arrays."""
        mx, my, mz = np.meshgrid(
            self.grid["X"], self.grid["Y"], self.grid["Z"], indexing="xy"
        )
        return {"X": mx, "Y": my, "Z": mz}

    @cached_property
    def normalized_mesh(self) -> dict[str, np.ndarray]:
        """Normalized 3D meshgrid arrays."""
        mx, my, mz = np.meshgrid(
            self.normalized_grid["X"],
            self.normalized_grid["Y"],
            self.normalized_grid["Z"],
            indexing="xy",
        )
        return {"X": mx, "Y": my, "Z": mz}

    @abstractmethod
    def prediction_points(self) -> np.ndarray:
        """Return (N, 3) array of points at which to predict."""
        ...

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"X=[{self.X.min}, {self.X.max}), "
            f"Y=[{self.Y.min}, {self.Y.max}), "
            f"Z=[{self.Z.min}, {self.Z.max}))"
        )


class RegularGrid3D(Grid3D):
    """Regular 3D grid with equal spacing in all axes.

    Args:
        x_min: Minimum X value.
        x_max: Maximum X value.
        y_min: Minimum Y value.
        y_max: Maximum Y value.
        z_min: Minimum Z value.
        z_max: Maximum Z value.
        gridres: Uniform grid resolution.
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
        super().__init__(
            x=GridAxis(Axis.X, x_min, x_max, gridres),
            y=GridAxis(Axis.Y, y_min, y_max, gridres),
            z=GridAxis(Axis.Z, z_min, z_max, gridres),
        )

    def prediction_points(self) -> np.ndarray:
        """Return all grid points as (N, 3) array."""
        m = self.mesh
        return np.column_stack([m["X"].ravel(), m["Y"].ravel(), m["Z"].ravel()])


class IrregularGrid3D(Grid3D):
    """3D grid with per-axis resolution and optional convex hull filtering.

    Args:
        x_min: Minimum X value.
        x_max: Maximum X value.
        x_res: X axis resolution.
        y_min: Minimum Y value.
        y_max: Maximum Y value.
        y_res: Y axis resolution.
        z_min: Minimum Z value.
        z_max: Maximum Z value.
        z_res: Z axis resolution.
        hull: Optional convex hull geometry for XY filtering.
    """

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
        hull: BaseGeometry | None = None,
    ):
        super().__init__(
            x=GridAxis(Axis.X, x_min, x_max, x_res),
            y=GridAxis(Axis.Y, y_min, y_max, y_res),
            z=GridAxis(Axis.Z, z_min, z_max, z_res),
        )
        self._hull = hull

    def prediction_points(self) -> np.ndarray:
        """Return grid points filtered by convex hull if available."""
        m = self.mesh
        points = np.column_stack([m["X"].ravel(), m["Y"].ravel(), m["Z"].ravel()])
        if self._hull is not None:
            from shapely import contains_xy

            mask = contains_xy(self._hull, points[:, 0], points[:, 1])
            filtered: np.ndarray = points[mask]
            return filtered
        return points


def create_grid(
    griddata: GridData,
    resolution: float | dict[str, float],
) -> Grid3D:
    """Factory to create the appropriate Grid3D from data and resolution.

    Args:
        griddata: Source data to derive grid extents from.
        resolution: Uniform float for RegularGrid3D,
            or per-axis dict for IrregularGrid3D.

    Returns:
        A Grid3D subclass instance.

    Raises:
        TypeError: If resolution type is unsupported.
    """
    specs = griddata.specs
    res = GridResolution.from_input(resolution)

    if isinstance(resolution, (int, float)):
        return RegularGrid3D(
            x_min=specs.xmin,
            x_max=specs.xmax,
            y_min=specs.ymin,
            y_max=specs.ymax,
            z_min=specs.zmin,
            z_max=specs.zmax,
            gridres=res.x,
        )
    if isinstance(resolution, dict):
        return IrregularGrid3D(
            x_min=specs.xmin,
            x_max=specs.xmax,
            x_res=res.x,
            y_min=specs.ymin,
            y_max=specs.ymax,
            y_res=res.y,
            z_min=specs.zmin,
            z_max=specs.zmax,
            z_res=res.z,
            hull=griddata.hull,
        )
    msg = f"resolution must be float or dict, got {type(resolution)}"
    raise TypeError(msg)
