"""Core data container for 3D interpolation."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from shapely.geometry.base import BaseGeometry

from .types import PreprocessingParams


class GridData:
    """Container for 3D grid data with spatial coordinates and values.

    Standardizes input DataFrames into a canonical format with
    MultiIndex (ID, X, Y, Z) and a single column V.

    Args:
        data: Source DataFrame with spatial data.
        ID: Column name for point identifier.
        X: Column name for X coordinate.
        Y: Column name for Y coordinate.
        Z: Column name for Z coordinate.
        V: Column name for value.
        preprocessing_params: Parameters from preprocessing applied to this data.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        *,
        ID: str = "ID",
        X: str = "X",
        Y: str = "Y",
        Z: str = "Z",
        V: str = "V",
        preprocessing_params: PreprocessingParams | None = None,
    ):
        self.preprocessing_params = preprocessing_params
        self.columns = {"ID": ID, "X": X, "Y": Y, "Z": Z, "V": V}
        self.data = self._set_data(data)

    def _set_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Standardize DataFrame to canonical format."""
        df = data.copy()[list(self.columns.values())]
        df = df.rename(columns={v: k for k, v in self.columns.items()})
        df = df.set_index(["ID", "X", "Y", "Z"])
        df["V"] = df["V"].astype(float)
        return df.sort_index(ascending=False)

    @property
    def specs(self) -> "GridDataSpecs":
        """Compute spatial and value extent statistics."""
        return GridDataSpecs.from_dataframe(self.data)

    @property
    def numpy_data(self) -> np.ndarray:
        """Return X, Y, Z, V as a numpy array."""
        return self.data.reset_index()[["X", "Y", "Z", "V"]].to_numpy()

    @property
    def hull(self) -> BaseGeometry:
        """Convex hull of XY coordinates as a shapely geometry."""
        from shapely import MultiPoint

        xy = self.data.reset_index()[["X", "Y"]].drop_duplicates().to_numpy()
        return MultiPoint(xy).convex_hull

    def __len__(self) -> int:
        return len(self.data)

    def __repr__(self) -> str:
        n_points = len(self.data)
        n_ids = self.data.index.get_level_values("ID").nunique()
        return f"GridData(points={n_points}, ids={n_ids})"


@dataclass(frozen=True)
class GridDataSpecs:
    """Precomputed spatial and value extent statistics.

    Computed once from a DataFrame rather than re-reading on every access.
    """

    xmin: float
    xmax: float
    ymin: float
    ymax: float
    zmin: float
    zmax: float
    vmin: float
    vmax: float

    @classmethod
    def from_dataframe(cls, data: pd.DataFrame) -> "GridDataSpecs":
        """Compute specs from a canonical GridData DataFrame."""
        df = data.reset_index()
        return cls(
            xmin=float(df["X"].min()),
            xmax=float(df["X"].max()),
            ymin=float(df["Y"].min()),
            ymax=float(df["Y"].max()),
            zmin=float(df["Z"].min()),
            zmax=float(df["Z"].max()),
            vmin=float(df["V"].min()),
            vmax=float(df["V"].max()),
        )

    @property
    def xsize(self) -> float:
        return abs(self.xmax - self.xmin)

    @property
    def ysize(self) -> float:
        return abs(self.ymax - self.ymin)

    @property
    def zsize(self) -> float:
        return abs(self.zmax - self.zmin)
