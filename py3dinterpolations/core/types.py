"""Type primitives for py3dinterpolations."""

from dataclasses import dataclass
from enum import StrEnum

import numpy as np


class Axis(StrEnum):
    """Spatial axis identifier."""

    X = "X"
    Y = "Y"
    Z = "Z"


class ModelType(StrEnum):
    """Supported interpolation model types."""

    ORDINARY_KRIGING = "ordinary_kriging"
    IDW = "idw"


class DownsamplingStatistic(StrEnum):
    """Supported downsampling statistics."""

    MEAN = "mean"
    MAX = "max"
    MIN = "min"
    MEDIAN = "median"
    SUM = "sum"
    QUANTILE75 = "quantile75"


@dataclass(frozen=True)
class NormalizationParams:
    """Min/max normalization parameters for a single axis."""

    min: float
    max: float


@dataclass(frozen=True)
class StandardizationParams:
    """Mean/std standardization parameters."""

    mean: float
    std: float


@dataclass(frozen=True)
class DownsamplingParams:
    """Downsampling parameters."""

    resolution: float


@dataclass(frozen=True)
class PreprocessingParams:
    """Collected preprocessing parameters from a preprocessing run."""

    downsampling: DownsamplingParams | None = None
    normalization: dict[Axis, NormalizationParams] | None = None
    standardization: StandardizationParams | None = None


@dataclass(frozen=True)
class GridResolution:
    """Grid resolution per axis.

    Use `uniform()` for equal resolution on all axes,
    or `from_input()` to accept either float or per-axis dict.
    """

    x: float
    y: float
    z: float

    @classmethod
    def uniform(cls, resolution: float) -> "GridResolution":
        """Create uniform resolution across all axes."""
        return cls(x=resolution, y=resolution, z=resolution)

    @classmethod
    def from_input(cls, resolution: float | dict[str, float]) -> "GridResolution":
        """Create from either a uniform float or per-axis dict."""
        if isinstance(resolution, (int, float)):
            return cls.uniform(float(resolution))
        if isinstance(resolution, dict):
            return cls(
                x=float(resolution["X"]),
                y=float(resolution["Y"]),
                z=float(resolution["Z"]),
            )
        msg = f"resolution must be float or dict, got {type(resolution)}"
        raise TypeError(msg)

    def __repr__(self) -> str:
        if self.x == self.y == self.z:
            return f"GridResolution(uniform={self.x})"
        return f"GridResolution(x={self.x}, y={self.y}, z={self.z})"


@dataclass
class InterpolationResult:
    """Result of an interpolation run."""

    interpolated: np.ndarray
    variance: np.ndarray | None = None
    probability: np.ndarray | None = None
