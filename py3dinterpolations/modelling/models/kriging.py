"""Ordinary Kriging 3D model via pykrige."""

import numpy as np
from pykrige.ok3d import OrdinaryKriging3D

from ...core.types import InterpolationResult
from .base import BaseModel


class KrigingModel(BaseModel):
    """Ordinary Kriging 3D wrapper around pykrige.

    pykrige fits at construction time, so fit() constructs the
    OrdinaryKriging3D instance.

    Args:
        **kriging_params: Parameters passed to OrdinaryKriging3D constructor.
    """

    def __init__(self, **kriging_params: object):
        self._params = kriging_params
        self._model: OrdinaryKriging3D | None = None

    def fit(self, x: np.ndarray, y: np.ndarray, z: np.ndarray, v: np.ndarray) -> None:
        """Fit by constructing the OrdinaryKriging3D model."""
        self._model = OrdinaryKriging3D(x, y, z, v, **self._params)

    def predict(
        self,
        grid_x: np.ndarray,
        grid_y: np.ndarray,
        grid_z: np.ndarray,
        **kwargs: object,
    ) -> InterpolationResult:
        """Execute kriging on the given grid arrays.

        Returns:
            InterpolationResult with interpolated and variance arrays.
            Shape is (len(grid_z), len(grid_y), len(grid_x)) per pykrige convention.
        """
        if self._model is None:
            msg = "Model must be fit before predicting"
            raise RuntimeError(msg)
        interpolated, variance = self._model.execute(
            style="grid",
            xpoints=grid_x,
            ypoints=grid_y,
            zpoints=grid_z,
            **kwargs,
        )
        return InterpolationResult(
            interpolated=interpolated,
            variance=variance,
        )

    @property
    def name(self) -> str:
        return "ordinary_kriging"
