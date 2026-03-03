"""Sklearn-compatible model wrapper."""

import numpy as np

from ...core.types import InterpolationResult
from .base import BaseModel


class SklearnModel(BaseModel):
    """Wrapper for any sklearn estimator with fit/predict interface.

    Handles classifiers (predict_proba) and regressors (predict).

    Args:
        estimator: A sklearn estimator instance.
        model_name: Human-readable name for this model.
    """

    def __init__(self, estimator: object, model_name: str = "sklearn"):
        self._estimator = estimator
        self._model_name = model_name

    def fit(self, x: np.ndarray, y: np.ndarray, z: np.ndarray, v: np.ndarray) -> None:
        """Fit the sklearn estimator."""
        X = np.column_stack([x, y, z])
        self._estimator.fit(X, v)

    def predict(
        self,
        grid_x: np.ndarray,
        grid_y: np.ndarray,
        grid_z: np.ndarray,
        **kwargs: object,
    ) -> InterpolationResult:
        """Predict on a regular grid.

        Returns:
            InterpolationResult with shape (len(grid_z), len(grid_y), len(grid_x))
            to match the convention of other models.
        """
        mx, my, mz = np.meshgrid(grid_x, grid_y, grid_z, indexing="ij")
        X = np.column_stack([mx.ravel(), my.ravel(), mz.ravel()])

        predictions = self._estimator.predict(X)
        interpolated = predictions.reshape(mx.shape)
        # Transpose from XYZ to ZYX to match pykrige convention
        interpolated = np.einsum("xyz->zyx", interpolated)

        probability = None
        if hasattr(self._estimator, "predict_proba"):
            proba = self._estimator.predict_proba(X)
            probability = proba.reshape((*mx.shape, -1))

        return InterpolationResult(
            interpolated=interpolated,
            probability=probability,
        )

    @property
    def name(self) -> str:
        return self._model_name
