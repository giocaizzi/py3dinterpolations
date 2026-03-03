"""Abstract base class for all interpolation models."""

from abc import ABC, abstractmethod

import numpy as np

from ...core.types import InterpolationResult


class BaseModel(ABC):
    """Interface for interpolation models.

    All models must implement fit() and predict() with consistent signatures.
    """

    @abstractmethod
    def fit(self, x: np.ndarray, y: np.ndarray, z: np.ndarray, v: np.ndarray) -> None:
        """Fit the model to training data.

        Args:
            x: X coordinates of training points.
            y: Y coordinates of training points.
            z: Z coordinates of training points.
            v: Values at training points.
        """
        ...

    @abstractmethod
    def predict(
        self,
        grid_x: np.ndarray,
        grid_y: np.ndarray,
        grid_z: np.ndarray,
        **kwargs: object,
    ) -> InterpolationResult:
        """Predict on 1D grid arrays.

        Args:
            grid_x: 1D array of X grid coordinates.
            grid_y: 1D array of Y grid coordinates.
            grid_z: 1D array of Z grid coordinates.

        Returns:
            Interpolation result with at least the interpolated field.
        """
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable model name."""
        ...
