"""Vectorized Inverse Distance Weighting (IDW) model."""

import numpy as np

from ...core.types import InterpolationResult
from .base import BaseModel

# Maximum number of prediction points per batch to avoid OOM
_BATCH_SIZE = 50_000


class IDWModel(BaseModel):
    """Vectorized IDW interpolation.

    Uses numpy broadcasting instead of Python loops for ~1000x speedup
    on typical workloads. Batches computation for memory safety.

    Args:
        power: Power parameter controlling distance decay. Higher values
            give more weight to nearby points.
        threshold: Distance below which a point is treated as coincident
            with a training point (exact interpolation).
    """

    def __init__(self, power: float = 1.0, threshold: float = 1e-10):
        self._power = power
        self._threshold = threshold
        self._points: np.ndarray | None = None
        self._values: np.ndarray | None = None

    def fit(self, x: np.ndarray, y: np.ndarray, z: np.ndarray, v: np.ndarray) -> None:
        """Store training data."""
        self._points = np.column_stack([x, y, z])
        self._values = v

    def _predict_batch(self, query_points: np.ndarray) -> np.ndarray:
        """Predict values for a batch of query points.

        Args:
            query_points: (M, 3) array of prediction coordinates.

        Returns:
            (M,) array of interpolated values.
        """
        assert self._points is not None
        assert self._values is not None

        # (M, N) distance matrix
        diff = query_points[:, np.newaxis, :] - self._points[np.newaxis, :, :]
        distances = np.linalg.norm(diff, axis=2)

        # Handle exact interpolation: if any query point coincides with a
        # training point, return that training value directly
        exact_mask = distances < self._threshold
        has_exact = exact_mask.any(axis=1)

        # Compute IDW weights: w_i = 1 / d_i^p
        # Avoid division by zero at exact matches (we handle those separately)
        safe_distances = np.where(exact_mask, 1.0, distances)
        weights = 1.0 / np.power(safe_distances, self._power)
        weights = np.where(exact_mask, 0.0, weights)

        denominator = weights.sum(axis=1)
        # Guard against all-zero denominator
        safe_denominator = np.where(denominator == 0, 1.0, denominator)
        result = (weights * self._values[np.newaxis, :]).sum(axis=1) / safe_denominator
        result = np.where(denominator == 0, np.nan, result)

        # For exact matches, use the first coincident training point value
        if has_exact.any():
            exact_indices = exact_mask[has_exact].argmax(axis=1)
            result[has_exact] = self._values[exact_indices]

        return result

    def predict(
        self,
        grid_x: np.ndarray,
        grid_y: np.ndarray,
        grid_z: np.ndarray,
        **kwargs: object,
    ) -> InterpolationResult:
        """Predict on a regular grid defined by 1D arrays.

        Returns:
            InterpolationResult with shape (len(grid_z), len(grid_y), len(grid_x))
            to match pykrige's output convention.
        """
        if self._points is None:
            msg = "Model must be fit before predicting"
            raise RuntimeError(msg)

        # Build meshgrid in ij (XYZ) indexing for computation
        mx, my, mz = np.meshgrid(grid_x, grid_y, grid_z, indexing="ij")
        query_points = np.column_stack([mx.ravel(), my.ravel(), mz.ravel()])

        # Batch processing for memory safety
        n_points = len(query_points)
        result = np.empty(n_points)
        for start in range(0, n_points, _BATCH_SIZE):
            end = min(start + _BATCH_SIZE, n_points)
            result[start:end] = self._predict_batch(query_points[start:end])

        # Reshape to (X, Y, Z) then transpose to (Z, Y, X) to match pykrige
        interpolated = result.reshape(mx.shape)
        interpolated = np.einsum("xyz->zyx", interpolated)

        return InterpolationResult(interpolated=interpolated, variance=None)

    @property
    def name(self) -> str:
        return "idw"
