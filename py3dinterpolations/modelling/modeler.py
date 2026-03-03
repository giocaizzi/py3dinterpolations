"""High-level modelling orchestrator."""

import logging

import numpy as np

from ..core.grid3d import Grid3D
from ..core.griddata import GridData
from ..core.types import InterpolationResult
from .models.base import BaseModel

logger = logging.getLogger(__name__)


class Modeler:
    """Orchestrates fitting a model and predicting on a 3D grid.

    Handles normalization-aware grid selection and standardization reversal.

    Args:
        griddata: Training data.
        grid: 3D grid for predictions.
        model: Fitted or unfitted BaseModel instance. Will be fit on construction.
    """

    def __init__(
        self,
        griddata: GridData,
        grid: Grid3D,
        model: BaseModel,
    ):
        self._griddata = griddata
        self._grid = grid
        self._model = model
        self._result: InterpolationResult | None = None

        # Fit the model on training data
        data = griddata.numpy_data
        self._model.fit(data[:, 0], data[:, 1], data[:, 2], data[:, 3])
        logger.info("Model %s fitted on %d points", model.name, len(data))

    @property
    def griddata(self) -> GridData:
        return self._griddata

    @property
    def grid(self) -> Grid3D:
        return self._grid

    @property
    def model(self) -> BaseModel:
        return self._model

    @property
    def result(self) -> InterpolationResult | None:
        return self._result

    def predict(self, **kwargs: object) -> np.ndarray:
        """Make predictions, handling normalization and standardization reversal.

        Returns:
            Interpolated numpy array.
        """
        logger.info("Starting prediction on grid %s", self._grid)

        # Use normalized grid if normalization was applied
        params = self._griddata.preprocessing_params
        if params is not None and params.normalization is not None:
            grid_arrays = self._grid.normalized_grid
        else:
            grid_arrays = self._grid.grid

        # Predict
        result = self._model.predict(
            grid_arrays["X"],
            grid_arrays["Y"],
            grid_arrays["Z"],
            **kwargs,
        )

        interpolated = result.interpolated
        variance = result.variance

        # Reverse standardization if it was applied
        if params is not None and params.standardization is not None:
            std_params = params.standardization
            interpolated = interpolated * std_params.std + std_params.mean
            if variance is not None:
                variance = variance * std_params.std + std_params.mean

        self._result = InterpolationResult(
            interpolated=interpolated,
            variance=variance,
            probability=result.probability,
        )

        # Also attach to grid
        self._grid.result = self._result

        logger.info("Prediction complete")
        return interpolated
