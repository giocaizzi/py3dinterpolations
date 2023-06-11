"""model wrapper for interpolation"""

import numpy as np

from ..core.griddata import GridData
from ..core.grid3d import Grid3D
from ..modelling.models import SUPPORTED_MODELS


class Modeler3D:
    """modeler for wrapping 3d models from different libraries

    currently only supports
    - pykrige

    allow future wrapping of sklearn models

    Defines a fit and a predict method, to be called
    by the Interpolator3D class.

    Ensures correct input and output for the Interpolator3D class.

    Args:
        griddata (GridData): GridData istance
        grid3d (Grid3D): Grid3D istance
        model_type (str): model type, statistical or deterministic,
            default `statistical`
        model_name (str): model name, default `ordinary_kriging`
        model_params (dict): model parameters

    Attributes:
        griddata (GridData): GridData istance
        grid3d (Grid3D): Grid3D istance
        model (object): model object
        _model_name (str): model name
        _model_type (str): model type
        results (dict): dictionary with interpolated and variance grids
    """

    griddata: GridData = None
    grid3d: Grid3D = None
    model = None
    _model_name = None
    _model_type = None
    results: dict = None

    def __init__(
        self,
        griddata: GridData = None,
        grid3d: Grid3D = None,
        model_type: str = "statistical",
        model_name: str = "oridnary_kriging",
        model_params: dict = None,
    ):
        # model type and name
        self._model_type = model_type
        self._model_name = model_name

        # griddata and grid3d
        self.griddata = griddata
        self.grid3d = grid3d

        # model
        self.model = SUPPORTED_MODELS[self._model_type][self._model_name](
            self.griddata.numpy_data[:, 0],  # x
            self.griddata.numpy_data[:, 1],  # y
            self.griddata.numpy_data[:, 2],  # z
            self.griddata.numpy_data[:, 3],  # value
            **model_params,
        )

    def fit(self):
        # pykrige fit automatically
        pass

    def predict(self, **kwargs) -> None:
        """makes predictions considering all past preprocessing

        - if preprocessing was applied, predict on normalized grid
        - if standardized data, reverse standardization
        - reshape from zxy to xyz (pykrige output)

        Args:
            grids_arrays (dict): dictionary with x, y, z grids 1d np.ndarray
        """
        # make predictions on normalized grid if preprocessing was applied
        if self.griddata.preprocessing_params is None:
            grids_arrays = self.grid3d.grid
        else:
            grids_arrays = self.grid3d.normalized_grid

        # predict
        interpolated, variance = self.model.execute(
            style="grid",
            xpoints=grids_arrays["X"],
            ypoints=grids_arrays["Y"],
            zpoints=grids_arrays["Z"],
            **kwargs,
        )

        # if standardized data, reverse standardization
        if "standardization" in self.griddata.preprocessing_params:
            interpolated = _reverse_standardized(
                interpolated, self.griddata.preprocessing_params["standardization"]
            )
            variance = _reverse_standardized(
                variance, self.griddata.preprocessing_params["standardization"]
            )

        # reshape fron zxy to xyz
        if self._model_name == "ordinary_kriging":
            # reshape pykrige output
            interpolated = _reshape_pykrige(interpolated)
            variance = _reshape_pykrige(variance)

        # save results
        self.results = {
            "interpolated": interpolated,
            "variance": variance,
        }

        # sets results  also in grid3d
        self.grid3d.results = self.results


def _reverse_standardized(data: np.ndarray, standardization: dict) -> np.ndarray:
    """reverse standardization of a 1d numpy array"""
    return data * standardization["std"] + standardization["mean"]


def _reshape_pykrige(ndarray: np.ndarray) -> np.ndarray:
    """reshape pykrige output to match the grid3d shape"""
    return np.einsum("ZXY->XYZ", ndarray)
