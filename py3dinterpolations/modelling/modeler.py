"""model wrapper for interpolation"""

import numpy as np
from typing import Union

from ..core.griddata import GridData
from ..core.grid3d import Grid3D
from .models import ModelsWrapper


class Modeler3D:
    """modeler for wrapping 3d models from different libraries

    Currently supports:
        Statistical:
            - Ordinary Kriging : `ordinary_kriging` (pykrige)
        Deterministic:
            - Inverse Distance Weighting : `idw`


    This class is designed to allow future of sklearn models.

    Defines a fit and a predict method, to be called
    when interpolating.

    Ensures correct input and output.

    Args:
        griddata (GridData): GridData istance
        grid3d (Grid3D): Grid3D istance
        model_name (str): model name, default `ordinary_kriging`
        model_params (dict): model parameters

    Attributes:
        griddata (GridData): GridData istance
        grid3d (Grid3D): Grid3D istance
        model (object): model object
        results (dict): dictionary with interpolated and variance grids
    """

    model: ModelsWrapper
    results: Union[dict, np.ndarray]

    def __init__(
        self,
        griddata: GridData,
        grid3d: Grid3D,
        model_name: str = "ordinary_kriging",
        model_params: dict = {},
    ):
        # griddata and grid3d
        self.griddata = griddata
        self.grid3d = grid3d

        # model
        self.model = ModelsWrapper(
            model_name,
            self.griddata.numpy_data[:, 0],  # x
            self.griddata.numpy_data[:, 1],  # y
            self.griddata.numpy_data[:, 2],  # z
            self.griddata.numpy_data[:, 3],  # value
            **model_params,
        )

    def predict(self, **kwargs) -> np.ndarray:
        """makes predictions considering all past preprocessing

        - if preprocessing was applied, predict on normalized grid
        - if standardized data, reverse standardization
        - reshape from zxy to xyz (pykrige output)

        Args:
            grids_arrays (dict): dictionary with x, y, z grids 1d np.ndarray

        Returns:
            interpolated (np.ndarray): interpolated grid
        """
        # make predictions on normalized grid if preprocessing was applied
        if self.griddata.preprocessing_params == {}:
            grids_arrays = self.grid3d.grid
        else:
            grids_arrays = self.grid3d.normalized_grid

        # predict
        #
        interpolated, variance = self.model.predict(
            grids_arrays["X"],
            grids_arrays["Y"],
            grids_arrays["Z"],
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
        if self.model._model_name == "ordinary_kriging":
            # reshape pykrige output
            interpolated = _reshape_pykrige(interpolated)
            variance = _reshape_pykrige(variance)

        # save results
        self.results = {
            "interpolated": interpolated,
            "variance": variance,
        }

        # sets results  also in associated grid3d
        self.grid3d.results = self.results

        # return interpolated grid
        return interpolated


def _reverse_standardized(data: np.ndarray, standardization: dict) -> np.ndarray:
    """reverse standardization of a 1d numpy array

    considers that data could be none, in which case returns an empty array
    """
    return data * standardization["std"] + standardization["mean"]


def _reshape_pykrige(ndarray: np.ndarray) -> np.ndarray:
    """reshape pykrige output to match the grid3d shape"""
    return np.einsum("ZXY->XYZ", ndarray)
