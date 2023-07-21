"""test interpolate function"""

import pytest
import numpy as np
from unittest.mock import patch

from py3dinterpolations.core.griddata import GridData
from py3dinterpolations.modelling.interpolator import interpolate
from py3dinterpolations.modelling.modeler import Modeler3D


def test_interpolate_model_params_no_preprocessing(test_data):
    """test interpolate with model_params"""
    gd = GridData(test_data)

    interpolated = interpolate(
        griddata=gd,
        model_name="ordinary_kriging",
        grid_resolution=5,
        model_params={
            "variogram_model": "linear",
            "nlags": 6,
            "weight": True,
        },
    )
    assert isinstance(interpolated, np.ndarray)


def test_interpolate_return_model(test_data):
    """test interpolate when return_model is True"""
    gd = GridData(test_data)

    interpolated, model = interpolate(
        griddata=gd,
        model_name="ordinary_kriging",
        grid_resolution=5,
        model_params={
            "variogram_model": "linear",
            "nlags": 6,
            "weight": True,
        },
        return_model=True,
    )
    assert isinstance(interpolated, np.ndarray)
    assert model is not None
    assert isinstance(model, Modeler3D)


def test_interpolate_model_params_preprocessing(test_data):
    """test interpolate with model_params and preprocessing"""
    gd = GridData(test_data)

    # mock preprocessing preprocess method
    with patch(
        "py3dinterpolations.modelling.interpolator.Preprocessing.preprocess",
        return_value=gd,
    ) as mock:
        interpolated = interpolate(
            griddata=gd,
            model_name="ordinary_kriging",
            grid_resolution=5,
            model_params={
                "variogram_model": "linear",
                "nlags": 6,
                "weight": True,
            },
            preprocess_kwags={"downsampling_res": 10},
        )
    mock.assert_called_once()
    assert isinstance(interpolated, np.ndarray)
