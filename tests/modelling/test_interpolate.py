"""test interpolate function"""

import pytest
import numpy as np
from unittest.mock import patch

from py3dinterpolations.core.griddata import GridData
from py3dinterpolations.modelling.interpolate import interpolate
from py3dinterpolations.modelling.modeler import Modeler


def test_interpolate_model_params_no_preprocessing(test_data):
    """test interpolate with model_params, returns Modeler"""
    gd = GridData(test_data)
    modeler = interpolate(
        griddata=gd,
        model_type="ordinary_kriging",
        grid_resolution=5,
        model_params={
            "variogram_model": "linear",
            "nlags": 6,
            "weight": True,
        },
    )
    assert isinstance(modeler, Modeler)
    assert modeler.result is not None
    assert isinstance(modeler.result.interpolated, np.ndarray)


def test_interpolate_model_params_preprocessing(test_data):
    """test interpolate with model_params and preprocessing"""
    gd = GridData(test_data)
    with patch(
        "py3dinterpolations.modelling.interpolate.Preprocessor.preprocess",
        return_value=gd,
    ) as mock:
        modeler = interpolate(
            griddata=gd,
            model_type="ordinary_kriging",
            grid_resolution=5,
            model_params={
                "variogram_model": "linear",
                "nlags": 6,
                "weight": True,
            },
            preprocessing={"downsampling_res": 10},
        )
    mock.assert_called_once()
    assert isinstance(modeler, Modeler)


def test_interpolate_default_params(test_data):
    """test that no params defaults to empty dict (model defaults)"""
    gd = GridData(test_data)
    modeler = interpolate(
        griddata=gd,
        model_type="ordinary_kriging",
        grid_resolution=5,
    )
    assert isinstance(modeler, Modeler)
    assert modeler.result is not None


def test_interpolate_both_params_raises(test_data):
    """test that ValueError is raised when both params given"""
    gd = GridData(test_data)
    with pytest.raises(ValueError, match="Cannot provide both"):
        interpolate(
            griddata=gd,
            model_type="ordinary_kriging",
            grid_resolution=5,
            model_params={"variogram_model": "linear"},
            model_params_grid={"variogram_model": ["linear", "spherical"]},
        )


def test_interpolate_idw(test_data):
    """test interpolate with IDW model"""
    gd = GridData(test_data)
    modeler = interpolate(
        griddata=gd,
        model_type="idw",
        grid_resolution=5,
        model_params={"power": 2},
    )
    assert isinstance(modeler, Modeler)
    assert modeler.result is not None
