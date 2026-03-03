"""test interpolate function with irregular (dict) grid resolution"""

import numpy as np

from py3dinterpolations.core.grid3d import IrregularGrid3D
from py3dinterpolations.core.griddata import GridData
from py3dinterpolations.modelling.interpolate import interpolate
from py3dinterpolations.modelling.modeler import Modeler

IRREGULAR_RESOLUTION = {"X": 5.0, "Y": 5.0, "Z": 2.0}


def test_interpolate_irregular_grid_idw(test_data):
    """interpolate with dict resolution creates IrregularGrid3D"""
    gd = GridData(test_data)
    modeler = interpolate(
        griddata=gd,
        model_type="idw",
        grid_resolution=IRREGULAR_RESOLUTION,
        model_params={"power": 2},
    )
    assert isinstance(modeler, Modeler)
    assert isinstance(modeler.grid, IrregularGrid3D)
    assert modeler.grid.X.res == 5.0
    assert modeler.grid.Z.res == 2.0
    assert modeler.result is not None
    assert isinstance(modeler.result.interpolated, np.ndarray)
    assert modeler.result.interpolated.ndim == 3


def test_interpolate_irregular_grid_kriging(test_data):
    """interpolate with dict resolution works for kriging"""
    gd = GridData(test_data)
    modeler = interpolate(
        griddata=gd,
        model_type="ordinary_kriging",
        grid_resolution=IRREGULAR_RESOLUTION,
        model_params={
            "variogram_model": "linear",
            "nlags": 6,
            "weight": True,
        },
    )
    assert isinstance(modeler, Modeler)
    assert isinstance(modeler.grid, IrregularGrid3D)
    assert modeler.result is not None
    assert modeler.result.variance is not None


def test_interpolate_irregular_grid_with_preprocessing(test_data):
    """interpolate with dict resolution and preprocessing"""
    gd = GridData(test_data)
    modeler = interpolate(
        griddata=gd,
        model_type="idw",
        grid_resolution=IRREGULAR_RESOLUTION,
        model_params={"power": 2},
        preprocessing={"downsampling_res": 10},
    )
    assert isinstance(modeler, Modeler)
    assert isinstance(modeler.grid, IrregularGrid3D)
    assert modeler.result is not None
    assert modeler.griddata.preprocessing_params is not None
