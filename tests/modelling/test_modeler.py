"""test models module"""

import pytest
from unittest.mock import patch

import numpy as np

from py3Dinterpolations.core.griddata import GridData
from py3Dinterpolations.core import create_regulargrid3d_from_griddata
from py3Dinterpolations.modelling.modeler import Modeler3D

scenarios = [
    (
        "ordinary_kriging",
        {
            "variogram_model": "linear",
            "nlags": 6,
            "weight": True,
        },
    ),
    ("idw", {"power": 2}),
]


@pytest.mark.parametrize("model_name,model_params", scenarios)
# test init of Modeler3D
def test_Modeler3D_init(model_name, model_params, test_data):
    """test that the correct model is initialised"""

    # quick testing without mocking, should be improved

    gd = GridData(test_data)
    g3d = create_regulargrid3d_from_griddata(gd, 1)
    m3d = Modeler3D(
        griddata=gd,
        grid3d=g3d,
        model_name=model_name,
        model_params=model_params,
    )
    assert m3d.model._model_name == model_name
    assert m3d.griddata == gd
    assert m3d.grid3d == g3d

    assert m3d.model is not None

    # assert m3d.grid3d object has no results
    assert m3d.results is None

    # predict
    m3d.predict()

    if m3d.model._model_type=="statistical":
        # assert m3d.grid3d object has results
        # statistical models return a dict, interpolated and variance arrays    
        assert isinstance(m3d.results["interpolated"], np.ndarray)
        assert isinstance(m3d.results["variance"], np.ndarray)
    
    if m3d.model._model_type=="deterministic":
        # assert m3d.grid3d object has results
        # deterministic models return an array
        assert isinstance(m3d.results, np.ndarray)


