"""test models module"""

import pytest
from unittest.mock import patch

import numpy as np

from py3Dinterpolations.core.griddata import GridData
from py3Dinterpolations.core import create_regulargrid3d_from_griddata
from py3Dinterpolations.modelling.modeler import Modeler3D, SUPPORTED_MODELS


TEST_MODEL = {
    "model_type": "statistical",
    "model_name": "ordinary_kriging",
}

TEST_MODEL_MODEL_PARAMS = {
    "variogram_model": "linear",
    "nlags": 6,
    "weight": True,
}


# test init of Modeler3D
def test_Modeler3D_init(test_data):
    """test that the correct model is initialised"""

    # quick testing without mocking, should be improved

    gd = GridData(test_data)
    g3d = create_regulargrid3d_from_griddata(gd, 1)
    m3d = Modeler3D(
        griddata=gd,
        grid3d=g3d,
        model_name=TEST_MODEL["model_name"],
        model_params=TEST_MODEL_MODEL_PARAMS,
    )
    assert m3d._model_type == TEST_MODEL["model_type"]
    assert m3d._model_name == TEST_MODEL["model_name"]
    assert m3d.griddata == gd
    assert m3d.grid3d == g3d

    assert m3d.model is not None

    # assert m3d.grid3d object has no results
    assert m3d.results is None

    # no testing of fit because in pykrige model is fitted at init
    # predict
    m3d.predict()

    # assert m3d.grid3d object has results
    assert isinstance(m3d.results["interpolated"], np.ndarray)
    assert isinstance(m3d.results["variance"], np.ndarray)

    # assert m3d.grid3d object has now results
    assert m3d.grid3d.results["interpolated"] is not None
    assert m3d.grid3d.results["variance"] is not None
