"""test Modeler class"""

import pytest
import numpy as np

from py3dinterpolations.core.griddata import GridData
from py3dinterpolations.core.grid3d import create_grid
from py3dinterpolations.modelling.modeler import Modeler
from py3dinterpolations.modelling.models import get_model
from py3dinterpolations.core.types import InterpolationResult

scenarios = [
    ("ordinary_kriging", {"variogram_model": "linear", "nlags": 6, "weight": True}),
    ("idw", {"power": 2}),
]


@pytest.mark.parametrize("model_name,model_params", scenarios)
def test_modeler_init_and_predict(model_name, model_params, test_data):
    """test that the correct model is initialised and predict works"""
    gd = GridData(test_data)
    grid = create_grid(gd, 5)
    model = get_model(model_name, **model_params)
    modeler = Modeler(griddata=gd, grid=grid, model=model)

    assert modeler.griddata is gd
    assert modeler.grid is grid
    assert modeler.model is model
    assert modeler.result is None

    # predict
    interpolated = modeler.predict()

    assert modeler.result is not None
    assert isinstance(modeler.result, InterpolationResult)
    assert isinstance(modeler.result.interpolated, np.ndarray)
    assert isinstance(interpolated, np.ndarray)
