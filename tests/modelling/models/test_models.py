"""test model registry"""

import pytest

from py3dinterpolations.modelling.models import get_model
from py3dinterpolations.modelling.models.kriging import KrigingModel
from py3dinterpolations.modelling.models.idw import IDWModel
from py3dinterpolations.core.types import ModelType


def test_get_model_kriging():
    model = get_model(ModelType.ORDINARY_KRIGING, variogram_model="linear")
    assert isinstance(model, KrigingModel)
    assert model.name == "ordinary_kriging"


def test_get_model_idw():
    model = get_model(ModelType.IDW, power=2.0)
    assert isinstance(model, IDWModel)
    assert model.name == "idw"


def test_get_model_by_string():
    model = get_model("ordinary_kriging", variogram_model="linear")
    assert isinstance(model, KrigingModel)


def test_get_model_invalid():
    with pytest.raises(ValueError):
        get_model("not_a_model")
