"""test models module"""

import pytest

import numpy as np

from py3Dinterpolations.modelling.models import get_model_type

def test_get_model_type():
    """test get_model_type"""
    assert get_model_type("ordinary_kriging") == "statistical"
    assert get_model_type("idw") == "deterministic"
    with pytest.raises(ValueError):
        get_model_type("not_a_model")