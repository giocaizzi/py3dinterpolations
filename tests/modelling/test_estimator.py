"""test estimator module"""

import pytest
from unittest.mock import patch
import itertools

from sklearn.model_selection import GridSearchCV

from py3dinterpolations.core.griddata import GridData
from py3dinterpolations.modelling.estimator import Estimator

PARAMS = {
    "method": ["ordinary3d"],
    "variogram_model": ["linear", "power", "gaussian"],
    "nlags": [2, 4, 6, 8, 10],
    "weight": [True, False],
}

@pytest.mark.skip
def test_Estimator(test_data):
    """test Estimator class initialization"""
    gd = GridData(test_data)
    estimator = Estimator(gd, PARAMS)
    assert isinstance(estimator.estimator, GridSearchCV)
    assert isinstance(estimator.best_params, dict)
    assert isinstance(estimator.best_score, float)
