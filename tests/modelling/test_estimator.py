"""test estimator module"""

import pytest
from unittest.mock import patch, MagicMock

from py3dinterpolations.core.griddata import GridData
from py3dinterpolations.modelling.estimator import Estimator


PARAMS = {
    "method": ["ordinary3d"],
    "variogram_model": ["linear", "power", "gaussian"],
    "nlags": [2, 4, 6, 8, 10],
    "weight": [True, False],
}


def test_estimator(test_data):
    """test Estimator class with mocked GridSearchCV"""
    gd = GridData(test_data)

    mock_cv = MagicMock()
    mock_cv.best_params_ = {"method": "ordinary3d", "variogram_model": "linear", "nlags": 6, "weight": True}
    mock_cv.best_score_ = -1.5
    mock_cv.cv_results_ = {"mean_test_score": [-1.5]}

    with patch("py3dinterpolations.modelling.estimator.GridSearchCV", return_value=mock_cv):
        estimator = Estimator(gd, PARAMS)

    assert isinstance(estimator.best_params, dict)
    assert isinstance(estimator.best_score, float)
    assert isinstance(estimator.cv_results, dict)
    mock_cv.fit.assert_called_once()
