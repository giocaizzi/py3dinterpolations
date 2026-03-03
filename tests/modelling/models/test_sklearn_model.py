"""test SklearnModel wrapper"""

import numpy as np
import pytest
from sklearn.neighbors import KNeighborsRegressor

from py3dinterpolations.modelling.models.sklearn_model import SklearnModel


def test_sklearn_model_regressor():
    rng = np.random.default_rng(42)
    n = 50
    x = rng.uniform(0, 10, n)
    y = rng.uniform(0, 10, n)
    z = rng.uniform(0, 5, n)
    v = x + y + z  # simple linear relationship

    model = SklearnModel(KNeighborsRegressor(n_neighbors=3), model_name="knn")
    model.fit(x, y, z, v)

    grid_x = np.arange(0, 10, 2.0)
    grid_y = np.arange(0, 10, 2.0)
    grid_z = np.arange(0, 5, 1.0)

    result = model.predict(grid_x, grid_y, grid_z)
    assert result.interpolated.shape == (len(grid_z), len(grid_y), len(grid_x))
    assert result.probability is None
    assert model.name == "knn"
