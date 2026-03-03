"""test IDW model (vectorized)"""

import numpy as np
import pytest

from py3dinterpolations.modelling.models.idw import IDWModel


def test_simple_idw():
    rng = np.random.default_rng(42)
    n_points = 20
    x = rng.uniform(0, 10, n_points)
    y = rng.uniform(0, 20, n_points)
    z = rng.uniform(0, 5, n_points)
    v = rng.random(n_points)

    resolution = 0.5
    grid_x = np.arange(x.min(), x.max(), resolution)
    grid_y = np.arange(y.min(), y.max(), resolution)
    grid_z = np.arange(z.min(), z.max(), resolution)

    model = IDWModel(power=1.0)
    model.fit(x, y, z, v)
    result = model.predict(grid_x, grid_y, grid_z)

    assert isinstance(result.interpolated, np.ndarray)
    # Shape should be (Z, Y, X) per pykrige convention
    expected_shape = (len(grid_z), len(grid_y), len(grid_x))
    assert result.interpolated.shape == expected_shape
    assert result.variance is None


def test_idw_exact_interpolation():
    """At a training point, IDW should return the exact training value."""
    x = np.array([0.0, 1.0, 2.0])
    y = np.array([0.0, 1.0, 2.0])
    z = np.array([0.0, 1.0, 2.0])
    v = np.array([10.0, 20.0, 30.0])

    model = IDWModel(power=2.0)
    model.fit(x, y, z, v)

    # Query exactly at training points
    result = model.predict(
        np.array([0.0]),
        np.array([0.0]),
        np.array([0.0]),
    )
    assert np.isclose(result.interpolated.flatten()[0], 10.0)


def test_idw_must_fit_before_predict():
    model = IDWModel()
    with pytest.raises(RuntimeError, match="fit"):
        model.predict(np.array([0.0]), np.array([0.0]), np.array([0.0]))


def test_idw_power_effect():
    """Higher power should produce more localized interpolation."""
    rng = np.random.default_rng(99)
    n = 10
    x = rng.uniform(0, 10, n)
    y = rng.uniform(0, 10, n)
    z = rng.uniform(0, 10, n)
    v = rng.uniform(0, 100, n)

    grid_x = np.arange(0, 10, 2.0)
    grid_y = np.arange(0, 10, 2.0)
    grid_z = np.arange(0, 10, 2.0)

    model_low = IDWModel(power=1.0)
    model_low.fit(x, y, z, v)
    result_low = model_low.predict(grid_x, grid_y, grid_z)

    model_high = IDWModel(power=4.0)
    model_high.fit(x, y, z, v)
    result_high = model_high.predict(grid_x, grid_y, grid_z)

    # Higher power should produce different (more extreme) values
    assert not np.allclose(result_low.interpolated, result_high.interpolated)
