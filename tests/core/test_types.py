"""test type primitives"""

import pytest
import numpy as np

from py3dinterpolations.core.types import (
    Axis,
    ModelType,
    DownsamplingStatistic,
    NormalizationParams,
    StandardizationParams,
    DownsamplingParams,
    PreprocessingParams,
    GridResolution,
    InterpolationResult,
)


def test_axis_enum():
    assert Axis.X == "X"
    assert Axis.Y == "Y"
    assert Axis.Z == "Z"
    assert Axis("X") == Axis.X


def test_model_type_enum():
    assert ModelType.ORDINARY_KRIGING == "ordinary_kriging"
    assert ModelType.IDW == "idw"
    assert ModelType("idw") == ModelType.IDW


def test_downsampling_statistic_enum():
    assert DownsamplingStatistic.MEAN == "mean"
    assert DownsamplingStatistic.QUANTILE75 == "quantile75"


def test_normalization_params_frozen():
    params = NormalizationParams(min=0.0, max=10.0)
    assert params.min == 0.0
    assert params.max == 10.0
    with pytest.raises(AttributeError):
        params.min = 5.0  # type: ignore[misc]


def test_standardization_params():
    params = StandardizationParams(mean=5.0, std=2.5)
    assert params.mean == 5.0
    assert params.std == 2.5


def test_preprocessing_params_defaults():
    params = PreprocessingParams()
    assert params.downsampling is None
    assert params.normalization is None
    assert params.standardization is None


def test_grid_resolution_uniform():
    res = GridResolution.uniform(5.0)
    assert res.x == 5.0
    assert res.y == 5.0
    assert res.z == 5.0
    assert "uniform=5.0" in repr(res)


def test_grid_resolution_from_input_float():
    res = GridResolution.from_input(5.0)
    assert res.x == 5.0
    assert res.y == 5.0
    assert res.z == 5.0


def test_grid_resolution_from_input_dict():
    res = GridResolution.from_input({"X": 1.0, "Y": 2.0, "Z": 3.0})
    assert res.x == 1.0
    assert res.y == 2.0
    assert res.z == 3.0
    assert "x=1.0" in repr(res)


def test_grid_resolution_from_input_invalid():
    with pytest.raises(TypeError):
        GridResolution.from_input("invalid")


def test_grid_resolution_rejects_zero():
    with pytest.raises(ValueError, match="positive"):
        GridResolution(x=0, y=1, z=1)


def test_grid_resolution_rejects_negative():
    with pytest.raises(ValueError, match="positive"):
        GridResolution(x=1, y=-1, z=1)


def test_grid_resolution_from_input_rejects_zero():
    with pytest.raises(ValueError, match="positive"):
        GridResolution.from_input(0.0)


def test_interpolation_result():
    data = np.zeros((3, 4, 5))
    result = InterpolationResult(interpolated=data)
    assert result.interpolated is data
    assert result.variance is None
    assert result.probability is None
