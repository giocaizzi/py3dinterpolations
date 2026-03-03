"""test utility functions"""

import pandas as pd
from pandas.testing import assert_series_equal

from py3dinterpolations.modelling.utils import standardize, normalize
from py3dinterpolations.core.types import NormalizationParams, StandardizationParams


def test_standardize():
    """test that standardize returns correct values"""
    data = pd.DataFrame({"V": [1, 2, 3, 4, 5, 6, 7, 8, 9]})
    output_series, output_params = standardize(data["V"])

    assert_series_equal(
        output_series,
        pd.Series(
            [
                -1.460593,
                -1.095445,
                -0.730297,
                -0.365148,
                0.000000,
                0.365148,
                0.730297,
                1.095445,
                1.460593,
            ],
            name="V",
        ),
    )

    assert isinstance(output_params, StandardizationParams)
    assert abs(output_series.mean()) < 1e-10
    assert abs(output_series.std() - 1.0) < 1e-10
    assert output_params.mean == 5.0
    assert output_params.std == 2.7386127875258306


def test_normalize():
    """test that normalize returns correct values"""
    data = pd.Series([10, 20, 30, 40, 50], name="X")
    output_series, output_params = normalize(data)

    assert isinstance(output_params, NormalizationParams)
    assert output_params.min == 10.0
    assert output_params.max == 50.0
    assert output_series.min() == 0.0
    assert output_series.max() == 1.0
