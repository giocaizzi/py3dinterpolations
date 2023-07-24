### test _standardize function
from py3dinterpolations.modelling.utils import _standardize


import pandas as pd
from pandas.testing import assert_series_equal


def test_standardize():
    """test that the standardize function returns the right values"""
    data = pd.DataFrame({"V": [1, 2, 3, 4, 5, 6, 7, 8, 9]})
    output = _standardize(data["V"])
    assert_series_equal(
        output[0],
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

    # return type
    assert isinstance(output, tuple)

    def compare_floats(a, b, eps=1e-10):
        return abs(a - b) < eps

    # standardized output
    assert compare_floats(output[0].mean(), 0.0)
    assert compare_floats(output[0].std(), 1.0)

    # tuple of type
    assert isinstance(output[1], dict)
    assert output[1] == {"mean": 5.0, "std": 2.7386127875258306}
