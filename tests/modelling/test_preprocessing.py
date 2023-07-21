"""test preprocessing class"""

import pytest
from unittest.mock import patch
import itertools
import pandas as pd
from pandas.testing import assert_series_equal, assert_frame_equal


from py3dinterpolations.core.griddata import GridData
from py3dinterpolations.modelling.preprocessing import Preprocessing
from py3dinterpolations.modelling.preprocessing import (
    reverse_preprocessing,
    _standardize,
)


### test _standardize function
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
    assert isinstance(output[1], dict)
    assert output[1] == {"mean": 5.0, "std": 2.7386127875258306}


def test_preprocessing_init(test_data):
    # test that the preprocessing class is initialized correctly
    # with the right attributes
    # and preprocssing is not actually run
    gd = GridData(test_data)
    preprocess = Preprocessing(
        gd,
    )
    # griddata is set correctly
    assert preprocess.griddata == gd
    # preprocessing parameters are set correctly
    assert preprocess.downsampling_res is None
    assert preprocess.standardize_v is True
    assert preprocess.normalize_xyz is True
    # preprocessing is not run
    assert preprocess.preprocessing_params == {}


# like this, all scenarios are tested,
# also the False, False, None
# original data is at 1 m resolution so the 4 m downsampling_res
PREPROCESSING_COMBINATIONS = list(
    itertools.product([4, None], [True, False], [True, False])
)


@pytest.mark.parametrize(
    "downsampling_res, normalize_xyz, standardize_v",
    PREPROCESSING_COMBINATIONS,
)
def test_Preprocessing_preprocess_subcalls(
    downsampling_res, normalize_xyz, standardize_v, test_data
):
    """test that the preprocessing class calls the right methods"""
    # this tests uses PREPROCESSING_COMBINATIONS to test all possible
    # combinations of downsampling_res, normalize_xyz, standardize_v
    # Create a mock for each of the three methods
    with patch(
        "py3dinterpolations.modelling.preprocessing.Preprocessing._downsample_data"
    ) as mock_downsample, patch(
        "py3dinterpolations.modelling.preprocessing.Preprocessing._normalize_xyz"
    ) as mock_normalize, patch(
        "py3dinterpolations.modelling.preprocessing.Preprocessing._standardize_v"
    ) as mock_standardize:
        # Create a Preprocessing object with different arguments
        gd = GridData(test_data)
        pp = Preprocessing(
            gd,
            downsampling_res=downsampling_res,
            normalize_xyz=normalize_xyz,
            standardize_v=standardize_v,
        )

        # Call the preprocess() method
        pp.preprocess()

        if downsampling_res is not None:
            # assert mock_downsample was called once
            mock_downsample.assert_called_once()

        if normalize_xyz:
            # assert mock_normalize was called once
            mock_normalize.assert_called_once()

        if standardize_v:
            # assert mock_standardize was called once
            mock_standardize.assert_called_once()


@pytest.mark.parametrize(
    "downsampling_res, normalize_xyz, standardize_v",
    PREPROCESSING_COMBINATIONS,
)
def test_Preprocessing_preprocess_output(
    downsampling_res, normalize_xyz, standardize_v, test_data
):
    """test that the preprocessing class restitute a griddata object"""
    # with same carachetristics as the original one,
    # if downsample with also less data

    gd = GridData(test_data)
    pp_gd = Preprocessing(
        gd,
        downsampling_res=downsampling_res,
        normalize_xyz=normalize_xyz,
        standardize_v=standardize_v,
    ).preprocess()

    # is a griddata object
    assert isinstance(pp_gd, GridData)

    if downsampling_res is not None:
        # has less data
        assert len(gd.data) > len(pp_gd.data)

    # in any case it would not change number of unique ids
    assert len(gd.data.index.get_level_values("ID").unique()) == len(
        pp_gd.data.index.get_level_values("ID").unique()
    )


@pytest.mark.skip
# test reverse preprocessing method
@pytest.mark.parametrize(
    "downsampling_res, normalize_xyz, standardize_v",
    PREPROCESSING_COMBINATIONS,
)
def test_reverse_preprocessing(
    downsampling_res, normalize_xyz, standardize_v, test_data
):
    gd = GridData(test_data)
    pp_gd = Preprocessing(
        gd,
        downsampling_res=downsampling_res,
        normalize_xyz=normalize_xyz,
        standardize_v=standardize_v,
    ).preprocess()
    # reverse preprocessing
    gd_rev = reverse_preprocessing(pp_gd)

    # can check if they are equal to original only if downsampling is not None
    if downsampling_res is None:
        print(gd.data, gd_rev.data)
        # datframes should be equal
        assert_frame_equal(gd.data, gd_rev.data)
    else:
        pytest.skip("NOT IMPLEMENTED YET")

    # todo: assert something if downsampling is not None
