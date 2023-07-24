"""test preprocessing class"""

import pytest
from unittest.mock import patch
import itertools
from pandas.testing import assert_frame_equal


from py3dinterpolations.core.griddata import GridData
from py3dinterpolations.modelling.preprocessor import (
    Preprocessor,
    reverse_preprocessing,
)


def test_Preprocessor_init(test_data):
    # test that the Preprocessor class is initialized correctly
    # with the right attributes
    # and preprocssing is not actually run
    gd = GridData(test_data)
    preprocess = Preprocessor(
        gd,
    )
    # griddata is set correctly
    assert preprocess.griddata == gd

    # Preprocessor Default parameters are set correctly
    assert preprocess.downsampling_res is None
    assert preprocess.standardize_v is True
    assert preprocess.normalize_xyz is True

    # preprocessing is not run
    assert preprocess.preprocessor_params == {}


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
def test_Preprocessor_preprocess_output(
    downsampling_res, normalize_xyz, standardize_v, test_data
):
    """test that the Preprocessor class restitute a griddata object"""
    # with same carachetristics as the original one,
    # if downsample with also less data

    gd = GridData(test_data)
    pp_gd = Preprocessor(
        gd,
        downsampling_res=downsampling_res,
        normalize_xyz=normalize_xyz,
        standardize_v=standardize_v,
    ).preprocess()

    # output is a griddata object
    assert isinstance(pp_gd, GridData)

    # if downsampling_res is not None,
    # less data is expected
    if downsampling_res is not None:
        # has less data
        assert len(gd.data) > len(pp_gd.data)

    # normalize means that are between 0 and 1
    if normalize_xyz:
        # X Y Z are normalized
        df = pp_gd.data.copy().reset_index()
        assert df["X"].min() == 0
        assert df["X"].max() == 1
        assert df["Y"].min() == 0
        assert df["Y"].max() == 1
        assert df["Z"].min() == 0
        assert df["Z"].max() == 1

    def compare_floats(a, b, eps=1e-10):
        return abs(a - b) < eps

    # standardize means that mean is 0 and std is 1
    if standardize_v:
        # V is standardized
        assert compare_floats(pp_gd.data["V"].mean(), 0.0)
        assert compare_floats(pp_gd.data["V"].std(), 1.0)

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
    pp_gd = Preprocessor(
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
