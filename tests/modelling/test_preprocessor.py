"""test preprocessing class"""

import pytest
import itertools
from pandas.testing import assert_frame_equal

from py3dinterpolations.core.griddata import GridData
from py3dinterpolations.core.types import PreprocessingParams
from py3dinterpolations.modelling.preprocessor import (
    Preprocessor,
    reverse_preprocessing,
)


def test_preprocessor_init(test_data):
    gd = GridData(test_data)
    preprocess = Preprocessor(gd)
    assert preprocess.griddata is gd
    assert preprocess.downsampling_res is None
    assert preprocess.standardize_v is True
    assert preprocess.normalize_xyz is True


# original data is at 1 m resolution so the 4 m downsampling_res
PREPROCESSING_COMBINATIONS = list(
    itertools.product([4, None], [True, False], [True, False])
)


@pytest.mark.parametrize(
    "downsampling_res, normalize_xyz, standardize_v",
    PREPROCESSING_COMBINATIONS,
)
def test_preprocessor_preprocess_output(
    downsampling_res, normalize_xyz, standardize_v, test_data
):
    """test that the Preprocessor class returns a griddata object"""
    gd = GridData(test_data)
    pp_gd = Preprocessor(
        gd,
        downsampling_res=downsampling_res,
        normalize_xyz=normalize_xyz,
        standardize_v=standardize_v,
    ).preprocess()

    assert isinstance(pp_gd, GridData)
    assert isinstance(pp_gd.preprocessing_params, PreprocessingParams)

    if downsampling_res is not None:
        assert len(gd.data) > len(pp_gd.data)
        assert pp_gd.preprocessing_params.downsampling is not None

    if normalize_xyz:
        df = pp_gd.data.copy().reset_index()
        assert df["X"].min() == 0
        assert df["X"].max() == 1
        assert df["Y"].min() == 0
        assert df["Y"].max() == 1
        assert df["Z"].min() == 0
        assert df["Z"].max() == 1
        assert pp_gd.preprocessing_params.normalization is not None

    if standardize_v:
        assert abs(pp_gd.data["V"].mean()) < 1e-10
        assert abs(pp_gd.data["V"].std() - 1.0) < 1e-10
        assert pp_gd.preprocessing_params.standardization is not None

    # same number of unique ids
    assert len(gd.data.index.get_level_values("ID").unique()) == len(
        pp_gd.data.index.get_level_values("ID").unique()
    )


@pytest.mark.parametrize(
    "downsampling_res, normalize_xyz, standardize_v",
    PREPROCESSING_COMBINATIONS,
)
def test_reverse_preprocessing(downsampling_res, normalize_xyz, standardize_v, test_data):
    """test reverse_preprocessing restores original data"""
    gd = GridData(test_data)
    pp_gd = Preprocessor(
        gd,
        downsampling_res=downsampling_res,
        normalize_xyz=normalize_xyz,
        standardize_v=standardize_v,
    ).preprocess()

    gd_rev = reverse_preprocessing(pp_gd)

    if downsampling_res is None:
        # dataframes should be equal when no downsampling
        assert_frame_equal(gd.data, gd_rev.data)
    else:
        # with downsampling, at least check we can reverse
        assert isinstance(gd_rev, GridData)
        assert gd_rev.preprocessing_params is None


def test_reverse_preprocessing_no_params(test_data):
    """test that ValueError is raised when no params"""
    gd = GridData(test_data)
    with pytest.raises(ValueError, match="No preprocessing"):
        reverse_preprocessing(gd)
