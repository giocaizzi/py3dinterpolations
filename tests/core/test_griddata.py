"""test griddata class"""

import pytest
from unittest.mock import patch

from py3dinterpolations.core.griddata import GridData

import pandas as pd


@pytest.mark.parametrize(
    "test_data,colnames,preprocessor_params",
    [
        ("griddata_default_colnames.csv", ["ID", "X", "Y", "Z", "V"], None),
        ("griddata_default_colnames.csv", ["ID", "X", "Y", "Z", "V"], {"foo": "bar"}),
        ("griddata_custom_colnames.csv", ["IID", "XX", "YY", "ZZ", "VV"], None),
        (
            "griddata_custom_colnames.csv",
            ["IID", "XX", "YY", "ZZ", "VV"],
            {"foo": "bar"},
        ),
    ],
    indirect=["test_data"],
)
def test_GridData_init(test_data, colnames, preprocessor_params):
    """test GridData initalization with custom column names"""
    # avoid setting specs

    if preprocessor_params is None:
        griddata = GridData(
            test_data,
            ID=colnames[0],
            X=colnames[1],
            Y=colnames[2],
            Z=colnames[3],
            V=colnames[4],
        )
    else:
        griddata = GridData(
            test_data,
            ID=colnames[0],
            X=colnames[1],
            Y=colnames[2],
            Z=colnames[3],
            V=colnames[4],
            preprocessor_params=preprocessor_params,
        )

    # assert preprocessor_params is not passed
    if preprocessor_params == None:
        # assert preprocessor_params is empty dict
        assert griddata.preprocessor_params == {}
    else:
        # saved preprocessor_params
        assert isinstance(griddata.preprocessor_params, dict)
        assert griddata.preprocessor_params == preprocessor_params

    # assert default column names mapping
    assert griddata.columns == {
        "ID": colnames[0],
        "X": colnames[1],
        "Y": colnames[2],
        "Z": colnames[3],
        "V": colnames[4],
    }

    # assert data is in GridData.data standard
    assert isinstance(griddata.data, pd.DataFrame)
    # assert index
    assert len(griddata.data.index.names) == 4
    assert griddata.data.index.names == ["ID", "X", "Y", "Z"]
    # assert columns
    assert len(griddata.data.columns) == 1
    assert griddata.data.columns == ["V"]
    assert griddata.data["V"].dtype == "float64"


# test fails when prerpocessor_params is not dict
def test_GridData_init_fail(test_data):
    with pytest.raises(ValueError):
        gd = GridData(test_data, preprocessor_params="foo")
