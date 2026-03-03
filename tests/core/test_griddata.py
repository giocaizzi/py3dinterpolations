"""test GridData class"""

import pytest

from py3dinterpolations.core.griddata import GridData, GridDataSpecs
from py3dinterpolations.core.types import PreprocessingParams

import pandas as pd


@pytest.mark.parametrize(
    "test_data,colnames,pp_params",
    [
        ("griddata_default_colnames.csv", ["ID", "X", "Y", "Z", "V"], None),
        (
            "griddata_default_colnames.csv",
            ["ID", "X", "Y", "Z", "V"],
            PreprocessingParams(),
        ),
        ("griddata_custom_colnames.csv", ["IID", "XX", "YY", "ZZ", "VV"], None),
        (
            "griddata_custom_colnames.csv",
            ["IID", "XX", "YY", "ZZ", "VV"],
            PreprocessingParams(),
        ),
    ],
    indirect=["test_data"],
)
def test_griddata_init(test_data, colnames, pp_params):
    """test GridData initialization with custom column names"""
    griddata = GridData(
        test_data,
        ID=colnames[0],
        X=colnames[1],
        Y=colnames[2],
        Z=colnames[3],
        V=colnames[4],
        preprocessing_params=pp_params,
    )

    if pp_params is None:
        assert griddata.preprocessing_params is None
    else:
        assert isinstance(griddata.preprocessing_params, PreprocessingParams)

    # assert default column names mapping
    assert griddata.columns == {
        "ID": colnames[0],
        "X": colnames[1],
        "Y": colnames[2],
        "Z": colnames[3],
        "V": colnames[4],
    }

    # data format
    assert isinstance(griddata.data, pd.DataFrame)
    assert len(griddata.data.index.names) == 4
    assert griddata.data.index.names == ["ID", "X", "Y", "Z"]
    assert len(griddata.data.columns) == 1
    assert griddata.data.columns.tolist() == ["V"]
    assert griddata.data["V"].dtype == "float64"


def test_griddata_specs(test_data):
    gd = GridData(test_data)
    specs = gd.specs
    assert isinstance(specs, GridDataSpecs)
    assert specs.xmin <= specs.xmax
    assert specs.ymin <= specs.ymax
    assert specs.zmin <= specs.zmax
    assert specs.vmin <= specs.vmax
    assert specs.xsize >= 0
    assert specs.ysize >= 0
    assert specs.zsize >= 0


def test_griddata_hull(test_data):
    gd = GridData(test_data)
    hull = gd.hull
    assert hull is not None
    assert hull.area > 0


def test_griddata_len(test_data):
    gd = GridData(test_data)
    assert len(gd) == len(gd.data)


def test_griddata_repr(test_data):
    gd = GridData(test_data)
    r = repr(gd)
    assert "GridData" in r
    assert "points=" in r
