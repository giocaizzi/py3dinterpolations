"""test regulargrid3d object and its properties"""

import pytest

from py3dinterpolations.core.grid3d import RegularGrid3D, GridAxis, Grid3D

REGULARGRID3D_PARAMETERS = {
    "x_min": 0.0,
    "x_max": 1.0,
    "y_min": 0.0,
    "y_max": 1.0,
    "z_min": 0.0,
    "z_max": 1.0,
    "gridres": 0.1,
}


def test_regulargrid3d():
    rg3d = RegularGrid3D(**REGULARGRID3D_PARAMETERS)
    assert isinstance(rg3d, Grid3D)
    assert isinstance(rg3d.X, GridAxis)
    assert rg3d.X.min == REGULARGRID3D_PARAMETERS["x_min"]
    assert rg3d.X.max == REGULARGRID3D_PARAMETERS["x_max"]
    assert rg3d.X.res == REGULARGRID3D_PARAMETERS["gridres"]
    assert isinstance(rg3d.Y, GridAxis)
    assert rg3d.Y.min == REGULARGRID3D_PARAMETERS["y_min"]
    assert rg3d.Y.max == REGULARGRID3D_PARAMETERS["y_max"]
    assert rg3d.Y.res == REGULARGRID3D_PARAMETERS["gridres"]
    assert isinstance(rg3d.Z, GridAxis)
    assert rg3d.Z.min == REGULARGRID3D_PARAMETERS["z_min"]
    assert rg3d.Z.max == REGULARGRID3D_PARAMETERS["z_max"]
    assert rg3d.Z.res == REGULARGRID3D_PARAMETERS["gridres"]
    assert rg3d.results == {}
    assert isinstance(rg3d.grid, dict)
    assert isinstance(rg3d.mesh, dict)
    assert isinstance(rg3d.normalized_grid, dict)
