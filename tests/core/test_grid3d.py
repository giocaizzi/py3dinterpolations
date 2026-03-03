"""test Grid3D objects"""

import pytest
import numpy as np

from py3dinterpolations.core.grid3d import Grid3D, GridAxis, RegularGrid3D
from py3dinterpolations.core.types import Axis

GRID3D_PARAMETERS = [
    {
        "x_min": 0.0,
        "x_max": 10.0,
        "y_min": 0.0,
        "y_max": 5.0,
        "z_min": 0.0,
        "z_max": 8.0,
        "gridres": 0.1,
    },
]


@pytest.mark.parametrize("kwargs", GRID3D_PARAMETERS)
def test_grid3d_init_and_properties(kwargs):
    """test RegularGrid3D (concrete) init and properties"""
    g3d = RegularGrid3D(**kwargs)

    assert isinstance(g3d.X, GridAxis)
    assert g3d.X.min == kwargs["x_min"]
    assert g3d.X.max == kwargs["x_max"]
    assert g3d.X.res == kwargs["gridres"]
    assert isinstance(g3d.Y, GridAxis)
    assert g3d.Y.min == kwargs["y_min"]
    assert g3d.Y.max == kwargs["y_max"]
    assert g3d.Y.res == kwargs["gridres"]
    assert isinstance(g3d.Z, GridAxis)
    assert g3d.Z.min == kwargs["z_min"]
    assert g3d.Z.max == kwargs["z_max"]
    assert g3d.Z.res == kwargs["gridres"]

    # no results yet
    assert g3d.result is None

    # grid
    assert isinstance(g3d.grid, dict)
    assert isinstance(g3d.grid["X"], np.ndarray)
    assert isinstance(g3d.grid["Y"], np.ndarray)
    assert isinstance(g3d.grid["Z"], np.ndarray)
    assert g3d.grid["X"].ndim == 1
    assert g3d.grid["Y"].ndim == 1
    assert g3d.grid["Z"].ndim == 1

    # mesh
    assert isinstance(g3d.mesh, dict)
    assert g3d.mesh["X"].ndim == 3
    assert g3d.mesh["Y"].ndim == 3
    assert g3d.mesh["Z"].ndim == 3

    # mesh matches grid
    assert np.all(g3d.grid["X"] == g3d.mesh["X"][0, :, 0])
    assert np.all(g3d.grid["Y"] == g3d.mesh["Y"][:, 0, 0])
    assert np.all(g3d.grid["Z"] == g3d.mesh["Z"][0, 0, :])

    # normalized_grid
    assert isinstance(g3d.normalized_grid, dict)
    assert g3d.normalized_grid["X"].ndim == 1
    assert g3d.normalized_grid["Y"].ndim == 1
    assert g3d.normalized_grid["Z"].ndim == 1


def test_grid3d_is_abstract():
    """Grid3D cannot be instantiated directly"""
    with pytest.raises(TypeError):
        Grid3D(
            GridAxis(Axis.X, 0, 1, 0.1),
            GridAxis(Axis.Y, 0, 1, 0.1),
            GridAxis(Axis.Z, 0, 1, 0.1),
        )


def test_grid3d_get_axis():
    g3d = RegularGrid3D(0, 1, 0, 1, 0, 1, 0.1)
    assert g3d.get_axis("X") == g3d.X
    assert g3d.get_axis(Axis.Y) == g3d.Y
    assert g3d.get_axis("Z") == g3d.Z
    with pytest.raises(ValueError):
        g3d.get_axis("W")


def test_grid3d_repr():
    g3d = RegularGrid3D(0, 10, 0, 5, 0, 8, 0.1)
    r = repr(g3d)
    assert "RegularGrid3D" in r
    assert "X=" in r
