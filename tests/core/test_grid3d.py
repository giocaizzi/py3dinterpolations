"""test grid3d objects"""

import pytest

import numpy as np

# parent class
from py3Dinterpolations.core.grid3d import Grid3D

# child classes
from py3Dinterpolations.core.grid3d import RegularGrid3D

# positive values
TEST_DATA_1 = {
    "xmin": 0.0,
    "xmax": 5.0,
    "ymin": 0.0,
    "ymax": 10.0,
    "zmin": 0.0,
    "zmax": 15.0,
}
# negative values
TEST_DATA_2 = {
    "xmin": -10.0,
    "xmax": 0.0,
    "ymin": -10.0,
    "ymax": 0.0,
    "zmin": -10.0,
    "zmax": 0.0,
}
# across zero
TEST_DATA_3 = {
    "xmin": -10.0,
    "xmax": 10.0,
    "ymin": -10.0,
    "ymax": 10.0,
    "zmin": -10.0,
    "zmax": 10.0,
}

GRIDRES = {"X": 0.1, "Y": 0.1, "Z": 0.1}


# test the base class


@pytest.mark.parametrize(
    "kwargs",
    [
        TEST_DATA_1,
        TEST_DATA_2,
        TEST_DATA_3,
    ],
)
def test_grid3d_init_ands_proprieties(kwargs):
    """test init of grid3d object and its properties"""
    g3d = Grid3D(**kwargs)
    # xyz min max properties
    assert g3d.xmin == kwargs["xmin"]
    assert g3d.xmax == kwargs["xmax"]
    assert g3d.ymin == kwargs["ymin"]
    assert g3d.ymax == kwargs["ymax"]
    assert g3d.zmin == kwargs["zmin"]
    assert g3d.zmax == kwargs["zmax"]

    # bbox
    assert g3d.bbox == (
        kwargs["xmin"],
        kwargs["xmax"],
        kwargs["ymin"],
        kwargs["ymax"],
        kwargs["zmin"],
        kwargs["zmax"],
    )

    # assert that grid, gridres, mesh are None or empty
    # because needs to be set when calling general
    # _set_baseclass_attributes method or individual methods

    assert g3d.grid == {}
    assert g3d.gridres == {}
    assert g3d.mesh == {}


# test grid3d.gridres setter


@pytest.mark.parametrize(
    "kwargs",
    [
        TEST_DATA_1,
        TEST_DATA_2,
        TEST_DATA_3,
    ],
)
def test_grid3d_gridres_setter(kwargs):
    """test setting the gridres dictionary"""
    g3d = Grid3D(**kwargs)
    # gridres setter
    g3d.gridres = GRIDRES
    assert g3d.gridres == GRIDRES


# test grid3d gridding


@pytest.mark.parametrize(
    "kwargs",
    [
        TEST_DATA_1,
        TEST_DATA_2,
        TEST_DATA_3,
    ],
)
def test_grid3d_set_grid(kwargs):
    """test setting the grid"""
    g3d = Grid3D(**kwargs)

    # grid3d object does not have gridres attribute set
    # tested for regular grid
    g3d.gridres = GRIDRES
    g3d._set_grid()

    # test resulting grid arrays and dimensions
    assert isinstance(g3d.grid, dict)
    assert list(g3d.grid.keys()) == ["X", "Y", "Z"]
    assert isinstance(g3d.grid["X"], np.ndarray)
    assert g3d.grid["X"].ndim == 1
    assert isinstance(g3d.grid["Y"], np.ndarray)
    assert g3d.grid["Y"].ndim == 1
    assert isinstance(g3d.grid["Z"], np.ndarray)
    assert g3d.grid["Z"].ndim == 1


# test grid3d set grid raises error if gridres is not set


@pytest.mark.parametrize(
    "kwargs",
    [
        TEST_DATA_1,
        TEST_DATA_2,
        TEST_DATA_3,
    ],
)
def test_grid3d_set_grid_raises_error(kwargs):
    """test setting the grid without gridres"""
    g3d = Grid3D(**kwargs)
    with pytest.raises(AttributeError):
        g3d._set_grid()


# test normalized grid property


@pytest.mark.parametrize(
    "kwargs",
    [
        TEST_DATA_1,
        TEST_DATA_2,
        TEST_DATA_3,
    ],
)
def test_grid3d_normalized_grid(kwargs):
    """test normalized grid property"""
    g3d = Grid3D(**kwargs)

    # grid3d object does not have gridres attribute set
    # tested for regular grid
    g3d.gridres = GRIDRES
    g3d._set_grid()

    # test resulting grid arrays and dimensions
    assert isinstance(g3d.normalized_grid, dict)
    assert list(g3d.normalized_grid.keys()) == ["X", "Y", "Z"]
    assert isinstance(g3d.normalized_grid["X"], np.ndarray)
    assert g3d.normalized_grid["X"].ndim == 1
    assert isinstance(g3d.normalized_grid["Y"], np.ndarray)
    assert g3d.normalized_grid["Y"].ndim == 1
    assert isinstance(g3d.normalized_grid["Z"], np.ndarray)
    assert g3d.normalized_grid["Z"].ndim == 1

    # test normalized grid values
    assert g3d.normalized_grid["X"].min() == 0.0
    assert g3d.normalized_grid["X"].max() == 1.0
    assert g3d.normalized_grid["Y"].min() == 0.0
    assert g3d.normalized_grid["Y"].max() == 1.0
    assert g3d.normalized_grid["Z"].min() == 0.0
    assert g3d.normalized_grid["Z"].max() == 1.0


# test normalized grid property
# raise error if grid not set


@pytest.mark.parametrize(
    "kwargs",
    [
        TEST_DATA_1,
        TEST_DATA_2,
        TEST_DATA_3,
    ],
)
def test_grid3d_normalized_grid_raises_error(kwargs):
    """test normalized grid property if grid"""
    g3d = Grid3D(**kwargs)
    with pytest.raises(AttributeError):
        g3d.normalized_grid


# TODO : test grid3d relative grid property
# TODO : test grid3d n_cells property


# test grid3d set_mesh


@pytest.mark.parametrize(
    "kwargs",
    [
        TEST_DATA_1,
        TEST_DATA_2,
        TEST_DATA_3,
    ],
)
def test_grid3d_set_mesh(kwargs):
    """test setting the mesh"""
    g3d = Grid3D(**kwargs)

    # grid3d object does not have gridres attribute set
    # tested for regular grid
    g3d.gridres = GRIDRES
    g3d._set_grid()
    g3d._set_mesh()

    # test resulting grid arrays and dimensions
    assert isinstance(g3d.mesh, dict)
    assert list(g3d.mesh.keys()) == ["X", "Y", "Z"]
    assert isinstance(g3d.mesh["X"], np.ndarray)
    assert g3d.mesh["X"].ndim == 3
    assert isinstance(g3d.mesh["Y"], np.ndarray)
    assert g3d.mesh["Y"].ndim == 3
    assert isinstance(g3d.mesh["Z"], np.ndarray)
    assert g3d.mesh["Z"].ndim == 3


# TODO: test mesh indexing
