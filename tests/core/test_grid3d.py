"""test grid3d objects"""

import pytest

import numpy as np

# parent class
from py3dinterpolations.core.grid3d import Grid3D, GridAxis

# tested only at constant resoultion

GRID3D_PARAMETERS = [
    {
        "x_min": 0.0,
        "x_max": 10.0,
        "x_res": 0.1,
        "y_min": 0.0,
        "y_max": 5.0,
        "y_res": 0.1,
        "z_min": 0.0,
        "z_max": 8.0,
        "z_res": 0.1,
    },
]


# test the base class


@pytest.mark.parametrize("kwargs", GRID3D_PARAMETERS)
def test_grid3d_init_ands_proprieties(kwargs):
    """test init of grid3d object and its properties"""
    g3d = Grid3D(**kwargs)

    # assert that each X,Y,Z axis is a GridAxis object
    assert isinstance(g3d.X, GridAxis)
    assert g3d.X.min == kwargs["x_min"]
    assert g3d.X.max == kwargs["x_max"]
    assert g3d.X.res == kwargs["x_res"]
    assert isinstance(g3d.Y, GridAxis)
    assert g3d.Y.min == kwargs["y_min"]
    assert g3d.Y.max == kwargs["y_max"]
    assert g3d.Y.res == kwargs["y_res"]
    assert isinstance(g3d.Z, GridAxis)
    assert g3d.Z.min == kwargs["z_min"]
    assert g3d.Z.max == kwargs["z_max"]
    assert g3d.Z.res == kwargs["z_res"]

    # assert has not results
    assert g3d.results == {}

    # assert grid is a dictionary
    assert isinstance(g3d.grid, dict)
    # assert grid has X,Y,Z keys and are numpy arrays
    assert isinstance(g3d.grid["X"], np.ndarray)
    assert isinstance(g3d.grid["Y"], np.ndarray)
    assert isinstance(g3d.grid["Z"], np.ndarray)
    # 1d arrays
    assert g3d.grid["X"].ndim == 1
    assert g3d.grid["Y"].ndim == 1
    assert g3d.grid["Z"].ndim == 1

    # assert mesh is a dictionary
    assert isinstance(g3d.mesh, dict)
    # assert mesh has X,Y,Z keys and are numpy arrays
    assert isinstance(g3d.mesh["X"], np.ndarray)
    assert isinstance(g3d.mesh["Y"], np.ndarray)
    assert isinstance(g3d.mesh["Z"], np.ndarray)
    # 3d arrays
    assert g3d.mesh["X"].ndim == 3
    assert g3d.mesh["Y"].ndim == 3
    assert g3d.mesh["Z"].ndim == 3

    # assert that the mesh is the same as the grid
    # numpy conventions with xyz
    # In the 3-D case with inputs of length M, N and P, 
    #  outputs are of shape (N, M, P) for ‘xy’ indexing
    assert np.all(g3d.grid["X"] == g3d.mesh["X"][0, :, 0]) 
    assert np.all(g3d.grid["Y"] == g3d.mesh["Y"][:, 0, 0])
    assert np.all(g3d.grid["Z"] == g3d.mesh["Z"][0, 0, :])

    # normalized_grid is a dictionary
    assert isinstance(g3d.normalized_grid, dict)
    # normalized_grid has X,Y,Z keys and are numpy arrays
    assert isinstance(g3d.normalized_grid["X"], np.ndarray)
    assert isinstance(g3d.normalized_grid["Y"], np.ndarray)
    assert isinstance(g3d.normalized_grid["Z"], np.ndarray)
    # 1d arrays
    assert g3d.normalized_grid["X"].ndim == 1
    assert g3d.normalized_grid["Y"].ndim == 1
    assert g3d.normalized_grid["Z"].ndim == 1
