"""test RegularGrid3D and IrregularGrid3D"""

import pytest
import numpy as np

from py3dinterpolations.core.grid3d import (
    RegularGrid3D,
    IrregularGrid3D,
    Grid3D,
    GridAxis,
    create_grid,
)
from py3dinterpolations.core.griddata import GridData


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
    assert rg3d.result is None
    assert isinstance(rg3d.grid, dict)
    assert isinstance(rg3d.mesh, dict)
    assert isinstance(rg3d.normalized_grid, dict)


def test_regulargrid3d_prediction_points():
    rg3d = RegularGrid3D(**REGULARGRID3D_PARAMETERS)
    points = rg3d.prediction_points()
    assert points.ndim == 2
    assert points.shape[1] == 3
    expected_n = len(rg3d.grid["X"]) * len(rg3d.grid["Y"]) * len(rg3d.grid["Z"])
    assert points.shape[0] == expected_n


def test_irregulargrid3d():
    ig3d = IrregularGrid3D(
        x_min=0.0, x_max=1.0, x_res=0.1,
        y_min=0.0, y_max=1.0, y_res=0.2,
        z_min=0.0, z_max=1.0, z_res=0.25,
    )
    assert isinstance(ig3d, Grid3D)
    assert ig3d.X.res == 0.1
    assert ig3d.Y.res == 0.2
    assert ig3d.Z.res == 0.25
    points = ig3d.prediction_points()
    assert points.ndim == 2
    assert points.shape[1] == 3


def test_irregulargrid3d_with_hull(test_data):
    gd = GridData(test_data)
    hull = gd.hull
    specs = gd.specs
    ig3d = IrregularGrid3D(
        x_min=specs.xmin, x_max=specs.xmax, x_res=5.0,
        y_min=specs.ymin, y_max=specs.ymax, y_res=5.0,
        z_min=specs.zmin, z_max=specs.zmax, z_res=5.0,
        hull=hull,
    )
    points_with_hull = ig3d.prediction_points()
    ig3d_no_hull = IrregularGrid3D(
        x_min=specs.xmin, x_max=specs.xmax, x_res=5.0,
        y_min=specs.ymin, y_max=specs.ymax, y_res=5.0,
        z_min=specs.zmin, z_max=specs.zmax, z_res=5.0,
    )
    points_no_hull = ig3d_no_hull.prediction_points()
    # Hull filtering should reduce or equal number of points
    assert len(points_with_hull) <= len(points_no_hull)


def test_create_grid_float(test_data):
    gd = GridData(test_data)
    grid = create_grid(gd, 5.0)
    assert isinstance(grid, RegularGrid3D)


def test_create_grid_dict(test_data):
    gd = GridData(test_data)
    grid = create_grid(gd, {"X": 5.0, "Y": 5.0, "Z": 2.0})
    assert isinstance(grid, IrregularGrid3D)
    assert grid.X.res == 5.0
    assert grid.Z.res == 2.0


def test_create_grid_invalid(test_data):
    gd = GridData(test_data)
    with pytest.raises(TypeError):
        create_grid(gd, "invalid")
