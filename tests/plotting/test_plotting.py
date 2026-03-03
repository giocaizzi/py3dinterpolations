"""test plotting module"""

from unittest.mock import patch

import plotly.graph_objs as go
import pytest
from matplotlib.figure import Figure

from py3dinterpolations.core.griddata import GridData
from py3dinterpolations.modelling.interpolate import interpolate
from py3dinterpolations.modelling.preprocessor import Preprocessor
from py3dinterpolations.plotting.downsampling import plot_downsampling
from py3dinterpolations.plotting.plot_2d import plot_2d_model
from py3dinterpolations.plotting.plot_3d import plot_3d_model

REGULAR_RES = 5
IRREGULAR_RES = {"X": 5.0, "Y": 5.0, "Z": 2.0}
IDW_PARAMS = {"power": 2}


def _make_modeler(test_data, grid_resolution, preprocessing=None):
    """Create a modeler with IDW for fast tests."""
    gd = GridData(test_data)
    return interpolate(
        griddata=gd,
        model_type="idw",
        grid_resolution=grid_resolution,
        model_params=IDW_PARAMS,
        preprocessing=preprocessing,
    )


# --- plot_downsampling ---


def test_plot_downsampling(test_data):
    gd = GridData(test_data)
    preprocessed_gd = Preprocessor(
        griddata=gd,
        normalize_xyz=True,
        standardize_v=True,
        downsampling_res=1,
    ).preprocess()

    with patch(
        "py3dinterpolations.plotting.downsampling.reverse_preprocessing",
        return_value=GridData(test_data.sample(n=50, random_state=42)),
    ):
        fig = plot_downsampling(gd, preprocessed_gd)

    assert isinstance(fig, Figure)
    n = len(gd.data.index.get_level_values("ID").unique())
    visible_axes = [ax for ax in fig.axes if ax.get_visible()]
    assert len(visible_axes) == n


# --- plot_2d_model ---


class TestPlot2DModel:
    """Tests for plot_2d_model with regular and irregular grids."""

    def test_regular_grid_no_preprocessing(self, test_data):
        modeler = _make_modeler(test_data, REGULAR_RES)
        fig = plot_2d_model(modeler)
        assert isinstance(fig, Figure)

    def test_irregular_grid_no_preprocessing(self, test_data):
        modeler = _make_modeler(test_data, IRREGULAR_RES)
        fig = plot_2d_model(modeler)
        assert isinstance(fig, Figure)

    def test_regular_grid_with_preprocessing(self, test_data):
        modeler = _make_modeler(
            test_data, REGULAR_RES, preprocessing={"downsampling_res": 10}
        )
        fig = plot_2d_model(modeler)
        assert isinstance(fig, Figure)

    def test_irregular_grid_with_preprocessing(self, test_data):
        modeler = _make_modeler(
            test_data, IRREGULAR_RES, preprocessing={"downsampling_res": 10}
        )
        fig = plot_2d_model(modeler)
        assert isinstance(fig, Figure)

    def test_with_points(self, test_data):
        modeler = _make_modeler(test_data, REGULAR_RES)
        fig = plot_2d_model(modeler, plot_points=True)
        assert isinstance(fig, Figure)

    def test_with_points_annotated(self, test_data):
        modeler = _make_modeler(test_data, REGULAR_RES)
        fig = plot_2d_model(modeler, plot_points=True, annotate_points=True)
        assert isinstance(fig, Figure)

    @pytest.mark.parametrize("axis", ["X", "Y", "Z"])
    def test_all_axes(self, test_data, axis):
        modeler = _make_modeler(test_data, REGULAR_RES)
        fig = plot_2d_model(modeler, axis=axis)
        assert isinstance(fig, Figure)

    def test_invalid_axis_raises(self, test_data):
        modeler = _make_modeler(test_data, REGULAR_RES)
        with pytest.raises(KeyError):
            plot_2d_model(modeler, axis="W")


# --- plot_3d_model ---


class TestPlot3DModel:
    """Tests for plot_3d_model with regular and irregular grids."""

    def test_regular_grid_no_preprocessing(self, test_data):
        modeler = _make_modeler(test_data, REGULAR_RES)
        fig = plot_3d_model(modeler)
        assert isinstance(fig, go.Figure)

    def test_irregular_grid_no_preprocessing(self, test_data):
        modeler = _make_modeler(test_data, IRREGULAR_RES)
        fig = plot_3d_model(modeler)
        assert isinstance(fig, go.Figure)

    def test_regular_grid_with_preprocessing(self, test_data):
        modeler = _make_modeler(
            test_data, REGULAR_RES, preprocessing={"downsampling_res": 10}
        )
        fig = plot_3d_model(modeler)
        assert isinstance(fig, go.Figure)

    def test_irregular_grid_with_preprocessing(self, test_data):
        modeler = _make_modeler(
            test_data, IRREGULAR_RES, preprocessing={"downsampling_res": 10}
        )
        fig = plot_3d_model(modeler)
        assert isinstance(fig, go.Figure)

    def test_with_points_no_preprocessing(self, test_data):
        modeler = _make_modeler(test_data, REGULAR_RES)
        fig = plot_3d_model(modeler, plot_points=True)
        assert isinstance(fig, go.Figure)
        # Volume + Scatter3d
        assert len(fig.data) == 2

    def test_with_points_with_preprocessing(self, test_data):
        modeler = _make_modeler(
            test_data, REGULAR_RES, preprocessing={"downsampling_res": 10}
        )
        fig = plot_3d_model(modeler, plot_points=True)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 2
