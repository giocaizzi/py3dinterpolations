"""test plotting module"""

import pytest
from unittest.mock import patch

from matplotlib.figure import Figure

from py3dinterpolations.core.griddata import GridData
from py3dinterpolations.modelling.preprocessor import Preprocessor
from py3dinterpolations.plotting.downsampling import plot_downsampling


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
