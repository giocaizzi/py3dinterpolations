"""test plotting module"""

import pytest
import matplotlib
from unittest.mock import patch

from py3Dinterpolations.core.griddata import GridData
from py3Dinterpolations.modelling.preprocessing import Preprocessing
from py3Dinterpolations.plotting.plotting import plot_downsampling


# test plot downsampling method
# using pytest test_data fixture
# patching the call to reverse_preprocessing
# to make the test independent of the method
def test_plot_downsampling(test_data):
    gd = GridData(test_data)
    preprocessed_gd = Preprocessing(
        griddata=gd,
        normalize_xyz=True,
        standardize_v=True,
        downsampling_res=1,
    ).preprocess()

    # patch the reverse_preprocessing function
    with patch(
        "py3Dinterpolations.plotting.plotting.reverse_preprocessing",
        return_value=GridData(test_data.sample(n=500, random_state=42)),
    ):
        fig = plot_downsampling(gd, preprocessed_gd)

    # assert is figure
    assert isinstance(fig, matplotlib.figure.Figure)

    # assert figure has n visible axes
    n = len(gd.data.index.get_level_values("ID").unique())
    visible_axes = [ax for ax in fig.axes if ax.get_visible()]
    assert len(visible_axes) == n

    # # assert each axes has 2 lines
