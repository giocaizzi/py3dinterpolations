"""test plotting module"""

import pytest
from unittest.mock import patch

from matplotlib.figure import Figure

from py3dinterpolations.core.griddata import GridData
from py3dinterpolations.modelling.preprocessing import Preprocessing
from py3dinterpolations.plotting.plotting import plot_downsampling


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
    # returning  a GriddData object
    # created with a sample of the test_data
    with patch(
        "py3dinterpolations.plotting.plotting.reverse_preprocessing",
        return_value=GridData(test_data.sample(n=50, random_state=42)),
    ):
        fig = plot_downsampling(gd, preprocessed_gd)

    # assert is figure
    assert isinstance(fig, Figure)

    # assert figure has n visible axes
    n = len(gd.data.index.get_level_values("ID").unique())
    visible_axes = [ax for ax in fig.axes if ax.get_visible()]
    assert len(visible_axes) == n

    # # TODO assert each axes has 2 lines
