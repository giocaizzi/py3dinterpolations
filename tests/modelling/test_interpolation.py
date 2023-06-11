"""test interpolation module"""

import pytest
from unittest.mock import patch

from py3Dinterpolations.core.griddata import GridData

from py3Dinterpolations.modelling.interpolation import Interpolator3D


# test init of Interpolator3D
# mocking calls to Modeler3D
def test_Interpolator3D_init(test_data):
    """test that the correct model is initialised"""

    gd = GridData(test_data)

    with patch(
        "py3Dinterpolations.modelling.modeler.Modeler3D.__init__", return_value=None
    ) as mock_method:
        i3d = Interpolator3D(
            griddata=gd,
            model_type="statistical",
            model_name="ordinary_kriging",
            model_params={
                "variogram_model": "linear",
                "nlags": 6,
                "weight": True,
            },
            grid_resolution=1,
        )
        # fitted once
        mock_method.assert_called_once()
        # with the correct arguments
        # does not consider ordering of kwargs
        mock_method.assert_called_with(
            griddata=gd,
            grid3d=i3d.grid3d,
            model_type="statistical",
            model_name="ordinary_kriging",
            model_params={
                "variogram_model": "linear",
                "nlags": 6,
                "weight": True,
            },
        )
