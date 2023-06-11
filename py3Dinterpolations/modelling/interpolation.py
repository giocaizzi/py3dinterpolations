"""Modeler3D class"""

from ..core.griddata import GridData
from ..core import create_regulargrid3d_from_griddata
from .preprocessing import Preprocessing
from .modeler import Modeler3D

from copy import deepcopy


class Interpolator3D:
    """3D interpolator

    Supports the following interpolation models:

        - Statistical
            - Ordinary Kriging (pykrige.ok3d.OrdinaryKriging3D): model = `ordinary_kriging`::

                    model_params ={
                            "variogram_model": "spherical",
                            "nlags": 15,
                            "weight": True,
                            "exact_values": False,
                    }


    Args:
        griddata (GridData): GridData istance
        model_name (str): model name
        model_params (dict): model parameters
        grid_resolution (float): grid resolution for interpolation
        preprocess_kwags (dict): preprocessing parameters

    Attributes:
        griddata (GridData): GridData istance
        model_name (str): model name
        grid_resolution (float): grid resolution for interpolation
        grid3d (Grid3D): Grid3D istance
        results (Model3DResults): Model3DResults istance
    """

    griddata = None
    model_type = None
    model_name = None
    model_params = None
    grid_resolution = None
    grid3d = None
    results = None

    def __init__(
        self,
        griddata: GridData,
        model_type: str,
        model_name: str,
        model_params: dict,
        grid_resolution: float,
        preprocess_kwags: dict = None,
    ):
        """initialize Interpolator3D object"""

        # save original grid3d
        self.grid3d = create_regulargrid3d_from_griddata(griddata, grid_resolution)

        # preprocess griddata if needed
        if preprocess_kwags is not None:
            # preprocess
            griddata = Preprocessing(griddata, **preprocess_kwags).preprocess()

        # save griddata
        self.griddata = griddata

        # init model
        self.model = Modeler3D(
            griddata=griddata,
            grid3d=self.grid3d,
            model_type=model_type,
            model_name=model_name,
            model_params=model_params,
        )

    def interpolate(self, **kwargs) -> None:
        """execute interpolation

        Sets the results attribute within the model attribute

        Args:
            **kwargs: keyword arguments to be passed to the model predict method

        """
        # predict
        self.model.predict(
            **kwargs,
        )
