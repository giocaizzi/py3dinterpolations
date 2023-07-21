"""modelling classes and functions"""

from typing import Union, Tuple
import numpy as np

from ..core.griddata import GridData
from ..core.grid3d import Grid3D
from ..core import create_regulargrid3d_from_griddata
from ..modelling.modeler import Modeler3D
from ..modelling.preprocessing import Preprocessing


def interpolate(
    griddata: GridData,
    model_name: str,
    model_params: dict,
    grid_resolution: float,
    preprocess_kwags: dict = {},
    predict_kwags: dict = {},
    return_model: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, Modeler3D, Grid3D, GridData]]:
    """interpolate griddata

    Interpolate griddata using a Modeler3D instance that wraps all supported
    models.
    model_name and model_params.

    The 3D grid is retrived from the training data.

    If requested, the griddata is preprocessed using the Preprocessing class.

    Args:
        griddata (GridData): griddata to interpolate
        model_name (str): model name
        model_params (dict): model parameters
        grid_resolution (float): grid resolution
        preprocess_kwags (dict, optional): preprocessing parameters. Defaults to None.
        predict_kwags (dict, optional): prediction parameters. Defaults to None.
        return_model (bool, optional): return model. Defaults to False.

    Returns:
        Union[np.ndarray, Tuple[np.ndarray, Modeler3D, Grid3D, GridData]]:
            interpolated griddata or all components

    """
    # retrive associated grid
    grid3d = create_regulargrid3d_from_griddata(griddata, grid_resolution)

    # preprocess griddata if needed
    if preprocess_kwags is not None:
        # preprocess
        griddata = Preprocessing(griddata, **preprocess_kwags).preprocess()

    # init Modeler3D
    model = Modeler3D(
        griddata=griddata,
        grid3d=grid3d,
        model_name=model_name,
        model_params=model_params,
    )

    # make predictions
    predictions = model.predict(**predict_kwags)

    # return model
    if return_model:
        return predictions, model
    else:
        return predictions
