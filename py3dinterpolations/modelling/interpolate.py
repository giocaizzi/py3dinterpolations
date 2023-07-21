"""interpolator"""

from typing import Union, Tuple
import numpy as np

from ..core.griddata import GridData
from ..core import create_regulargrid3d_from_griddata
from .modeler import Modeler
from .preprocessing import Preprocessing
from .estimator import Estimator


def interpolate(
    griddata: GridData,
    model_name: str,
    grid_resolution: float,
    model_params: dict = {},
    model_params_grid: dict = {},
    preprocess_kwags: dict = {},
    predict_kwags: dict = {},
    return_model: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, Modeler]]:
    """interpolate griddata

    Interpolate griddata using a Modeler instance that wraps all supported
    models. The moedel is selected with the argument `model_name`.

    If the `model_params` is passed, then the model is initialized with those
    parameters. Otherwise, to make a search for the best parameters, use the
    `model_params_grid` argument.

    The 3D grid is retrived from the training data.
    At the moment features only a regular grid.

    If requested, the griddata is preprocessed using the Preprocessing class.


    Args:
        griddata (GridData): griddata to interpolate
        model_name (str): model name
        grid_resolution (float): grid resolution
        model_params (dict, optional): model parameters. Defaults to {}.
        preprocess_kwags (dict, optional): preprocessing parameters. Defaults to {}.
        predict_kwags (dict, optional): prediction parameters. Defaults to {}.
        return_model (bool, optional): return model. Defaults to False.

    Returns:
        Union[np.ndarray, Tuple[np.ndarray, Modeler]]:
            interpolated griddata, optionally with model

    """
    if model_params == {} and model_params_grid == {}:
        raise ValueError("either model_params or model_params_grid must be passed")

    if model_params != {} and model_params_grid != {}:
        raise ValueError("model_params and model_params_grid cannot be passed together")

    # retrive associated grid
    grid3d = create_regulargrid3d_from_griddata(griddata, grid_resolution)

    # preprocess griddata if needed
    if preprocess_kwags != {}:
        # preprocessor
        preprocessor = Preprocessing(griddata, **preprocess_kwags)
        # get new griddata
        griddata = preprocessor.preprocess()

    if model_params == {}:
        # implemented only for ordinary_kriging
        if model_name != "ordinary_kriging":
            raise NotImplementedError(
                "Parameter search is only supported for ordinary_kriging"
            )

        # estimate
        est = Estimator(griddata, model_params_grid)

        # TODO: krige wrapper, method key param
        #   that is not needed for the estimator
        #   find a flexible way to handle this, probably a good way is to use directly
        #   the sci-kit wrapper Krige()
        model_params = est.best_params
        model_params.pop("method")

    # init Modeler
    model = Modeler(
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
