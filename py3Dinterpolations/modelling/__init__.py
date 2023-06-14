"""modelling classes and functions"""

from ..core.griddata import GridData
from ..core import create_regulargrid3d_from_griddata

from ..modelling.modeler import Modeler3D
from ..modelling.preprocessing import Preprocessing


def interpolate(
    griddata: GridData,
    model_type: str,
    model_name: str,
    model_params: dict,
    grid_resolution: float,
    preprocess_kwags: dict = None,      
    predict_kwags: dict = None, 
    return_components: bool = False,
):
    
    # retrive associated grid
    grid3d = create_regulargrid3d_from_griddata(griddata, grid_resolution)

    # preprocess griddata if needed
    if preprocess_kwags is not None:
        # preprocess
        griddata = Preprocessing(griddata, **preprocess_kwags).preprocess()

    # init model
    model = Modeler3D(
        griddata=griddata,
        grid3d=grid3d,
        model_type=model_type,
        model_name=model_name,
        model_params=model_params,
    )

    predictions = model.predict(**predict_kwags)

    if return_components:
        return predictions, model, grid3d, griddata
    else:
        return predictions