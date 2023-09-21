"""models for 3D interpolation"""

import numpy as np
from typing import Union

# EXTERNAL
from pykrige.ok3d import OrdinaryKriging3D
from sklearn.gaussian_process import GaussianProcessClassifier

# INTERNAL
from .idw import IDW


SUPPORTED_MODELS = {
        "ordinary_kriging": OrdinaryKriging3D,  # pykrige
        "idw": IDW,
        "gaussian_process_classifier": GaussianProcessClassifier,
}



class ModelWrapper:
    """model wrappper

    This class is designed to allow wrapping of different models
    from different libraries, both external and internal to py3Dinterpolations.

    Defines a fit and a predict method, to be called
    when interpolating.

    Args:
        model_name (str): model name
        X (np.ndarray): X data
        Y (np.ndarray): Y data
        \\*\\*kwargs: kwargs for model

    Attributes:
        model_name (str): model name
        model (object): model object

    """
    model = None

    def __init__(self, model_name: str,X:np.ndarray,Y:np.ndarray, **kwargs):
        self._model_name = model_name
        self._X = X
        self._Y = Y

        # when itiatin the model, some are already fitted (pykrige) others not (sklearn)
        self.model = SUPPORTED_MODELS[self.model_name](**kwargs)

        if self.model_name == "gaussian_process_classifier":
            self.fit(X=self.X, y=self.Y)
    
    @property
    def model_name(self):
        return self._model_name
    
    @property
    def X(self):
        return self._X
    
    @property
    def Y(self):
        return self._Y

    def fit(self, *args, **kwargs):
        # pykrige is already fitted
        # deterministic models are not
        self.model.fit(*args, **kwargs)

    def predict(self, *args, **kwargs) -> Union[tuple, np.ndarray]:
        """predict method

        Execute predictions for the model.

        Args:
            \\*args: args for model
            \\*\\*kwargs: kwargs for model

        Returns:
            Union[tuple, np.ndarray]: only predictions 
                or prediction and variance grids
        """
        if self.model_name == "ordinary_kriging":
            # returns both interpolated and variance grids
            # quickfix for weird behacviour
            # unpack first three args as x, y, z
            # returns predictions and variance grids
            x, y, z, *args = args
            return self.model.execute(
                *args, style="grid", xpoints=x, ypoints=y, zpoints=z, **kwargs
            )
        elif self.model_name == "idw":
            # idw does not return variance, only a single grid
            return self.model.compute(*args, **kwargs), np.ndarray([])
        elif self.model_name == "gaussian_process_classifier":
            # skleran model
            return self.model.predict(*args, **kwargs)
        else:
            # if gets here, model not supported
            raise ValueError(f"model {self._model_name} not supported")
