"""models for 3D interpolation"""

from typing import Union

# EXTERNAL
from pykrige.ok3d import OrdinaryKriging3D

# INTERNAL
from .idw import IDW


SUPPORTED_MODELS = {
    "statistical": {
        "ordinary_kriging": OrdinaryKriging3D,  # pykrige
    },
    "deterministic": {
        "idw": IDW,
    },
}


def get_model_type(inner_key: str):
    """get model type

    Raises:
        ValueError: if model not supported
    """
    for outer_key in SUPPORTED_MODELS:
        if inner_key in SUPPORTED_MODELS[outer_key]:
            return outer_key
    # if gets here, model not supported
    raise ValueError(f"model {inner_key} not supported")


class ModelsWrapper:
    """model wrappper

    This class is designed to allow wrapping of different models
    from different libraries, both external and internal to py3Dinterpolations.

    Defines a fit and a predict method, to be called
    when interpolating.

    Args:
        model_name (str): model name
        *args: args for model
        **kwargs: kwargs for model

    Attributes:
        model_type (str): model type
        model_name (str): model name

    """

    _model_name: Union[str, None] = None
    _model_type: Union[str, None] = None
    model = None

    @property
    def model_type(self):
        return self._model_type

    @property
    def model_name(self):
        return self._model_name

    @model_type.setter
    def model_type(self, model_type: str):
        self._model_type = model_type

    @model_name.setter
    def model_name(self, model_name: str):
        self._model_name = model_name
        self._model_type = get_model_type(self._model_name)

    def __init__(self, model_name: str, *args, **kwargs):
        self.model_name = model_name
        self.model = SUPPORTED_MODELS[self.model_type][self.model_name](*args, **kwargs)

    def fit(self):
        # pykrige is already fitted
        # deterministic models are not
        pass

    def predict(self, *args, **kwargs):
        """predict method

        Execute predictions for the model.

        Args:
            *args: args for model
            \*\*kwargs: kwargs for model
        """
        if self.model_name == "ordinary_kriging":
            return self.model.execute(style="grid", *args, **kwargs)
        if self.model_name == "idw":
            return self.model.compute(*args, **kwargs)
