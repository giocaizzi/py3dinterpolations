"""models for 3D interpolation"""

# EXTERNAL
from pykrige.ok3d import OrdinaryKriging3D

# INTERNAL
from .idw import IDW


SUPPORTED_MODELS = {
    "statistical": {
        "ordinary_kriging": OrdinaryKriging3D,  # pykrige
    },
    "deteministic": {
        "idw": IDW,
    },
}

def get_model_type(inner_key: str):
    """get model type"""
    for outer_key in SUPPORTED_MODELS:
        if inner_key in SUPPORTED_MODELS[outer_key]:
            return outer_key
        else:
            raise ValueError(f"model {inner_key} not supported")