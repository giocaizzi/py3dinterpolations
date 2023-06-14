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
