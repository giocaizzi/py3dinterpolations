"""models supported by py3Dinterpolations"""

import numpy as np
from pykrige.ok3d import OrdinaryKriging3D


SUPPORTED_MODELS = {
    "statistical": {
        "ordinary_kriging": OrdinaryKriging3D,  # pykrige
    },
    "deteministic": {},
}


class InverseDistanceWeighting:
    def __init__(self):
        pass
