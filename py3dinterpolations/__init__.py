"""quick 3D interpolation with python"""

__version__ = "1.0.0"

__author__ = "Giorgio Caizzi"
__copyright__ = "Giorgio Caizzi, 2023"
__license__ = "MIT"
__maintainer__ = __author__
__email__ = "giocaizzi@gmail.com"

from .core.griddata import GridData
from .core.types import ModelType
from .modelling.interpolate import interpolate
from .plotting import plot_2d_model, plot_3d_model

__all__ = [
    "GridData",
    "ModelType",
    "__version__",
    "interpolate",
    "plot_2d_model",
    "plot_3d_model",
]
