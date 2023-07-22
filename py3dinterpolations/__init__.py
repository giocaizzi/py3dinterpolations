"""quick 3D interpolation with python"""

# Version
__version__ = "0.4.0"

# Credits
__author__ = "Giorgio Caizzi"
__copyright__ = "Giorgio Caizzi, 2023"
__license__ = "MIT"
__maintainer__ = __author__
__email__ = "giocaizzi@gmail.com"

from .core.griddata import GridData
from .modelling.interpolate import interpolate
from .plotting.plotting import plot_3d_model, plot_2d_model

__all__ = ["__version__", "GridData", "interpolate", "plot_3d_model", "plot_2d_model"]
