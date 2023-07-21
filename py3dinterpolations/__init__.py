"""quick 3D interpolation with python"""

# Version
__version__ = "0.2.4"

# Credits
__author__ = "Giorgio Caizzi"
__copyright__ = "Giorgio Caizzi, 2023"
__license__ = "MIT"
__maintainer__ = __author__
__email__ = "giocaizzi@gmail.com"

from .core.griddata import GridData
from .modelling import interpolate

__all__ = ["__version__", "GridData", "interpolate"]