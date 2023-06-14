"""models supported by py3Dinterpolations"""

import numpy as np
from pykrige.ok3d import OrdinaryKriging3D


SUPPORTED_MODELS = {
    "statistical": {
        "ordinary_kriging": OrdinaryKriging3D,  # pykrige
    },
    "deteministic": {},
}


def compute_distance_matrix(
    x0: np.ndarray,
    y0: np.ndarray,
    z0: np.ndarray,
    x1: np.ndarray,
    y1: np.ndarray,
    z1: np.ndarray,
) -> np.ndarray:
    """compute distance matrix

    Compute distance matrix from pairwise coordinates

    Args:
        x0 (np.ndarray): x coordinates of the set of data points
        y0 (np.ndarray): y coordinates of the set of data points
        z0 (np.ndarray): z coordinates of the set of data points
        x1 (np.ndarray): x coordinates of the interpolation points
        y1 (np.ndarray): y coordinates of the interpolation points
        z1 (np.ndarray): z coordinates of the interpolation points
    """
    # observations
    obs = np.vstack((x0, y0, z0)).T
    # data points
    interp = np.vstack((x1, y1, z1)).T

    # Make a distance matrix between pairwise observations
    # Note: from <http://stackoverflow.com/questions/1871536>
    d0 = np.subtract.outer(obs[:, 0], interp[:, 0])
    # data points
    d1 = np.subtract.outer(obs[:, 1], interp[:, 1])
    return np.hypot(d0, d1)


class DeterministicModel:
    """base class for all deterministic models

    This class is inspired by the pykrige OrdinaryKriging3D class.
    This class requires as initialization parameters:
        - x, y, z: the coordinates of the points where the model will be evaluated
        - values: the values of the points where the model will be evaluated
        - model_parameters: the parameters of the model

    Args:
        x (np.ndarray): x coordinates of the points where the model will be evaluated
        y (np.ndarray): y coordinates of the points where the model will be evaluated
        z (np.ndarray): z coordinates of the points where the model will be evaluated
        values (np.ndarray): values of the points where the model will be evaluated

    Attributes:
        x (np.ndarray): x coordinates of the points where the model will be evaluated
        y (np.ndarray): y coordinates of the points where the model will be evaluated
        z (np.ndarray): z coordinates of the points where the model will be evaluated
        values (np.ndarray): values of the points where the model will be evaluated
    """

    def __init__(
        self,
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray,
        values: np.ndarray,
    ):
        self.x = x
        self.y = y
        self.z = z
        self.values = values

    def compute(self):
        """predict values at the points where the model will be evaluated

        This method is overriden by the child classes.
        """
        pass


class SimpleIDW(DeterministicModel):
    """Simple IDW Model object

    Offers tweaking of the power parameter of the IDW model.

    Args:
        x (np.ndarray): x coordinates of the points where the model will be evaluated
        y (np.ndarray): y coordinates of the points where the model will be evaluated
        z (np.ndarray): z coordinates of the points where the model will be evaluated
        values (np.ndarray): values of the points where the model will be evaluated
        power (float): power of the IDW model

    Attributes:
        power (float): power of the IDW model
        distance_matrix (np.ndarray): distance matrix between observations and data points
    """

    power: float = 1
    distance_matrix: np.ndarray = None

    def __init__(
        self,
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray,
        values: np.ndarray,
        power: float = 1,
    ):
        super().__init__(x=x, y=y, z=z, values=values)

    def compute(self, gridX: np.ndarray, gridY: np.ndarray, gridZ: np.ndarray):
        """compute IDW grid

        Args:
            gridX (np.ndarray): x coordinates of the grid
            gridY (np.ndarray): y coordinates of the grid
            gridZ (np.ndarray): z coordinates of the grid
        """
        self.distance_matrix = compute_distance_matrix(
            x0=self.x,
            y0=self.y,
            z0=self.z,
            x1=gridX,
            y1=gridY,
            z1=gridZ,
        )

        weights = 1.0 / (self.distance_matrix + 1e-12) ** self.power

        # Make weights sum to one
        weights /= weights.sum(axis=0)

        # Multiply the weights for each interpolated point by all observed Z-values
        # grid1 = grid1.reshape((ny, nx))
        return np.dot(weights.T, self.values)
