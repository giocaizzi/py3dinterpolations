"""Inverse Distance Weighting (IDW) model"""

import numpy as np
import math

from .deterministic import DeterministicModel


class IDW(DeterministicModel):
    """Simple IDW Model object

    IDW: Inverse Distance Weigthting

    Offers tweaking of the power parameter of the IDW model.

    Could be really slow for large datasets, due to the use of loops.

    TODO: Remove loopings use a vectorized approach

    Args:
        x (np.ndarray): x coordinates of the points where the model will be evaluated
        y (np.ndarray): y coordinates of the points where the model will be evaluated
        z (np.ndarray): z coordinates of the points where the model will be evaluated
        values (np.ndarray): values of the points where the model will be evaluated
        power (float): power of the IDW model

    Attributes:
        power (float): power of the IDW model
        distance_matrix (np.ndarray): distance matrix between
            observations and data points
    """

    def __init__(
        self,
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray,
        values: np.ndarray,
        power: float = 1,
    ):
        # initialize parent class
        super().__init__(x=x, y=y, z=z, values=values)

        # set power
        self.power = power

    def _compute_point(
        self,
        x: float,
        y: float,
        z: float,
        power: float,
        threshold: float = 0.0000000001,
    ) -> float:
        """find value at a point

        Loops through all the data points to estimate the value at a point
        based on the Inverse Distance Weighting (IDW) method.
        Might be really slow for large datasets.

        TODO: Implement parametrization
            This implementation however offers easy advanced parametrization with:
            - n_points: number of points to use for the IDW estimation
            - max_distance: maximum distance to use for the IDW estimation
        """
        nominator = 0
        denominator = 0

        # distance in 3d
        for i in range(0, len(self.values)):
            dist = math.sqrt(
                (x - self.x[i]) * (x - self.x[i])
                + (y - self.y[i]) * (y - self.y[i])
                + (z - self.z[i]) * (z - self.z[i])
            )

            # If the point is really close to one of the data points,
            # return the data point value to avoid singularities
            # EXACT interpolations
            if dist < threshold:
                return self.values[i]

            nominator = nominator + (self.values[i] / pow(dist, power))
            denominator = denominator + (1 / pow(dist, power))

        # Return NaN if the denominator is zero
        if denominator > 0:
            value = nominator / denominator
        else:
            value = np.nan
        return value

    def compute(
        self, gridx: np.ndarray, gridy: np.ndarray, gridz: np.ndarray
    ) -> np.ndarray:
        meshx, meshy, meshz = np.meshgrid(gridx, gridy, gridz, indexing="ij")

        # Create a new mesh to store the coordinates
        new_mesh = np.zeros_like(meshx)

        # Iterate over the indices of the arrays
        for i in range(meshx.shape[0]):
            for j in range(meshx.shape[1]):
                for k in range(meshx.shape[2]):
                    # Store the coordinates in the new mesh
                    new_mesh[i, j, k] = self._compute_point(
                        x=meshx[i, j, k],
                        y=meshy[i, j, k],
                        z=meshz[i, j, k],
                        power=self.power,
                    )

        # quickfix for grid in shape outputting a gird in (zyx)
        # pykrige outputs zyx
        new_mesh = np.einsum("xyz->zyx", new_mesh)

        return new_mesh
