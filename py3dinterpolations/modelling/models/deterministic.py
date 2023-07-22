"""base classes for all models"""

import numpy as np

# TODO: make this a custom sklearn class


class DeterministicModel:
    """base class for all deterministic models

    This class requires as initialization parameters:
        - x, y, z: the coordinates of the points where the model will be evaluated
        - values: the values of the points where the model will be evaluated

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
        # deterministic model saves coordinates and values as attributes
        self.x = x
        self.y = y
        self.z = z
        self.values = values

    def compute(self):
        """compute the model

        This method should be implemented in the child classes
        """
        pass
