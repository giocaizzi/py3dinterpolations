import numpy as np


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
        This method has to be overriden by the child classes.
        """
        pass
