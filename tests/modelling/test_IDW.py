"""test IDW model"""

import numpy as np
import pytest

from py3Dinterpolations.modelling.models.models import SimpleIDW


def test_simple_idw():
    # Generate x, y, and z arrays with the specified ranges

    n_points = 20
    x = np.random.uniform(0, 10, n_points)
    y = np.random.uniform(0, 20, n_points)
    z = np.random.uniform(0, 5, n_points)

    # Generate random values to interpolate
    v = np.random.rand(n_points)

    # equally spaced grid
    resoution = 0.5
    gridx = np.arange(x.min(), x.max(), resoution)
    gridy = np.arange(y.min(), y.max(), resoution)
    gridz = np.arange(z.min(), z.max(), resoution)


    # Create a SimpleIDW model and fit it to the data
    model = SimpleIDW(x,y,z,v)
    results = model.compute(gridx, gridy, gridz)

    assert isinstance(results, np.ndarray)
