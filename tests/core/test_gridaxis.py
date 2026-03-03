"""test GridAxis frozen dataclass"""

import pytest
import numpy as np

from py3dinterpolations.core.grid3d import GridAxis
from py3dinterpolations.core.types import Axis

AXIS_PARAMETERS = [
    (Axis.X, 0, 1, 10),
    (Axis.Y, 0.5, 12.7, 0.33),
]


@pytest.mark.parametrize("axis_name, min_val, max_val, res", AXIS_PARAMETERS)
def test_gridaxis(axis_name, min_val, max_val, res):
    axis = GridAxis(
        axis_name,
        min_val,
        max_val,
        res,
    )

    assert axis.name == axis_name
    assert axis.min == min_val
    assert axis.max == max_val
    assert axis.res == res
    assert isinstance(axis.grid, np.ndarray)


def test_gridaxis_is_frozen():
    axis = GridAxis(Axis.X, 0, 1, 0.1)
    with pytest.raises(AttributeError):
        axis.min = 5  # type: ignore[misc]


def test_gridaxis_repr():
    axis = GridAxis(Axis.X, 0.0, 10.0, 0.5)
    assert "X" in repr(axis)
    assert "0.0" in repr(axis)
