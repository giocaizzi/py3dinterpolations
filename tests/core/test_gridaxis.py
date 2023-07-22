"""test GridAxis class"""

import pytest
import numpy as np

from py3dinterpolations.core.grid3d import GridAxis

AXIS_PARAMETERS = [
    ("X", 0, 1, 10),
    ("Y", 0.5, 12.7, 0.33),
]


@pytest.mark.parametrize("axis_name, min, max, res", AXIS_PARAMETERS)
def test_GridAxis(axis_name, min, max, res):
    """test gridaxis initialization"""

    axis = GridAxis(
        axis_name,
        min,
        max,
        res,
    )

    assert axis.name == axis_name
    assert axis.min == min
    assert axis.max == max
    assert axis.res == res
    assert isinstance(axis.grid, np.ndarray)
