"""test GridRes class and its properties"""

import pytest

from py3dinterpolations.core.grid3d import GridRes


@pytest.mark.parametrize(
    "resolutions,exprected_res,expected_x,expected_y,expected_z",
    [
        ({"X": 0.1, "Y": 0.1, "Z": 0.1}, 0.1, 0.1, 0.1, 0.1),
        ({"X": 0.1, "Y": 0.1, "Z": 0.2}, {"X": 0.1, "Y": 0.1, "Z": 0.2}, 0.1, 0.1, 0.2),
    ],
)
def test_gridres_init_and_properties(resolutions, exprected_res, expected_x, expected_y, expected_z):
    """test init of gridres object and its properties"""
    gridres = GridRes(resolutions)

    # calling resolution resutrns a float if all resolutions are equal
    # otherwise a dictionary
    assert gridres.resolutions == exprected_res
    # subscripting resolution will always reutrn a float
    assert gridres["X"] == expected_x
    assert gridres["Y"] == expected_y
    assert gridres["Z"] == expected_z
