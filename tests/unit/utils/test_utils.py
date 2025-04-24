import pytest

from dftpy.utils.utils import *


@pytest.mark.parametrize(
    "rho, n, expected",
    [
        (0, 1, 0),
        (1, 2, 2),
        (1, 3, 3),
    ],
)
def test_calc_rho(rho, n, expected):
    assert calc_rho(rho, N=n) == expected


def test_calc_drho():
    assert calc_drho(1, 3) == pytest.approx(6.0)
