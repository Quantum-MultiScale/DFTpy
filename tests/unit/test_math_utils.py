import pytest
from dftpy.math_utils import *

@pytest.mark.parametrize(
    "ecut, expected",
    [
        (1, 2.221441469079183),
        (2, 1.5707963267948966),
    ]
)
def test_calc_rho(ecut, expected):
    assert ecut2spacing(ecut) == pytest.approx(expected)
