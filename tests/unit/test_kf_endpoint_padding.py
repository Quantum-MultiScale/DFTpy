import numpy as np
import pytest

from dftpy.field import DirectField
from dftpy.functional import Functional
from dftpy.functional.kedf.hc import get_kflist
from dftpy.grid import DirectGrid

# A nearly uniform density with a small cosine along x. A whole plane of grid
# points then sits exactly at the minimum (and maximum) density, so the local
# kF there equals the first (and, for linear spacing, the last) kF checkpoint.
N, L = 16, 10.0


def cosine_density():
    grid = DirectGrid(lattice=np.eye(3) * L, nr=[N, N, N])
    x = np.arange(N) / N
    profile = 0.03 * (1.0 + 0.01 * np.cos(2 * np.pi * x))
    rho = DirectField(grid=grid, griddata_3d=np.broadcast_to(profile[:, None, None], (N, N, N)).copy())
    return rho, int(np.argmin(profile))


def test_hc_kflist_brackets_extreme_kf():
    rho, _ = cosine_density()
    kf = 2.0 * np.cbrt(3.0 * np.pi ** 2 * rho)
    kf0 = 2.0 * np.cbrt(3.0 * np.pi ** 2 * rho.amean())
    kflists = get_kflist(kf, kf0, delta=0.01, kfmin=kf.amin(), kfmax=kf.amax())
    # The padding must place every grid kF strictly inside the checkpoint range.
    assert kflists[0] < kf.amin()
    assert kflists[-1] > kf.amax()


def test_lmgp_potential_continuous_at_minimum_density():
    rho, imin = cosine_density()
    out = Functional(type="KEDF", name="LMGP").compute(rho, calcType={"V"})
    v = np.asarray(out.potential)[:, 0, 0]
    # Grid points at the minimum density must receive the interpolated nonlocal
    # potential, so the potential there matches its neighbours to within a small
    # fraction of its variation along x instead of jumping.
    neighbours = 0.5 * (v[imin - 1] + v[imin + 1])
    assert abs(v[imin] - neighbours) < 0.1 * np.ptp(np.delete(v, imin))
