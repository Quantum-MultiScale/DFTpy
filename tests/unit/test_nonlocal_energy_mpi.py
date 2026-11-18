"""MPI tests for nonlocal/GGA KEDF and rVV10 energies on a non-uniform density.

``func(rho).energy`` must be the global energy on every rank, as it already
is for TF, vW, XC, Hartree and the local pseudopotential. A uniform density
cannot test this because the nonlocal term vanishes, so the density here is
modulated.
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

pytest.importorskip("mpi4py")

from dftpy.field import DirectField
from dftpy.functional import Functional, TotalFunctional
from dftpy.functional.xc import RVV10NL
from dftpy.grid import DirectGrid
from dftpy.mpi import MP

GRID_NR = (16, 16, 16)
LATTICE = np.eye(3) * 7.6
ENERGY_RTOL = 1e-10

KEDF_CASES = {
    "WT": dict(name="WT"),
    "SM": dict(name="SM"),
    "MGP": dict(name="MGP"),
    "LWT": dict(name="LWT"),
    "HC": dict(name="HC"),
    "LKT": dict(name="LKT"),
    "TFvW": dict(name="TFvW", y=1.0),
}


def _build_density(mp: MP) -> DirectField:
    grid = DirectGrid(lattice=LATTICE, nr=list(GRID_NR), mp=mp)
    s = grid.s
    rho = DirectField(grid=grid)
    rho[:] = 0.03 * (
        1.0
        + 0.3 * np.cos(2 * np.pi * s[0])
        + 0.2 * np.cos(2 * np.pi * (s[1] + s[2]))
        + 0.1 * np.sin(4 * np.pi * s[2])
    )
    return rho


def _energies(mp: MP) -> dict:
    rho = _build_density(mp)
    out = {}
    for key, kwargs in KEDF_CASES.items():
        out[key] = float(Functional(type="KEDF", **kwargs)(rho, calcType={"E", "V"}).energy)
    out["rVV10-NL"] = float(RVV10NL()(rho, calcType={"E", "V"}).energy)
    total = TotalFunctional(
        KEDF=Functional(type="KEDF", name="WT"),
        XC=Functional(type="XC", name="LDA"),
        HARTREE=Functional(type="HARTREE"),
    )
    out["TOTAL(WT+LDA+HARTREE)"] = float(total(rho, calcType={"E"}).energy)
    return out


@pytest.mark.mpi
def test_nonlocal_kedf_energy_mpi():
    """Every rank reports the serial energy for nonlocal and GGA KEDFs."""

    mp = MP(parallel=True)
    if mp.size < 2:
        pytest.skip("need at least 2 MPI ranks")

    ref = mp.comm.bcast(_energies(MP(parallel=False)) if mp.is_root else None, root=0)
    result = _energies(mp)

    # Checked on every rank: a rank-local sum differs from rank to rank.
    for key, expected in ref.items():
        assert_allclose(result[key], expected, rtol=ENERGY_RTOL, err_msg=f"{key} energy")
