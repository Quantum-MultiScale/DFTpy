"""MPI test: every stress term equals the serial tensor on every rank."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose

pytest.importorskip("mpi4py")

from dftpy.field import DirectField
from dftpy.formats.vasp import read_POSCAR
from dftpy.functional import Functional, TotalFunctional
from dftpy.functional.pseudo import LocalPseudo
from dftpy.grid import DirectGrid
from dftpy.mpi import MP

DATA = Path(__file__).resolve().parents[2] / "examples" / "DATA"
PP_AL = DATA / "al.lda.recpot"
POSCAR_AL = DATA / "fcc.vasp"
GRID_NR = (16, 16, 16)
STRESS_ATOL = 1e-10

KEDF_CASES = {
    "TFvW": dict(name="TFvW", y=1.0),
    "WT": dict(name="WT"),
    "GGA-LKT": dict(name="GGA", k_str="LKT"),
}


def _stresses(mp: MP, pme: bool) -> dict:
    ions = read_POSCAR(POSCAR_AL, names=["Al"])
    grid = DirectGrid(lattice=ions.cell, nr=list(GRID_NR), mp=mp)
    pseudo = LocalPseudo(grid=grid, ions=ions, PP_list={"Al": PP_AL}, PME=pme)
    s = grid.s
    rho = DirectField(grid=grid)
    rho[:] = pseudo.ions.get_ncharges() / ions.cell.volume * (
        1.0
        + 0.3 * np.cos(2 * np.pi * s[0])
        + 0.2 * np.cos(2 * np.pi * (s[1] + s[2]))
        + 0.1 * np.sin(4 * np.pi * s[2])
    )
    out = {}
    for label, kwargs in KEDF_CASES.items():
        total = TotalFunctional(
            KEDF=Functional(type="KEDF", **kwargs),
            XC=Functional(type="XC", name="LDA"),
            HARTREE=Functional(type="HARTREE"),
            PSEUDO=pseudo,
        )
        for key, value in total.get_stress(rho, split=True).items():
            out[f"{label}:{key}"] = np.array(value, dtype=float)
    return out


@pytest.mark.mpi
@pytest.mark.parametrize("pme", [True, False])
def test_stress_terms_match_serial_mpi(pme):
    if not POSCAR_AL.is_file() or not PP_AL.is_file():
        pytest.skip("missing fcc.vasp or al.lda.recpot in examples/DATA")
    mp = MP(parallel=True)
    if mp.size < 2:
        pytest.skip("need at least 2 MPI ranks")

    ref = mp.comm.bcast(_stresses(MP(parallel=False), pme) if mp.is_root else None, root=0)
    result = _stresses(mp, pme)

    assert set(result) == set(ref)
    # Checked on every rank: the tensor must be global, not a per-rank share.
    for key, expected in ref.items():
        assert_allclose(result[key], expected, rtol=0, atol=STRESS_ATOL, err_msg=key)
