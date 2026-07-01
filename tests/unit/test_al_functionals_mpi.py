"""MPI tests for KEDF, XC, HARTREE, and PSEUDO on FCC Al.

Each functional is evaluated on a decomposed grid (2+ MPI ranks) using the
same call patterns as the rest of the codebase (``func(rho).energy``,
``func.forces(rho)``) and compared to a serial reference built on rank 0.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose

pytest.importorskip("mpi4py")

from dftpy.field import DirectField
from dftpy.formats.vasp import read_POSCAR
from dftpy.functional import Functional
from dftpy.functional.martyna_tuckerman import MartynaTuckerman
from dftpy.functional.pseudo import LocalPseudo
from dftpy.grid import DirectGrid
from dftpy.mpi import MP

DATA = Path(__file__).resolve().parents[2] / "examples" / "DATA"
PP_AL = DATA / "al.lda.recpot"
POSCAR_AL = DATA / "fcc.vasp"
GRID_NR = (16, 16, 16)
ENERGY_RTOL = 1e-10
ENERGY_ATOL = 1e-10
POT_RTOL = 1e-10
POT_ATOL = 1e-10
FORCE_RTOL = 1e-8
FORCE_ATOL = 1e-8


def _require_al_data() -> None:
    if not POSCAR_AL.is_file() or not PP_AL.is_file():
        pytest.skip("missing fcc.vasp or al.lda.recpot in examples/DATA")


def _uniform_valence_density(grid: DirectGrid, ions, pseudo: LocalPseudo) -> DirectField:
    rho = DirectField(grid=grid)
    ne = pseudo.ions.get_ncharges()
    rho[:] = ne / ions.cell.volume
    return rho


def _build_system(mp: MP):
    ions = read_POSCAR(POSCAR_AL, names=["Al"])
    grid = DirectGrid(lattice=ions.cell, nr=list(GRID_NR), mp=mp)
    pseudo = LocalPseudo(grid=grid, ions=ions, PP_list={"Al": PP_AL}, PME=True)
    rho = _uniform_valence_density(grid, ions, pseudo)
    mt = MartynaTuckerman(grid)
    return ions, grid, rho, pseudo, mt


def _functional_cases(ions, grid, rho, pseudo, mt):
    return {
        "KEDF": Functional(type="KEDF", name="TFvW", y=1.0),
        "XC": Functional(type="XC", name="LDA"),
        "HARTREE": Functional(type="HARTREE"),
        "HARTREE_MT": Functional(type="HARTREE", mt=mt),
        "PSEUDO": pseudo,
        "PSEUDO_MT": LocalPseudo(
            grid=grid, ions=ions, PP_list={"Al": PP_AL}, PME=True, mt=mt
        ),
    }


def _eval_functional(func, rho):
    """Energy and potential via ``func(rho)``; forces via ``func.forces(rho)``."""

    func_out = func(rho)
    energy = func_out.energy
    potential = func_out.potential
    if rho.mp.is_mpi:
        potential = potential.gather()
    potential = np.asarray(potential)
    forces = None
    if hasattr(func, "forces"):
        forces = np.asarray(func.forces(rho), dtype=np.float64)
    return {"energy": energy, "potential": potential, "forces": forces}


def _serial_reference() -> dict:
    mp = MP(parallel=False)
    ions, grid, rho, pseudo, mt = _build_system(mp)
    ref = {}
    for name, func in _functional_cases(ions, grid, rho, pseudo, mt).items():
        ref[name] = _eval_functional(func, rho)
    return ref


@pytest.mark.mpi
def test_al_kedf_xc_hartree_pseudo_mpi():
    """KEDF, XC, Hartree, and local PP match serial on a decomposed Al grid."""

    _require_al_data()
    mp = MP(parallel=True)
    if mp.size < 2:
        pytest.skip("need at least 2 MPI ranks")

    ref = mp.comm.bcast(_serial_reference() if mp.is_root else None, root=0)
    ions, grid, rho, pseudo, mt = _build_system(mp)

    for name, func in _functional_cases(ions, grid, rho, pseudo, mt).items():
        result = _eval_functional(func, rho)

        if mp.is_root:
            assert_allclose(
                result["energy"],
                ref[name]["energy"],
                rtol=ENERGY_RTOL,
                atol=ENERGY_ATOL,
                err_msg=f"{name} energy",
            )
            assert_allclose(
                result["potential"],
                ref[name]["potential"],
                rtol=POT_RTOL,
                atol=POT_ATOL,
                err_msg=f"{name} potential",
            )
            if result["forces"] is not None:
                assert_allclose(
                    result["forces"],
                    ref[name]["forces"],
                    rtol=FORCE_RTOL,
                    atol=FORCE_ATOL,
                    err_msg=f"{name} forces",
                )

    mp.comm.Barrier()
