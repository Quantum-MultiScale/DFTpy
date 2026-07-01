"""Gaussian valence HF + center forces under MPI decomposition."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose

pytest.importorskip("mpi4py")

from dftpy.density.gaussian_valence import (
    gaussian_valence_center_forces,
    gaussian_valence_density,
)
from dftpy.formats.vasp import read_POSCAR
from dftpy.functional import Functional, TotalFunctional
from dftpy.functional.martyna_tuckerman import MartynaTuckerman
from dftpy.functional.pseudo import LocalPseudo
from dftpy.grid import DirectGrid
from dftpy.ions import Ions
from dftpy.mpi import MP

DATA = Path(__file__).resolve().parents[2] / "examples" / "DATA"

# Small Mg dimer in the Mg8 box: 6 FD energies vs 24 for the full cluster.
GRID_NR = (12, 12, 12)
FD_ATOL = 5e-5
FD_EPS = 3.0e-4
GAUSSIAN_SIGMA = 1.35


def _mg_dimer_ions(poscar: Path) -> Ions:
    ref = read_POSCAR(poscar, names=["Mg"])
    ions = Ions(
        symbols=["Mg", "Mg"],
        positions=ref.get_positions()[:2],
        cell=ref.cell,
    )
    ions.set_charges([2.0, 2.0])
    return ions


def _build_evaluator(
    ions: Ions,
    grid: DirectGrid,
    pp_file: Path,
    mt: MartynaTuckerman,
) -> TotalFunctional:
    pseudo = LocalPseudo(
        grid=grid, ions=ions, PP_list={"Mg": pp_file}, PME=True, mt=mt
    )
    return TotalFunctional(
        KE=Functional(type="KEDF", name="TFvW", y=1.0),
        XC=Functional(type="XC", name="LDA"),
        HARTREE=Functional(type="HARTREE", mt=mt),
        PSEUDO=pseudo,
    )


def _energy_adiabatic(
    evaluator: TotalFunctional,
    grid: DirectGrid,
    ions: Ions,
    *,
    sigma: float,
    ne,
) -> float:
    evaluator.PSEUDO.restart(ions=ions, grid=grid)
    rho = gaussian_valence_density(
        grid, ions, sigma=sigma, total_valence_electrons=ne
    )
    return evaluator.Energy(rho)


def _finite_difference_forces(
    evaluator: TotalFunctional,
    grid: DirectGrid,
    ions: Ions,
    *,
    sigma: float,
    ne,
    eps: float,
) -> np.ndarray:
    nat = ions.nat
    f_fd = np.zeros((nat, 3), dtype=np.float64)
    pos0 = ions.get_positions().copy()
    for ia in range(nat):
        for j in range(3):
            pos = pos0.copy()
            pos[ia, j] += eps
            ip = ions.copy()
            ip.set_positions(pos)
            e_p = _energy_adiabatic(
                evaluator, grid, ip, sigma=sigma, ne=ne
            )

            pos = pos0.copy()
            pos[ia, j] -= eps
            im = ions.copy()
            im.set_positions(pos)
            e_m = _energy_adiabatic(
                evaluator, grid, im, sigma=sigma, ne=ne
            )
            f_fd[ia, j] = -(e_p - e_m) / (2.0 * eps)
    return f_fd


@pytest.mark.mpi
def test_gaussian_valence_hf_plus_center_forces_mpi():
    """HF + Pulay center forces match adiabatic FD on a decomposed grid."""

    mp = MP(parallel=True)
    if mp.size < 2:
        pytest.skip("need at least 2 MPI ranks")

    poscar = DATA / "Mg8.vasp"
    pp_file = DATA / "mg.lda.recpot"
    if not poscar.is_file() or not pp_file.is_file():
        pytest.skip("missing Mg8 or mg.lda.recpot in examples/DATA")

    ions = _mg_dimer_ions(poscar)
    grid = DirectGrid(lattice=ions.cell, nr=list(GRID_NR), mp=mp)
    mt = MartynaTuckerman(grid)
    evaluator = _build_evaluator(ions, grid, pp_file, mt)
    ne = evaluator.PSEUDO.ions.get_ncharges()

    rho = gaussian_valence_density(
        grid, ions, sigma=GAUSSIAN_SIGMA, total_valence_electrons=ne
    )
    pot = evaluator.get_energy_potential(rho, calcType={"V"}).potential
    f_hf = np.asarray(evaluator.get_forces(rho, ions=ions), dtype=np.float64)
    f_ctr = gaussian_valence_center_forces(
        grid, ions, GAUSSIAN_SIGMA, pot, ne
    )
    f_tot = f_hf + f_ctr
    f_fd = _finite_difference_forces(
        evaluator,
        grid,
        ions,
        sigma=GAUSSIAN_SIGMA,
        ne=ne,
        eps=FD_EPS,
    )

    if mp.is_root:
        assert_allclose(f_tot, f_fd, atol=FD_ATOL, rtol=0.0)
    mp.comm.Barrier()
