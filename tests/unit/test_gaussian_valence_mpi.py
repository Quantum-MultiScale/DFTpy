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
from dftpy.mpi import MP

DATA = Path(__file__).resolve().parents[2] / "examples" / "DATA"


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

    ions = read_POSCAR(poscar, names=["Mg"])
    grid = DirectGrid(lattice=ions.cell, nr=[18, 18, 16], mp=mp)
    mt = MartynaTuckerman(grid)
    pseudo = LocalPseudo(
        grid=grid, ions=ions, PP_list={"Mg": pp_file}, PME=True, mt=mt
    )
    evaluator = TotalFunctional(
        KE=Functional(type="KEDF", name="LKT"),
        XC=Functional(type="XC", name="LDA"),
        HARTREE=Functional(type="HARTREE", mt=mt),
        PSEUDO=pseudo,
    )
    ne = float(pseudo.ions.get_ncharges())
    sigma = 1.35
    eps = 3.0e-4

    rho = gaussian_valence_density(grid, ions, sigma=sigma, total_valence_electrons=ne)
    pot = evaluator.get_energy_potential(rho, calcType={"V"}).potential
    f_hf = np.asarray(evaluator.get_forces(rho, ions=ions), dtype=np.float64)
    f_ctr = gaussian_valence_center_forces(grid, ions, sigma, pot, ne)
    f_tot = f_hf + f_ctr

    pos0 = ions.get_positions().copy()
    f_fd = np.zeros((ions.nat, 3), dtype=np.float64)
    for ia in range(ions.nat):
        for j in range(3):
            pos = pos0.copy()
            pos[ia, j] += eps
            ip = ions.copy()
            ip.set_positions(pos)
            rho_p = gaussian_valence_density(
                grid, ip, sigma=sigma, total_valence_electrons=ne
            )
            e_p = _evaluator_at_ions(ip, grid, pp_file, mt).Energy(rho_p)

            pos = pos0.copy()
            pos[ia, j] -= eps
            im = ions.copy()
            im.set_positions(pos)
            rho_m = gaussian_valence_density(
                grid, im, sigma=sigma, total_valence_electrons=ne
            )
            e_m = _evaluator_at_ions(im, grid, pp_file, mt).Energy(rho_m)
            f_fd[ia, j] = -(e_p - e_m) / (2.0 * eps)

    if mp.is_root:
        assert_allclose(f_tot, f_fd, atol=5e-5, rtol=0.0)
    mp.comm.Barrier()


def _evaluator_at_ions(ions, grid, pp_file, mt):
    pseudo = LocalPseudo(
        grid=grid, ions=ions, PP_list={"Mg": pp_file}, PME=True, mt=mt
    )
    return TotalFunctional(
        KE=Functional(type="KEDF", name="LKT"),
        XC=Functional(type="XC", name="LDA"),
        HARTREE=Functional(type="HARTREE", mt=mt),
        PSEUDO=pseudo,
    )
