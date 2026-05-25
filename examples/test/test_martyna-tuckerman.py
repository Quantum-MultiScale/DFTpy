#!/usr/bin/env python3
"""Martyna–Tuckerman regressions (QE-style ``W(G)``, orthorhombic cells, Mg cluster forces).

Tests are **module-level** ``test_*`` functions only (no ``unittest.TestCase`` subclass), so
collection lists exactly what this file defines.

Run from repo root, e.g. ``pytest examples/test/test_martyna-tuckerman.py`` or
``python examples/test/test_martyna-tuckerman.py``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose

from common import dftpy_data_path

from dftpy.config.config import DefaultOption
from dftpy.density.gaussian_valence import (
    gaussian_valence_center_forces,
    gaussian_valence_density,
)
from dftpy.field import DirectField
from dftpy.formats.vasp import read_POSCAR
from dftpy.functional import Functional, TotalFunctional
from dftpy.functional.hartree import Hartree
from dftpy.functional.martyna_tuckerman import MartynaTuckerman, _ws_dist_corner_brute_chunks, ws_dist_corner
from dftpy.functional.pseudo import LocalPseudo
from dftpy.grid import DirectGrid
from dftpy.ions import Ions


def isolated_gaussian_hartree_self_energy(sigma: float) -> float:
    """Hartree self-energy (Ha) for ∫ρ=1 Gaussian width σ in free space: 1/(2√π σ)."""
    return float(1.0 / (2.0 * np.sqrt(np.pi) * sigma))


def _cubic_grid(L: float = 5.0, n: int = 24) -> DirectGrid:
    lattice = np.eye(3) * L
    return DirectGrid(lattice=lattice, nr=[n, n, n], origin=np.array([L / 2, L / 2, L / 2]))


def test_default_config_contains_martyna_tuckerman_section():
    conf = DefaultOption()
    mt = conf["MARTYNA_TUCKERMAN"]
    assert mt["enable"] is False
    assert mt["alpha"] is None


def test_direct_grid_r_mic_cell_center_near_zero():
    g = _cubic_grid()
    mid = (g.nr[0] // 2, g.nr[1] // 2, g.nr[2] // 2)
    assert g.r_mic[mid] < 1e-10
    assert g.rmic[mid] == pytest.approx(g.r_mic[mid])


def test_martyna_build_wg_finite_at_gamma():
    g = _cubic_grid()
    mt = MartynaTuckerman(g)
    wg = mt.wg
    assert np.isfinite(wg[0, 0, 0])
    recip = g.get_reciprocal()
    assert np.all(np.isfinite(wg[recip.mask_serial]))


def test_hartree_mt_changes_gaussian_energy():
    g = _cubic_grid()
    sigma = 0.85
    rr = g.rr
    rho = DirectField(
        grid=g, griddata_3d=(2.0 * np.pi * sigma**2) ** (-1.5) * np.exp(-rr / (2.0 * sigma**2))
    )
    e_pw = Hartree()(rho, calcType={"E"}).energy
    e_mt = Hartree(mt=MartynaTuckerman(g))(rho, calcType={"E"}).energy
    assert abs(e_pw - e_mt) > 1e-10


def test_hartree_mt_pw_difference_matches_wg_contribution():
    L, n, sigma = 6.5, 32, 1.05
    lattice = np.eye(3) * L
    g = DirectGrid(lattice=lattice, nr=[n, n, n], origin=np.array([L / 2, L / 2, L / 2]))
    rho = DirectField(grid=g, griddata_3d=(2.0 * np.pi * sigma**2) ** (-1.5) * np.exp(-g.r_mic**2 / (2.0 * sigma**2)))
    rho /= rho.integral()
    mt = MartynaTuckerman(g)
    recip = g.get_reciprocal()
    kern_mt = mt.coulomb_kernel(recip)
    kern_pw = recip.invgg * 4.0 * np.pi
    rho_g = rho.fft()
    v_wg_r = (rho_g * (kern_mt - kern_pw)).ifft(force_real=True)
    delta_wg = 0.5 * np.einsum("ijk, ijk->", v_wg_r, rho, optimize=True) * g.dV
    e_pw = Hartree()(rho, calcType={"E"}).energy
    e_mt = Hartree(mt=mt)(rho, calcType={"E"}).energy
    assert_allclose(e_mt - e_pw, delta_wg, rtol=1e-10, atol=1e-10)


def test_mt_hartree_gaussian_near_continuum_qe_ws():
    sigma = 0.6
    L = 25.0
    n = 96
    lattice = np.eye(3) * L
    g = DirectGrid(lattice=lattice, nr=[n, n, n], origin=np.array([L / 2, L / 2, L / 2]))
    rho = DirectField(grid=g, griddata_3d=np.exp(-g.r_mic**2 / (2.0 * sigma**2)))
    rho /= rho.integral()
    e_ref = isolated_gaussian_hartree_self_energy(sigma)
    e_mt = Hartree(mt=MartynaTuckerman(g))(rho, calcType={"E"}).energy
    assert_allclose(e_mt, e_ref, rtol=0.115, atol=5e-2)


def test_ws_dist_corner_orthogonal_equals_brute_cubic_small():
    rng = np.random.default_rng(0)
    lattice = np.diag((4.0 + rng.random(3))).astype(np.float64)
    n = 9
    g = DirectGrid(lattice=lattice, nr=[n, n, n])
    pts = np.moveaxis(g.r, 0, -1).reshape(-1, 3)[::47]
    lat = np.asarray(g.lattice, dtype=np.float64)
    brute = _ws_dist_corner_brute_chunks(pts, lat, nmax=4)
    r_xyz = pts.T.reshape(3, -1)
    corner = ws_dist_corner(r_xyz, lat, nmax=4)
    assert_allclose(corner.reshape(-1), brute, rtol=0.0, atol=5e-10)


def test_mt_local_correction_recipro_shape_and_finite():
    g = _cubic_grid()
    ions = Ions(symbols=["H", "H"], positions=np.array([[1.3, 0.1, 0.2], [2.9, 0.4, 0.55]]), cell=g.lattice)
    ions.set_charges([1.0, -1.0])
    corr = MartynaTuckerman(g).local_pp_correction_reciprocal(ions)
    recip = g.get_reciprocal()
    assert corr.shape == recip.q.shape
    assert np.all(np.isfinite(corr[recip.mask_serial]))


def test_mt_ion_ewald_energy_sign_symmetry_same_geometry():
    g = _cubic_grid()
    mt = MartynaTuckerman(g)
    p = np.array([[1.8, 0.95, 0.6], [2.95, 0.15, 0.9]])
    ions1 = Ions(symbols=["H", "H"], positions=p.copy(), cell=g.lattice)
    ions1.set_charges([1.0, -1.0])
    ions2 = Ions(symbols=["H", "H"], positions=p.copy(), cell=g.lattice)
    ions2.set_charges([-1.0, 1.0])
    assert_allclose(mt.ion_ewald_energy(ions1), mt.ion_ewald_energy(ions2), rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("use_pme", [True, False])
def test_mg_cluster_totalfunctional_mt_localpseudo_energy(use_pme: bool):
    """Mg8 + ``mg.lda.recpot`` + TF/LDA + Hartree(MT): total energy finite with Gaussian ``rho``.

    Parametrized over ``LocalPseudo(..., PME=…)`` (Bspline spreading vs direct structure factors).

    Valence density is a sum of Gaussians (width ``sigma``) centered on each Mg with
    minimum-image separation, then scaled so :math:`\\int \\rho = N_e`. This keeps charge
    localized away from cell faces, which matches how MT screening is meant to be used.

    Forces: ``get_forces`` (HF at fixed grid ρ) plus ``gaussian_valence_center_forces`` vs
    adiabatic FD (ρ rebuilt per displacement). Optimized ρ checks: ``verify_mt_al4_cluster.py``.
    """

    poscar = Path(dftpy_data_path) / "Mg8.vasp"
    pp_file = Path(dftpy_data_path) / "mg.lda.recpot"
    assert poscar.is_file(), f"missing {poscar}"
    assert pp_file.is_file(), f"missing {pp_file}"

    ions = read_POSCAR(poscar, names=["Mg"])
    nat = ions.nat
    assert nat == 8

    grid = DirectGrid(lattice=ions.cell, nr=[18, 18, 16])
    mt = MartynaTuckerman(grid)

    pseudo_ref = LocalPseudo(grid=grid, ions=ions, PP_list={"Mg": pp_file}, PME=use_pme, mt=mt)
    ne = float(pseudo_ref.ions.get_ncharges())

    sigma = 1.35

    def rho_for(ii: Ions) -> DirectField:
        return gaussian_valence_density(grid, ii, sigma=sigma, total_valence_electrons=ne)

    def build_evaluator(ii: Ions) -> TotalFunctional:
        pseudo = LocalPseudo(grid=grid, ions=ii, PP_list={"Mg": pp_file}, PME=use_pme, mt=mt)
        ke = Functional(type="KEDF", name="TF")
        xc = Functional(type="XC", name="LDA")
        har = Functional(type="HARTREE", mt=mt)
        return TotalFunctional(KE=ke, XC=xc, HARTREE=har, PSEUDO=pseudo)

    evaluator = build_evaluator(ions)
    rho = rho_for(ions)
    assert rho.integral() == pytest.approx(ne, rel=0.0, abs=1e-9)

    energy = evaluator.Energy(rho)
    assert np.isfinite(energy)
    assert abs(energy) < 1.0e6

    pot = evaluator.get_energy_potential(rho, calcType={"V"}).potential
    f_hf = evaluator.get_forces(rho, ions=ions)
    f_ctr = gaussian_valence_center_forces(grid, ions, sigma, pot, ne)
    f_tot = f_hf + f_ctr

    eps = 3.0e-4
    pos0 = ions.get_positions().copy()
    f_fd = np.zeros((nat, 3), dtype=np.float64)
    for ia in range(nat):
        for j in range(3):
            pos = pos0.copy()
            pos[ia, j] += eps
            ip = ions.copy()
            ip.set_positions(pos)
            e_p = build_evaluator(ip).Energy(rho_for(ip))
            pos = pos0.copy()
            pos[ia, j] -= eps
            im = ions.copy()
            im.set_positions(pos)
            e_m = build_evaluator(im).Energy(rho_for(im))
            f_fd[ia, j] = -(e_p - e_m) / (2.0 * eps)
    assert_allclose(f_tot, f_fd, atol=5e-5, rtol=0.0)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, *sys.argv[1:]]))
