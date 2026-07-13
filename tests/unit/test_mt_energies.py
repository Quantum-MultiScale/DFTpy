"""Light integration tests for Martyna–Tuckerman energies.

Tests Hartree, local pseudopotential, and Ewald ion–ion energies
with and without PME (particle mesh Ewald) to ensure consistent results.
Uses a small Al4 cluster in a cubic cell with a Gaussian valence density.
"""

import numpy as np
import pytest
from pathlib import Path

from dftpy.field import DirectField
from dftpy.functional.hartree import Hartree
from dftpy.functional.martyna_tuckerman import MartynaTuckerman
from dftpy.functional.pseudo import LocalPseudo
from dftpy.ewald import ewald
from dftpy.grid import DirectGrid
from dftpy.ions import Ions
from dftpy.math_utils import ecut2nr


DATA_DIR = Path(__file__).resolve().parents[2] / "examples" / "DATA"
PP_FILE = DATA_DIR / "al.lda.recpot"


def _skip_if_no_pp():
    if not PP_FILE.is_file():
        pytest.skip(f"missing {PP_FILE}")


@pytest.fixture()
def al4_setup():
    """Al4 FCC cluster in a 12 bohr cubic cell with Gaussian valence density."""
    _skip_if_no_pp()

    L = 12.0
    lattice = np.eye(3) * L
    positions = np.array([
        [0.0, 0.0, 0.0],
        [2.0, 2.0, 0.0],
        [2.0, 0.0, 2.0],
        [0.0, 2.0, 2.0],
    ]) + L / 2 - 1.0  # centered in cell

    ions = Ions(symbols=["Al"] * 4, positions=positions, cell=lattice)
    nr = ecut2nr(ecut=600 / 27.2114, lattice=lattice)
    grid = DirectGrid(lattice=lattice, nr=nr, full=True)
    mt = MartynaTuckerman(grid, alpha=0.9)

    # Gaussian valence density: sum of Gaussians centered on atoms, normalized to N_e
    sigma = 1.2
    rho_arr = np.zeros(tuple(grid.nr), dtype=np.float64)
    r = grid.r
    for pos in positions:
        dr2 = sum((r[i] - pos[i]) ** 2 for i in range(3))
        rho_arr += np.exp(-dr2 / (2 * sigma ** 2))
    rho = DirectField(grid=grid, griddata_3d=rho_arr)
    # N_e will be set after PP tells us the valence
    return ions, grid, mt, rho, sigma


class TestHartreeEnergy:
    """Hartree energy must be the same with/without MT up to the wg correction."""

    def test_mt_hartree_is_finite(self, al4_setup):
        ions, grid, mt, rho, _ = al4_setup
        ne = 12.0
        rho *= ne / rho.integral()
        e = Hartree(mt=mt)(rho, calcType={"E"}).energy
        assert np.isfinite(e)
        assert e > 0

    def test_hartree_mt_minus_plain_equals_wg_contribution(self, al4_setup):
        ions, grid, mt, rho, _ = al4_setup
        ne = 12.0
        rho *= ne / rho.integral()
        e_mt = Hartree(mt=mt)(rho, calcType={"E"}).energy
        e_pw = Hartree()(rho, calcType={"E"}).energy
        recip = grid.get_reciprocal()
        rho_g = rho.fft()
        wg_contrib = 0.5 * np.real(
            (np.conj(rho_g) * rho_g * mt.wg).integral()
        )
        np.testing.assert_allclose(e_mt - e_pw, wg_contrib, rtol=1e-10)


class TestPseudoEnergy:
    """Local pseudopotential energy with MT: PME and non-PME must agree."""

    def test_pseudo_energy_pme_vs_direct(self, al4_setup):
        ions, grid, mt, rho, _ = al4_setup
        pseudo_pme = LocalPseudo(
            grid=grid, ions=ions, PP_list={"Al": str(PP_FILE)}, PME=True, mt=mt
        )
        ne = float(ions.get_ncharges())
        rho *= ne / rho.integral()

        pseudo_direct = LocalPseudo(
            grid=grid, ions=ions, PP_list={"Al": str(PP_FILE)}, PME=False, mt=mt
        )

        e_pme = pseudo_pme(rho, calcType={"E"}).energy
        e_direct = pseudo_direct(rho, calcType={"E"}).energy
        np.testing.assert_allclose(e_pme, e_direct, rtol=1e-6)

    def test_pseudo_energy_mt_vs_plain(self, al4_setup):
        ions, grid, mt, rho, _ = al4_setup
        ions.set_charges([3.0] * 4)
        ne = float(ions.get_ncharges())
        rho *= ne / rho.integral()

        e_mt = LocalPseudo(
            grid=grid, ions=ions, PP_list={"Al": str(PP_FILE)}, PME=False, mt=mt
        )(rho, calcType={"E"}).energy
        e_plain = LocalPseudo(
            grid=grid, ions=ions, PP_list={"Al": str(PP_FILE)}, PME=False, mt=None
        )(rho, calcType={"E"}).energy

        # They should differ (MT adds -wg*S_tot correction)
        assert abs(e_mt - e_plain) > 1e-6

    def test_pseudo_energy_mt_pme_vs_mt_direct(self, al4_setup):
        """PME and direct must agree even when MT correction is present."""
        ions, grid, mt, rho, _ = al4_setup
        ions.set_charges([3.0] * 4)
        ne = float(ions.get_ncharges())
        rho *= ne / rho.integral()

        e_pme = LocalPseudo(
            grid=grid, ions=ions, PP_list={"Al": str(PP_FILE)}, PME=True, mt=mt
        )(rho, calcType={"E"}).energy
        e_direct = LocalPseudo(
            grid=grid, ions=ions, PP_list={"Al": str(PP_FILE)}, PME=False, mt=mt
        )(rho, calcType={"E"}).energy
        np.testing.assert_allclose(e_pme, e_direct, rtol=1e-6)


class TestEwaldIonIonEnergy:
    """Ewald ion–ion energy: PME vs direct, with and without MT."""

    def test_ewald_pme_vs_direct_no_mt(self, al4_setup):
        ions, grid, mt, _, _ = al4_setup
        ions.set_charges([3.0] * 4)
        e_pme = ewald(ions=ions, grid=grid, PME=True, mt=None)
        e_direct = ewald(ions=ions, grid=grid, PME=False, mt=None)
        np.testing.assert_allclose(e_pme.energy, e_direct.energy, rtol=1e-6)

    def test_ewald_pme_vs_direct_with_mt(self, al4_setup):
        ions, grid, mt, _, _ = al4_setup
        ions.set_charges([3.0] * 4)
        e_pme = ewald(ions=ions, grid=grid, PME=True, mt=mt)
        e_direct = ewald(ions=ions, grid=grid, PME=False, mt=mt)
        np.testing.assert_allclose(e_pme.energy, e_direct.energy, rtol=1e-6)

    def test_ewald_mt_adds_ion_ewald_energy(self, al4_setup):
        ions, grid, mt, _, _ = al4_setup
        ions.set_charges([3.0] * 4)
        ew_plain = ewald(ions=ions, grid=grid, PME=False, mt=None)
        ew_mt = ewald(ions=ions, grid=grid, PME=False, mt=mt)
        delta_corr = ew_mt.Energy_corr() - ew_plain.Energy_corr()
        np.testing.assert_allclose(
            ew_mt.energy,
            ew_plain.energy + mt.ion_ewald_energy(ions) + delta_corr,
            rtol=1e-10,
        )

    def test_ewald_mt_forces_fd(self, al4_setup):
        """MT Ewald forces vs finite difference."""
        ions, grid, mt, _, _ = al4_setup
        ions.set_charges([3.0] * 4)
        f_an = mt.ion_ewald_forces(ions)
        disp = 1e-4
        f_fd = np.zeros_like(f_an)
        for ia in range(ions.nat):
            for a in range(3):
                pos_p = ions.positions.copy()
                pos_m = ions.positions.copy()
                pos_p[ia, a] += disp
                pos_m[ia, a] -= disp
                ip = Ions(symbols=ions.symbols, positions=pos_p, cell=grid.lattice)
                ip.set_charges(ions.charges)
                im = Ions(symbols=ions.symbols, positions=pos_m, cell=grid.lattice)
                im.set_charges(ions.charges)
                f_fd[ia, a] = -(mt.ion_ewald_energy(ip) - mt.ion_ewald_energy(im)) / (2 * disp)
        np.testing.assert_allclose(f_an, f_fd, atol=1e-4)

    def test_neutral_g0_cancellation(self, al4_setup):
        """For a neutral system, the G=0 contributions of Hartree + PP + Ewald cancel."""
        ions, grid, mt, rho, _ = al4_setup
        ne = 12.0
        ions.set_charges([3.0] * 4)
        rho *= ne / rho.integral()

        wg0 = mt.wg[0, 0, 0]
        rho_g0 = float(np.real(rho.fft()[0, 0, 0]))
        stot_g0 = float(np.real(ions.total_strf(grid.get_reciprocal())[0, 0, 0]))
        vol = grid.volume

        hartree_g0 = 0.5 * wg0 * rho_g0 ** 2 / vol
        pseudo_g0 = -wg0 * rho_g0 * stot_g0 / vol
        ewald_g0 = 0.5 * wg0 * stot_g0 ** 2 / vol
        total_g0 = hartree_g0 + pseudo_g0 + ewald_g0

        # For neutral: rho_g0 == stot_g0 == N_e, so total == 0
        np.testing.assert_allclose(rho_g0, stot_g0, rtol=1e-10)
        np.testing.assert_allclose(total_g0, 0.0, atol=1e-10)
