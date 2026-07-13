import numpy as np
import pytest

from dftpy.config.config import DefaultOption

from dftpy.field import DirectField
from dftpy.ewald import ewald
from dftpy.functional.hartree import Hartree
from dftpy.functional.martyna_tuckerman import (
    MartynaTuckerman,
    _ws_dist_corner_brute_chunks,
    ws_dist_corner,
    ws_dist_corner_grid,
)
from dftpy.grid import DirectGrid
from dftpy.ions import Ions


def isolated_gaussian_hartree_self_energy(sigma):
    """
    Coulomb Hartree energy (atomic units, free space) for a normalized 3D Gaussian

        ρ(r) = (2π σ²)⁻³/² exp(−r² / (2σ²)),       ∫ ρ = 1

    excluding the infinite self-interaction singularity handled by cancelling
    constructions; the finite self-energy is

        E = 1 / (2 √π σ).
    """
    return 1.0 / (2.0 * np.sqrt(np.pi) * sigma)


def test_default_config_contains_martyna_tuckerman_section():
    conf = DefaultOption()
    mt = conf["MARTYNA_TUCKERMAN"]
    assert mt["enable"] is False
    assert mt["alpha"] is None


@pytest.fixture()
def cubic_grid_serial():
    L = 5.0
    n = 24
    lattice = np.eye(3) * L
    return DirectGrid(lattice=lattice, nr=[n, n, n], origin=np.array([L / 2, L / 2, L / 2]))


def test_direct_grid_r_mic_cell_center_near_zero(cubic_grid_serial):
    g = cubic_grid_serial
    mid = (g.nr[0] // 2, g.nr[1] // 2, g.nr[2] // 2)
    assert g.r_mic[mid] < 1e-10
    assert g.rmic[mid] == pytest.approx(g.r_mic[mid])


def test_martyna_build_wg_finite_at_gamma(cubic_grid_serial):
    """Appendix B: finite \(g\to 0\) screening after singularity cancellation — not forced to zero."""

    mt = MartynaTuckerman(cubic_grid_serial)
    wg = mt.wg
    assert np.isfinite(wg[0, 0, 0])
    reciprocal = cubic_grid_serial.get_reciprocal()
    wg_sel = wg[reciprocal.mask_serial]
    assert np.all(np.isfinite(wg_sel))


def test_hartree_mt_changes_gaussian_energy(cubic_grid_serial):
    g = cubic_grid_serial
    sigma = 0.85
    rr = g.rr
    rho = DirectField(grid=g, griddata_3d=(2.0 * np.pi * sigma**2) ** (-1.5) * np.exp(-rr / (2.0 * sigma**2)))
    e_pw = Hartree()(rho, calcType={"E"}).energy
    mt = MartynaTuckerman(g)
    e_mt = Hartree(mt=mt)(rho, calcType={"E"}).energy
    assert abs(e_pw - e_mt) > 1e-10


def test_hartree_mt_pw_difference_matches_wg_contribution():
    """E_MT − E_PW must equal the Hartree contribution from multiplying ρ_G by w_G only."""
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
    np.testing.assert_allclose(e_mt - e_pw, delta_wg, rtol=1e-10, atol=1e-10)


def test_mt_hartree_gaussian_near_continuum_qe_ws():
    """MT Coulomb Hartree must reproduce the isolated Gaussian self-energy.

    For a normalized Gaussian ρ(r) with σ = 0.6 bohr in a 25-bohr box on a 96³ grid,
    the MT Hartree energy should match E = 1/(2√π σ) to better than 1e-4 relative.
    """

    sigma = 0.6
    L = 25.0
    n = 96
    lattice = np.eye(3) * L
    g = DirectGrid(lattice=lattice, nr=[n, n, n], origin=np.array([L / 2, L / 2, L / 2]))
    rho = DirectField(grid=g, griddata_3d=np.exp(-g.r_mic**2 / (2.0 * sigma**2)))
    rho /= rho.integral()
    e_ref = isolated_gaussian_hartree_self_energy(sigma)
    e_mt = Hartree(mt=MartynaTuckerman(g))(rho, calcType={"E"}).energy
    np.testing.assert_allclose(e_mt, e_ref, rtol=1e-4)


def test_ws_dist_corner_orthogonal_equals_brute_cubic_small():
    """Orthogonal fast path agrees with brute translation minimization."""

    rng = np.random.default_rng(0)
    lattice = np.diag((4.0 + rng.random(3))).astype(np.float64)
    n = 9
    g = DirectGrid(lattice=lattice, nr=[n, n, n])
    pts = np.moveaxis(g.r, 0, -1).reshape(-1, 3)[::47]
    lat = np.asarray(g.lattice, dtype=np.float64)
    brute = _ws_dist_corner_brute_chunks(pts, lat, nmax=4)
    r_xyz = pts.T.reshape(3, -1)
    corner = ws_dist_corner(r_xyz, lat, nmax=4)
    np.testing.assert_allclose(corner.reshape(-1), brute, rtol=0.0, atol=5e-10)


def test_mt_local_correction_recipro_shape_and_finite(cubic_grid_serial):
    g = cubic_grid_serial
    ions = Ions(symbols=["H", "H"], positions=np.array([[1.3, 0.1, 0.2], [2.9, 0.4, 0.55]]), cell=g.lattice)
    ions.set_charges([1.0, -1.0])
    corr = MartynaTuckerman(g).local_pp_correction_reciprocal(ions)
    qshape = cubic_grid_serial.get_reciprocal().q.shape
    assert corr.shape == qshape
    reciprocal = cubic_grid_serial.get_reciprocal()
    assert np.all(np.isfinite(corr[reciprocal.mask_serial]))


def test_mt_ion_ewald_energy_sign_symmetry_same_geometry(cubic_grid_serial):
    """|sum Z S|^2 invariant under swapping signs on every ion yields same MT ion energy."""

    g = cubic_grid_serial
    mt = MartynaTuckerman(g)
    p = np.array([[1.8, 0.95, 0.6], [2.95, 0.15, 0.9]])
    ions1 = Ions(symbols=["H", "H"], positions=p.copy(), cell=g.lattice)
    ions1.set_charges([1.0, -1.0])
    ions2 = Ions(symbols=["H", "H"], positions=p.copy(), cell=g.lattice)
    ions2.set_charges([-1.0, 1.0])
    assert mt.ion_ewald_energy(ions1) == pytest.approx(mt.ion_ewald_energy(ions2), rel=1e-12)


def test_mt_ion_ewald_energy_matches_explicit_reciprocal_sum(cubic_grid_serial):
    g = cubic_grid_serial
    mt = MartynaTuckerman(g)
    ions = Ions(symbols=["H"], positions=np.array([[1.2, 0.3, 0.4]]), cell=g.lattice)
    ions.set_charges([2.0])
    recip = g.get_reciprocal()
    stot = ions.total_strf(recip)
    strf_sq = np.real(np.conjugate(stot) * stot)
    explicit = np.sum(strf_sq[recip.mask] * mt.wg[recip.mask]) / g.volume
    explicit -= 0.5 * strf_sq[0, 0, 0] * mt.wg[0, 0, 0] / g.volume
    np.testing.assert_allclose(mt.ion_ewald_energy(ions), explicit, rtol=0.0, atol=1e-12)


def test_mt_ion_ewald_forces_finite_difference(cubic_grid_serial):
    g = cubic_grid_serial
    mt = MartynaTuckerman(g)
    ions = Ions(symbols=["H", "H"], positions=np.array([[1.5, 0.8, 0.5], [2.6, 0.2, 0.7]]), cell=g.lattice)
    ions.set_charges([1.0, -0.5])
    disp = 1e-4
    f_an = mt.ion_ewald_forces(ions)
    f_fd = np.zeros_like(f_an)
    for ia in range(ions.nat):
        for a in range(3):
            plus = ions.positions.copy()
            minus = ions.positions.copy()
            plus[ia, a] += disp
            minus[ia, a] -= disp
            ip = Ions(symbols=ions.symbols, positions=plus, cell=g.lattice)
            ip.set_charges(ions.charges)
            im = Ions(symbols=ions.symbols, positions=minus, cell=g.lattice)
            im.set_charges(ions.charges)
            f_fd[ia, a] = -(mt.ion_ewald_energy(ip) - mt.ion_ewald_energy(im)) / (2.0 * disp)
    np.testing.assert_allclose(f_an, f_fd, rtol=2e-4, atol=2e-4)


def test_ewald_with_mt_includes_ion_screening_energy(cubic_grid_serial):
    g = cubic_grid_serial
    mt = MartynaTuckerman(g)
    ions = Ions(symbols=["H"], positions=np.array([[1.0, 1.0, 1.0]]), cell=g.lattice)
    ions.set_charges([1.5])
    ew_plain = ewald(ions=ions, grid=g, mt=None)
    ew_mt = ewald(ions=ions, grid=g, mt=mt)
    delta_corr = ew_mt.Energy_corr() - ew_plain.Energy_corr()
    np.testing.assert_allclose(
        ew_mt.energy,
        ew_plain.energy + mt.ion_ewald_energy(ions) + delta_corr,
        rtol=1e-10,
        atol=1e-10,
    )


def test_ws_dist_corner_cvp_equals_brute_triclinic():
    """Babai + shell MIC agrees with brute translation search on grid nodes."""

    L = 3.7
    skew = np.array([[L, 0.0, 0.0], [0.5 * L, L, 0.0], [0.0, 0.0, L]], dtype=np.float64)
    g = DirectGrid(lattice=skew, nr=[10, 10, 10])
    pts = np.moveaxis(g.r, 0, -1).reshape(-1, 3)
    lat = np.asarray(g.lattice, dtype=np.float64)
    brute = _ws_dist_corner_brute_chunks(pts, lat, nmax=5)
    corner = ws_dist_corner_grid(g)
    np.testing.assert_allclose(corner.reshape(-1), brute, rtol=0.0, atol=5e-10)


def test_ws_dist_corner_cvp_equals_brute_random_triclinic():
    """CVP path matches brute on random fractional points in the unit cell."""

    rng = np.random.default_rng(1)
    lat = np.array(
        [
            [4.2, 0.0, 0.0],
            [1.1, 3.8, 0.0],
            [0.3, 0.7, 5.1],
        ],
        dtype=np.float64,
    )
    n_pts = 500
    frac = rng.random((n_pts, 3))
    pts = frac @ lat
    brute = _ws_dist_corner_brute_chunks(pts, lat, nmax=5)
    corner = ws_dist_corner(pts.T.reshape(3, -1), lat)
    np.testing.assert_allclose(corner.reshape(-1), brute, rtol=0.0, atol=5e-10)


def test_martyna_tuckerman_builds_wg_on_triclinic_cell():
    L = 3.7
    skew = np.array([[L, 0.0, 0.0], [0.5 * L, L, 0.0], [0.0, 0.0, L]], dtype=np.float64)
    grid = DirectGrid(lattice=skew, nr=[8, 8, 8])
    mt = MartynaTuckerman(grid)
    wg = mt.wg
    assert np.isfinite(wg[0, 0, 0])
    reciprocal = grid.get_reciprocal()
    assert np.all(np.isfinite(wg[reciprocal.mask_serial]))
