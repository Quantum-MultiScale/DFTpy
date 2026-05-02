import numpy as np
import pytest

from dftpy.config.config import DefaultOption

from dftpy.field import DirectField
from dftpy.functional.hartree import Hartree
from dftpy.functional.martyna_tuckerman import MartynaTuckerman
from dftpy.grid import DirectGrid
from dftpy.ions import Ions


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


def test_martyna_build_wg_zero_at_gamma(cubic_grid_serial):
    mt = MartynaTuckerman(cubic_grid_serial)
    wg = mt.wg
    assert wg[0, 0, 0] == 0.0
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


def test_mt_local_correction_recipro_shape_and_finite(cubic_grid_serial):
    g = cubic_grid_serial
    ions = Ions(symbols=["H", "H"], positions=np.array([[1.3, 0.1, 0.2], [2.9, 0.4, 0.55]]), cell=g.lattice)
    ions.set_charges([1.0, -1.0])
    corr = MartynaTuckerman(g).local_pp_correction_reciprocal(ions)
    qshape = cubic_grid_serial.get_reciprocal().q.shape
    assert corr.shape == qshape
    reciprocal = cubic_grid_serial.get_reciprocal()
    assert np.all(np.isfinite(corr[reciprocal.mask_serial]))


def test_mt_ewald_energy_sign_symmetry_same_geometry(cubic_grid_serial):
    """|sum Z S|^2 invariant under swapping signs on every ion yields same MT ion energy."""

    g = cubic_grid_serial
    mt = MartynaTuckerman(g)
    p = np.array([[1.8, 0.95, 0.6], [2.95, 0.15, 0.9]])
    ions1 = Ions(symbols=["H", "H"], positions=p.copy(), cell=g.lattice)
    ions1.set_charges([1.0, -1.0])
    ions2 = Ions(symbols=["H", "H"], positions=p.copy(), cell=g.lattice)
    ions2.set_charges([-1.0, 1.0])
    e1 = mt.ion_ewald_energy(ions1)
    e2 = mt.ion_ewald_energy(ions2)
    assert e1 == pytest.approx(e2, rel=1e-12)


def test_mt_ewald_forces_have_expected_shape(cubic_grid_serial):
    g = cubic_grid_serial
    mt = MartynaTuckerman(g)
    ions = Ions(symbols=["H"], positions=np.zeros((1, 3)), cell=g.lattice)
    ions.set_charges([2.5])
    f = mt.ion_ewald_forces(ions)
    assert f.shape == (ions.nat, 3)
    assert np.all(np.isfinite(f))
