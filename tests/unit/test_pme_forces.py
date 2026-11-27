"""Cartesian energy derivatives for PME forces on general cell matrices.

The density is fixed during ionic displacements.  Every displaced evaluation
uses new Ewald/pseudopotential objects so that cached structure factors cannot
hide a force error.  These helpers are also used by the MPI regression tests.
"""

from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose

from dftpy.ewald import ewald
from dftpy.field import DirectField
from dftpy.functional.pseudo import LocalPseudo, ReadPseudo
from dftpy.grid import DirectGrid
from dftpy.ions import Ions
from dftpy.mpi import MP


DATA = Path(__file__).resolve().parents[2] / "examples" / "DATA"
GRID_NR = (32, 28, 24)
COMPONENTS = ("ewald_reciprocal", "ewald", "pseudo")
ORTHORHOMBIC_CELL = np.diag([8.4, 9.1, 10.2])
ROTATION = np.array([[0.8, -0.6, 0.0], [0.36, 0.48, -0.8], [0.48, 0.64, 0.6]])
CELLS = {
    "triclinic": np.array([[8.4, 1.3, -0.7], [-0.6, 9.1, 1.5], [1.2, -0.8, 10.2]]),
    "orthorhombic": ORTHORHOMBIC_CELL,
    "rotated_orthorhombic": ORTHORHOMBIC_CELL @ ROTATION,
}
SCALED_POSITIONS = np.array([[0.013, 0.231, 0.347], [0.427, 0.563, 0.689], [0.791, 0.857, 0.923]])
FD_ATOL = 2.0e-8
PME_ATOL = 5.0e-7


def make_system(mp, cell=None, nr=GRID_NR):
    """Build a nonzero-force Al system with a positive, fixed periodic density."""
    if cell is None:
        cell = CELLS["triclinic"]
    readpp = ReadPseudo({"Al": DATA / "al.lda.recpot"}, comm=mp.comm)
    ions = Ions(
        symbols="Al3",
        cell=cell,
        scaled_positions=SCALED_POSITIONS,
        charges=np.full(3, readpp.zval["Al"]),
    )
    grid = DirectGrid(lattice=ions.cell, nr=nr, mp=mp)
    s = grid.s
    modulation = (
        1.0
        + 0.20 * np.cos(2.0 * np.pi * (s[0] + s[1]))
        + 0.15 * np.sin(2.0 * np.pi * (s[1] - s[2]))
        + 0.10 * np.cos(2.0 * np.pi * (s[0] + 2.0 * s[2]) + 0.37)
    )
    rho = DirectField(grid=grid, griddata_3d=modulation * ions.get_ncharges() / grid.volume)
    return {"ions": ions, "grid": grid, "rho": rho, "readpp": readpp}


def evaluate(system, component, pme=True, positions=None, forces=True):
    """Return global energy and force, including the proper MPI reductions."""
    ions = system["ions"].copy()
    if positions is not None:
        ions.positions[:] = positions
    grid = system["grid"]
    mp = grid.mp
    force = None
    if component == "pseudo":
        pseudo = LocalPseudo(grid=grid, ions=ions, readpp=system["readpp"], PME=pme)
        energy = pseudo(system["rho"], calcType={"E"}).energy
        if forces:
            force = pseudo.forces(system["rho"])
    elif component in ("ewald", "ewald_reciprocal"):
        electrostatic = ewald(grid=grid, ions=ions, PME=pme, precision=1.0e-10)
        if component == "ewald":
            energy = mp.asum(electrostatic.energy)
            if forces:
                force = electrostatic.forces
        else:
            energy = mp.asum(electrostatic.Energy_rec_PME() if pme else electrostatic.Energy_rec())
            if forces:
                force = mp.vsum(electrostatic.Forces_rec_PME() if pme else electrostatic.Forces_rec())
    else:
        raise ValueError("Unknown force component: " + component)
    return float(energy), None if force is None else np.asarray(force)


def finite_difference(system, component, pme=True, step=1.0e-4):
    """Differentiate energy by Cartesian displacements with the cell/density fixed."""
    positions = system["ions"].positions
    force = np.empty_like(positions)
    for atom in range(len(positions)):
        for axis in range(3):
            plus = positions.copy()
            minus = positions.copy()
            plus[atom, axis] += step
            minus[atom, axis] -= step
            e_plus, _ = evaluate(system, component, pme=pme, positions=plus, forces=False)
            e_minus, _ = evaluate(system, component, pme=pme, positions=minus, forces=False)
            force[atom, axis] = -(e_plus - e_minus) / (2.0 * step)
    return force


@pytest.mark.parametrize("cell_name", CELLS)
@pytest.mark.parametrize("component", COMPONENTS)
@pytest.mark.parametrize("pme", [False, True], ids=["direct", "pme"])
def test_cartesian_energy_derivative(cell_name, component, pme):
    system = make_system(MP(parallel=False), cell=CELLS[cell_name])
    _, analytic = evaluate(system, component, pme=pme)
    coarse = finite_difference(system, component, pme=pme, step=1.0e-3)
    fine = finite_difference(system, component, pme=pme, step=5.0e-4)

    # A symmetric/equilibrium configuration with vanishing forces would miss
    # the transpose error.  All Cartesian directions must carry a signal.
    assert np.all(np.max(np.abs(analytic), axis=0) > 1.0e-3)
    assert_allclose(analytic, fine, atol=FD_ATOL, rtol=0)
    assert_allclose(coarse, fine, atol=FD_ATOL, rtol=0)
    coarse_error = np.max(np.abs(analytic - coarse))
    fine_error = np.max(np.abs(analytic - fine))
    # Central differences converge quadratically until floating-point
    # cancellation dominates the energy difference.
    assert fine_error <= 0.35 * coarse_error + 2.0e-9


@pytest.mark.parametrize("cell_name", CELLS)
@pytest.mark.parametrize("component", COMPONENTS)
def test_pme_matches_direct_sum(cell_name, component):
    system = make_system(MP(parallel=False), cell=CELLS[cell_name])
    direct_energy, direct_force = evaluate(system, component, pme=False)
    pme_energy, pme_force = evaluate(system, component, pme=True)
    assert_allclose(pme_energy, direct_energy, atol=PME_ATOL, rtol=0)
    assert_allclose(pme_force, direct_force, atol=PME_ATOL, rtol=0)


@pytest.mark.parametrize("component", COMPONENTS)
def test_rotated_orthorhombic_forces(component):
    """A rigid rotation preserves energies and rotates Cartesian force rows."""
    system = make_system(MP(parallel=False), cell=CELLS["orthorhombic"])
    rotated = make_system(MP(parallel=False), cell=CELLS["rotated_orthorhombic"])
    energy, force = evaluate(system, component)
    rotated_energy, rotated_force = evaluate(rotated, component)
    assert_allclose(rotated_energy, energy, atol=2.0e-11, rtol=0)
    assert_allclose(rotated_force, force @ ROTATION, atol=2.0e-11, rtol=0)
