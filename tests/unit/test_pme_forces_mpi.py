"""PME force regressions on genuinely decomposed two- and four-rank grids.

Run with ``mpiexec -n 2 python -m pytest --with-mpi`` (or ``-n 4``).
The serial reference uses an independent ``MP(parallel=False)`` grid.  The
shared evaluator sums only rank-local Ewald energies/reciprocal forces;
public Ewald and local-pseudopotential forces are already global.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

pytest.importorskip("mpi4py")

from dftpy.mpi import MP

from .test_pme_forces import (
    CELLS,
    COMPONENTS,
    FD_ATOL,
    evaluate,
    finite_difference,
    make_system,
)

pytestmark = pytest.mark.mpi


@pytest.fixture(scope="module")
def mp():
    parallel = MP(parallel=True)
    if parallel.size < 2:
        pytest.skip("requires mpiexec with at least two ranks")
    yield parallel
    parallel.free()


def _serial_reference(mp, cell, pme):
    """Broadcast a reference or an error so a rank-zero failure cannot hang."""
    outcome = None
    if mp.is_root:
        try:
            system = make_system(MP(parallel=False), cell=cell)
            reference = {
                component: evaluate(system, component, pme=pme)
                for component in COMPONENTS
            }
            outcome = (reference, None)
        except Exception as exc:
            outcome = (None, f"{type(exc).__name__}: {exc}")
    reference, error = mp.comm.bcast(outcome, root=0)
    if error is not None:
        pytest.fail(f"serial reference failed: {error}")
    return reference


@pytest.mark.parametrize("cell_name", tuple(CELLS))
@pytest.mark.parametrize("pme", [False, True], ids=["direct", "pme"])
def test_parallel_forces_and_energies_match_serial(mp, cell_name, pme):
    """Every rank receives the same physical result as the serial solver."""
    cell = CELLS[cell_name]
    reference = _serial_reference(mp, cell, pme)
    system = make_system(mp, cell=cell)
    result = {
        component: evaluate(system, component, pme=pme)
        for component in COMPONENTS
    }

    # Complete all collectives before asserting.  Inspect every rank on every
    # process, so a rank-specific failure cannot strand peers in the next test.
    rank_results = mp.comm.allgather(result)
    for rank, actual in enumerate(rank_results):
        for component in COMPONENTS:
            energy, forces = actual[component]
            expected_energy, expected_forces = reference[component]
            context = f"{component}, {cell_name}, PME={pme}, rank={rank}"
            assert_allclose(
                energy, expected_energy, rtol=2e-11, atol=2e-11,
                equal_nan=False, err_msg=context + " energy",
            )
            assert_allclose(
                forces, expected_forces, rtol=2e-11, atol=2e-11,
                equal_nan=False, err_msg=context + " forces",
            )


@pytest.mark.parametrize("component", COMPONENTS)
def test_parallel_pme_forces_match_cartesian_finite_difference(mp, component):
    """Differentiate the global PME energy at fixed, nonuniform density."""
    system = make_system(mp, cell=CELLS["triclinic"])
    _, forces = evaluate(system, component, pme=True)
    finite_difference_forces = finite_difference(
        system, component, pme=True, step=1e-4,
    )
    rank_results = mp.comm.allgather((forces, finite_difference_forces))
    for rank, (analytic, numerical) in enumerate(rank_results):
        assert np.max(np.abs(analytic)) > 1e-4
        assert_allclose(
            analytic, numerical, rtol=0.0, atol=FD_ATOL,
            equal_nan=False,
            err_msg=f"{component}, rank={rank}: Cartesian energy derivative",
        )
