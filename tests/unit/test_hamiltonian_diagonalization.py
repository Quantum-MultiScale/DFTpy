"""Serial and MPI checks for reduced-reciprocal Hamiltonian diagonalization."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from dftpy.field import DirectField
from dftpy.grid import DirectGrid
from dftpy.mpi import MP
from dftpy.td.hamiltonian import Hamiltonian


POTENTIAL = 0.2


def _hamiltonian(mp):
    grid = DirectGrid(lattice=np.eye(3) * 8.0, nr=[8, 8, 8], mp=mp)
    potential = DirectField(grid=grid)
    potential[:] = POTENTIAL
    return Hamiltonian(potential)


def _diagonalize_reduced(mp):
    return _hamiltonian(mp).diagonalize(
        numeig=1,
        reciprocal=True,
        ecut=2.0,
        tol=1.0e-10,
        maxiter=20,
        block_size=4,
        matvec_block_size=2,
    )


def _diagonalize_full_real():
    eigenvalues, _ = _hamiltonian(MP(parallel=False)).diagonalize(
        numeig=1,
        reciprocal=False,
        tol=1.0e-10,
    )
    return eigenvalues


def _check_ground_state(eigenvalues, wavefunctions):
    assert_allclose(eigenvalues, [POTENTIAL], rtol=0.0, atol=1.0e-10)
    norm = (np.abs(wavefunctions[0]) ** 2).integral()
    assert_allclose(norm, 1.0, rtol=0.0, atol=1.0e-12)


def test_hamiltonian_diagonalization_serial():
    reduced_eigenvalues, wavefunctions = _diagonalize_reduced(MP(parallel=False))
    assert_allclose(
        reduced_eigenvalues, _diagonalize_full_real(), rtol=0.0, atol=1.0e-6
    )
    _check_ground_state(reduced_eigenvalues, wavefunctions)


@pytest.mark.mpi
def test_hamiltonian_diagonalization_mpi():
    pytest.importorskip("mpi4py")
    mp = MP(parallel=True)
    if mp.size < 2:
        pytest.skip("need at least 2 MPI ranks")

    full_eigenvalues = mp.comm.bcast(
        _diagonalize_full_real() if mp.is_root else None, root=0
    )
    reduced_eigenvalues, wavefunctions = _diagonalize_reduced(mp)
    assert_allclose(
        reduced_eigenvalues, full_eigenvalues, rtol=0.0, atol=1.0e-6
    )
    _check_ground_state(reduced_eigenvalues, wavefunctions)
    mp.comm.Barrier()
