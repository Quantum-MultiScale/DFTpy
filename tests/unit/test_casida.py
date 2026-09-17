"""Unit tests for vectorized and matrix-free Casida diagonalization."""

import numpy as np
from numpy.testing import assert_allclose

from dftpy.field import DirectField
from dftpy.grid import DirectGrid
from dftpy.mpi import MP
from dftpy.td.casida import Casida


class _KernelResult:
    def __init__(self, grid):
        self.v2rho2 = DirectField(grid=grid, rank=3)
        self.v2rho2[:] = 0.0


def _zero_kernel(density, calcType=None):
    return _KernelResult(density.grid)


def _casida_inputs():
    grid = DirectGrid(
        lattice=np.eye(3) * 6.0,
        nr=[4, 4, 4],
        mp=MP(parallel=False),
    )
    rho = DirectField(grid=grid)
    rho[:] = 2.0 / grid.volume

    x, y, z = grid.r
    orbitals = []
    for values in (np.ones(grid.nr), np.sin(x), np.sin(y), np.sin(z)):
        psi = DirectField(grid=grid, griddata_3d=values)
        psi /= np.sqrt((psi * psi).integral())
        orbitals.append(psi)
    return rho, np.array([0.0, 0.4, 0.7, 1.1]), orbitals


def test_casida_dense_and_matrix_free_modes():
    rho, eigenvalues, orbitals = _casida_inputs()

    dense = Casida(rho, _zero_kernel)
    dense.build_matrix(4, eigenvalues, orbitals, scale=0.0)
    dense_modes, _, _ = dense(k_modes=2)

    matrix_free = Casida(rho, _zero_kernel)
    matrix_free.build_matrix(
        4, eigenvalues, orbitals, scale=0.0, matrix_free=True
    )
    iterative_modes, _, _ = matrix_free(k_modes=2)

    assert_allclose(dense_modes, eigenvalues[1:3], atol=1.0e-12)
    assert_allclose(iterative_modes, dense_modes, atol=1.0e-12)
