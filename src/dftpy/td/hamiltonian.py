import numpy as np
from scipy.sparse.linalg import LinearOperator, eigsh
from typing import Tuple

from dftpy.constants import SPEED_OF_LIGHT
from dftpy.field import DirectField, ReciprocalField
from dftpy.grid import DirectGrid
from dftpy.time_data import timer
from dftpy.td.lobpcg import lobpcg as lobpcg_solver
from dftpy.td.lobpcg import lowest_diagonal_guesses
from dftpy.td.operator import Operator


class ReducedReciprocalBasis:
    """Local part of the distributed plane-wave basis below ``ecut``."""

    def __init__(self, reci_grid, ecut):
        if ecut is None or ecut <= 0.0:
            raise ValueError("ecut must be a positive number.")

        self.reci_grid = reci_grid
        self.mp = reci_grid.mp
        kinetic = 0.5 * np.asarray(reci_grid.gg)
        self.mask = kinetic <= ecut
        self.kinetic = kinetic[self.mask].astype(np.float64)
        self.gg = np.asarray(reci_grid.gg)[self.mask].astype(np.float64)
        self.local_npw = int(self.kinetic.size)
        self.npw = int(self.mp.vsum(self.local_npw))

    def expand(self, coeff):
        full = np.zeros(self.reci_grid.nr, dtype=np.complex128)
        full[self.mask] = coeff
        return full

    def expand_block(self, coeffs):
        coeffs = np.asarray(coeffs, dtype=np.complex128)
        full = np.zeros((coeffs.shape[0], *self.reci_grid.nr), dtype=np.complex128)
        full[:, self.mask] = coeffs
        return full

    def project(self, field_g):
        return np.asarray(field_g)[self.mask].astype(np.complex128, copy=False)

    def project_block(self, field_g):
        data = np.asarray(field_g)
        if data.ndim == 3:
            return data[self.mask][None, :].astype(np.complex128, copy=False)
        return data[:, self.mask].astype(np.complex128, copy=False)


class Hamiltonian(Operator):
    """
    Hamiltonian: 
    .. math::
        \hat{H} = \frac{1}{2}\left(-i\nabla-\frac{\mathbf{A}}{c}\right)^2 + v
    """

    def __init__(self, v=None, A=None, full=True):
        """

        Parameters
        ----------
        v: DirectField
            Effective scalar potential
        A: np.ndarray
            Vector potential which is constant in space
        full: bool
            Must be True if complex numbers are involved in the computation

        """
        self.full = full
        self.v = v
        self.A = A

    @property
    def v(self):
        return self._v

    @property
    def A(self):
        return self._A

    @v.setter
    def v(self, v):
        if isinstance(v, DirectField):
            self._v = v
            if v.grid.full == self.full:
                self.grid = v.grid
            else:
                self.grid = DirectGrid(lattice=v.grid.lattice, nr=v.grid.nr, origin=v.grid.origin, full=self.full, mp=v.grid.mp)
        elif v is None:
            self._v = None
            self.grid = None
        else:
            raise TypeError("v must be a DFTpy DirectField.")

    @A.setter
    def A(self, new_A):
        if new_A is None:
            self._A = None
        else:
            if np.size(new_A) != 3:
                raise AttributeError('Size of the A must be 3.')
            self._A = np.asarray(new_A)
            ones = np.ones(self.grid.nr)
            self.a_field = DirectField(self.grid, rank=3, griddata_3d=np.asarray(
                [self._A[0] * ones, self._A[1] * ones, self._A[2] * ones]) / SPEED_OF_LIGHT)

    def __call__(self, psi, force_real=None, sigma=0.025):
        """
        Performs the Hamiltonian operation on psi.

        Parameters
        ----------
        psi: DirectField or ReciprocalField
            The wavefunction
        force_real: None or bool
            Determines the force_real flag for inverse FFT. If set to None, the flag will be determined base on
            whether psi is real or complex.
        sigma: float
            The smearing factor for gradient.

        Returns
        -------
        result: DirectField or ReciprocalField
        The result

        """
        if isinstance(psi, DirectField):
            if force_real is None:
                if np.isrealobj(psi):
                    force_real = True
                else:
                    force_real = False
            if self._A is None:
                return -0.5 * psi.laplacian(force_real=force_real, sigma=sigma) + self.v * psi
            else:
                psi_fft = psi.fft()
                intermediate_result1 = 0.5 * psi_fft.grid.gg * psi_fft
                intermediate_result1 *= np.exp(-psi_fft.grid.gg * sigma * sigma / 4.0)
                intermediate_result2 = - psi_fft.grid.g * psi_fft
                intermediate_result2 *= np.exp(-psi_fft.grid.gg * sigma * sigma / 4.0)
                return intermediate_result1.ifft(force_real=force_real) + self.a_field.dot(
                    intermediate_result2.ifft(force_real=force_real)) + 0.5 * self.a_field.dot(
                    self.a_field) * psi + self.v * psi
                # return -0.5 * psi.laplacian(force_real = force_real, sigma=sigma) + 0.5 * self.a_field.dot(self.a_field) * psi + 1.0j * self.a_field.dot(psi.gradient(flag = "supersmooth", force_real = force_real, sigma=sigma)) + self.v * psi
                # return -0.5 * psi.laplacian(force_real = force_real, sigma=sigma) + 0.5 * self.a_field.dot(self.a_field) * psi + 0.5j * self.a_field.dot(psi.gradient(force_real = force_real)) + 0.5j * (self.a_field * psi).divergence(force_real=force_real) + self.v * psi

        elif isinstance(psi, ReciprocalField):
            gg = psi.grid.gg
            kinetic = 0.5 * gg * np.exp(-gg * sigma**2 / 2.0) * psi
            potential = (self.v * psi.ifft()).fft()
            return kinetic + potential #0.5 * psi.grid.gg * psi + (self.v * psi.ifft()).fft()
        else:
            raise TypeError("psi must be a DFTpy DirectField or ReciprocalField.")

    def reduced_reciprocal_matvec(self, coeff, basis, sigma=0.025):
        coeff = np.asarray(coeff, dtype=np.complex128)
        kinetic = basis.kinetic * np.exp(-basis.gg * sigma**2 / 2.0) * coeff
        psi_g = ReciprocalField(
            basis.reci_grid,
            rank=1,
            griddata_3d=basis.expand(coeff),
        )
        potential_g = (self.v * psi_g.ifft(force_real=False)).fft()
        return kinetic + basis.project(potential_g)

    def reduced_reciprocal_matvec_block(self, coeffs, basis, sigma=0.025,
                                        chunk_size=None):
        """Apply the reduced Hamiltonian to a block of local vectors."""
        if len(coeffs) == 0:
            return []
        coeffs = np.asarray(coeffs, dtype=np.complex128)
        chunk_size = int(chunk_size or coeffs.shape[0])
        kinetic_factor = basis.kinetic[None, :] * np.exp(
            -basis.gg[None, :] * sigma**2 / 2.0
        )
        result = []
        for start in range(0, coeffs.shape[0], chunk_size):
            block = coeffs[start:start + chunk_size]
            psi_g = ReciprocalField(
                basis.reci_grid,
                rank=block.shape[0],
                griddata_3d=basis.expand_block(block),
            )
            potential_g = (self.v * psi_g.ifft(force_real=False)).fft()
            hblock = kinetic_factor * block + basis.project_block(potential_g)
            result.extend(row.copy() for row in hblock)
        return result

    def _diagonalize_reduced(
            self, basis, numeig, sigma, tol, maxiter, max_subspace,
            initial_field, damping, block_size, matvec_block_size):
        """Run the same distributed LOBPCG algorithm in serial or MPI."""
        block_size = block_size or max(2 * numeig, numeig + 4)
        initial_vectors = []
        if initial_field is not None:
            data = np.asarray(initial_field)
            guess_r = DirectField(
                initial_field.grid,
                rank=1,
                griddata_3d=np.sqrt(np.maximum(data.real, 0.0)),
            )
            initial_vectors.append(basis.project(guess_r.fft()))
        initial_vectors.extend(
            lowest_diagonal_guesses(basis.kinetic, block_size, basis.mp)
        )
        average_potential = float(self.v.integral() / self.grid.volume)
        eigenvalues, coeffs, _ = lobpcg_solver(
            matvec=lambda vector: self.reduced_reciprocal_matvec(
                vector, basis, sigma=sigma
            ),
            matvec_block=lambda vectors: self.reduced_reciprocal_matvec_block(
                vectors, basis, sigma=sigma, chunk_size=matvec_block_size
            ),
            diagonal=basis.kinetic,
            numeig=numeig,
            mp=basis.mp,
            tol=tol,
            maxiter=maxiter,
            max_subspace=max_subspace,
            initial_vectors=initial_vectors,
            diagonal_shift=average_potential,
            damping=damping,
            block_size=block_size,
        )
        return eigenvalues, coeffs

    @timer('Diagonalize')
    def diagonalize(
            self,
            numeig: int,
            return_eigenvectors: bool = True,
            reciprocal: bool = False,
            sigma: float = 0.025,
            ecut: float = None,
            tol: float = 1.0e-8,
            maxiter: int = 80,
            max_subspace: int = None,
            initial_field=None,
            damping: float = 0.35,
            block_size: int = None,
            matvec_block_size: int = None,
    ) -> Tuple:
        """
        Diagonalize the Hamiltonian and returns the lowest eigenvalues and optionally eigenvectors

        Parameters
        ----------
        numeig: int
            Number of eigenvalues to return.
        return_eigenvectors: bool
            Determine whether the eigenvectors will be returned.
        reciprocal: bool
            Determine the eigenvectors are calculated in real or reciprocal space.
        sigma: float
            The smearing factor for the kinetic operator.
        ecut: float
            If given with reciprocal=True, diagonalize in the reduced reciprocal
            basis satisfying 0.5 * |G|^2 <= ecut.

        Returns
        -------
        Tuple (eigenvalue_list, ) or (eigenvalue_list, psi_list)
        eigenvalue_list: List[int]
            The list of eigenvalues
        psi_list: List[DirectField or ReciprocalField]
            The list of eigenvectors

        """

        if ecut is not None and not reciprocal:
            raise ValueError("ecut is only supported with reciprocal=True.")

        if reciprocal and ecut is not None:
            reci_grid = self.grid.get_reciprocal()
            basis = ReducedReciprocalBasis(reci_grid, ecut)

            if basis.npw <= numeig:
                raise ValueError("ecut is too small for the requested numeig.")

            eigenvalue_list, psis = self._diagonalize_reduced(
                basis, numeig, sigma, tol, maxiter, max_subspace,
                initial_field, damping, block_size, matvec_block_size,
            )

            if return_eigenvectors:
                order = np.argsort(eigenvalue_list)
                eigenvalue_list = eigenvalue_list[order]
                psis = [psis[i] for i in order]
                psi_list = []
                for i in range(numeig):
                    psi = ReciprocalField(
                        reci_grid,
                        rank=1,
                        griddata_3d=basis.expand(psis[i]),
                    )
                    psi = psi.ifft(force_real=False)
                    psi = psi / np.sqrt((np.real(psi) * np.real(psi) + np.imag(psi) * np.imag(psi)).integral())
                    psi_list.append(psi)
                return eigenvalue_list, psi_list
            else:
                return np.sort(eigenvalue_list),

        if reciprocal:
            reci_grid = self.grid.get_reciprocal()
            size = reci_grid.nnr
            dtype = np.complex128
        else:
            size = self.grid.nnr
            dtype = np.float64

        A = LinearOperator((size, size), dtype=dtype, matvec=self.scipy_matvec_utils(reciprocal=reciprocal, sigma=sigma))

        if return_eigenvectors:
            eigenvalue_list, psis = eigsh(A, k=numeig, which='SA', return_eigenvectors=True)
            order = np.argsort(eigenvalue_list)
            eigenvalue_list = eigenvalue_list[order]
            psis = psis[:, order]
            psi_list = []
            for i in range(numeig):
                if reciprocal:
                    psi = ReciprocalField(reci_grid, rank=1, griddata_3d=np.reshape(psis[:, i], reci_grid.nr))
                    psi = psi.ifft()
                else:
                    psi = DirectField(self.grid, rank=1, griddata_3d=np.reshape(psis[:, i], self.grid.nr))
                psi = psi / np.sqrt((np.real(psi) * np.real(psi) + np.imag(psi) * np.imag(psi)).integral())
                psi_list.append(psi)
            return eigenvalue_list, psi_list
        else:
            eigenvalue_list = eigsh(A, k=numeig, which='SA', return_eigenvectors=False)
            return np.sort(eigenvalue_list),

    def energy(self, psi):
        """
        Calculate the expectation value of Hamiltonian on wavefunction psi

        Parameters
        ----------
        psi: DirectField or ReciprocalField
            The wavefunction

        Returns
        -------
        energy: DirectField or ReciprocalField

        """
        return np.real(np.conj(psi) * self(psi)).integral()
