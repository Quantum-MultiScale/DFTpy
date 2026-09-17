import numpy as np
from typing import Tuple

from dftpy.constants import SPEED_OF_LIGHT
from dftpy.field import DirectField, ReciprocalField
from dftpy.grid import DirectGrid
from dftpy.time_data import timer
from dftpy.td.lobpcg import lobpcg as lobpcg_solver
from dftpy.td.lobpcg import lowest_diagonal_guesses
from dftpy.td.operator import Operator


class DistributedReducedReciprocalBasis:
    """Local part of the plane-wave basis 0.5 * |G|^2 <= ecut."""

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

    def expand(self, coeff_local):
        """Put local reduced coefficients back on the local reciprocal FFT grid."""
        full = np.zeros(self.reci_grid.nr, dtype=np.complex128)
        full[self.mask] = coeff_local
        return full

    def expand_block(self, coeffs_local):
        """Put a block of local reduced coefficients back on the reciprocal FFT grid."""
        coeffs_local = np.asarray(coeffs_local, dtype=np.complex128)
        full = np.zeros((coeffs_local.shape[0], *self.reci_grid.nr), dtype=np.complex128)
        full[:, self.mask] = coeffs_local
        return full

    def project(self, field_g):
        """Project the local reciprocal FFT grid onto the local reduced basis."""
        return np.asarray(field_g)[self.mask].astype(np.complex128, copy=False)

    def project_block(self, field_g):
        """Project a rank>1 reciprocal FFT field onto the local reduced basis."""
        data = np.asarray(field_g)
        if data.ndim == 3:
            return data[self.mask][None, :].astype(np.complex128, copy=False)
        return data[:, self.mask].astype(np.complex128, copy=False)


class HamiltonianMPI(Operator):
    """Simple MPI reduced-reciprocal Hamiltonian with block Davidson."""

    def __init__(self, v=None, A=None, full=True):
        self.full = full
        self.v = v
        self.A = A

    @property
    def v(self):
        return self._v

    @v.setter
    def v(self, v):
        if isinstance(v, DirectField):
            self._v = v
            if v.grid.full == self.full:
                self.grid = v.grid
            else:
                self.grid = DirectGrid(
                    lattice=v.grid.lattice,
                    nr=v.grid.nr,
                    origin=v.grid.origin,
                    full=self.full,
                    mp=v.grid.mp,
                )
        elif v is None:
            self._v = None
            self.grid = None
        else:
            raise TypeError("v must be a DFTpy DirectField.")

    @property
    def A(self):
        return self._A

    @A.setter
    def A(self, new_A):
        if new_A is None:
            self._A = None
            return
        if np.size(new_A) != 3:
            raise AttributeError("Size of A must be 3.")
        self._A = np.asarray(new_A)
        ones = np.ones(self.grid.nr)
        self.a_field = DirectField(
            self.grid,
            rank=3,
            griddata_3d=np.asarray(
                [self._A[0] * ones, self._A[1] * ones, self._A[2] * ones]
            ) / SPEED_OF_LIGHT,
        )

    def __call__(self, psi, force_real=None, sigma=0.025):
        """Keep the same real/reciprocal behavior as Hamiltonian."""
        if isinstance(psi, DirectField):
            if force_real is None:
                force_real = bool(np.isrealobj(psi))
            if self._A is None:
                return -0.5 * psi.laplacian(force_real=force_real, sigma=sigma) + self.v * psi
            psi_fft = psi.fft()
            gg = psi_fft.grid.gg
            intermediate1 = 0.5 * gg * psi_fft
            intermediate1 *= np.exp(-gg * sigma * sigma / 4.0)
            intermediate2 = -psi_fft.grid.g * psi_fft
            intermediate2 *= np.exp(-gg * sigma * sigma / 4.0)
            return (
                intermediate1.ifft(force_real=force_real)
                + self.a_field.dot(intermediate2.ifft(force_real=force_real))
                + 0.5 * self.a_field.dot(self.a_field) * psi
                + self.v * psi
            )

        if isinstance(psi, ReciprocalField):
            gg = psi.grid.gg
            kinetic = 0.5 * gg * np.exp(-gg * sigma**2 / 2.0) * psi
            potential = (self.v * psi.ifft(force_real=False)).fft()
            return kinetic + potential

        raise TypeError("psi must be a DFTpy DirectField or ReciprocalField.")

    def reduced_reciprocal_matvec(self, coeff_local, basis, sigma=0.025):
        """Distributed H|psi>: local kinetic + distributed FFT potential."""
        coeff_local = np.asarray(coeff_local, dtype=np.complex128)

        kinetic = basis.kinetic * np.exp(-basis.gg * sigma**2 / 2.0) * coeff_local
        psi_g = ReciprocalField(
            basis.reci_grid,
            rank=1,
            griddata_3d=basis.expand(coeff_local),
        )
        potential_g = (self.v * psi_g.ifft(force_real=False)).fft()
        return kinetic + basis.project(potential_g)

    def reduced_reciprocal_matvec_block(self, coeffs_local, basis, sigma=0.025, chunk_size=None):
        """Distributed block H|psi>: apply the Hamiltonian to many vectors."""
        if len(coeffs_local) == 0:
            return []

        coeffs = np.asarray(coeffs_local, dtype=np.complex128)
        chunk_size = int(chunk_size or coeffs.shape[0])
        kinetic_factor = basis.kinetic[None, :] * np.exp(-basis.gg[None, :] * sigma**2 / 2.0)
        result = []
        for start in range(0, coeffs.shape[0], chunk_size):
            stop = min(start + chunk_size, coeffs.shape[0])
            block = coeffs[start:stop]
            kinetic = kinetic_factor * block
            psi_g = ReciprocalField(
                basis.reci_grid,
                rank=block.shape[0],
                griddata_3d=basis.expand_block(block),
            )
            potential_g = (self.v * psi_g.ifft(force_real=False)).fft()
            hblock = kinetic + basis.project_block(potential_g)
            result.extend(hblock[i].copy() for i in range(hblock.shape[0]))
        return result

    @staticmethod
    def dot_global(x_local, y_local, mp):
        local = np.vdot(x_local, y_local)
        return mp.vsum(local)

    @classmethod
    def norm_global(cls, x_local, mp):
        return float(np.sqrt(cls.dot_global(x_local, x_local, mp).real))

    @classmethod
    def orthonormalize(cls, v_local, vectors, mp, tol=1.0e-12):
        v_local = np.asarray(v_local, dtype=np.complex128).copy()
        for q_local in vectors:
            v_local -= cls.dot_global(q_local, v_local, mp) * q_local
        norm = cls.norm_global(v_local, mp)
        if norm < tol:
            return None
        return v_local / norm

    @staticmethod
    def combine_vectors(coeffs, vectors):
        out = np.zeros_like(vectors[0])
        for c, v in zip(coeffs, vectors):
            out += c * v
        return out

    def low_kinetic_guesses(self, basis, nvec):
        """Unit vectors on the globally lowest local kinetic-energy G modes."""
        mp = basis.mp
        if mp.is_mpi:
            gathered = mp.comm.allgather(basis.kinetic)
        else:
            gathered = [basis.kinetic]
        modes = []
        for rank, kinetic in enumerate(gathered):
            modes.extend((float(value), rank, idx) for idx, value in enumerate(kinetic))
        modes = sorted(modes)[:nvec]

        guesses = []
        for _, owner, idx in modes:
            guess = np.zeros(basis.local_npw, dtype=np.complex128)
            if mp.rank == owner:
                guess[idx] = 1.0
            guesses.append(guess)
        return guesses

    def field_guess(self, field, basis):
        """Project a real-space field, for example sqrt(rho), into the reduced G basis."""
        data = np.asarray(field)
        guess_r = DirectField(
            field.grid,
            rank=1,
            griddata_3d=np.sqrt(np.maximum(data.real, 0.0)),
        )
        return basis.project(guess_r.fft())

    def average_potential(self):
        """Cell average of the local potential, used as a simple preconditioner shift."""
        return float(self.v.integral() / self.grid.volume)

    def hermiticity_test(self, basis, sigma=0.025, ntrials=3, seed=13579):
        """Check <x|Hy> = <Hx|y> for random distributed coefficient vectors."""
        mp = basis.mp
        rng = np.random.default_rng(seed + 104729 * mp.rank)
        results = []

        for _ in range(ntrials):
            x = rng.normal(size=basis.local_npw) + 1j * rng.normal(size=basis.local_npw)
            y = rng.normal(size=basis.local_npw) + 1j * rng.normal(size=basis.local_npw)
            x /= self.norm_global(x, mp)
            y /= self.norm_global(y, mp)

            hx = self.reduced_reciprocal_matvec(x, basis, sigma=sigma)
            hy = self.reduced_reciprocal_matvec(y, basis, sigma=sigma)
            lhs = self.dot_global(x, hy, mp)
            rhs = self.dot_global(hx, y, mp)
            diff = lhs - rhs
            scale = max(abs(lhs), abs(rhs), 1.0)
            results.append(
                {
                    "lhs": lhs,
                    "rhs": rhs,
                    "abs_error": float(abs(diff)),
                    "rel_error": float(abs(diff) / scale),
                }
            )

        return results

    def lobpcg(
        self,
        basis,
        numeig,
        sigma=0.025,
        tol=1.0e-8,
        maxiter=80,
        max_subspace=None,
        initial_field=None,
        damping=0.35,
        block_size=None,
        matvec_block_size=None,
    ):
        """Distributed LOBPCG wrapper for the reduced reciprocal Hamiltonian."""
        matvec = lambda v: self.reduced_reciprocal_matvec(v, basis, sigma=sigma)
        matvec_block = lambda vectors: self.reduced_reciprocal_matvec_block(
            vectors, basis, sigma=sigma, chunk_size=matvec_block_size
        )
        block_size = block_size or max(2 * numeig, numeig + 4)
        initial_vectors = []
        if initial_field is not None:
            initial_vectors.append(self.field_guess(initial_field, basis))
        initial_vectors.extend(lowest_diagonal_guesses(basis.kinetic, block_size, basis.mp))
        theta, ritz, _ = lobpcg_solver(
            matvec=matvec,
            diagonal=basis.kinetic,
            numeig=numeig,
            mp=basis.mp,
            tol=tol,
            maxiter=maxiter,
            max_subspace=max_subspace,
            initial_vectors=initial_vectors,
            diagonal_shift=self.average_potential(),
            damping=damping,
            block_size=block_size,
            matvec_block=matvec_block,
        )
        return theta, ritz

    @timer("Diagonalize MPI")
    def diagonalize(
        self,
        numeig: int,
        return_eigenvectors: bool = True,
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
        """Diagonalize in a distributed reduced reciprocal basis."""
        reci_grid = self.grid.get_reciprocal()
        basis = DistributedReducedReciprocalBasis(reci_grid, ecut)

        if basis.npw <= numeig:
            raise ValueError("ecut is too small for the requested numeig.")

        eigenvalues, coeffs = self.lobpcg(
            basis,
            numeig=numeig,
            sigma=sigma,
            tol=tol,
            maxiter=maxiter,
            max_subspace=max_subspace,
            initial_field=initial_field,
            damping=damping,
            block_size=block_size,
            matvec_block_size=matvec_block_size,
        )

        if not return_eigenvectors:
            return eigenvalues,

        psi_list = []
        for coeff_local in coeffs:
            psi = ReciprocalField(
                reci_grid,
                rank=1,
                griddata_3d=basis.expand(coeff_local),
            ).ifft(force_real=False)
            norm = np.sqrt((np.real(psi) * np.real(psi) + np.imag(psi) * np.imag(psi)).integral())
            psi_list.append(psi / norm)

        return eigenvalues, psi_list
