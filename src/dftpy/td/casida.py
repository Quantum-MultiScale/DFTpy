import numpy as np
from scipy.linalg import eigh
from scipy.sparse.linalg import LinearOperator, eigsh

from dftpy.field import DirectField
from dftpy.functional import Functional
from dftpy.time_data import timer


class Casida(object):

    def __init__(self, rho0, E_v_Evaluator):
        if not isinstance(rho0, DirectField):
            raise TypeError("rho0 must be an rank-1 or -2 DFTpy DirectField.")
        if rho0.rank == 2:
            self.rho0 = rho0
            self.polarized = True
        elif rho0.rank == 1:
            self.rho0 = DirectField(rho0.grid, rank=2, griddata_3d=np.stack([rho0 / 2, rho0 / 2], axis=0))
            self.polarized = False
        else:
            raise AttributeError("rho0 must be an rank-1 or -2 DFTpy DirectField.")
        self.N = rho0.integral()
        self.grid = rho0.grid
        self.functional = E_v_Evaluator
        # For spin-unpolarized GS, rho is doubled into (rho/2, rho/2). Evaluating XC at
        # rank-2 density forces the LibXC LDA path for V2. Use the total density
        # (rank 1) instead so built-in LDA can supply v2rho2 without pylibxc.
        if self.polarized:
            self.fkxc = self.functional(self.rho0, calcType=['V2']).v2rho2
        else:
            rho_tot = self.rho0[0] + self.rho0[1]
            v2 = self.functional(rho_tot, calcType=['V2']).v2rho2
            if v2.rank == 1:
                self.fkxc = DirectField(
                    self.grid,
                    rank=3,
                    griddata_3d=np.stack([v2, v2, v2], axis=0),
                )
            else:
                self.fkxc = v2
        self.fkxc_singlet = DirectField(
            self.grid,
            rank=1,
            griddata_3d=(self.fkxc[0] + self.fkxc[1]) / 2.0,
        )
        self.fkxc_triplet = DirectField(
            self.grid,
            rank=1,
            griddata_3d=(self.fkxc[0] - self.fkxc[1]) / 2.0,
        )
        self.fkxc_singlet_r = np.asarray(self.fkxc_singlet).ravel()
        self.fkxc_triplet_r = np.asarray(self.fkxc_triplet).ravel()

    def calc_k(self, psi_i, psi_j, vh, scale=1.0):
        if not self.polarized:
            return (psi_i * (scale * vh + self.fkxc_singlet * psi_j)).integral()
        else:
            raise Exception('Spin polarized Casida is not implemented')

    def calc_k_tri(self, psi_i, psi_j):
        if not self.polarized:
            return (psi_i * self.fkxc_triplet * psi_j).integral()
        else:
            raise Exception('Spin polarized Casida is not implemented')

    def calc_mu(self, psi_1, psi_2, direc=0):
        x = psi_1.grid.r[direc]
        mu = (psi_1 * x * psi_2).integral()
        return mu

    def calc_tdensities_and_tcoeff(self, mode, psi_list, eigs,
                                   x_minus_y_list):
        """Return the transition density and normalized orbital weights."""
        del eigs  # Kept in the public signature for backwards compatibility.
        transition_density = psi_list[0].copy()
        transition_density[:] = 0.0
        weights = []
        for i, psi in enumerate(psi_list[1:]):
            coeff = x_minus_y_list[int(mode)][i]
            transition_density += self.N * coeff * psi_list[0] * np.conj(psi)
            weights.append(abs(coeff) ** 2)

        weights = np.asarray(weights, dtype=np.float64)
        total = np.sum(weights)
        if total == 0.0:
            raise ValueError("Transition coefficients cannot all be zero.")
        weights /= total
        return transition_density, weights

    @timer('Casida Matrix')
    def build_matrix(self, num_psi, eigs, psi_list, scale=1.0,
                     calc_triplet=False, build_ab=False, matrix_free=False):
        """Build the Casida matrix, or retain data for matrix-free products."""
        if self.polarized:
            raise NotImplementedError('Spin polarized Casida is not implemented')
        if num_psi != len(psi_list) or num_psi != len(eigs):
            raise ValueError("num_psi must match eigs and psi_list.")

        num_modes = num_psi - 1
        omega = eigs[1:] - eigs[0]
        if np.any(omega <= 0.0):
            raise ValueError("Excited-state eigenvalues must exceed the ground state.")

        grid = self.grid
        P = np.empty((num_modes, grid.nnr), dtype=np.float64)
        for i in range(num_modes):
            P[i] = np.asarray(psi_list[0] * psi_list[i + 1]).real.ravel()

        self.omega = omega
        self.sqrtomega = np.sqrt(omega)
        self.scale = scale
        positions = np.asarray(grid.r).reshape(3, -1)
        mu = grid.mp.vsum(P @ positions.T * grid.dV)
        self.x, self.y, self.z = mu[:, 0], mu[:, 1], mu[:, 2]

        if matrix_free:
            if num_modes < 2:
                raise ValueError("Matrix-free Casida requires at least two modes.")
            self.P = P
            self.matrix_free = True
            for attr in ('k', 'c', 'a', 'b', 'c_tri', 'a_tri',
                         'b_tri', 'k_tri'):
                if hasattr(self, attr):
                    delattr(self, attr)
            return

        self.matrix_free = False
        hartree = Functional(type='HARTREE')
        W = np.empty_like(P)
        for j in range(num_modes):
            pj = DirectField(grid, rank=1, griddata_3d=P[j].reshape(grid.nr))
            vh = hartree(pj, calcType=['V']).potential
            W[j] = (scale * np.asarray(vh).real.ravel()
                    + self.fkxc_singlet_r * P[j])

        k = grid.mp.vsum(P @ W.T * grid.dV)
        self.k = 0.5 * (k + k.T)
        sw = self.sqrtomega
        self.c = (2.0 * self.N * self.k * np.outer(sw, sw)
                  + np.diag(omega * omega))
        if build_ab:
            self.a = self.N * self.k + np.diag(omega)
            self.b = self.N * self.k
        if calc_triplet:
            k_tri = grid.mp.vsum((P * self.fkxc_triplet_r) @ P.T * grid.dV)
            self.k_tri = 0.5 * (k_tri + k_tri.T)
            self.c_tri = (2.0 * self.N * self.k_tri * np.outer(sw, sw)
                          + np.diag(omega * omega))
            if build_ab:
                self.a_tri = self.N * self.k_tri + np.diag(omega)
                self.b_tri = self.N * self.k_tri

    @timer('Casida')
    def __call__(self, calc_triplet=False, k_modes=None):
        if getattr(self, 'matrix_free', False):
            if calc_triplet:
                raise NotImplementedError(
                    'Matrix-free Casida does not support triplet modes.'
                )
            P = self.P
            grid = self.grid
            hartree = Functional(type='HARTREE')
            vh_P = np.empty_like(P)
            for j in range(P.shape[0]):
                pj = DirectField(
                    grid, rank=1, griddata_3d=P[j].reshape(grid.nr)
                )
                vh_P[j] = np.asarray(
                    hartree(pj, calcType=['V']).potential
                ).real.ravel()

            sw = self.sqrtomega

            def matvec(vector):
                weighted = sw * vector
                rho = weighted @ P
                vh = weighted @ vh_P
                kv = grid.mp.vsum(
                    P @ ((self.scale * vh + self.fkxc_singlet_r * rho)
                         * grid.dV)
                )
                return 2.0 * self.N * sw * kv + self.omega**2 * vector

            n = P.shape[0]
            k = n - 1 if k_modes is None else int(k_modes)
            if not 1 <= k < n:
                raise ValueError("Matrix-free k_modes must satisfy 1 <= k_modes < num_modes.")
            omega2, z_list = eigsh(
                LinearOperator((n, n), dtype=np.float64, matvec=matvec),
                k=k,
                which='SA',
            )
            order = np.argsort(omega2)
            omega2, z_list = omega2[order], z_list[:, order]
            return self._casida_results(omega2, z_list)

        if not hasattr(self, 'c'):
            raise Exception("Matrix is not built yet. Run build_matrix first.")
        if calc_triplet and not hasattr(self, 'c_tri'):
            raise Exception("Matrix for triplet is not built yet. Run build_matrix with calc_triplet=True first.")
        omega2, z_list = eigh(self.c)
        if k_modes is not None:
            k = int(k_modes)
            if not 1 <= k <= len(omega2):
                raise ValueError("k_modes must satisfy 1 <= k_modes <= num_modes.")
            omega2, z_list = omega2[:k], z_list[:, :k]
        omega, f, x_minus_y_list = self._casida_results(omega2, z_list)

        if calc_triplet:
            omega2_tri, z_list_tri = eigh(self.c_tri)
            omega_tri, f_tri, xy_tri = self._casida_results(
                omega2_tri, z_list_tri
            )
            omega = np.concatenate((omega, omega_tri))
            f = np.concatenate((f, f_tri))
            x_minus_y_list += xy_tri

        return omega, f, x_minus_y_list

    def _casida_results(self, omega2, z_list):
        omega = np.real(np.sqrt(omega2))
        mu = np.vstack((self.x, self.y, self.z))
        dipoles = mu @ (self.sqrtomega[:, None] * z_list)
        strengths = 2.0 / 3.0 * self.N * np.sum(dipoles**2, axis=0)
        x_minus_y = (z_list / self.sqrtomega[:, None]).T.tolist()
        return omega, strengths, x_minus_y

    def tda(self, calc_triplet=False):
        if not hasattr(self, 'a'):
            raise Exception("Matrix is not built yet. Run build_matrix with build_ab=True first.")
        if calc_triplet and not hasattr(self, 'a_tri'):
            raise Exception("Matrix for triplet is not built yet. Run build_matrix with calc_triplet=True first.")
        omega, x_list = eigh(self.a)
        mu = np.vstack((self.x, self.y, self.z))
        dipoles = mu @ (self.a @ x_list)
        f = 2.0 / 3.0 * self.N * np.sum(dipoles**2, axis=0)

        if calc_triplet:
            omega_tri, x_list_tri = eigh(self.a_tri)
            dipoles_tri = mu @ (self.a_tri @ x_list_tri)
            f_tri = 2.0 / 3.0 * self.N * np.sum(dipoles_tri**2, axis=0)
            omega = np.real(np.concatenate((omega, omega_tri)))
            f = np.concatenate((f, f_tri))

        return omega, f
