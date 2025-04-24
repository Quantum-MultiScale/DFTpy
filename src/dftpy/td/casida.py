import numpy as np
from scipy.linalg import eigh

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
            self.rho0 = DirectField(
                rho0.grid, rank=2, griddata_3d=np.stack([rho0 / 2, rho0 / 2], axis=0)
            )
            self.polarized = False
        else:
            raise AttributeError("rho0 must be an rank-1 or -2 DFTpy DirectField.")
        self.N = rho0.integral()
        self.grid = rho0.grid
        self.functional = E_v_Evaluator
        self.fkxc = self.functional(self.rho0, calcType=['V2']).v2rho2

    def calc_k(self, psi_i, psi_j, vh):
        if not self.polarized:
            fkxc = DirectField(
                self.grid, rank=1, griddata_3d=(self.fkxc[0] + self.fkxc[1]) / 2.0
            )
            return (psi_i * (vh + fkxc * psi_j)).integral()
        else:
            raise Exception('Spin polarized Casida is not implemented')

    def calc_k_tri(self, psi_i, psi_j):
        if not self.polarized:
            fkxc = DirectField(
                self.grid, rank=1, griddata_3d=(self.fkxc[0] - self.fkxc[1]) / 2.0
            )
            return (psi_i * fkxc * psi_j).integral()
        else:
            raise Exception('Spin polarized Casida is not implemented')

    def calc_mu(self, psi_1, psi_2, direc=0):
        x = psi_1.grid.r[direc]
        mu = (psi_1 * x * psi_2).integral()
        return mu

    @timer('Casida Matrix')
    def build_matrix(self, num_psi, eigs, psi_list, calc_triplet=False, build_ab=False):
        hartree = Functional(type='HARTREE')
        self.c = np.empty([num_psi - 1, num_psi - 1], dtype=np.float64)
        if build_ab:
            self.a = np.empty([num_psi - 1, num_psi - 1], dtype=np.float64)
            self.b = np.empty([num_psi - 1, num_psi - 1], dtype=np.float64)
        if calc_triplet:
            self.c_tri = np.empty([num_psi - 1, num_psi - 1], dtype=np.float64)
            if build_ab:
                self.a_tri = np.empty([num_psi - 1, num_psi - 1], dtype=np.float64)
                self.b_tri = np.empty([num_psi - 1, num_psi - 1], dtype=np.float64)
        omega = eigs[1:] - eigs[0]
        for j in range(1, num_psi):
            psi_j = psi_list[0] * psi_list[j]
            vh = hartree(psi_j, calcType=['V']).potential
            for i in range(j, num_psi):
                psi_i = psi_list[0] * psi_list[i]
                k = self.calc_k(psi_i, psi_j, vh)
                self.c[i - 1, j - 1] = (
                    k * self.N * 2.0 * np.sqrt(omega[i - 1] * omega[j - 1])
                )
                if build_ab:
                    self.a[i - 1, j - 1] = k * self.N
                    self.b[i - 1, j - 1] = k * self.N
                if calc_triplet:
                    k_tri = self.calc_k_tri(psi_i, psi_j)
                    self.c_tri[i - 1, j - 1] = (
                        k_tri * self.N * 2.0 * np.sqrt(omega[i - 1] * omega[j - 1])
                    )
                    if build_ab:
                        self.a_tri[i - 1, j - 1] = k_tri * self.N
                        self.b_tri[i - 1, j - 1] = k_tri * self.N
                if not i == j:
                    self.c[j - 1, i - 1] = self.c[i - 1, j - 1]
                    if build_ab:
                        self.a[j - 1, i - 1] = self.a[i - 1, j - 1]
                        self.b[j - 1, i - 1] = self.b[i - 1, j - 1]
                    if calc_triplet:
                        self.c_tri[j - 1, i - 1] = self.c_tri[i - 1, j - 1]
                        if build_ab:
                            self.a_tri[j - 1, i - 1] = self.a_tri[i - 1, j - 1]
                            self.b_tri[j - 1, i - 1] = self.b_tri[i - 1, j - 1]

        self.c += np.identity(num_psi - 1) * (omega * omega)
        if build_ab:
            self.a += np.identity(num_psi - 1) * omega
        if calc_triplet:
            self.c_tri += np.identity(num_psi - 1) * (omega * omega)
            if build_ab:
                self.a_tri += np.identity(num_psi - 1) * omega
        self.sqrtomega = np.sqrt(omega)
        # sprint(self.sqrtomega)
        # sprint(self.c)
        # sprint(self.c_tri)

        self.x = np.empty(num_psi - 1, dtype=np.float64)
        self.y = np.empty(num_psi - 1, dtype=np.float64)
        self.z = np.empty(num_psi - 1, dtype=np.float64)
        for i in range(1, num_psi):
            self.x[i - 1] = self.calc_mu(psi_list[0], psi_list[i])
            self.y[i - 1] = self.calc_mu(psi_list[0], psi_list[i], direc=1)
            self.z[i - 1] = self.calc_mu(psi_list[0], psi_list[i], direc=2)

    @timer('Casida')
    def __call__(self, calc_triplet=False):
        if not hasattr(self, 'c'):
            raise Exception("Matrix is not built yet. Run build_matrix first.")
        if calc_triplet and not hasattr(self, 'c_tri'):
            raise Exception(
                "Matrix for triplet is not built yet. Run build_matrix with calc_triplet=True first."
            )
        num_modes = np.shape(self.c)[0]

        omega2, z_list = eigh(self.c)
        omega = np.sqrt(omega2)
        omega = np.real(omega)

        x_minus_y_list = []

        f = np.empty(num_modes, dtype=np.float64)
        for i in range(num_modes):
            tmp = np.sum(self.x * self.sqrtomega * z_list[:, i])
            f[i] = tmp * tmp
            tmp = np.sum(self.y * self.sqrtomega * z_list[:, i])
            f[i] += tmp * tmp
            tmp = np.sum(self.z * self.sqrtomega * z_list[:, i])
            f[i] += tmp * tmp

            f[i] = f[i] * 2.0 / 3.0 * self.N

            x_minus_y_list.append(z_list[:, i] / self.sqrtomega)

        if calc_triplet:
            omega2, z_list = eigh(self.c_tri)
            omega_tri = np.sqrt(omega2)
            omega_tri = np.real(omega_tri)

            f_tri = np.empty(num_modes, dtype=np.float64)
            for i in range(num_modes):
                tmp = np.sum(self.x * self.sqrtomega * z_list[:, i])
                f_tri[i] = tmp * tmp
                tmp = np.sum(self.y * self.sqrtomega * z_list[:, i])
                f_tri[i] += tmp * tmp
                tmp = np.sum(self.z * self.sqrtomega * z_list[:, i])
                f_tri[i] += tmp * tmp

                f_tri[i] = f_tri[i] * 2.0 / 3.0 * self.N

                x_minus_y_list.append(z_list[:, i] / self.sqrtomega)

            omega = np.real(np.concatenate((omega, omega_tri)))
            f = np.concatenate((f, f_tri))

        return omega, f, x_minus_y_list

    def tda(self, calc_triplet=False):
        if not hasattr(self, 'a'):
            raise Exception("Matrix is not built yet. Run build_matrix first.")
        if calc_triplet and not hasattr(self, 'a_tri'):
            raise Exception(
                "Matrix for triplet is not built yet. Run build_matrix with calc_triplet=True first."
            )
        num_modes = np.shape(self.a)[0]

        omega, x_list = eigh(self.a)

        f = np.empty(num_modes, dtype=np.float64)
        for i in range(num_modes):
            # tmp = np.sum(self.x * self.sqrtomega * self.sqrtomega * x_list[:,i])
            tmp = np.sum(self.x * np.matmul(self.a, x_list[:, i]))
            f[i] = tmp * tmp
            # tmp = np.sum(self.y * self.sqrtomega * self.sqrtomega * x_list[:,i])
            tmp = np.sum(self.y * np.matmul(self.a, x_list[:, i]))
            f[i] += tmp * tmp
            # tmp = np.sum(self.z * self.sqrtomega * self.sqrtomega * x_list[:,i])
            tmp = np.sum(self.z * np.matmul(self.a, x_list[:, i]))
            f[i] += tmp * tmp

            f[i] = f[i] * 2.0 / 3.0 * self.N

        if calc_triplet:
            omega_tri, x_list = eigh(self.a_tri)

            f_tri = np.empty(num_modes, dtype=np.float64)
            for i in range(num_modes):
                tmp = np.sum(self.x * self.sqrtomega * self.sqrtomega * x_list[:, i])
                f_tri[i] = tmp * tmp
                tmp = np.sum(self.y * self.sqrtomega * self.sqrtomega * x_list[:, i])
                f_tri[i] += tmp * tmp
                tmp = np.sum(self.z * self.sqrtomega * self.sqrtomega * x_list[:, i])
                f_tri[i] += tmp * tmp

                f_tri[i] = f_tri[i] * 2.0 / 3.0 * self.N

            omega = np.real(np.concatenate((omega, omega_tri)))
            f = np.concatenate((f, f_tri))

        return omega, f
