from typing import Tuple

import numpy as np
from scipy.sparse.linalg import LinearOperator, eigsh

from dftpy.constants import SPEED_OF_LIGHT
from dftpy.field import DirectField, ReciprocalField
from dftpy.grid import DirectGrid
from dftpy.td.operator import Operator
from dftpy.time_data import timer


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
        v : DirectField
            Effective scalar potential
        A : np.ndarray
            Vector potential which is constant in space
        full : bool
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

    @A.setter
    def A(self, new_A):
        if new_A is None:
            self._A = None
        else:
            if np.size(new_A) != 3:
                raise AttributeError('Size of the A must be 3.')
            self._A = np.asarray(new_A)
            ones = np.ones(self.grid.nr)
            self.a_field = DirectField(
                self.grid,
                rank=3,
                griddata_3d=np.asarray(
                    [self._A[0] * ones, self._A[1] * ones, self._A[2] * ones]
                )
                / SPEED_OF_LIGHT,
            )

    def __call__(self, psi, force_real=None, sigma=0.025):
        """
        Performs the Hamiltonian operation on psi.

        Parameters
        ----------
        psi : DirectField or ReciprocalField
            The wavefunction
        force_real : None or bool
            Determines the force_real flag for inverse FFT. If set to None, the flag will be determined base on
            whether psi is real or complex.
        sigma : float
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
                return (
                    -0.5 * psi.laplacian(force_real=force_real, sigma=sigma)
                    + self.v * psi
                )
            else:
                psi_fft = psi.fft()
                intermediate_result1 = 0.5 * psi_fft.grid.gg * psi_fft
                intermediate_result1 *= np.exp(-psi_fft.grid.gg * sigma * sigma / 4.0)
                intermediate_result2 = -psi_fft.grid.g * psi_fft
                intermediate_result2 *= np.exp(-psi_fft.grid.gg * sigma * sigma / 4.0)
                return (
                    intermediate_result1.ifft(force_real=force_real)
                    + self.a_field.dot(intermediate_result2.ifft(force_real=force_real))
                    + 0.5 * self.a_field.dot(self.a_field) * psi
                    + self.v * psi
                )
                # return -0.5 * psi.laplacian(force_real = force_real, sigma=sigma) + 0.5 * self.a_field.dot(self.a_field) * psi + 1.0j * self.a_field.dot(psi.gradient(flag = "supersmooth", force_real = force_real, sigma=sigma)) + self.v * psi
                # return -0.5 * psi.laplacian(force_real = force_real, sigma=sigma) + 0.5 * self.a_field.dot(self.a_field) * psi + 0.5j * self.a_field.dot(psi.gradient(force_real = force_real)) + 0.5j * (self.a_field * psi).divergence(force_real=force_real) + self.v * psi

        elif isinstance(psi, ReciprocalField):
            return 0.5 * psi.grid.gg * psi + (self.v * psi.ifft()).fft()
        else:
            raise TypeError("psi must be a DFTpy DirectField or ReciprocalField.")

    @timer('Diagonalize')
    def diagonalize(
        self, numeig: int, return_eigenvectors: bool = True, reciprocal: bool = False
    ) -> Tuple:
        """
        Diagonalize the Hamiltonian and returns the lowest eigenvalues and optionally eigenvectors

        Parameters
        ----------
        numeig : int
            Number of eigenvalues to return.
        return_eigenvectors : bool
            Determine whether the eigenvectors will be returned.
        reciprocal : bool
            Determine the eigenvectors are calculated in real or reciprocal space.

        Returns
        -------
        Tuple (eigenvalue_list, ) or (eigenvalue_list, psi_list)
        eigenvalue_list: List[int]
            The list of eigenvalues
        psi_list: List[DirectField or ReciprocalField]
            The list of eigenvectors

        """

        if reciprocal:
            reci_grid = self.grid.get_reciprocal()
            size = reci_grid.nnr
            dtype = np.complex128
        else:
            size = self.grid.nnr
            dtype = np.float64

        A = LinearOperator(
            (size, size),
            dtype=dtype,
            matvec=self.scipy_matvec_utils(reciprocal=reciprocal),
        )

        if return_eigenvectors:
            eigenvalue_list, psis = eigsh(
                A, k=numeig, which='SA', return_eigenvectors=return_eigenvectors
            )
            psi_list = []
            for i in range(numeig):
                if reciprocal:
                    psi = ReciprocalField(
                        reci_grid,
                        rank=1,
                        griddata_3d=np.reshape(psis[:, i], reci_grid.nr),
                    )
                    psi = psi.ifft()
                else:
                    psi = DirectField(
                        self.grid,
                        rank=1,
                        griddata_3d=np.reshape(psis[:, i], self.grid.nr),
                    )
                psi = psi / np.sqrt(
                    (
                        np.real(psi) * np.real(psi) + np.imag(psi) * np.imag(psi)
                    ).integral()
                )
                psi_list.append(psi)
            return eigenvalue_list, psi_list
        else:
            eigenvalue_list, psis = eigsh(
                A, k=numeig, which='SA', return_eigenvectors=return_eigenvectors
            )
            return (eigenvalue_list,)

    def energy(self, psi):
        """
        Calculate the expectation value of Hamiltonian on wavefunction psi

        Parameters
        ----------
        psi : DirectField or ReciprocalField
            The wavefunction

        Returns
        -------
        energy: DirectField or ReciprocalField

        """
        return np.real(np.conj(psi) * self(psi)).integral()
