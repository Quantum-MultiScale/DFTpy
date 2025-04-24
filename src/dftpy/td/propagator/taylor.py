from typing import Tuple, Union

import numpy as np

from dftpy.field import DirectField, ReciprocalField
from dftpy.mpi.utils import sprint
from dftpy.td.hamiltonian import Hamiltonian
from dftpy.td.operator import Operator
from dftpy.td.propagator.abstract_propagator import AbstractPropagator
from dftpy.time_data import timer


class Taylor(AbstractPropagator):
    """
    Taylor propagator for real-time propagation
    """

    def __init__(
        self, hamiltonian: Operator, interval: float, order: int = 1, **kwargs
    ) -> None:
        """
        Parameters
        ----------
        hamiltonian : Hamiltonian
            the time-dependent Hamiltonian
        interval : float
            the time interval for one time step
        order : int
            the order of Taylor expansion

        """
        super(Taylor, self).__init__(hamiltonian, interval)
        self.order = order

    @timer('Taylor-Propagator')
    def __call__(self, psi0: Union[DirectField, ReciprocalField]) -> Tuple:
        """
        Perform one step of propagation.

        Parameters
        ----------
        psi0 : DirectField or ReciprocalField
            the initial wavefunction.

        Returns
        -------
        A tuple (psi1, status)
        psi1: DirectField or ReciprocalField, same as psi0
            the final wavefunction.
        status: int
            0: no issue, 1: has NaN issue

        """
        n_elec_0 = (psi0 * np.conj(psi0)).integral()
        psi1 = psi0

        new_psi = psi0
        for i_order in range(self.order):
            new_psi = -1j * self.interval / (i_order + 1) * self.hamiltonian(new_psi)
            if np.isnan(new_psi).any():
                sprint(
                    "Warning: taylor propagator exits on order {0:d} due to NaN in new psi.".format(
                        i_order
                    )
                )
                psi1 = psi1 + new_psi
                return psi1, 1
            psi1 = psi1 + new_psi

        n_elec_1 = (psi1 * np.conj(psi1)).integral()
        psi1 = psi1 * np.sqrt(n_elec_0 / n_elec_1)

        return psi1, 0
