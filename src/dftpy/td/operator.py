from abc import ABC, abstractmethod
from typing import Callable

import numpy as np

from dftpy.field import BaseField, DirectField, ReciprocalField
from dftpy.grid import DirectGrid


class Operator(ABC):
    """
    Abstract class for operators which act on a DirectField or a ReciprocalField
    """

    def __init__(self, grid: DirectGrid) -> None:
        """
        Parameters
        ----------
        grid : DirectGrid
            the grid the operator acts on

        """
        self.grid = grid

    @abstractmethod
    def __call__(self, psi: BaseField) -> BaseField:
        """
        Performs the operation on a wavefunction, needs to be implemented by subclasses

        Parameters
        ----------
        psi : DirectField or ReciprocalField
            the wavefunction the operator acts on

        Returns
        -------
        DirectField or ReciprocalField, same as psi
            The resulting wavefunction

        """
        pass

    def scipy_matvec_utils(self, reciprocal: bool = False) -> Callable:
        """
        Utility function that generates a matvec function for SciPy to perform the operation on psi.ravel()

        Parameters
        ----------
        reciprocal : bool
            whether the input field is a DirectField or a ReciprocalField

        Returns
        -------
        _scipy_matvec: Callable
            the function that can be passed as the matvec function for SciPy LinearOperator.

        """
        if reciprocal:
            reci_grid = self.grid.get_reciprocal()

        def _scipy_matvec(psi_: np.ndarray) -> np.ndarray:
            if reciprocal:
                psi = ReciprocalField(
                    reci_grid, rank=1, griddata_3d=np.reshape(psi_, reci_grid.nr)
                )
            else:
                psi = DirectField(
                    self.grid, rank=1, griddata_3d=np.reshape(psi_, self.grid.nr)
                )
            prod = self(psi)
            return prod.ravel()

        return _scipy_matvec
