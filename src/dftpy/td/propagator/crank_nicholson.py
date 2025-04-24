from typing import Tuple, Union

import numpy as np
import scipy.sparse.linalg as linalg
from scipy.sparse.linalg import LinearOperator

import dftpy.linear_solver
from dftpy.field import BaseField, DirectField, ReciprocalField
from dftpy.mpi.utils import sprint
from dftpy.td.hamiltonian import Hamiltonian
from dftpy.td.operator import Operator
from dftpy.td.propagator.abstract_propagator import AbstractPropagator
from dftpy.time_data import timer


class CrankNicholsonOperator(Operator):
    """
    Operator that performs (1+i*\hat(H)*dt/2)
    """

    def __init__(self, hamiltonian: Hamiltonian, interval: float) -> None:
        """
        Parameters
        ----------
        hamiltonian : Hamiltonian
            the time-dependent Hamiltonian
        interval : float
            the time interval for one time step
        """
        super(CrankNicholsonOperator, self).__init__(hamiltonian.grid)
        self.hamiltonian = hamiltonian
        self.interval = interval

    @property
    def hamiltonian(self) -> Hamiltonian:
        return self._hamiltonian

    @hamiltonian.setter
    def hamiltonian(self, hamiltonian: Hamiltonian) -> None:
        self.grid = hamiltonian.grid
        self._hamiltonian = hamiltonian

    def __call__(self, psi: BaseField) -> BaseField:
        """
        Parameters
        ----------
        psi : DirectField or ReciprocalField
            The wavefunction the operator acts on
        DirectField or ReciprocalField, same as psi
            The resulting wavefunction
        """
        return psi + 1j * self.hamiltonian(psi) * self.interval / 2.0


class CrankNicholson(AbstractPropagator):
    """
        Crank-Nicholson propagator for real-time propagation
        Eq. (4.23) of C. A. Ullrich, Time-dependent density-functional theory: concepts and applications (OUP Oxford,
    2011)
    """

    LinearSolverDict = {
        "bicg_scipy": {"func": linalg.bicg, "scipy": True},
        "bicgstab_scipy": {"func": linalg.bicgstab, "scipy": True},
        "cg_scipy": {"func": linalg.cg, "scipy": True},
        "cgs_scipy": {"func": linalg.cgs, "scipy": True},
        "gmres_scipy": {"func": linalg.gmres, "scipy": True},
        "lgmres_scipy": {"func": linalg.lgmres, "scipy": True},
        "minres_scipy": {"func": linalg.minres, "scipy": True},
        "qmr_scipy": {"func": linalg.qmr, "scipy": True},
        "bicg": {"func": dftpy.linear_solver.bicg, "scipy": False},
        "bicgstab": {"func": dftpy.linear_solver.bicgstab, "scipy": False},
        "cg": {"func": dftpy.linear_solver.cg, "scipy": False},
    }

    def __init__(
        self,
        hamiltonian: Hamiltonian,
        interval: float,
        linearsolver: str = "cg",
        tol: float = 1e-8,
        maxiter: int = 100,
        atol: Union[float, None] = None,
        **kwargs,
    ) -> None:
        """
        Parameters
        ----------
        hamiltonian : Hamiltionian
            the time-dependent Hamiltonian
        interval : float
            the time interval for one time step
        linearsolver : str
            the name of the linear solver to solve the Ax=b problem. Options:
            "bicg_scipy", "bicgstab_scipy", "cg_scipy", "cgs_scipy", "gmres_scipy", "lgmres_scipy", "minres_scipy",
            "qmr_scipy", "bicg", "bicgstab", "cg"
            Note: Scipy solvers only works for the serial version
        tol : float
            the tolerance of the linear solver
        maxiter : int
            the max number of iterations of the linear solver
        atol : float
            the absolute tolerance of the linear solver

        """
        super(CrankNicholson, self).__init__(hamiltonian, interval)
        self._a = CrankNicholsonOperator(self._hamiltonian, self._interval)
        self.linear_solver = linearsolver
        self.tol = tol
        self.maxiter = maxiter
        self.atol = atol

    @property
    def hamiltonian(self) -> Hamiltonian:
        return self._hamiltonian

    @hamiltonian.setter
    def hamiltonian(self, hamiltonian: Hamiltonian) -> None:
        self._hamiltonian = hamiltonian
        self._a = CrankNicholsonOperator(self._hamiltonian, self._interval)

    @property
    def interval(self) -> float:
        return self._interval

    @interval.setter
    def interval(self, interval: float) -> None:
        self._interval = interval
        self._a = CrankNicholsonOperator(self._hamiltonian, self._interval)

    @property
    def linear_solver(self) -> str:
        return self._linear_solver_name

    @linear_solver.setter
    def linear_solver(self, linear_solver: str) -> None:
        """
        Set the linear solver

        Parameters
        ----------
        linear_solver : the name of the linear solver to solve the Ax=b problem

        """
        if linear_solver not in self.LinearSolverDict:
            raise AttributeError("{0:s} is not a linear solver".format(linear_solver))
        self._linear_solver_name = linear_solver
        self._linear_solver = self.LinearSolverDict[linear_solver]['func']
        self._scipy = self.LinearSolverDict[linear_solver]['scipy']

    def _calc_b(
        self, psi: Union[DirectField, ReciprocalField]
    ) -> Union[DirectField, ReciprocalField]:
        return psi - 1j * self.hamiltonian(psi) * self.interval / 2.0

    @timer('Crank-Nicholson-Propagator')
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
            0: converged, others: not converged

        """

        if self._scipy:
            reciprocal = isinstance(psi0, ReciprocalField)
            size = self.hamiltonian.grid.nnr
            b = self._calc_b(psi0).ravel()
            mat = LinearOperator(
                shape=(size, size),
                dtype=psi0.dtype,
                matvec=self._a.scipy_matvec_utils(reciprocal=reciprocal),
            )
            psi1_, status = self._linear_solver(
                mat,
                b,
                x0=psi0.ravel(),
                tol=self.tol,
                maxiter=self.maxiter,
                atol=self.atol,
            )
            psi1 = DirectField(
                grid=self.hamiltonian.grid,
                rank=1,
                griddata_3d=np.reshape(psi1_, self.hamiltonian.grid.nr),
                cplx=True,
            )
        else:
            b = self._calc_b(psi0)
            psi1, status = self._linear_solver(
                self._a, b, psi0, self.tol, self.maxiter, atol=self.atol, mp=psi0.mp
            )

        if status:
            sprint("Linear solver did not converge. Info: {0:d}".format(status))

        return psi1, status
