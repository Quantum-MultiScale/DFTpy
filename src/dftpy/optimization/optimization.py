from abc import ABC, abstractmethod
from functools import partial

import numpy as np

from dftpy.constants import ENERGY_CONV
from dftpy.field import DirectField
from dftpy.math_utils import (
    LBFGS,
    Brent,
    get_direction_CG,
    get_direction_GD,
    get_direction_LBFGS,
    line_search,
    minimize,
)
from dftpy.mpi import sprint
from dftpy.time_data import TimeData, timer


class AbstractOptimization(ABC):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def __call__(self, optimization_options={}):
        pass

    @abstractmethod
    def get_direction(self):
        pass

    @abstractmethod
    def optimize_rho(self):
        pass


class Optimization(AbstractOptimization):
    """
    Class handling electron density optimization.

    Attributes
    ---------
    optimization_method: string

    optimization_options: dict
            kwargs for the minim. method

    EnergyEvaluator: TotalEnergyAndPotential class


    guess_rho: DirectField, optional
            an initial guess for the electron density

     Example
     -------
     EE = TotalEnergyAndPotential(...)
     opt = Optimization(EnergyEvaluator=EE)
     new_rho = opt(guess_rho)
    """

    def __init__(
        self,
        optimization_method="CG-HS",
        optimization_options=None,
        EnergyEvaluator=None,
        guess_rho=None,
    ):

        self.rho = guess_rho

        if optimization_options is None:
            self.optimization_options = {}
        else:
            self.optimization_options = optimization_options

        default_options = {
            "maxcor": 6,
            "ftol": 1.0e-7,
            "xtol": 1.0e-12,
            "maxfun": 50,
            "maxiter": 200,
            "maxls": 30,
            "econv": 1.0e-5,
            "c1": 1e-4,
            "c2": 0.2,
            "algorithm": "EMM",
            "vector": "Orthogonalization",
            "ncheck": 2,
            "h0": 1.0,
        }
        for key in default_options:
            if key not in self.optimization_options:
                self.optimization_options[key] = default_options[key]

        if EnergyEvaluator is None:
            raise AttributeError("Must provide an energy evaluator")
        else:
            self.EnergyEvaluator = EnergyEvaluator

        self.optimization_method = optimization_method
        self.lphi = False
        self.comm = None

    def get_direction_TN(self, res0, phi=None, mu=None, density=None, spin=1, **kwargs):
        if self.nspin > 1:
            rho = density.copy()
        direction = np.zeros_like(res0)
        epsi = 1.0e-9 * self.mp.comm.size
        res = -res0.copy()
        p = res.copy()
        r0Norm = self.mp.einsum("ijk, ijk->", res, res)
        r1Norm = r0Norm
        rConv = r0Norm * 0.1
        stat = "NOTCONV"
        Best = direction
        rLists = [1e10]
        for it in range(self.optimization_options["maxfun"]):
            phi1 = phi + epsi * p
            rho1 = phi1 * phi1
            if self.nspin > 1:
                rho[spin] = rho1
                func = self.EnergyEvaluator(
                    rho, calcType={"V"}, phi=phi1, lphi=self.lphi
                )
                Ap = ((func.potential[spin] - mu) * phi1 - res0) / epsi
            else:
                func = self.EnergyEvaluator(
                    rho1, calcType={"V"}, phi=phi1, lphi=self.lphi
                )
                Ap = ((func.potential - mu) * phi1 - res0) / epsi
            pAp = self.mp.einsum("ijk, ijk->", p, Ap)
            if pAp < 0.0:
                if it == 0:
                    direction = r0Norm / pAp * p
                    stat = "WARN"
                else:
                    stat = "FAILED"
                    sprint(
                        "!WARN : pAp small than zero :iter = ", it, pAp, comm=self.comm
                    )
                break
            alpha = r0Norm / pAp
            direction += alpha * p
            res -= alpha * Ap
            r1Norm = self.mp.einsum("ijk, ijk->", res, res)
            # sprint('it', it, rConv, r1Norm, comm=self.comm)
            if r1Norm < min(rLists):
                Best = direction.copy()
            rLists.append(r1Norm)
            if r1Norm < rConv:
                stat = "CONV"
                break
            elif r1Norm > 1000 * min(rLists[:-1]):
                stat = "WARN : Not reduce"
                direction = Best
                break
            elif it > 10 and abs(r0Norm - r1Norm) < 0.1 * r0Norm:
                stat = "WARN : Change too small"
                direction = Best
                break
            beta = r1Norm / r0Norm
            r0Norm = r1Norm
            p = res + beta * p
        number = it + 1

        return direction, number

    def get_direction_LBFGS(
        self, resA, dirA=None, phi=None, method="CG-HS", lbfgs=None, mu=None, **kwargs
    ):
        number = 1
        direction = np.zeros_like(resA[-1])
        q = -resA[-1]
        alphaList = np.zeros(len(lbfgs.s))
        for i in range(len(lbfgs.s) - 1, 0, -1):
            alpha = lbfgs.rho[i] * self.mp.einsum("ijk, ijk->", lbfgs.s[i], q)
            alphaList[i] = alpha
            q -= alpha * lbfgs.y[i]

        if not lbfgs.H0:
            if len(lbfgs.s) < 1:
                gamma = 1.0
            else:
                gamma = self.mp.einsum(
                    "ijk, ijk->", lbfgs.s[-1], lbfgs.y[-1]
                ) / self.mp.einsum("ijk, ijk->", lbfgs.y[-1], lbfgs.y[-1])
            direction = gamma * q
        else:
            direction = lbfgs.H0 * q

        for i in range(len(lbfgs.s)):
            beta = lbfgs.rho[i] * self.mp.einsum("ijk->", lbfgs.y[i] * direction)
            direction += lbfgs.s[i] * (alphaList[i] - beta)

        return direction, number

    def get_direction_DIIS(self, resA, **kwargs):
        direction = -resA[-1]
        return direction

    def get_direction(
        self, resA, dirA=None, phi=None, method="CG-HS", lbfgs=None, mu=None
    ):
        number = 1
        if method[0:2] == "CG":
            direction = get_direction_CG(
                resA, dirA=dirA, phi=phi, method=method, lbfgs=lbfgs, mu=mu
            )
        elif method == "LBFGS":
            direction = get_direction_LBFGS(
                resA, dirA=dirA, phi=phi, method=method, lbfgs=lbfgs, mu=mu
            )
        elif method == "GD":
            direction = get_direction_GD(
                resA, dirA=dirA, phi=phi, method=method, lbfgs=lbfgs, mu=mu
            )
        elif method == "DIIS":
            direction = get_direction_GD(
                resA, dirA=dirA, phi=phi, method=method, lbfgs=lbfgs, mu=mu
            )
        elif method == "TN":
            if self.nspin > 1:
                density = phi * phi
                direction, number = self.get_direction_TN(
                    resA[-1][0], phi=phi[0], mu=mu[0], density=density, spin=0
                )
                for i in range(1, self.nspin):
                    d1, n1 = self.get_direction_TN(
                        resA[-1][i], phi=phi[i], mu=mu[i], density=density, spin=i
                    )
                    d1 = direction
                    direction = np.vstack((direction, d1))
                    number += n1
                direction = DirectField(
                    grid=self.rho.grid, griddata_3d=direction, rank=self.nspin
                )
            else:
                direction, number = self.get_direction_TN(resA[-1], phi=phi, mu=mu)
        else:
            raise AttributeError("The %s direction method not implemented." % method)
        return direction, number

    def OrthogonalNormalization(self, p, phi, Ne=None, vector="Orthogonalization"):
        if vector == "Orthogonalization":
            if Ne is None:
                Ne = (phi * phi).integral()
            N = Ne
            # ptest = p + phi ; # N = (ptest * ptest).integral()
            factor = (p * phi).integral() / Ne
            if self.nspin > 1:
                factor = factor[:, None, None, None]
            p -= factor * phi
            pNorm = (p * p).integral()
            theta = np.sqrt(pNorm / N)

            factor = np.sqrt(Ne / pNorm)
            if self.nspin > 1:
                factor = factor[:, None, None, None]
            p *= factor
        else:
            theta = 0.01
            if self.nspin > 1:
                theta = np.ones(self.nspin) * theta
        return p, theta

    def ValueAndDerivative(
        self,
        phi,
        p,
        theta,
        Ne=None,
        algorithm="EMM",
        vector="Orthogonalization",
        func=None,
    ):
        if Ne is None:
            Ne = (phi * phi).integral()
        if self.nspin > 1:
            theta = theta[:, None, None, None]

        if vector == "Orthogonalization":
            newphi = phi * np.cos(theta) + p * np.sin(theta)
            newrho = newphi * newphi
        else:  # Scaling
            newphi = phi + p * theta
            newrho = newphi * newphi
            norm = Ne / newrho.integral()
            if self.nspin > 1:
                norm = norm[:, None, None, None]
            newrho *= norm
            newphi *= np.sqrt(norm)

        if func is not None:
            f = func

        if algorithm == "EMM":
            if func is None:
                f = self.EnergyEvaluator(
                    newrho, calcType={"E", "V"}, phi=newphi, lphi=self.lphi
                )
            value = f.energy
        else:  # RMM
            if func is None:
                f = self.EnergyEvaluator(
                    newrho, calcType={"E", "V"}, phi=newphi, lphi=self.lphi
                )
            # mu = (f.potential * newrho).integral() / Ne
            mu = self.get_chemical_potential(
                f.potential, newrho, phi=newphi, lphi=self.lphi
            )
            if self.nspin > 1:
                mu = mu[:, None, None, None]
            if algorithm == "RMM":
                residual = (f.potential - mu) * newphi
                try:
                    resN = (
                        self.mp.einsum(
                            "..., ...->", residual, residual, optimize='optimal'
                        )
                        * phi.grid.dV
                    )
                except Exception:
                    resN = self.mp.sum(residual * residual) * phi.grid.dV
                value = resN
            elif algorithm == "CMM":
                value = mu

        if vector == "Orthogonalization":
            p2 = p * np.cos(theta) - phi * np.sin(theta)
        else:
            p2 = p
        if self.nspin == 1:
            grad = (
                2.0
                * self.mp.einsum("ijk, ijk, ijk->", f.potential, newphi, p2)
                * phi.grid.dV
            )
        else:
            grad = (
                2.0
                * self.mp.einsum("lijk, lijk, lijk->l", f.potential, newphi, p2)
                * phi.grid.dV
            )

        # sprint('theta', np.asarray(theta).ravel(), value, grad, comm=self.comm)
        return [value, grad, newphi, f]

    @timer('Optimize')
    def optimize_rho(
        self, guess_rho=None, guess_phi=None, lphi=False, lsfun="line_search"
    ):
        if guess_rho is None and self.rho is None:
            raise AttributeError("Must provide a guess density")
        elif guess_rho is not None:
            self.rho = guess_rho
        rho = self.rho.copy()
        self.nspin = rho.rank
        converged = 1  # if >0 means not converged
        self.lphi = lphi
        # -----------------------------------------------------------------------
        xtol = self.optimization_options["xtol"]
        maxls = self.optimization_options["maxls"]
        c1 = self.optimization_options["c1"]
        c2 = self.optimization_options["c2"]
        theta = 0.1
        if self.nspin > 1:
            theta = np.ones(self.nspin) * theta
        # -----------------------------------------------------------------------
        self.mp = rho.grid.mp
        self.comm = self.mp.comm
        # -----------------------------------------------------------------------
        EnergyHistory = []
        if guess_phi is None:
            phi = rho.copy()
            mask = rho > 0
            mask2 = np.invert(mask)
            phi[mask] = np.sqrt(rho[mask])
            phi[mask2] = 1e-300
            factor = np.sqrt(self.mp.sum(rho) / self.mp.sum(phi * phi))
            phi *= factor
        else:
            phi = guess_phi.copy()
        rho[:] = phi * phi
        func = self.EnergyEvaluator(rho, calcType=['E', 'V'], phi=phi, lphi=self.lphi)
        # mu = (func.potential * rho).integral() / rho.N
        mu = self.get_chemical_potential(func.potential, rho, phi=phi, lphi=self.lphi)
        if self.nspin > 1:
            mus = mu[:, None, None, None]
        else:
            mus = mu
        residual = (func.potential - mus) * phi
        residualA = []
        residualA.append(residual)
        directionA = []
        energy = func.energy
        EnergyHistory.append(energy)

        CostTime = TimeData.Time("Optimize")

        fmt = "{:8s}{:24s}{:16s}{:16s}{:8s}{:8s}{:16s}".format(
            "Step", "Energy(a.u.)", "dE", "dP", "Nd", "Nls", "Time(s)"
        )
        sprint(fmt, comm=self.comm)
        dE = energy
        try:
            resN = (
                self.mp.einsum("..., ...->", residual, residual, optimize='optimal')
                * rho.grid.dV
            )
        except Exception:
            resN = float(self.mp.sum(residual * residual) * rho.grid.dV)
        fmt = "{:<8d}{:<24.12E}{:<16.6E}{:<16.6E}{:<8d}{:<8d}{:<16.6E}".format(
            0, energy, dE, resN, 1, 1, CostTime
        )
        sprint(fmt, comm=self.comm)
        Bound = self.optimization_options["maxcor"]

        if self.optimization_method == "LBFGS":
            lbfgs = LBFGS(H0=self.optimization_options["h0"], Bound=Bound)
        else:
            lbfgs = None

        if self.optimization_method == "DIIS":
            self.optimization_options["vector"] = "Scaling"
            lsfun = 'DIIS'
        elif self.optimization_method == "Mixing":
            self.optimization_options["vector"] = "Scaling"
            lsfun = 'mixing'

        for it in range(1, self.optimization_options["maxiter"]):
            p, NumDirectrion = self.get_direction(
                residualA,
                directionA,
                phi=phi,
                method=self.optimization_method,
                lbfgs=lbfgs,
                mu=mu,
            )
            p, theta0 = self.OrthogonalNormalization(
                p, phi, vector=self.optimization_options["vector"]
            )

            if self.nspin > 1:
                lsfun = 'minimize'  # only minimize support nspin>1
                thetaDeriv0 = (
                    self.mp.einsum("lijk, lijk, lijk ->l", func.potential, phi, p) * 2.0
                )
                if any(thetaDeriv0 > 0):
                    gradf = True
                else:
                    gradf = False
            else:
                thetaDeriv0 = (
                    self.mp.einsum("ijk, ijk, ijk->", func.potential, phi, p) * 2.0
                )
                if thetaDeriv0 > 0:
                    gradf = True
                else:
                    gradf = False
            # -----------------------------------------------------------------------
            if gradf:
                sprint("!WARN: Change to steepest decent", comm=self.comm)
                p = -residualA[-1]
                p, theta0 = self.OrthogonalNormalization(
                    p, phi, vector=self.optimization_options["vector"]
                )

            if self.nspin == 1:
                theta = min(theta0, theta)

            fun_value_deriv = partial(
                self.ValueAndDerivative,
                phi,
                p,
                algorithm=self.optimization_options["algorithm"],
                vector=self.optimization_options["vector"],
            )

            if lsfun == "line_search":
                func0 = fun_value_deriv(0.0, func=func)
                theta, task, NumLineSearch, valuederiv = line_search(
                    fun_value_deriv,
                    alpha0=theta,
                    func0=func0,
                    c1=c1,
                    c2=c2,
                    amax=np.pi,
                    amin=0.0,
                    xtol=xtol,
                    maxiter=maxls,
                )
            elif lsfun == "minimize":
                if self.nspin > 1:
                    method = 'powell'
                    func0 = fun_value_deriv(np.zeros(self.nspin), func=func)
                else:
                    method = 'cg'
                    func0 = fun_value_deriv(0.0, func=func)
                theta, task, NumLineSearch, valuederiv = minimize(
                    fun_value_deriv,
                    alpha0=theta,
                    func0=func0,
                    c1=c1,
                    c2=c2,
                    amax=np.pi,
                    amin=0.0,
                    xtol=xtol,
                    maxiter=maxls,
                    method=method,
                )
            elif lsfun == "brent":
                theta, task, NumLineSearch, valuederiv = Brent(
                    fun_value_deriv, theta, brack=(0.0, theta), tol=1e-8, full_output=1
                )
            else:
                if lsfun == 'DIIS':
                    theta = 0.1
                    newphi = phi + p * theta
                elif lsfun == 'mixing':
                    newphi = phi + p * theta

                newrho = newphi * newphi
                norm = rho.N / newrho.integral()
                newrho *= norm
                newphi *= np.sqrt(norm)
                newfunc = self.EnergyEvaluator(
                    newrho, calcType={"V"}, phi=newphi, lphi=self.lphi
                )
                NumLineSearch = 1
                valuederiv = [0, 0, newphi, newfunc]

            if theta is None:
                converged = 1
                sprint("!!!ERROR : Line-Search Failed!!!", comm=self.comm)
                sprint(
                    "!!!ERROR : Density Optimization NOT Converged  !!!", comm=self.comm
                )
                break
                # sprint('!WARN: Line-search failed and change to steepest decent', comm=self.comm)
                # theta = np.ones(self.nspin)*1E-3
                # valuederiv = fun_value_deriv(theta)

            newphi = valuederiv[2]
            newfunc = valuederiv[3]
            old_phi, phi = phi, newphi

            rho = phi * phi
            func = newfunc
            # if self.optimization_options["algorithm"] == 'RMM' :
            # f = self.EnergyEvaluator(rho, calcType = ['E'], phi = phi, lphi = self.lphi)
            # func.energy = f.energy
            # func = self.EnergyEvaluator(rho, calcType = ['E', 'V'], phi = phi, lphi = self.lphi)
            # mu = (func.potential * rho).integral() / rho.N
            mu = self.get_chemical_potential(
                func.potential, rho, phi=phi, lphi=self.lphi
            )
            if self.nspin > 1:
                mus = mu[:, None, None, None]
            else:
                mus = mu
            residual = (func.potential - mus) * phi
            # -----------------------------------------------------------------------
            if self.optimization_method == "DIIS":
                p = -residual
                phi = phi + p * theta
                rho = phi * phi
                norm = rho.N / rho.integral()
                rho *= norm
                phi *= np.sqrt(norm)
                func = self.EnergyEvaluator(
                    rho, calcType={"E", "V"}, phi=phi, lphi=self.lphi
                )
                # mu = (func.potential * rho).integral() / rho.N
                mu = self.get_chemical_potential(
                    func.potential, rho, phi=phi, lphi=self.lphi
                )
                if self.nspin > 1:
                    mus = mu[:, None, None, None]
                else:
                    mus = mu
                residual = (func.potential - mus) * phi

            residualA.append(residual)

            if self.optimization_method == "LBFGS":
                lbfgs.update(phi - old_phi, residualA[-1] - residualA[-2])

            energy = func.energy
            EnergyHistory.append(energy)
            CostTime = TimeData.Time("Optimize")
            dE = EnergyHistory[-1] - EnergyHistory[-2]
            try:
                resN = (
                    self.mp.einsum("..., ...->", residual, residual, optimize='optimal')
                    * phi.grid.dV
                )
            except Exception:
                resN = float(self.mp.sum(residual * residual) * phi.grid.dV)
            fmt = "{:<8d}{:<24.12E}{:<16.6E}{:<16.6E}{:<8d}{:<8d}{:<16.6E}".format(
                it, energy, dE, resN, NumDirectrion, NumLineSearch, CostTime
            )
            sprint(fmt, comm=self.comm)
            if self.check_converge(EnergyHistory):
                converged = 0
                sprint("#### Density Optimization Converged ####", comm=self.comm)
                break

            directionA.append(p)
            if len(residualA) > 2:
                residualA.pop(0)
            if len(directionA) > 2:
                directionA.pop(0)
        else:
            converged = 2
            sprint("!WARN: Not converged, but reached max steps", comm=self.comm)

        sprint('Chemical potential (a.u.):', mu, comm=self.comm)
        sprint(
            'Chemical potential (eV)  :',
            mu * ENERGY_CONV['Hartree']['eV'],
            comm=self.comm,
        )
        self.mu = mu
        self.rho = rho
        self.functional = func
        self.converged = converged
        self.phi = phi
        return rho

    def check_converge(self, EnergyHistory, **kwargs):
        flag = False
        econv = self.optimization_options["econv"]
        ncheck = self.optimization_options["ncheck"]
        E = EnergyHistory[-1]
        if econv is not None:
            if len(EnergyHistory) - 1 < ncheck:
                return flag
            for i in range(ncheck):
                dE = abs(EnergyHistory[-2 - i] - E)
                if abs(dE) > econv:
                    return flag
        flag = True
        return flag

    def get_chemical_potential(self, potential, rho, phi=None, lphi=False):
        mu = (potential * rho).integral() / rho.N
        return mu

    def __call__(self, guess_rho=None, calcType={"E", "V"}, guess_phi=None, lphi=False):
        return self.optimize_rho(guess_rho=guess_rho, guess_phi=guess_phi, lphi=lphi)
