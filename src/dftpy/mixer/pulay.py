import numpy as np
from scipy import linalg

from dftpy.mpi import sprint

from .mixer import AbstractMixer, SpecialPrecondition


class PulayMixer(AbstractMixer):
    def __init__(
        self,
        predtype=None,
        predcoef=[0.8, 1.0, 1.0],
        maxm=5,
        coef=0.7,
        delay=0,
        predecut=None,
        restarted=False,
        **kwargs,
    ):
        self.pred = SpecialPrecondition(predtype, predcoef, predecut=predecut)
        self._delay = delay
        self.maxm = maxm
        self.coef = coef
        self.restarted = restarted
        self.restart()
        self.comm = None

    def __call__(self, nin, nout, coef=None):
        results = self.compute(nin, nout, coef)
        return results

    def restart(self):
        self._iter = 0
        self.dr_mat = None
        self.dn_mat = None
        self.prev_density = None
        self.prev_residual = None

    def residual(self, nin, nout):
        res = nout - nin
        return res

    def compute(self, nin, nout, coef=None):
        """
        Ref : G. Kresse and J. Furthmuller, Comput. Mat. Sci. 6, 15-50 (1996).
        """
        self.comm = nin.grid.mp.comm
        if coef is None:
            coef = self.coef
        self._iter += 1

        sprint('mixing parameters : ', coef, comm=self.comm, level=1)
        r = nout - nin
        # -----------------------------------------------------------------------
        if self._iter == 1 and self._delay < 1:
            res = r * coef
            results = self.pred(nin, nout, residual=res)
        elif self._iter > self._delay:
            if self.restarted and (self._iter - self._delay) % (self.maxm + 1) == 0:
                self.dr_mat = None
                self.dn_mat = None
                sprint('Restart history of the mixer', comm=self.comm, level=3)
            dn = nin - self.prev_density
            dr = r - self.prev_residual
            if self.dr_mat is None:
                self.dr_mat = dr.reshape((-1, *dr.shape))
                self.dn_mat = dn.reshape((-1, *dn.shape))
            elif len(self.dr_mat) < self.maxm:
                self.dr_mat = np.concatenate((self.dr_mat, dr.reshape((-1, *r.shape))))
                self.dn_mat = np.concatenate((self.dn_mat, dn.reshape((-1, *r.shape))))
            else:
                self.dr_mat = np.roll(self.dr_mat, -1, axis=0)
                self.dr_mat[-1] = dr
                self.dn_mat = np.roll(self.dn_mat, -1, axis=0)
                self.dn_mat[-1] = dn

            ns = len(self.dr_mat)
            amat = np.empty((ns, ns))
            b = np.empty((ns))
            for i in range(ns):
                for j in range(i + 1):
                    amat[i, j] = (self.dr_mat[i] * self.dr_mat[j]).sum()
                    amat[j, i] = amat[i, j]
                b[i] = -(self.dr_mat[i] * r).sum()

            amat = nin.mp.vsum(amat)
            b = nin.mp.vsum(b)

            try:
                x = linalg.solve(amat, b, assume_a='sym')
                # sprint('x', x, comm=self.comm)
                for i in range(ns):
                    if i == 0:
                        drho = x[i] * self.dn_mat[i]
                        res = r + x[i] * self.dr_mat[i]
                    else:
                        drho += x[i] * self.dn_mat[i]
                        res += x[i] * self.dr_mat[i]
                # -----------------------------------------------------------------------
                # res[:] = filter_density(res)
                # -----------------------------------------------------------------------
                res *= coef
                results = self.pred(nin, nout, drho, res, coef)
            except Exception:
                res = r * coef
                sprint('!WARN : Change to linear mixer', comm=self.comm)
                sprint('amat', amat, comm=self.comm)
                results = self.pred(nin, nout, residual=res, coef=coef)
        else:
            results = nout.copy()
            sprint('delay : use output density', comm=self.comm)
        # -----------------------------------------------------------------------
        results = self.format_density(results, nin)
        # -----------------------------------------------------------------------
        self.prev_density = nin + 0.0
        self.prev_residual = r + 0.0

        return results

    def update_predcoef(self, nin, tol=1e-2):
        rho0 = np.mean(nin)
        kf = (3.0 * rho0 * np.pi**2) ** (1.0 / 3.0)
        sprint('kf', kf, self.pred.predcoef[1], comm=self.comm)
        if abs(kf - self.pred.predcoef[1]) > tol:
            self.pred.predcoef[1] = kf
            self.pred._matrix = None
            sprint('Embed restart pulay coef', comm=self.comm)
            self.dr_mat = None
            self.dn_mat = None
            sprint('Restart history of the mixer', comm=self.comm)
