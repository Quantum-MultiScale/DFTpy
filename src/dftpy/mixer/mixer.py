from abc import ABC, abstractmethod

import numpy as np

from dftpy.constants import ZERO
from dftpy.density import normalization_density
from dftpy.field import DirectField, ReciprocalField
from dftpy.mpi import sprint


class SpecialPrecondition:
    def __init__(
        self,
        predtype='inverse_kerker',
        predcoef=[0.8, 1.0, 1.0],
        grid=None,
        predecut=None,
        **kwargs,
    ):
        self.predtype = predtype
        self._init_predcoef(predcoef, predtype)
        self._ecut = predecut
        self._grid = grid
        self._matrix = None
        self._direct = False
        self._mask = False

    def _init_predcoef(self, predcoef=[], predtype='inverse_kerker'):
        nl = len(predcoef)
        if nl == 0:
            predcoef = [0.8, 1.0, 1.0]
        elif nl == 1:
            predcoef.extend([1.0, 1.0])
        elif nl == 2:
            predcoef.extend([1.0])
        self.predcoef = predcoef

    @property
    def matrix(self):
        if self._matrix is None:
            if self.predtype is None:
                self._matrix = ReciprocalField(self.grid.get_reciprocal()) + 1.0
            elif self.predtype == 'kerker':
                self._matrix = self.kerker()
            elif self.predtype == 'inverse_kerker':
                self._matrix = self.inverse_kerker()
            elif self.predtype == 'resta':
                self._matrix = self.resta()
        return self._matrix

    @property
    def grid(self):
        return self._grid

    @property
    def comm(self):
        self.grid.mp.comm

    @grid.setter
    def grid(self, value):
        self._grid = value
        self._matrix = None
        self._mask = None

    @property
    def mask(self):
        if self._mask is None:
            recip_grid = self.grid.get_reciprocal()
            gg = recip_grid.gg
            self._mask = np.zeros(recip_grid.nr, dtype='bool')
            if self._ecut is not None:
                if self._ecut < 2:
                    gmax = (
                        max(gg[:, 0, 0].max(), gg[0, :, 0].max(), gg[0, 0, :].max()) + 2
                    )
                    gmax = self.grid.mp.amax(gmax)
                else:
                    gmax = 2.0 * self._ecut
                self._mask[gg > gmax] = True
                sprint('Density mixing gmax', gmax, self._ecut, comm=self.comm)
        return self._mask

    def kerker(self):
        a0 = self.predcoef[0]
        q0 = self.predcoef[1] ** 2
        amin = self.predcoef[2]
        recip_grid = self.grid.get_reciprocal()
        gg = recip_grid.gg
        preg = a0 * np.minimum(gg / (gg + q0), amin)
        preg = ReciprocalField(recip_grid, data=preg)
        return preg

    def inverse_kerker(self):
        b0 = self.predcoef[0] ** 2
        recip_grid = self.grid.get_reciprocal()
        gg = recip_grid.gg
        qflag = True if gg[0, 0, 0] < ZERO else False
        if qflag:
            gg[0, 0, 0] = 1.0
        preg = b0 / gg + 1.0
        if qflag:
            gg[0, 0, 0] = 0.0
            preg[0, 0, 0] = 0.0
        preg = ReciprocalField(recip_grid, data=preg)
        return preg

    def resta(self):
        epsi = self.predcoef[0]
        q0 = self.predcoef[1] ** 2
        rs = self.predcoef[2]
        recip_grid = self.grid.get_reciprocal()
        gg = recip_grid.gg
        q = recip_grid.q
        qflag = True if q[0, 0, 0] < ZERO else False
        if qflag:
            q[0, 0, 0] = 1.0
        preg = (q0 * np.sin(q * rs) / (epsi * q * rs) + gg) / (q0 + gg)
        if qflag:
            q[0, 0, 0] = 0.0
            preg[0, 0, 0] = 1.0
        preg = ReciprocalField(recip_grid, data=preg)
        return preg

    def __call__(self, nin, nout, drho=None, residual=None, coef=0.7):
        results = self.compute(nin, nout, drho, residual, coef)
        return results

    def compute(self, nin, nout, drho=None, residual=None, coef=0.7):
        if self.grid is None:
            self.grid = nin.grid
        nin_g = nin.fft()
        results = nin_g.copy()
        if drho is not None:
            dr = DirectField(self.grid, data=drho)
            results += dr.fft()
        if residual is not None:
            res = DirectField(self.grid, data=residual)
            results += res.fft() * self.matrix
        # -----------------------------------------------------------------------
        if nin.mp.asum(self.mask) > 0:
            # Linear mixing for high-frequency part
            results[self.mask] = (
                nin_g[self.mask] * (1 - coef) + coef * nout.fft()[self.mask]
            )
        # -----------------------------------------------------------------------
        return results.ifft(force_real=True)

    def add(self, density, residual=None, grid=None):
        if grid is not None:
            self.grid = grid
        if not self._direct:
            den = DirectField(self.grid, data=density)
            if residual is None:
                results = (self.matrix * den.fft()).ifft(force_real=True)
            else:
                res = DirectField(self.grid, data=residual)
                results = (den.fft() + self.matrix * res.fft()).ifft(force_real=True)
        else:
            raise AttributeError("Real-space matrix will implemented soon")
        return results


class AbstractMixer(ABC):
    """
    This is a template class for mixer
    """

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def __call__(self):
        pass

    def restart(self, *arg, **kwargs):
        pass

    def format_density(self, results, nin):
        ncharge = nin.integral()
        results = normalization_density(
            results, ncharge=ncharge, grid=nin.grid, method='no'
        )
        return results
