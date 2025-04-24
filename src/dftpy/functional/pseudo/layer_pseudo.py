import numpy as np
from scipy.interpolate import splev, splrep

from dftpy.density.density import AtomicDensity
from dftpy.field import DirectField
from dftpy.functional.external_potential import ExternalPotential
from dftpy.grid import DirectGrid
from dftpy.ions import Ions


class LayerPseudo(ExternalPotential):

    def __init__(
        self, vr=None, r=None, grid: DirectGrid = None, ions: Ions = None, **kwargs
    ):
        super().__init__()
        self.grid = grid
        self.ions = ions
        if self.ions is not None:
            self.key = self.ions[0].symbol
        self._vr = vr
        self.r = r
        if r is not None:
            self.r_cut = r[-1]

    @property
    def vr(self):
        return self._vr

    @vr.setter
    def vr(self, new_vr):
        self._vr = new_vr
        self.update_v()

    @property
    def v(self):
        if self._v is not None:
            return self._v
        self.update_v()
        return self._v

    def update_v(self):
        spl = splrep(self.r, self.vr)
        self._v = DirectField(self.grid)
        self.calc_dis = AtomicDensity(self.grid, self.ions, self.key, self.r_cut)
        for dis in self.calc_dis.distance():
            dv0 = splev(dis['dists'], spl)
            dv1 = np.zeros_like(dv0)
            dv1[dis['index']] = dv0[dis['index']]
            self._v[
                dis['l123A'][0][dis['mask']],
                dis['l123A'][1][dis['mask']],
                dis['l123A'][2][dis['mask']],
            ] += dv1.ravel()[dis['mask']]

    # def __init__(self, **kwargs):
    #     self._gp = {}  # 1D PP grid g-space
    #     self._vp = {}  # PP on 1D PP grid
    #     self._vloc_interp = {}  # Interpolates recpot PP
    #     self._zval = {}
    #
    # @property
    # def vloc_interp(self):
    #     if not self._vloc_interp:
    #         """get the representation of PP
    #
    #                 Args:
    #                     key: Atomic symbol
    #                     k: The degree of the spline fit of splrep, should keep use 3.
    #                 """
    #         from scipy.interpolate import splrep
    #         vloc_interp = splrep(self._gp[key][1:], self._vp[key][1:], k=k)
    #         self._vloc_interp[key] = vloc_interp
    #     return self._vloc_interp
    #
    # @property
    # def gp(self):
    #     if not self._gp:
    #         raise AttributeError("Must init ReadPseudo")
    #     return self._gp
    #
    # @property
    # def vp(self):
    #     if not self._vp:
    #         raise AttributeError("Must init ReadPseudo")
    #     return self._vp
    #
    # def get_Zval(self, ions):
    #     pass
