"""
Martyna–Tuckerman reciprocal-space Coulomb screening (J. Chem. Phys. 110, 2810 (1999)).

Uses :attr:`DirectGrid.r_mic` for the spherically smoothed Coulomb numerator in real space.

Pass the **same** :class:`MartynaTuckerman` instance into both ``Functional(type='HARTREE', mt=m)``
and ``Functional(type='PSEUDO', ..., mt=m)`` so Coulomb kernels, ionic potentials, and
``ewald`` share one cached :math:`W(\\mathbf G)`.

References
----------
Martyna & Tuckerman, *J. Chem. Phys.* **110**, 2810 (1999).
Quantum ESPRESSO ``PW/src/martyna_tuckerman.f90``.
"""

from __future__ import annotations

import numpy as np
from scipy import special


def optimal_alpha_beta(gg_max: float, tol: float = 1e-7) -> tuple[float, float]:
    """Decrease *alpha* until a truncation bound analogous to QE ``init_wg_corr``.)

    gg_max :
        Largest :math:`|G|^2` sampled on this mesh (masked half-sphere for rFFT).
    """

    pi = np.pi
    alpha = 2.9
    upperbound = 1.0
    while upperbound > tol:
        alpha -= 0.1
        if alpha <= 0:
            raise ValueError(
                "MartynaTuckerman: alpha tuning failed; increase FFT resolution or cutoff."
            )
        upperbound = np.sqrt(2.0 * alpha / pi) * special.erfc(np.sqrt(gg_max / (4.0 * alpha)))
    beta = 0.5 / alpha
    return float(alpha), float(beta)


def smooth_coulomb_r(r: np.ndarray, alpha: float, eps: float = 1.0e-6) -> np.ndarray:
    """QE ``smooth_coulomb_r`` — long-range Coulomb fragment ``erf(sqrt(\\alpha)\\, r)/ r``."""

    r = np.asarray(r, dtype=np.float64)
    out = np.empty_like(r, dtype=np.float64)
    mask = r > eps
    out[mask] = special.erf(np.sqrt(alpha) * r[mask]) / r[mask]
    out[~mask] = 2.0 / np.sqrt(np.pi) * np.sqrt(alpha)
    return out


def smooth_coulomb_g(gg: np.ndarray, alpha: float, beta: float, eps: float = 1.0e-6) -> np.ndarray:
    """QE ``smooth_coulomb_g`` with |G|\\ :sup:`2` matching ``ReciprocalGrid.gg``."""

    gg = np.asarray(gg, dtype=np.float64)
    fpi = 4.0 * np.pi
    out = np.empty_like(gg, dtype=np.float64)
    mask = gg > eps
    out[mask] = fpi * np.exp(-gg[mask] / (4.0 * alpha)) / gg[mask]
    out[~mask] = -fpi * (1.0 / (4.0 * alpha) + 2.0 * beta / 4.0)
    return out


class MartynaTuckerman:
    """Cache :math:`W(G)` screening weights compatible with PW-style MT corrections."""

    __slots__ = ("_alpha", "_beta", "_grid", "_user_alpha", "_wg")

    def __init__(self, grid, alpha: float | None = None):
        self._grid = grid
        self._user_alpha = alpha
        self._wg = None
        self._alpha = None
        self._beta = None

    def invalidate_cache(self) -> None:
        self._wg = None

    @property
    def grid(self):
        return self._grid

    @grid.setter
    def grid(self, value):
        self._grid = value
        self.invalidate_cache()

    def _ensure_wg(self) -> None:
        if self._wg is None:
            self._build_wg()

    def _build_wg(self) -> None:
        from dftpy.field import DirectField

        reciprocal = self._grid.get_reciprocal()
        gg = reciprocal.gg

        if self._user_alpha is None:
            mask_sel = reciprocal.mask_serial
            gg_max = float(np.max(gg[mask_sel]))
            self._alpha, self._beta = optimal_alpha_beta(gg_max)
        else:
            self._alpha = float(self._user_alpha)
            self._beta = 0.5 / self._alpha

        aux_r = DirectField(grid=self._grid, griddata_3d=smooth_coulomb_r(self._grid.r_mic, self._alpha))
        aux_g = aux_r.fft()

        wg = np.real(aux_g) - smooth_coulomb_g(gg, self._alpha, self._beta)
        damp = np.exp(-0.25 * gg / self._alpha)
        wg *= damp**2
        wg[0, 0, 0] = 0.0
        self._wg = wg

    @property
    def wg(self) -> np.ndarray:
        self._ensure_wg()
        return self._wg

    @property
    def alpha_conv(self) -> float:
        self._ensure_wg()
        return self._alpha

    @property
    def beta(self) -> float:
        self._ensure_wg()
        return self._beta

    def coulomb_kernel(self, reciprocal_grid) -> np.ndarray:
        """``4 \\pi / |G|^2 + W(G)`` ready to multiply ``\\rho(G)`` (atomic units); G=0 cleared."""

        invgg = reciprocal_grid.invgg
        wg = np.ascontiguousarray(self.wg, dtype=np.float64)
        kern = 4.0 * np.pi * invgg + wg
        kern[0, 0, 0] = 0.0
        return kern

    def local_pp_correction_reciprocal(self, ions):
        r"""Additive local PP reciprocal potential

        \( \Delta \tilde v_{\mathrm{loc}}(\mathbf G)
          = -\frac{1}{\Omega}\, W(\mathbf G)\sum_I Z_I e^{-i\mathbf G\cdot\mathbf R_I} \).

        Returned array matches the layout of ``ReciprocalGrid.q``.
        """

        reciprocal_grid = self._grid.get_reciprocal()
        wg = np.ascontiguousarray(self.wg, dtype=np.float64).astype(np.complex128, copy=True)
        summed = np.zeros_like(wg, dtype=np.complex128)
        omega = float(self._grid.volume)
        for ii in range(ions.nat):
            Z = ions.charges[ii]
            if Z != 0.0:
                summed += Z * ions.strf(reciprocal_grid, ii)
        return summed * wg * (-1.0 / omega)

    def ion_ewald_energy(self, ions, reciprocal_grid=None) -> float:
        r"""MT ion–ion energy :math:`\frac{\Omega}{2}\sum |\rho_{\mathrm{ion}}|^2 W` with ρ_ion = (Σ Z S)/Ω."""

        if reciprocal_grid is None:
            reciprocal_grid = self._grid.get_reciprocal()
        mask = reciprocal_grid.mask
        wg = self.wg
        S_tot = np.zeros_like(wg, dtype=np.complex128)
        for ii in range(ions.nat):
            S_tot += ions.charges[ii] * ions.strf(reciprocal_grid, ii)
        rho_ion = S_tot / self._grid.volume
        return 0.5 * self._grid.volume * float(np.real(np.sum((np.abs(rho_ion[mask]) ** 2) * wg[mask])))

    def ion_ewald_forces(self, ions, reciprocal_grid=None) -> np.ndarray:
        """Derivative of :meth:`ion_ewald_energy` w.r.t. ion positions."""

        if reciprocal_grid is None:
            reciprocal_grid = self._grid.get_reciprocal()
        wg = np.ascontiguousarray(self.wg, dtype=np.float64)
        mask = reciprocal_grid.mask
        gvec = reciprocal_grid.g
        S_tot = np.zeros_like(wg, dtype=np.complex128)
        for ii in range(ions.nat):
            S_tot += ions.charges[ii] * ions.strf(reciprocal_grid, ii)

        F_rec = np.zeros((ions.nat, 3))
        for ia in range(ions.nat):
            Ion_strf = ions.charges[ia] * ions.strf(reciprocal_grid, ia)
            pref = wg[mask] * (
                Ion_strf.real[mask] * S_tot.imag[mask] - Ion_strf.imag[mask] * S_tot.real[mask]
            )
            F_rec[ia] = np.einsum("ij,j->i", gvec[:, mask], pref)
        F_rec *= 2.0 / self._grid.volume
        return F_rec
