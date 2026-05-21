"""
Martyna–Tuckerman reciprocal-space Coulomb screening (J. Chem. Phys. 110, 2810 (1999)).

Uses QE-style shortest periodic distance from lattice sites (corner-origin fractional
``DirectGrid.s``, same folding as minimum-image distance from corner lattice points)
to build the real-space smooth Coulomb fragment.

Pass the **same** :class:`MartynaTuckerman` instance into both ``Functional(type='HARTREE', mt=m)``
and ``Functional(type='PSEUDO', ..., mt=m)`` so Coulomb kernels, ionic potentials, and
``ewald`` share one cached :math:`W(\\mathbf G)`.

References
----------
Martyna & Tuckerman, *J. Chem. Phys.* **110**, 2810 (1999) (Sec. II, Appendix A--B):
screening \(\hat f_{\mathrm{screen}}(\mathbf g)=\bar f^{(\mathrm{long})}-\tilde f^{(\mathrm{long})}\) for the
``erf``/``erfc`` split of \(1/r\); **no extra** \(\exp(-G^2/(4\alpha))\) factor applied *after* forming \(W'\).
\(W(\mathbf G)=\mathcal{R}[A(\mathbf G)]-\tilde f_\alpha(\mathbf G)\) from Appendix B; \(g\to 0\) singularities of the
long-range Coulomb piece cancel between terms (their Eq. B3), leaving **finite** \(\bar f\) and kernel limits —
the Coulomb kernel therefore keeps **\(K(\mathbf 0)=W(\mathbf 0)\)** when the bare \(4\pi/G^2\) pole is omitted via
``invgg[0]=0``, rather than forcing \(K(\mathbf 0)=0\).

:class:`MartynaTuckerman` accepts any nonsingular ``DirectGrid.lattice``. Orthorhombic
primitive cells use a fast per-axis minimum-image path for ``ws_dist_corner``; general
cells use Babai nearest-plane plus a small integer shell (QE ``ws_dist`` semantics).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import special

from dftpy.constants import environ
from dftpy.mpi import sprint

from dftpy.field import ReciprocalField

print=False
if environ["LOGLEVEL"] >= 2:
    print=True

def optimal_alpha_beta(gg_max: float, tol: float = 1e-7) -> tuple[float, float]:
    """Tune *alpha* using a truncation bound analogous to QE ``init_wg_corr``.

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
    if print: sprint("MT alpha: ", alpha) 
    return float(alpha), float(beta)


def _lattice_primitive_orthogonal(lat: np.ndarray, atol: float = 1.0e-9) -> bool:
    """True if direct Bravais rows are pairwise orthogonal (orthorhombic primitives)."""

    metric = lat @ lat.T
    diag = np.diag(np.diagonal(metric))
    return bool(np.allclose(metric, diag, atol=atol))


def _ws_dist_nmax_auto(
    lattice_rows: np.ndarray, pts: np.ndarray | None = None, pad: int = 2
) -> int:
    """Upper bound on integer shifts for MIC-from-corner on a periodic grid."""

    lat = np.asarray(lattice_rows, dtype=np.float64)
    inv = np.linalg.inv(lat)
    corners = np.array(
        [[i, j, k] for i in (0.0, 1.0) for j in (0.0, 1.0) for k in (0.0, 1.0)],
        dtype=np.float64,
    )
    frac_extent = int(np.ceil(np.max(np.abs(corners @ inv.T)))) + pad
    if pts is not None:
        pts = np.asarray(pts, dtype=np.float64).reshape(-1, 3)
        frac_extent = max(frac_extent, int(np.ceil(np.max(np.abs(pts @ inv.T)))) + pad)
    return max(frac_extent, 2)


def _ws_dist_corner_orthogonal_fractional(s: np.ndarray, lattice_rows: np.ndarray) -> np.ndarray:
    """MIC-from-corner using fractional coords ``s`` (shape ``(3, ...)``, values in ``[0, 1)``)."""

    lattice_rows = np.asarray(lattice_rows, dtype=np.float64)
    frac = (np.asarray(s, dtype=np.float64) + 0.5) % 1.0 - 0.5
    mic = np.einsum("j...,jk->k...", frac, lattice_rows)
    return np.sqrt(np.maximum(0.0, np.sum(mic * mic, axis=0)))


def _ws_dist_corner_orthogonal(r_xyz: np.ndarray, lattice_rows: np.ndarray) -> np.ndarray:
    """Fast MIC-from-corner distances for orthogonal primitive rows (QE ``ws_dist`` limit)."""

    inv_lat = np.linalg.inv(lattice_rows)
    frac = np.einsum("k...,kj->...j", r_xyz, inv_lat)
    frac = np.moveaxis(frac, -1, 0)
    return _ws_dist_corner_orthogonal_fractional(frac, lattice_rows)


def _ws_dist_corner_orthogonal_grid(grid) -> np.ndarray:
    """Orthorhombic MIC-from-corner on ``DirectGrid.s`` (fast path)."""

    return _ws_dist_corner_orthogonal_fractional(grid.s, grid.lattice)


@dataclass(frozen=True)
class _LatticeMicSetup:
    """Cached Gram–Schmidt data for Babai CVP on a fixed Bravais matrix."""

    lattice: np.ndarray
    gram: np.ndarray
    basis: np.ndarray
    bstar: np.ndarray
    norms_sq: np.ndarray


def _mic_lattice_setup(lattice_rows: np.ndarray) -> _LatticeMicSetup:
    """One-time Babai preparation: minimize ``||(s+n)@L||`` over integer ``n``."""

    lat = np.asarray(lattice_rows, dtype=np.float64)
    basis = lat.T.copy()
    bstar = np.zeros_like(basis)
    norms_sq = np.zeros(3, dtype=np.float64)
    for i in range(3):
        v = basis[:, i].copy()
        for j in range(i):
            mu_ij = np.dot(basis[:, i], bstar[:, j]) / norms_sq[j]
            v -= mu_ij * bstar[:, j]
        bstar[:, i] = v
        norms_sq[i] = np.dot(v, v)
    return _LatticeMicSetup(
        lattice=lat,
        gram=lat @ lat.T,
        basis=basis,
        bstar=bstar,
        norms_sq=norms_sq,
    )


def _babai_nearest_coeff(setup: _LatticeMicSetup, target: np.ndarray) -> np.ndarray:
    """Integer coefficients ``n`` with ``n @ L ≈ target`` (Babai nearest-plane)."""

    target = np.asarray(target, dtype=np.float64).reshape(3)
    coeff = np.zeros(3, dtype=np.int64)
    residual = target.copy()
    for i in range(2, -1, -1):
        nsq = setup.norms_sq[i]
        if nsq < 1e-30:
            ci = 0
        else:
            ci = int(np.round(np.dot(residual, setup.bstar[:, i]) / nsq))
        coeff[i] = ci
        residual -= ci * setup.basis[:, i]
    return coeff


def _shell_offsets(shell: int) -> np.ndarray:
    """Integer shifts ``δ`` for ``n = n₀ + δ`` with ``|δ_i| ≤ shell``."""

    offs = np.arange(-shell, shell + 1, dtype=np.int64)
    i1, i2, i3 = np.meshgrid(offs, offs, offs, indexing="ij")
    return np.stack([i1.ravel(), i2.ravel(), i3.ravel()], axis=1)


def _ws_dist_sq_fractional(
    s: np.ndarray,
    setup: _LatticeMicSetup,
    shell: int = 1,
) -> np.ndarray:
    """Squared MIC distances for fractional coords ``s`` (shape ``(3, N)``)."""

    lat = setup.lattice
    gram = setup.gram
    s_flat = np.asarray(s, dtype=np.float64).reshape(3, -1).T
    n_pts = s_flat.shape[0]
    deltas = _shell_offsets(shell)
    n_shell = deltas.shape[0]
    best_sq = np.full(n_pts, np.inf, dtype=np.float64)

    chunk = 8192
    for start in range(0, n_pts, chunk):
        end = min(start + chunk, n_pts)
        sub = s_flat[start:end]
        cart = sub @ lat
        n0 = np.empty((end - start, 3), dtype=np.int64)
        for i, r in enumerate(cart):
            n0[i] = _babai_nearest_coeff(setup, -r)
        cand = n0[:, None, :] + deltas[None, :, :]
        vec = sub[:, None, :] + cand
        sq = np.einsum("nij,jk,nik->ni", vec, gram, vec, optimize=True)
        best_sq[start:end] = np.min(sq, axis=1)

    return best_sq


def _ws_dist_corner_cvp_fractional(
    s: np.ndarray,
    lattice_rows: np.ndarray,
    shell: int = 1,
    setup: _LatticeMicSetup | None = None,
) -> np.ndarray:
    """MIC-from-corner on fractional nodes via Babai + local integer shell."""

    if setup is None:
        setup = _mic_lattice_setup(lattice_rows)
    dist_sq = _ws_dist_sq_fractional(s, setup, shell=shell)
    return np.sqrt(np.maximum(dist_sq, 0.0))


def _ws_dist_corner_cvp_pts(
    pts: np.ndarray,
    lattice_rows: np.ndarray,
    shell: int = 1,
    setup: _LatticeMicSetup | None = None,
) -> np.ndarray:
    """MIC-from-corner for Cartesian rows ``pts`` (``N``, 3)."""

    lat = np.asarray(lattice_rows, dtype=np.float64)
    if setup is None:
        setup = _mic_lattice_setup(lat)
    inv_lat = np.linalg.inv(lat)
    s_flat = np.asarray(pts, dtype=np.float64).reshape(-1, 3) @ inv_lat
    dist_sq = _ws_dist_sq_fractional(s_flat.T, setup, shell=shell)
    return np.sqrt(np.maximum(dist_sq, 0.0))


def _ws_dist_corner_cvp_grid(grid, shell: int = 1) -> np.ndarray:
    """MIC-from-corner on ``DirectGrid.s`` (Babai + shell, O(27N) per shell width)."""

    setup = _mic_lattice_setup(grid.lattice)
    return _ws_dist_corner_cvp_fractional(grid.s, grid.lattice, shell=shell, setup=setup)


def _ws_dist_corner_grid(
    grid,
    nmax: int | None = None,
    shell: int = 1,
) -> np.ndarray:
    """MIC-from-corner on all FFT nodes of ``grid``."""

    lat = np.asarray(grid.lattice, dtype=np.float64)
    if _lattice_primitive_orthogonal(lat):
        return _ws_dist_corner_orthogonal_grid(grid)
    if nmax is not None:
        pts = np.moveaxis(grid.r, 0, -1).reshape(-1, 3)
        dist = _ws_dist_corner_brute_chunks(pts, lat, nmax=nmax)
        return dist.reshape(tuple(grid.nr))
    return _ws_dist_corner_cvp_grid(grid, shell=shell)


def _ws_dist_corner_brute_chunks(
    pts: np.ndarray, lattice_rows: np.ndarray, nmax: int = 6, chunk: int = 16384
) -> np.ndarray:
    """QE ``ws_dist`` via brute integer translations; chunked to limit memory."""

    ip = np.arange(-nmax, nmax + 1, dtype=np.float64)
    i1, i2, i3 = np.meshgrid(ip, ip, ip, indexing="ij")
    shifts = (
        i1[..., None] * lattice_rows[0] + i2[..., None] * lattice_rows[1] + i3[..., None] * lattice_rows[2]
    ).reshape(-1, 3)
    n_pts = pts.shape[0]
    best_sq = np.empty(n_pts, dtype=np.float64)
    shift_norm_sq = np.sum(shifts * shifts, axis=1)
    for start in range(0, n_pts, chunk):
        end = min(start + chunk, n_pts)
        sub = pts[start:end]
        dot_rs = np.einsum("bd,sd->bs", sub, shifts, optimize=True)
        norms_r = np.sum(sub * sub, axis=1, keepdims=True)
        cand = norms_r + shift_norm_sq[None, :] + 2.0 * dot_rs
        best_sq[start:end] = np.clip(np.min(cand, axis=1), 0.0, np.inf)
    return np.sqrt(best_sq)


def ws_dist_corner(
    r_xyz: np.ndarray, lattice_rows: np.ndarray, nmax: int | None = None, shell: int = 1
) -> np.ndarray:
    """
    QE ``ws_dist``: shortest periodic distance ``min_n |\\mathbf r + \\sum_i n_i \\mathbf a_i|``.

    Mirrors QE ``PW/src/martyna_tuckerman.f90`` + ``Modules/ws_base.f90``: Coulomb numerator
    is evaluated on corner-origin Cartesian nodes (``DirectGrid.r``), folded toward **origin**
    lattice sites, unlike :attr:`~dftpy.grid.DirectGrid.r_mic` which folds toward ``(½,½,½)``.

    Orthorhombic primitive rows use per-axis fractional wrapping; other cells use Babai
    nearest-plane plus a ``(2·shell+1)³`` integer correction shell (default ``shell=1`` → 27
    trials). Pass a positive *nmax* to force the legacy brute translation search instead.
    """

    assert r_xyz.ndim >= 1 and r_xyz.shape[0] == 3
    orig_shape = r_xyz.shape
    lattice_rows = np.asarray(lattice_rows, dtype=np.float64)

    if _lattice_primitive_orthogonal(lattice_rows):
        dist = _ws_dist_corner_orthogonal(r_xyz, lattice_rows)
    elif nmax is not None:
        pts = np.moveaxis(r_xyz, 0, -1).reshape(-1, 3)
        dist = _ws_dist_corner_brute_chunks(pts, lattice_rows, nmax=nmax)
    else:
        pts = np.moveaxis(r_xyz, 0, -1).reshape(-1, 3)
        dist = _ws_dist_corner_cvp_pts(pts, lattice_rows, shell=shell)
    return np.asarray(dist.reshape(orig_shape[1:]), dtype=np.float64)


def ws_dist_corner_grid(
    grid, nmax: int | None = None, shell: int = 1
) -> np.ndarray:
    """Like :func:`ws_dist_corner` but takes a ``DirectGrid`` and uses ``grid.s``."""

    return np.asarray(_ws_dist_corner_grid(grid, nmax=nmax, shell=shell), dtype=np.float64)


def _build_wg_cache(mt: "MartynaTuckerman") -> None:
    """Populate ``mt._wg`` and convolution parameters from ``mt._grid``."""

    from dftpy.field import DirectField

    grid = mt._grid
    reciprocal = grid.get_reciprocal()
    gg = reciprocal.gg

    if mt._user_alpha is None:
        mask_sel = reciprocal.mask_serial
        gg_max = np.max(gg[mask_sel])
        mt._alpha, mt._beta = optimal_alpha_beta(gg_max)
    else:
        mt._alpha = mt._user_alpha
        mt._beta = 0.5 / mt._alpha

    ws_r = _ws_dist_corner_grid(grid)
    aux_r = DirectField(grid=grid, griddata_3d=smooth_coulomb_r(ws_r, mt._alpha))
    aux_g = aux_r.fft().real
    sprint("aux_g", aux_g[0,0,0])
    #aux_g[gg <= 1e-8]*= 1.0/grid.volume

    wg = aux_g - smooth_coulomb_g(gg, mt._alpha, mt._beta)
    sprint("wg", wg[0,0,0])
    #wg[gg <= 1e-8]*= 1.0/grid.volume
    mt._wg = ReciprocalField(grid=reciprocal, data=wg) 
    #* np.exp(-gg*mt._beta/4.0)**2 # same as mt in QE to cut off high Gs.


def _touch_wg(mt: "MartynaTuckerman") -> None:
    if mt._wg is None:
        _build_wg_cache(mt)


def smooth_coulomb_r(r: np.ndarray, alpha: float, eps: float = 1.0e-6) -> np.ndarray:
    """QE ``smooth_coulomb_r`` — long-range Coulomb fragment ``erf(sqrt(\\alpha)\\, r)/ r``."""

    r = np.asarray(r, dtype=np.float64)
    out = np.empty_like(r, dtype=np.float64)
    mask = r > eps
    out[mask] = special.erf(np.sqrt(alpha) * r[mask]) / r[mask]
    out[~mask] = 2.0 / np.sqrt(np.pi) * np.sqrt(alpha)
    return out


def smooth_coulomb_g(gg: np.ndarray, alpha: float, beta: float, eps: float = 1.0e-8) -> np.ndarray:
    """QE ``smooth_coulomb_g`` with |G|\\ :sup:`2` matching ``ReciprocalGrid.gg``."""

    fpi = 4.0 * np.pi
    out = np.empty_like(gg, dtype=np.float64)
    mask = gg > eps
    out[mask] = fpi * np.exp(-gg[mask] / (4.0 * alpha)) / gg[mask]
    out[~mask] =  -fpi/ (4.0 * alpha) #* (1.0 / (4.0 * alpha) + 2.0 * beta / 4.0)
    return out


class MartynaTuckerman:
    """Cached \(W(\mathbf G)=\mathcal{R}[A]-\tilde f_\alpha\) screening (Martyna & Tuckerman 1999, Appendix B).

    Works with any nonsingular ``DirectGrid`` lattice (general triclinic or orthorhombic).
    """

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

    @property
    def wg(self) -> np.ndarray:
        _touch_wg(self)
        return self._wg

    @property
    def alpha_conv(self) -> float:
        _touch_wg(self)
        return self._alpha

    @property
    def beta(self) -> float:
        _touch_wg(self)
        return self._beta

    def coulomb_kernel(self, reciprocal_grid) -> np.ndarray:
        r"""\(K(\mathbf G)=4\pi/|\mathbf G|^2+W(\mathbf G)\) for the Hartree solve (atomic units).
        Strict MT gauge: the finite \(W(\mathbf 0)\) term is kept, i.e. no special override
        of ``kern[0,0,0]``.
        """
        invgg = reciprocal_grid.invgg
        #sprint(self.wg[0,0,0],np.pi/self._alpha)
        kern = 4.0 * np.pi * invgg + self.wg
        return kern

    def local_pp_correction_reciprocal(self, ions):
        r"""Additive local PP reciprocal potential (DFTpy ``ions.strf`` convention).

        QE writes \(\Delta\tilde v_{\mathrm{loc}} = -W(\mathbf G)\tilde\rho_{\mathrm{ion}}(\mathbf G)\)
        with \(\tilde\rho_{\mathrm{ion}}=\Omega^{-1}\sum_I Z_I S_I(\mathbf G)\). DFTpy uses
        dimensionless \(S_I(\mathbf G)=\exp(-i\mathbf G\cdot\mathbf R_I)\) in ``total_strf``, so
        no extra \(1/\Omega\) is applied here (same convention as ``ewald.Energy_rec``).
        """

        reciprocal_grid = self._grid.get_reciprocal()
        summed = ions.total_strf(reciprocal_grid)
        return summed * self.wg * (-1.0)

    def ion_ewald_energy(self, ions, *, mask=None) -> float:
        r"""Additive MT ion--ion reciprocal energy (DFTpy reciprocal sums).

        With \(S^{\mathrm{tot}}(\mathbf G)=\sum_I Z_I S_I(\mathbf G)\) from ``total_strf``,

        .. math::

           E_{\mathrm{II,MT}}
           = \frac{1}{2\Omega}\sum_{\mathbf G} W(\mathbf G)\,|S^{\mathrm{tot}}(\mathbf G)|^2,

        matching the \(4\pi/\Omega\) prefactor in ``ewald.Energy_rec`` for \(|S|^2\) sums.
        \(W(\mathbf G)\) is unchanged by the FFT ``dV`` scaling (same units as \(4\pi/G^2\)).
        """

        reciprocal_grid = self._grid.get_reciprocal()
        if mask is None:
            mask = reciprocal_grid.mask
        stot = ions.total_strf(reciprocal_grid)
        strf_sq = np.real(np.conjugate(stot) * stot)
        return 0.5 * float(np.sum(strf_sq[mask] * self.wg[mask]) / self._grid.volume)

    def ion_ewald_forces(self, ions, *, mask=None) -> np.ndarray:
        r"""Forces from :meth:`ion_ewald_energy` (Hellmann--Feynman, DFTpy ``strf`` convention)."""

        reciprocal_grid = self._grid.get_reciprocal()
        if mask is None:
            mask = reciprocal_grid.mask
        wg = self.wg
        gvec = reciprocal_grid.g
        stot = ions.total_strf(reciprocal_grid)
        inv_vol = 1.0 / self._grid.volume
        forces = np.zeros((ions.nat, 3), dtype=np.float64)
        for ia in range(ions.nat):
            z = ions.charges[ia]
            si = ions.strf(reciprocal_grid, ia)
            for a in range(3):
                term = wg[mask] * np.real(
                    np.conjugate(stot[mask]) * (-1j * z * gvec[a][mask] * si[mask])
                )
                forces[ia, a] = -inv_vol * np.sum(term)
        return forces
