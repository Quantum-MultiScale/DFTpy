"""Gaussian valence density centered on ions (MIC) and Pulay forces from moving centers."""

from __future__ import annotations

import numpy as np

from dftpy.field import DirectField
from dftpy.grid import DirectGrid
from dftpy.ions import Ions


def gaussian_valence_density(
    grid: DirectGrid,
    ions: Ions,
    sigma: float,
    total_valence_electrons: float,
) -> DirectField:
    """Sum of Gaussians at ion sites (MIC); scaled so :math:`\\int \\rho = N_e`."""

    accum, _ = _gaussian_accum_and_mic(grid, ions, sigma)
    rho = DirectField(grid=grid, griddata_3d=accum)
    rho *= float(total_valence_electrons) / rho.integral()
    return rho


def gaussian_valence_center_forces(
    grid: DirectGrid,
    ions: Ions,
    sigma: float,
    pot: DirectField,
    total_valence_electrons: float,
) -> np.ndarray:
    r"""
    Forces from moving normalized Gaussian centers at fixed grid potential.

    .. math::

        F_{Ia} = -\int \frac{\delta E}{\delta\rho}(\mathbf r)\,
                 \frac{\partial\rho}{\partial R_{Ia}}(\mathbf r)\,\mathrm d^3r

    with :math:`\rho = (N_e / A)\sum_J g_J`, :math:`A=\int\sum_J g_J`, and
    :math:`\partial g_I/\partial R_{Ia} = g_I\, m_{Ia}/\sigma^2` (MIC offset :math:`m_I`).
    """

    accum, mic_per_ion = _gaussian_accum_and_mic(grid, ions, sigma)
    a0 = float(DirectField(grid=grid, griddata_3d=accum).integral())
    scale = float(total_valence_electrons) / a0
    inv_sigma2 = 1.0 / (sigma**2)
    nat = ions.nat
    forces = np.zeros((nat, 3), dtype=np.float64)
    for i in range(nat):
        g_arr, mic = mic_per_ion[i]
        for a in range(3):
            dg = g_arr * mic[a] * inv_sigma2
            int_dg = float(DirectField(grid=grid, griddata_3d=dg).integral())
            drho = scale * dg - (scale / a0) * int_dg * accum
            drho_field = DirectField(grid=grid, griddata_3d=drho)
            forces[i, a] = -(pot * drho_field).integral()
    return forces


def _gaussian_accum_and_mic(
    grid: DirectGrid, ions: Ions, sigma: float
) -> tuple[np.ndarray, list[tuple[np.ndarray, np.ndarray]]]:
    lattice = np.asarray(ions.cell, dtype=np.float64)
    inv_lat = np.linalg.inv(lattice)
    s_ions = ions.get_positions() @ inv_lat
    S = np.stack(grid.s, axis=0)
    pref = (2.0 * np.pi * sigma**2) ** (-1.5)
    accum = np.zeros(grid.nr, dtype=np.float64)
    mic_per_ion: list[tuple[np.ndarray, np.ndarray]] = []
    for i in range(ions.nat):
        ds = S - s_ions[i][:, np.newaxis, np.newaxis, np.newaxis]
        ds = (ds + 0.5) % 1.0 - 0.5
        mic = np.einsum("j...,jk->k...", ds, lattice)
        r2 = np.sum(mic * mic, axis=0)
        g = pref * np.exp(-r2 / (2.0 * sigma**2))
        mic_per_ion.append((g, mic))
        accum += g
    return accum, mic_per_ion
