from typing import Set

import numpy as np

from dftpy.field import DirectField
from dftpy.functional.functional_output import FunctionalOutput
from dftpy.math_utils import PowerInt


def JP1Potential(
    rho: DirectField,
    j: DirectField,
    cutoff: float = 1.0e-2,
    rho_cutoff: float = 1.0e-3,
    k: int = 2,
    **kwargs,
) -> DirectField:
    """ """
    k_F = np.cbrt(3.0 * np.pi**2.0 * rho)
    k_F[k_F < cutoff] = cutoff
    k_F_square = k_F * k_F
    k_F_fourth = k_F_square * k_F_square
    reciprocal_grid = j.grid.get_reciprocal()
    j_of_g = j.fft() * np.exp(-0.025 * reciprocal_grid.gg)
    iq_dot_j = 1j * j_of_g.dot(reciprocal_grid.g)
    term1 = iq_dot_j * reciprocal_grid.invq
    term2 = iq_dot_j * reciprocal_grid.q  # * np.exp(-100*gg)
    a = 10
    # term2 *= 1.0 / (1.0 + a * reciprocal_grid.gg)
    # term2 *= np.exp(-a * reciprocal_grid.gg)
    # potential = 6.0 / k_F_square * term1.ifft(force_real=True) + 1.0 / k_F_fourth * term2.ifft(force_real=True)
    potential = -6.0 / k_F_square * term1.ifft(
        force_real=True
    ) - 1.0 / k_F_fourth * term2.ifft(force_real=True)
    potential *= np.pi**3 / 12.0
    v_mask = 1.0 - 1.0 / (1.0 + PowerInt(rho / rho_cutoff, k))
    potential *= v_mask

    return potential


def JP1Potential_alt(
    rho: DirectField, j: DirectField, cutoff: float = 1.0e-4, **kwargs
) -> DirectField:
    """ """
    k_F = np.cbrt(3.0 * np.pi**2.0 * rho)
    k_F_of_q = k_F.fft()
    k_F_square = k_F * k_F
    k_F_square_of_q = k_F_square.fft()
    # rhotwothirds[rhotwothirds<rhotwothirds_cutoff] = rhotwothirds_cutoff
    reciprocal_grid = j.grid.get_reciprocal()
    g = reciprocal_grid.g
    sqrt_gg = np.sqrt(reciprocal_grid.gg)
    sqrt_gg[0, 0, 0] = 1.0
    iq_dot_j = 1j * j.fft().dot(g)
    # term1 = iq_dot_j / sqrt_gg / k_F_of_q
    term1 = iq_dot_j / sqrt_gg
    sqrt_gg[0, 0, 0] = 0.0
    term2 = iq_dot_j * sqrt_gg / k_F_square_of_q
    # potential = 6.0 / k_F * term1.ifft(force_real = True) + 1.0 / k_F_square * term2.ifft(force_real = True)
    potential = (
        6.0 / k_F_square * term1.ifft(force_real=True)
    )  # + 1.0 / k_F_square * term2.ifft(force_real = True)
    potential *= -np.pi**3 / 12.0

    return potential


def JP1(
    rho: DirectField, j: DirectField, calcType: Set[str], **kwargs
) -> FunctionalOutput:
    functional = FunctionalOutput(name="JP1")
    if "V" in calcType:
        functional.potential = JP1Potential(rho, j, **kwargs)
    return functional
