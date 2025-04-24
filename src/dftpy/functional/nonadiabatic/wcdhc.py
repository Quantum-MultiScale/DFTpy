from typing import Set

import numpy as np

from dftpy.field import DirectField
from dftpy.functional.functional_output import FunctionalOutput


def WCDHCPotential(
    rho: DirectField, j: DirectField, cutoff: float = 1.0e-4, **kwargs
) -> DirectField:
    """
    current-dependent dynamic kinetic energy potential
    Eq. 3 of PRL 121, 145001 (2018)
    """
    k_F = np.cbrt(3.0 * np.pi**2.0 * rho)
    k_F[k_F < cutoff] = cutoff
    reciprocal_grid = j.grid.get_reciprocal()
    g = reciprocal_grid.g
    sqrt_gg = np.sqrt(reciprocal_grid.gg)
    sqrt_gg[0, 0, 0] = 1.0
    # temp = 1j * j.fft().dot(g) / np.sqrt(gg) * np.exp(-0.5 * gg)
    temp = 1j * j.fft().dot(g) / sqrt_gg

    return np.pi**3.0 / (2.0 * k_F * k_F) * temp.ifft(force_real=True)


def WCDHC(
    rho: DirectField, j: DirectField, calcType: Set[str], **kwargs
) -> FunctionalOutput:
    functional = FunctionalOutput(name="WCDHC")
    if "V" in calcType:
        functional.potential = WCDHCPotential(rho, j, **kwargs)
    return functional


def nWCDHC(
    rho: DirectField, j: DirectField, calcType: Set[str], **kwargs
) -> FunctionalOutput:
    functional = FunctionalOutput(name="nWCDHC")
    if "V" in calcType:
        functional.potential = -WCDHCPotential(rho, j, **kwargs)
    return functional
