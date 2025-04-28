# Collection of finite temperature Thomas Fermi functional
from IPython.terminal.shortcuts.auto_suggest import accept_and_keep_cursor

from dftpy.field import DirectField
from dftpy.functional.functional_output import FunctionalOutput

__all__ = ['FT_WT', 'FT_WT']


def FT_WTPotential(rho, kernel, alpha=5.0 / 6.0, beta=5.0 / 6.0):
    """
    Finite Temperature WT Potentia
    """
    if (alpha - beta) > 1e-10:
        raise RuntimeError("alpha neq beta, FT_WTPotential not work!")
    rhoa = rho ** alpha
    rhob = rho ** beta
    frhob = rhob.fft()
    pot = (frhob * kernel).ifft()
    energy = (rhoa * pot).sum() * rho.grid.dv
    pot_out = (alpha + beta) * pot * rhoa / rho
    return energy, pot_out


def FT_WTEnergy(rho: DirectField, kernel, alpha=5.0 / 6.0, beta=5.0 / 6.0):
    """
    Finite Temperature WT Energy
    """
    rhoa = rho ** alpha
    rhob = rho ** beta
    frhob = rhob.fft()
    pot = (frhob * kernel).ifft()
    energy = (rhoa * pot).sum() * rho.grid.dv
    return energy


def FT_WTStress(rho, x=1.0, temperature=1e-3, **kwargs):
    """
    Finite Temperature WT Stress
    """
    for i in range(3):
        stress[i, i] = stress_ii
    return stress


def FT_WT(rho, calcType={"E", "V"}, temperature=1e-3, **kwargs):
    """
    temperature in eV 
    FT_T in Ha
    aaauu
    """
    # HARTREE2EV = Units.Ha
    # has changed in hartree
    # print( "temperature",temperature)
    FT_T = temperature
    OutFunctional = FunctionalOutput(name="FT_TF")
    if "E" in calcType:
        ene = FT_WTEnergy(rho, FT_T)
        OutFunctional.energy = ene
    if "V" in calcType:
        OutFunctional.potential = FT_WTPotential(rho, FT_T)
    return OutFunctional


def get_WT_kernel_table(kernel_table: dict, rho0: float, temperature: float,
                        max_eta: float, neta: int, delta_eta: float):
    kernel_table['max_eta'] = max_eta
    kernel_table['neta'] = neta
    kernel_table['delta_eta'] = delta_eta
    kernel_table['rho0'] = rho0
    kernel_table['temperature'] = temperature

    return True
