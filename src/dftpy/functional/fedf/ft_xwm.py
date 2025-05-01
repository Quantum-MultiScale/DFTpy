# Collection of finite temperature Thomas Fermi functional

import numpy as np

from dftpy.functional.functional_output import FunctionalOutput
from dftpy.math_utils import PowerInt
from dftpy.time_data import timer
from dftpy.field import DirectField
from dftpy.functional.fedf import ftk, ftk_dt, get_reduce_t
from dftpy.constants import Units

__all__ = ['FT_XWM', 'FT_XWMStress']

from examples.Pseudopotentials.lpps import rho_target


def FT_XWMPotential(rho, kernel1, kernel2, kappa: float = 0.0):
    """
    Finite Temperature XWM Potential
    """
    alpha = kappa + 5.0 / 6.0
    beta = kappa + 11.0 / 6.0
    rhoa = rho ** alpha
    rhob = rho ** beta
    frhoa = rhoa.fft()

    # part 1
    pot_tmp = (frhoa * kernel1).ifft()
    pot = 2.0 * alpha * pot_tmp * rhoa / rho

    # part 2
    frhob = rhob.fft()
    pot_tmp = (frhoa * kernel2).ifft()
    pot = pot + beta * pot_tmp * rhob / rho
    pot_tmp = (frhob * kernel2).ifft()
    pot = pot + alpha * pot_tmp * rhoa / rho

    return pot


def FT_XWMEnergy(rho, kernel1, kernel2, kappa: float = 0.0):
    """
    Finite Temperature XWM Energy
    """
    alpha = kappa + 5.0 / 6.0
    beta = kappa + 11.0 / 6.0
    rhoa = rho ** alpha
    rhob = rho ** beta
    frhoa = rhoa.fft()

    # part 1
    pot_tmp = (frhoa * kernel1).ifft()
    energy = sum(pot_tmp * rhoa)

    # part 2
    pot_tmp = (frhoa * kernel2).ifft()
    energy = energy + (pot_tmp * rhob)

    energy = energy * rho.grid.dV
    return energy


def FT_XWMStress(rho, x=1.0, temperature=1e-3, **kwargs):
    """
    Finite Temperature XWM Stress
    """
    for i in range(3):
        stress[i, i] = stress_ii
    return stress


def FT_XWM(rho, calcType={"E", "V"}, temperature=1e-3, **kwargs):
    """
    temperature in eV 
    FT_T in Ha 
    """
    # HARTREE2EV = Units.Ha
    # has changed in hartree
    # print( "temperature",temperature)
    FT_T = temperature
    OutFunctional = FunctionalOutput(name="FT_TF")
    if "E" in calcType:
        ene = FT_XWMEnergy(rho, FT_T)
        OutFunctional.energy = ene
    if "V" in calcType:
        OutFunctional.potential = FT_XWMPotential(rho, FT_T)
    return OutFunctional
