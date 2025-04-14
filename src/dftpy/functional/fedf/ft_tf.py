# Collection of finite temperature Thomas Fermi functional

import numpy as np

from dftpy.functional.functional_output import FunctionalOutput
from dftpy.math_utils import PowerInt
from dftpy.time_data import timer
from dftpy.field import DirectField
from dftpy.functional.fedf import FTK,FTK_dt,get_reduce_t
from dftpy.constants import Units
__all__ = ['FT_TF']

def FT_ThomasFermiPotential(rho,FT_T):
    """
    Finite Temperature Thomas-Fermi Potential
    """

    ctf = (3.0 / 10.0) * (3.0 * np.pi ** 2) ** (2.0 / 3.0)
    t = get_reduce_t(rho,FT_T)
    kappa = FTK(t)
    kappa_dt = FTK_dt(t)
    rho23 = PowerInt(rho, 2, 3)
    pot = ( (5.0/3.0) * rho23 * kappa 
        - (2.0/3.0) * rho23 * kappa_dt * t
        )
    pot *= ctf 

    return pot 

def FT_ThomasFermiEnergy(rho,FT_T):
    """
    Finite Temperature Thomas-Fermi Energy
    """

    t = get_reduce_t(rho,FT_T) 
    edens = PowerInt(rho, 5, 3)
    kappa = FTK(t)
    edens = kappa * edens
    edens *= (3.0 / 10.0) * (3.0 * np.pi ** 2) ** (2.0 / 3.0)
    ene = edens.sum() * rho.grid.dV

    return ene

def FT_TF(rho, x=1.0, calcType={"E", "V"}, temperature=1e-3, **kwargs):
    """
    temperature in eV 
    FT_T in Ha 
    """
    # HARTREE2EV = Units.Ha
    # has changed in hartree
    print( "temperature",temperature)
    FT_T = temperature  
    OutFunctional = FunctionalOutput(name="FT_TF")
    if "E" in calcType:
        ene = FT_ThomasFermiEnergy(rho,FT_T)
        OutFunctional.energy = ene * x 
    if "V" in calcType:
        OutFunctional.potential = FT_ThomasFermiPotential(rho,FT_T) * x 
    return OutFunctional 
