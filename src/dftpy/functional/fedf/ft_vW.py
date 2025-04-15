# Collection of finite temperature Thomas Fermi functional

import numpy as np

from dftpy.functional.functional_output import FunctionalOutput
from dftpy.math_utils import PowerInt
from dftpy.time_data import timer
from dftpy.field import DirectField
from dftpy.functional.fedf import FTK,FTK_dt,get_reduce_t
from dftpy.constants import Units
__all__ = ['FT_vW','FT_vWStress']

def FT_vWPotential(rho,FT_T):
    """
    Finite Temperature vW Potential
    """

    pot *=  

    return pot 

def FT_vWEnergy(rho,FT_T):
    """
    Finite Temperature vW Energy
    """

    return ene

def FT_vWStress(rho,x=1.0,temperature=1e-3,**kwargs):
    """
    Finite Temperature vW Stress
    """
    for i in range(3):
        stress[i, i] = stress_ii
    return stress 

def FT_vW(rho, calcType={"E", "V"}, temperature=1e-3, **kwargs):
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
        ene = FT_vWEnergy(rho,FT_T)
        OutFunctional.energy = ene 
    if "V" in calcType:
        OutFunctional.potential = FT_vWPotential(rho,FT_T)
    return OutFunctional 
