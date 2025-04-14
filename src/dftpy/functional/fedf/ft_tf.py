# Collection of finite temperature Thomas Fermi functional

import numpy as np

from dftpy.functional.functional_output import FunctionalOutput
from dftpy.math_utils import PowerInt
from dftpy.time_data import timer
from dftpy.field import DirectField
from dftpy.functional.fedf import FTK,FTK_dt,get_reduce_t


def FT_ThomasFermiPotential(rho,FT_T):
    """
    Finite Temperature Thomas-Fermi Potential
    """

    ctf = (3.0 / 10.0) * (5.0 / 3.0) * (3.0 * np.pi ** 2) ** (2.0 / 3.0)
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
    edens *= kappa 
    edens *= (3.0 / 10.0) * (3.0 * np.pi ** 2) ** (2.0 / 3.0)
    ene = edens.sum() * rho.grid.dV

    return ene 
