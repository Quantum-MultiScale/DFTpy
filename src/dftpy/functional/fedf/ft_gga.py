# Collection of finite temperature Thomas Fermi functional

import numpy as np

from dftpy.functional.functional_output import FunctionalOutput
from dftpy.math_utils import PowerInt
from dftpy.time_data import timer
from dftpy.field import DirectField
from dftpy.functional.fedf import FTK,FTK_dt,get_reduce_t
from dftpy.constants import Units
__all__ = ['FT_GGA','FT_GGAStress']

def FT_GGAPotential(rho,FT_T,functional: str = "LKT"):
    """
    Finite Temperature GGA Potential
    """

    pot *=  

    return pot 

def FT_GGAEnergy(rho,FT_T,functional: str = "LKT"):
    """
    Finite Temperature GGA Energy
    """
    ctf = (3.0/10.0) * ((3.0 * np.pi ** 2) ** (1.0 / 3.0)) ** 2.0
    tau_tf = ctf * PowerInt(rho,5,3)

    t = get_reduce_t(rho,FT_T)
    h = FTH(t)
    h_dt = FTH_dt(t)
    zeta = FTZETA(t)
    xi = FTXI(t)

    s2 = get_s2(rho)
    s2_tau   = s2*(h-t*h_dt)/xi
    s2_sigma = s2*(t*h_dt)/zeta
    
    fs_tau = get_Fs(s2_tau,functional)
    fs_sigma = get_Fs(s2_sigma,functional)

    fs_sigma = 2.0 - fs_sigma
    tau_gga  = tau_tf * ( xi * fs_tau - zeta * fs_sigma)
    
    ene = tau_gga.sum()

    return ene

def FT_GGAStress(rho,temperature=1e-3,functional: str = "LKT",**kwargs):
    """
    Finite Temperature GGA Stress
    """
    for i in range(3):
        stress[i, i] = stress_ii
    return stress 

def FT_GGA(rho:DirectField, functional: str = "LKT", calcType={"E", "V"}, temperature=1e-3, **kwargs):
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
        ene = FT_GGAEnergy(rho,FT_T,functional: str = "LKT")
        OutFunctional.energy = ene 
    if "V" in calcType:
        OutFunctional.potential = FT_GGAPotential(rho,FT_T,functional: str = "LKT")
    return OutFunctional

def get_Fs(s2:DirectField,functional: str = "LKT",need_ds2=False):
    """

    """
    if functional=="LKT" : 
        LKTa = 1.3 
        Fs = 1.0/np.cosh(lkta*np.sqrt(s2)) + 5.0/3.0*s2 
        if need_ds2:
            Fs_ds2 = - lkta*np.tanh(lkta*np.sqrt(S2))/np.cosh(lkta*np.sqrt(s2))/2.0/np.sqrt(s2) + 5.0/3.0

    if need_ds2:
        return Fs,Fs_ds2
    else
        return Fs 
    return 


def get_s2(rho,need_g=False) : 
    """
    
    """
    ckf = (3.0 * np.pi ** 2) ** (1.0 / 3.0)
    gtot  = rho.gradient() 
    gtot2 = (PowerInt(gtot[0], 2) + PowerInt(gtot[1], 2) + PowerInt(gtot[2], 2))
    s2 = gtot2/(4.0*ckf**2*PowerInt(rho,8,3))
    if not need_g : 
        return s2 
    s2dg = 1.0 / ( 4.0 * ckf**2 * rho**(8.0/3.0) )
    s2drho = -(8.0/3.0) * gtot2 / ( 4.0 * ckf**2 * rho**(11.0/3.0) )
    return s2,s2drho,s2dg,gtot 










