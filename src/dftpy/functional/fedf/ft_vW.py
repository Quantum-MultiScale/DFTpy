# Collection of finite temperature Thomas Fermi functional

import numpy as np

from dftpy.functional.functional_output import FunctionalOutput
from dftpy.math_utils import PowerInt
from dftpy.time_data import timer
from dftpy.field import DirectField
from dftpy.functional.fedf import FTH,FTH_dt,get_reduce_t

__all__ = ['FT_vW','FT_vWStress']

def get_vW_kedensity(rho):
    """
    positive define vW kinetic energy densiy  
    | \nabla \rho |^2 / (8 \rho) 
    """
    rhoGrad = rho.gradient()
    eden = (PowerInt(rhoGrad[0], 2) + PowerInt(rhoGrad[1], 2) + PowerInt(rhoGrad[2], 2)) / rho / 8.0
    return eden

def vW_GQ(rho,h):
    """
    the vW_GQ function moved from ATLAS 
    """
    sqrt_rho = PowerInt(rho, 1, 2)
    gsrho = sqrt_rho.gradient()
    
    grho_x   = ( (h*gsrho[0]).gradient(ipol=0) 
               + (h*gsrho[1]).gradient(ipol=1) 
               + (h*gsrho[2]).gradient(ipol=2) )
    pot = - 0.5 * grho_x/sqrt_rho 
    return pot

def FT_vWPotential(rho,FT_T):
    """
    Finite Temperature vW Potential
    
    tau -> ground state vW energy density 
    """
    t = get_reduce_t(rho,FT_T)
    h = FTH(t)
    h_dt = FTH_dt(t) 
    tau = get_vW_kedensity(rho) 
    pot = vW_GQ(rho,h)
    pot = pot + (-2.0/3.0) * tau * h_dt * t / rho

    return pot 

def FT_vWEnergy(rho,FT_T):
    """
    Finite Temperature vW Energy
    """
    t = get_reduce_t(rho,FT_T)
    h = FTH(t) 
    tau = get_vW_kedensity(rho)
    eden = tau * h * rho.grid.dV 
    ene = eden.sum() 
    return ene

def FT_vWStress(rho,x=1.0,temperature=1e-3,**kwargs):
    """
    Finite Temperature vW Stress
    """
    stress = 0.0 
    for i in range(3):
        stress[i, i] = stress_ii
    return stress 

def FT_vW(rho, y=1.0 ,calcType={"E", "V"}, temperature=1e-3, **kwargs):
    """
    temperature in eV 
    FT_T in Ha 
    """
    # HARTREE2EV = Units.Ha
    # has changed in hartree
    # print( "temperature",temperature)
    FT_T = temperature  
    OutFunctional = FunctionalOutput(name="FT_VW")
    if "E" in calcType:
        ene = FT_vWEnergy(rho,FT_T)
        OutFunctional.energy = ene * y 
    if "V" in calcType:
        OutFunctional.potential = FT_vWPotential(rho,FT_T) * y 
    return OutFunctional 
