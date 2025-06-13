# Collection of finite temperature Thomas Fermi functional

import numpy as np

from dftpy.functional.functional_output import FunctionalOutput
from dftpy.math_utils import PowerInt
from dftpy.functional.fedf import fth, fth_dt, get_reduce_t

__all__ = ['FT_vW', 'FT_vWStress']


def get_vW_kedensity(rho):
    """
    positive define vW kinetic energy densiy  
    | \nabla \rho |^2 / (8 \rho) 
    """
    sqrt_rho = rho ** 0.5
    srhoGrad = sqrt_rho.gradient()
    eden = (PowerInt(srhoGrad[0], 2) + PowerInt(srhoGrad[1], 2) + PowerInt(srhoGrad[2], 2)) / 2.0
    return eden


def vW_GQ(rho, h):
    """
    the vW_GQ function moved from ATLAS 
    """
    sqrt_rho = rho ** 0.5
    gsrho = sqrt_rho.gradient()

    """
    old version 
    grho_x   = ( (h*gsrho[0]).gradient(ipol=1) 
               + (h*gsrho[1]).gradient(ipol=2) 
               + (h*gsrho[2]).gradient(ipol=3) )
    """
    g = rho.grid.get_reciprocal().g
    grho_x = np.zeros_like(rho)
    for icar in range(0, 3):
        hx = h * gsrho[icar]
        hx_g = hx.fft()
        hx_g = 1j * g[icar] * hx_g
        hx = hx_g.ifft(force_real=True)
        grho_x = grho_x + hx
    pot = - 0.5 * grho_x / sqrt_rho
    return pot


def FT_vWPotential(rho, FT_T):
    """
    Finite Temperature vW Potential
    
    tau -> ground state vW energy density 
    """
    t = get_reduce_t(rho, FT_T)
    h = fth(t)
    h_dt = fth_dt(t)
    # print("h",h[1,1,1])
    # print("h_dt",h_dt[1,1,1])
    tau = get_vW_kedensity(rho)
    pot = vW_GQ(rho, h)
    pot = pot + (-2.0 / 3.0) * tau * h_dt * t / rho
    return pot


def FT_vWEnergy(rho, FT_T):
    """
    Finite Temperature vW Energy
    """
    t = get_reduce_t(rho, FT_T)
    h = fth(t)
    tau = get_vW_kedensity(rho)
    eden = tau * h
    ene = eden.sum() * rho.grid.dV
    return ene


def FT_vWStress(rho, x=1.0, temperature=1e-3, **kwargs):
    """
    Finite Temperature vW Stress
    """
    stress = np.zeros((3, 3))
    t = get_reduce_t(rho, temperature)
    h = fth(t)
    h_dt = fth_dt(t)
    dtdrho = 2.0 / 3.0 * t / rho
    grho = rho.gradient()
    grho2 = (PowerInt(grho[0], 2) + PowerInt(grho[1], 2) + PowerInt(grho[2], 2))
    stress_ii = (h_dt * grho2 / 8.0 * dtdrho).sum()
    for ii in range(0, 3):
        for jj in range(0, 3):
            stress[ii, jj] = np.sum(- h / (4.0 * rho) * grho[ii] * grho[jj])
    for i in range(3):
        stress[i, i] += stress_ii
    stress *= rho.grid.dV / rho.grid.volume
    return stress


def FT_vW(rho, y=1.0, calcType={"E", "V"}, temperature=1e-3, **kwargs):
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
        ene = FT_vWEnergy(rho, FT_T)
        OutFunctional.energy = ene * y
    #    print("E",ene) 
    if "V" in calcType:
        OutFunctional.potential = FT_vWPotential(rho, FT_T) * y
        #    print("V",OutFunctional.potential)
    return OutFunctional
