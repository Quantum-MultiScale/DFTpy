# Collection of finite temperature Thomas Fermi functional

import numpy as np

from dftpy.functional.functional_output import FunctionalOutput
from dftpy.math_utils import PowerInt
from dftpy.time_data import timer
from dftpy.field import DirectField
from dftpy.functional.fedf import * 
from dftpy.constants import Units
__all__ = ['FT_GGA','FT_GGAStress']

def FT_GGAPotential(rho,FT_T,functional: str = "LKT"):
    """
    Finite Temperature GGA Potential
    """
    gtot  = rho.gradient() 
    gtot2 = (PowerInt(gtot[0], 2) + PowerInt(gtot[1], 2) + PowerInt(gtot[2], 2))
    
    vke,kes,eke = FT_GGA_libxclike(rho,gtot2,FT_T,functional)

    kes = 2.0*kes 
    kes2 = np.zeros_like(kes)
    g = rho.grid.get_reciprocal().g
    for icar in range(0, 3): 
        hx =  kes*gtot[icar] 
        hx_g = hx.fft()
        hx_g = 1j * g[icar] * hx_g
        hx = hx_g.ifft(force_real = True)
        kes2 = kes2 + hx 

    pot = vke - kes2 

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
    
    ene = tau_gga.sum()*rho.grid.dV

    return ene

def FT_GGAStress(rho,temperature=1e-3,functional: str = "LKT",**kwargs):
    """
    Finite Temperature GGA Stress
    """
    stress = np.zeros((3,3))
    gtot  = rho.gradient() 
    gtot2 = (PowerInt(gtot[0], 2) + PowerInt(gtot[1], 2) + PowerInt(gtot[2], 2))
    
    vke,kes,eke = FT_GGA_libxclike(rho,gtot2,temperature,functional)

    kes = 2.0*kes 
    #kes2 = np.zeros_like(kes)
    #g = rho.grid.get_reciprocal().g
    #for icar in range(0, 3): 
    #    hx =  kes*gtot[icar] 
    #    hx_g = hx.fft()
    #    hx_g = 1j * g[icar] * hx_g
    #    hx = hx_g.ifft(force_real = True)
    #    kes2 = kes2 + hx 

    stress_ii_den = eke - vke * rho - kes*gtot2 

    for i in range(3):
        stress[i, i] = stress_ii_den.sum()

    for ii in range(0,3):
        for jj in range(0,3):
            stree_tmp = kes * gtot[ii] * gtot[jj]
            stress[ii,jj] = stress[ii,jj] - stree_tmp.sum()
    stress *= rho.grid.dV/rho.grid.volume
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
        ene = FT_GGAEnergy(rho,FT_T,functional)
        OutFunctional.energy = ene 
    if "V" in calcType:
        OutFunctional.potential = FT_GGAPotential(rho,FT_T,functional)
    return OutFunctional

def get_Fs(s2,functional: str = "LKT",need_ds2=False):
    """

    """
    if functional=="LKT" : 
        lkta = 1.3
        safe_s2=np.clip(s2,1e-15,500)
        Fs = 1.0/np.cosh(lkta*np.sqrt(safe_s2)) + 5.0/3.0*safe_s2
        if need_ds2:
            Fs_ds2 = (-lkta * np.tanh(lkta * np.sqrt(s2)) / 
                       (np.cosh(lkta * np.sqrt(s2)) * 2.0 * np.sqrt(s2))) + (5.0 / 3.0)
    elif functional=="VT84F" :
        mu = 2.778
        a  = mu - 40.0/27.0
        Fs   = 1 - mu * s2 * np.exp(-a*s2)/(1.0+mu*s2) + (1.0-np.exp(-a*s2*s2))*(1.0/s2 - 1.0)  + 5.0/3.0 * s2
        if need_ds2: 
            Fs_ds2 = ( 2.0*a*(1.0/s2-1.0) *s2 * np.exp(-a*s2*s2) 
                   - (1.0-np.exp(-a*s2*s2) )/ s2/s2        
                   + mu*mu * s2 * np.exp(-a*s2)/(mu*s2+1.0)/(mu*s2+1.0) 
                   - mu * np.exp(-a*s2)/(mu*s2+1.0)        
                   + a*mu * s2 * np.exp(-a*s2)/(mu*s2+1.0) + 5.0/3.0 )
    elif functional=="VW" :
        Fs   = 5.0/3.0 * s2 
        if need_ds2: 
            Fs_ds2 = (5.0 / 3.0)

    elif functional=="TFVW" :
        Fs   = 1.0 + 5.0/3.0 * s2
        if need_ds2:
            Fs_ds2 = (5.0 / 3.0)

    elif functional=="TF" :
        Fs   = 0.0 * s2
        if need_ds2:
            Fs_ds2 = Fs
        
    if need_ds2:
        return Fs,Fs_ds2
    else : 
        return Fs 


def get_s2(rho,need_g=False) : 
    """
    
    """
    ckf = (3.0 * np.pi ** 2) ** (1.0 / 3.0)
    gtot  = rho.gradient() 
    gtot2 = (PowerInt(gtot[0], 2) + PowerInt(gtot[1], 2) + PowerInt(gtot[2], 2))
    s2 = gtot2/(4.0*ckf**2*PowerInt(rho,8,3))
    s2[s2 < 1e-15] = 1e-15
    if not need_g : 
        return s2 
    s2dg = 1.0 / ( 4.0 * ckf**2 * rho**(8.0/3.0) )
    s2drho = -(8.0/3.0) * gtot2 / ( 4.0 * ckf**2 * rho**(11.0/3.0) )
    return s2,s2drho,s2dg,gtot 

def FT_GGA_libxclike(rho,sigma,FT_T,functional: str = "LKT") : 
    
    ctf = (3.0/10.0) * ((3.0 * np.pi ** 2) ** (1.0 / 3.0)) ** 2.0
    ckf = (3.0 * np.pi ** 2) ** (1.0 / 3.0)
    tau_tf = ctf * PowerInt(rho,5,3)

    s2 = sigma/(4.0*ckf**2*PowerInt(rho,8,3))
    s2[s2 < 1e-15] = 1e-15
    s2_dg = 1.0 / ( 4.0 * ckf**2 * rho**(8.0/3.0) ) 
    s2_drho = -(8.0/3.0) * s2 / rho 

    t = get_reduce_t(rho,FT_T)
    t_drho = - (2.0/3.0) * t / rho
    h = FTH(t)
    h_dt = FTH_dt(t)
    h_dt2 = FTH_dt2(t)
    zeta = FTZETA(t)
    zeta_dt = FTZETA_dt(t)
    xi = FTXI(t)
    xi_dt = FTXI_dt(t)
    ### debug 
    #print(rho[0,0,0],h_dt2[0,0,0],zeta[0,0,0],zeta_dt[0,0,0],xi[0,0,0],xi_dt[0,0,0])

    s2_tau   = s2*(h-t*h_dt)/xi
    s2_sigma = s2*(t*h_dt)/zeta

    fs_tau,fs_tau_ds2 = get_Fs(s2_tau,functional=functional,need_ds2=True)
    fs_sigma,fs_sigma_ds2 = get_Fs(s2_sigma,functional=functional,need_ds2=True)

    fs_sigma     = 2.0 - fs_sigma
    fs_sigma_ds2 = - fs_sigma_ds2
    
    tau_gga  = tau_tf * ( xi * fs_tau - zeta * fs_sigma) 
    
    eke = tau_gga

    vke = np.zeros_like(rho)
    
    # part 1 \tau_tf
    vke = vke + (5.0/3.0)*tau_gga/rho

    # part 2 t in outer zeta and xi

    vke = vke + tau_tf * t_drho * ( xi_dt * fs_tau - zeta_dt * fs_sigma)

    # part 3 t in zeta, xi, and h in inner s2
    s2_tau_pre   = -(t*xi*h_dt2 - t*h_dt*xi_dt + h*xi_dt)/(xi*xi)
    s2_sigma_pre = (zeta * (t * h_dt2 + h_dt ) - t * h_dt * zeta_dt )/(zeta*zeta)
    vke = vke + tau_tf * t_drho * s2 * (xi*fs_tau_ds2*s2_tau_pre  - zeta*fs_sigma_ds2*s2_sigma_pre)
    
    # part 4 rho in s2
    vke = vke + tau_tf * s2_drho * (xi*fs_tau_ds2*(h-t*h_dt)/xi  - zeta*fs_sigma_ds2*(t*h_dt)/zeta)

    # part 5 rho in sigma part 
    kes = tau_tf * s2_dg * (xi * fs_tau_ds2 * (h-t*h_dt)/xi - zeta*fs_sigma_ds2*(t*h_dt)/zeta )

    return vke,kes,eke
