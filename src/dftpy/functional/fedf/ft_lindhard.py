"""
the Lindhard function used in Nonlocal Finite temperature free energy functionals
"""
import numpy as np

from dftpy.math_utils import PowerInt
from dftpy.functional.fedf import ftk, ftk_dt
from dftpy.functional.kedf.kernel import LindhardFunction
def get_chemical_potential(rho:float,temp:float):
    ctf = (3.0 / 10.0) * (3.0 * np.pi ** 2) ** (2.0 / 3.0)
    t = np.array([2.0 * temp / (3.0 * np.pi ** 2.0 * rho) ** (2.0 / 3.0)])
    kappa = ftk(t)
    kappa_dt = ftk_dt(t)
    pot = (5.0 / 3.0* rho ** (2.0 / 3.0) * kappa[0] &
           - 2.0 / 3.0 * rho ** (2.0 / 3.0) * kappa_dt[0] * t[0])
    return pot * ctf

def ft_lindhard(eta:float,rho:float,temp:float,maxp:int,temp0=None) -> float:
    if eta < 1e-20: return 0.0
    lind = 0.0
    beta = 1.0 / temp
    if temp0 :
        beta0 = 1.0 /temp0
    else :
        beta0 = beta
    chemical_potential = get_chemical_potential(rho,temp)
    kf = (3.0 * np.pi**2 * rho)**(1.0/3.0)
    max_x = 60.0 / beta0 + chemical_potential
    min_x = -60.0 / beta0 + chemical_potential
    min_x = 1e-20 if min_x < 0.0 else min_x
    dx = (max_x - min_x) /float(maxp)
    lind = 0.0
    for ip in range(0,maxp):
        e = -dx/2+ dx*ip + min_x
        fake_kf = np.sqrt(2.0*e)
        fake_eta = eta*kf /fake_kf
        lind_x = 1.0 / LindhardFunction(fake_eta,0.0,0.0)
        aa = beta / (4 * np.cosh((e - chemical_potential) * beta / 2) ** 2.0)
        lind += aa*lind_x
    return lind
