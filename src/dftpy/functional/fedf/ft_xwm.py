# Collection of finite temperature Thomas Fermi functional

import numpy as np
from sympy.abc import kappa

from dftpy.functional.functional_output import FunctionalOutput
from dftpy.functional.fedf.ft_lindhard import *
from dftpy.math_utils import PowerInt
from dftpy.time_data import timer
from dftpy.field import DirectField
from dftpy.functional.fedf import ftk, ftk_dt, get_reduce_t
from dftpy.constants import Units

__all__ = ['FT_XWM', 'FT_XWMStress']


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
    energy = np.sum(pot_tmp * rhoa)

    # part 2
    pot_tmp = (frhoa * kernel2).ifft()
    energy = energy + np.sum(pot_tmp * rhob)

    energy = energy * rho.grid.dV
    return energy


def FT_XWMStress(rho, ke_kernel_saved=None, temperature=1e-3, **kwargs):
    """
    Finite Temperature XWM Stress
    """
    if 'kernel_table' not in ke_kernel_saved:
        raise RuntimeError("please calculate energy first, "
                           "then calculate stress")
    _check_stess_kernel_talbe(ke_kernel_saved,
                              rho0=np.mean(rho),
                              temperature=temperature)

    kernel1 = ke_kernel_saved['kernel1']
    kernel2 = ke_kernel_saved['kernel2']

    kernel_table = ke_kernel_saved['kernel_table']

    stress = np.zeros((3, 3))
    kappa = kernel_table['kappa']
    a = kappa + 5.0 / 6.0
    b = kappa + 11.0 / 6.0
    rhoa = rho ** a
    rhob = rho ** b
    frhoa = rhoa.fft()
    frhob = rhob.fft()
    """
    part 1 
    """
    etmp = (frhoa * kernel1).ifft()
    ee = -(2.0 / 3.0 + 2 * kappa) * np.sum(etmp * rhoa)
    for i in range(3):
        stress[i, i] += ee
    etmp = (frhoa * kernel2).ifft()
    ee = -(5.0 / 3.0 + 2 * kappa) * np.sum(etmp * rhob)
    for i in range(3):
        stress[i, i] += ee
    """
    part 2 
    """
    rho0 = kernel_table['rho0']
    q_norm = rho.grid.get_reciprocal().q
    q = rho.grid.get_reciprocal().g
    kernel11 = fill_kernel_via_table(rho0, q_norm,
                                     kernel_table['eta'],
                                     kernel_table['s_weta1'])
    kernel12 = fill_kernel_via_table(rho0, q_norm,
                                     kernel_table['eta'],
                                     kernel_table['s_wetaprho1'])
    kernel21 = fill_kernel_via_table(rho0, q_norm,
                                     kernel_table['eta'],
                                     kernel_table['s_weta2'])
    kernel22 = fill_kernel_via_table(rho0, q_norm,
                                     kernel_table['eta'],
                                     kernel_table['s_wetaprho2'])
    kernel12 *= rho0
    kernel22 *= rho0
    #    print("debug", kernel11[6, 6, 6], kernel12[6, 6, 6], kernel21[6, 6, 6],
    #         kernel22[6, 6, 6])
    for i in range(3):
        for j in range(3):
            peta = peta_peS(rho0, q_norm, q, i, j)
            kernelx1 = kernel11 * peta
            kernelx2 = kernel21 * peta
            if i == j:
                kernelx1 -= kernel12
                kernelx2 -= kernel22
            stress[i, j] += np.sum((frhoa * kernelx1).ifft() * rhoa)
            stress[i, j] += np.sum((frhoa * kernelx2).ifft() * rhob)

    stress *= rho.grid.dV / rho.grid.volume
    return stress


def FT_XWM(rho, ke_kernel_saved=None, calcType={"E", "V"}, kappa=0.0,
           xwm_beta=1.0, temperature=1e-3, **kwargs):
    """
    temperature in eV 
    FT_T in Ha 
    """
    # HARTREE2EV = Units.Ha
    # has changed in hartree
    # print( "temperature",temperature)

    OutFunctional = FunctionalOutput(name="FT_XWM")
    rho0 = np.mean(rho)
    neta = 10000
    max_eta = 50.0
    delta_eta = max_eta / (neta - 1)
    kernel1, kernel2 = _fill_kernel(rho, ke_kernel_saved, rho0, temperature,
                                    max_eta=max_eta,
                                    neta=neta, delta_eta=delta_eta, kappa=kappa,
                                    xwm_beta=xwm_beta)
    # print("k1,k2", kernel1[6, 6, 6], kernel2[6, 6, 6])
    if "E" in calcType:
        ene = FT_XWMEnergy(rho, kernel1, kernel2, kappa=kappa)
        OutFunctional.energy = ene
    if "V" in calcType:
        OutFunctional.potential = FT_XWMPotential(rho, kernel1, kernel2,
                                                  kappa=kappa)
    return OutFunctional


def get_XWM_kernel_table(kernel_table: dict, rho0: float, temperature: float,
                         max_eta: float, neta: int, delta_eta: float,
                         maxp=100000, kappa=0.0, xwm_beta=1.0) -> bool:
    if check_kernel_table(kernel_table, rho0, temperature): return False
    init_kernel_table(kernel_table, max_eta, neta, delta_eta, maxp)
    kernel_table['kappa'] = kappa
    kernel_table['xwm_beta'] = xwm_beta
    kernel_table['rho0'] = rho0
    kernel_table['temperature'] = temperature

    ## def coe1 :
    fact1 = np.pi ** 2 / ((3.0 * np.pi ** 2) ** (1.0 / 3.0))
    xwm_coe1 = 18.0 / (6 * kappa + 5.0) ** 2 / rho0 ** (2 * kappa)
    xwm_coe2 = 1.0 / rho0 ** (2 * kappa) / 2.0
    xwm_c11 = 1.0 / ((kappa + 5.0 / 6) * (kappa + 11.0 / 6.0))
    xwm_c12 = -rho0 / (kappa + 5.0 / 6) ** 2

    kf = (3.0 * np.pi ** 2 * rho0) ** (1.0 / 3.0)
    kf_drho = kf / 3.0 / rho0
    chem_pot = get_chemical_potential(rho0, temperature)
    chi_tf = fermi__1_2_elegent(chem_pot / temperature, maxp=maxp)

    chi_tf_drho = fermi__2_elegent_drho(chem_pot / temperature,
                                        temperature, rho0, maxp=maxp)
    chi_tf = 0.5 * (2.0 * temperature) ** (0.5) * chi_tf / kf
    chi_tf_drho = 1.0 / 2.0 * (2.0 * temperature) ** (0.5) * chi_tf_drho / kf
    chi_tf_drho = chi_tf_drho - chi_tf / rho0 / 3.0
    #    print("chi_tf_drho", chi_tf_drho)
    kernel_table['eta'] = np.zeros(neta)
    k1 = np.zeros(neta)
    k2 = np.zeros(neta)

    print("kernel table begin")
    for ii in range(0, neta):
        eta = ii * delta_eta
        if (eta < 1e-10):
            k1[ii] = 0.0
            k2[ii] = 0.0
            continue
        kernel_table['eta'][ii] = eta
        # part 1
        chi_vw = 4.0 / 3.0 / (2.0 * eta) ** 2
        chi_lr0 = ft_lindhard(eta, rho0, temperature, maxp=maxp)
        chi_lr = chi_lr0 / kf
        k1[ii] = 1.0 / chi_lr - 1.0 / chi_vw - 1.0 / chi_tf
        # part 2
        chi_lr_drho = ft_lindhard_drho(eta, rho0, temperature, maxp=maxp)
        chi_vw = 2.0 * eta ** 2.0 / rho0
        k2[ii] = ((kf_drho * chi_lr0 - kf * chi_lr_drho) / chi_lr0 ** 2.0
                  + chi_vw + chi_tf_drho / chi_tf ** 2.0)
    #    print("ieta", eta, ii, chi_vw, chi_lr_drho, k2[ii])
    #    print("ieta2", eta, ii, kf_drho, kf, chi_tf, chi_lr0)

    k1 = k1 * fact1 * xwm_coe1
    k2 = k2 * fact1 * xwm_coe2
    xwm_c12 = xwm_beta * xwm_c12
    xwm_c11 = xwm_beta * xwm_c11
    kernel_table['weta1'] = k1 + xwm_c12 * k2
    kernel_table['weta2'] = xwm_c11 * k2

    # print("66", k1[6], k2[6])

    return True


def _fill_kernel(rho, ke_kernel_saved: dict, rho0: float, temperature: float,
                 max_eta: float, neta: int, delta_eta: float,
                 maxp=100000, kappa=0.0, xwm_beta=1.0):
    if 'kernel_table' not in ke_kernel_saved:
        ke_kernel_saved['kernel_table'] = {}

    kernel_table_update = get_XWM_kernel_table(ke_kernel_saved['kernel_table'],
                                               rho0, temperature,
                                               max_eta, neta, delta_eta,
                                               maxp=maxp, kappa=kappa,
                                               xwm_beta=xwm_beta)

    if kernel_table_update or ke_kernel_saved.get('kernel1') is None:
        q_norm = rho.grid.get_reciprocal().q
        kernel1 = fill_kernel_via_table(rho0, q_norm,
                                        ke_kernel_saved['kernel_table']['eta'],
                                        ke_kernel_saved['kernel_table']['weta1']
                                        )
        kernel2 = fill_kernel_via_table(rho0, q_norm,
                                        ke_kernel_saved['kernel_table']['eta'],
                                        ke_kernel_saved['kernel_table']['weta2']
                                        )
        ke_kernel_saved['kernel1'] = kernel1
        ke_kernel_saved['kernel2'] = kernel2
    else:
        kernel1 = ke_kernel_saved['kernel1']
        kernel2 = ke_kernel_saved['kernel2']
    return kernel1, kernel2


def get_XWM_stress_kernel_table(kernel_table: dict, rho0: float,
                                temperature: float, delta_rho=1e-5, ft_dx=1000):
    kernel_table_col = [{}, {}, {}, {}]
    if rho0 < 2.0 * delta_rho:
        rho_1k = rho0 / ft_dx
    else:
        rho_1k = delta_rho
    rho_h = np.full(4, rho0)
    rho_h[0] -= 2.0 * rho_1k
    rho_h[1] -= 1.0 * rho_1k
    rho_h[2] += 1.0 * rho_1k
    rho_h[3] += 2.0 * rho_1k
    for i in range(4):
        rho013 = rho0 ** (1.0 / 3.0)
        rhoh13 = rho_h[i] ** (1.0 / 3.0)
        me1 = kernel_table['max_eta'] * rho013 / rhoh13
        de1 = kernel_table['delta_eta'] * rho013 / rhoh13
        get_XWM_kernel_table(kernel_table_col[i], rho_h[i], temperature,
                             me1, kernel_table['neta'],
                             de1, kernel_table['maxp'],
                             kernel_table['kappa'], kernel_table['xwm_beta'])

    kernel_table['s_wetaprho1'] = dfdx_5p(kernel_table_col[0]['weta1'],
                                          kernel_table_col[1]['weta1'],
                                          kernel_table_col[2]['weta1'],
                                          kernel_table_col[3]['weta1'], rho_1k)
    kernel_table['s_weta1'] = dfdr(kernel_table['neta'],
                                   kernel_table['delta_eta'],
                                   kernel_table['weta1'])

    kernel_table['s_wetaprho2'] = dfdx_5p(kernel_table_col[0]['weta2'],
                                          kernel_table_col[1]['weta2'],
                                          kernel_table_col[2]['weta2'],
                                          kernel_table_col[3]['weta2'], rho_1k)
    kernel_table['s_weta2'] = dfdr(kernel_table['neta'],
                                   kernel_table['delta_eta'],
                                   kernel_table['weta2'])
    return True


def _check_stess_kernel_talbe(ke_kernel_saved: dict, rho0: float,
                              temperature: float):
    if 'kernel_table' not in ke_kernel_saved:
        raise RuntimeError("error kernel_table")

    need_update = not check_kernel_table(ke_kernel_saved['kernel_table'],
                                         rho0, temperature)

    if need_update or 's_weta1' not in ke_kernel_saved['kernel_table']:
        get_XWM_stress_kernel_table(ke_kernel_saved['kernel_table'], rho0,
                                    temperature)

    return True
