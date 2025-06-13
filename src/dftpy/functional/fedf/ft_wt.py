# Collection of finite temperature Thomas Fermi functional
from IPython.terminal.shortcuts.auto_suggest import accept_and_keep_cursor
import numpy as np
from dftpy.functional.fedf.ft_lindhard import *
from dftpy.field import DirectField
from dftpy.functional.functional_output import FunctionalOutput
from dftpy.mpi import MP

__all__ = ['FT_WT', 'FT_WTStress']


def FT_WTPotential(rho, kernel, alpha=5.0 / 6.0, beta=5.0 / 6.0):
    """
    Finite Temperature WT Potentia
    """
    if (alpha - beta) > 1e-10:
        raise RuntimeError("alpha neq beta, FT_WTPotential not work!")
    rhoa = rho ** alpha
    rhob = rho ** beta
    frhob = rhob.fft()
    pot = (frhob * kernel).ifft()
    #    energy = (rhoa * pot).sum() * rho.grid.dV
    pot_out = (alpha + beta) * pot * rhoa / rho
    return pot_out


def FT_WTEnergy(rho: DirectField, kernel, alpha=5.0 / 6.0, beta=5.0 / 6.0):
    """
    Finite Temperature WT Energy
    """
    rhoa = rho ** alpha
    rhob = rho ** beta
    frhob = rhob.fft()
    pot = (frhob * kernel).ifft()
    energy = (rhoa * pot).sum() * rho.grid.dV
    return energy


def FT_WTStress(rho, ke_kernel_saved=None, temperature=1e-3,
                **kwargs):
    """
    Finite Temperature WT Stress
    """
    if 'kernel_table' not in ke_kernel_saved:
        raise RuntimeError("please calculate energy first, "
                           "then calculate stress")
    _check_stess_kernel_talbe(ke_kernel_saved,
                              rho0=rho.amean(),
                              temperature=temperature, mp=rho.grid.mp)
    kernel = ke_kernel_saved['kernel']
    kernel_table = ke_kernel_saved['kernel_table']
    #    print(kernel_table)
    """
    calcualte energy 
    """
    rhoa = rho ** kernel_table['alpha']
    rhob = rho ** kernel_table['beta']
    frhoa = rhoa.fft()
    frhob = rhob.fft()
    pot = (frhob * kernel).ifft()
    energy = np.sum(pot * rhoa)
    """
    stress part 1 
    """
    stress = np.zeros((3, 3))
    stress_ii = - 2.0 / 3.0 * energy
    for i in range(3):
        stress[i, i] = stress_ii
    #    print("s1", stress)
    """
    stress part 2.1
    """
    rho0 = rho.amean()
    q_norm = rho.grid.get_reciprocal().q
    q = rho.grid.get_reciprocal().g
    kernel_1 = fill_kernel_via_table(rho0, q_norm,
                                     kernel_table['eta'],
                                     kernel_table['s_weta'])
    for i in range(3):
        for j in range(3):
            peta = peta_pe(rho0, q_norm, q, i, j)
            s_kernel = kernel_1 * peta
            pot = (frhoa * s_kernel).ifft()
            stress[i, j] += np.sum(rhob * pot)

    #    print("s2", stress)
    """
    stress part 2.2
    """
    kernel_2 = fill_kernel_via_table(rho0, q_norm,
                                     kernel_table['eta'],
                                     kernel_table['s_wetaprho'])
    kernel_2 = kernel_2 * rho0
    #    print("k2", kernel_2[6, 6, 6])
    pot = (frhoa * kernel_2).ifft()
    for i in range(3):
        stress[i, i] -= np.sum(pot * rhob)
    #    print("s3", stress)

    stress *= rho.grid.dV / rho.grid.volume

    return stress


def FT_WT(rho, ke_kernel_saved, calcType={"E", "V"}, temperature=1e-3,
          **kwargs):
    """
    temperature in eV 
    FT_T in Ha
    """
    # HARTREE2EV = Units.Ha
    # has changed in hartree
    # print( "temperature",temperature)
    OutFunctional = FunctionalOutput(name="FT_WT")
    rho0 = rho.amean()
    neta = 10000
    max_eta = 50.0
    delta_eta = max_eta / (neta - 1)
    kernel = _fill_kernel(rho, ke_kernel_saved, rho0, temperature,
                          max_eta=max_eta,
                          neta=neta, delta_eta=delta_eta)
    if "E" in calcType:
        ene = FT_WTEnergy(rho, kernel)
        OutFunctional.energy = ene
    if "V" in calcType:
        OutFunctional.potential = FT_WTPotential(rho, kernel)
    return OutFunctional


def get_WT_kernel_table(kernel_table: dict, rho0: float, temperature: float,
                        max_eta: float, neta: int, delta_eta: float,
                        maxp=100000, alpha=5.0 / 6.0, beta=5.0 / 6.0,
                        mp=None) -> bool:
    if check_kernel_table(kernel_table, rho0, temperature): return False
    init_kernel_table(kernel_table, max_eta, neta, delta_eta, maxp)
    kernel_table['alpha'] = alpha
    kernel_table['beta'] = beta
    kf = (3.0 * np.pi ** 2 * rho0) ** (1.0 / 3.0)
    coef = np.pi ** 2.0 / (2.0 * beta * alpha) / rho0 ** (
            beta + alpha - 2.0) / kf
    chem_pot = get_chemical_potential(rho0, temperature)
    chi_tf = fermi__1_2_elegent(chem_pot / temperature, maxp=maxp)
    chi_tf = 0.5 * (2.0 * temperature) ** (0.5) * chi_tf / kf
    kernel_table['eta'] = np.zeros(neta)
    kernel_table['weta'] = np.zeros(neta)
    print("kernel table begin")
    if mp is None:
        print("mp is none")
        mp = MP()
    for ii in range(0, neta):
        if ii % mp.size != mp.rank: continue
        eta = ii * delta_eta
        if (eta < 1e-10):
            kernel_table['weta'][ii] = 0.0
            continue
        kernel_table['eta'][ii] = eta
        chi_vw = 4.0 / 3.0 / (2.0 * eta) ** 2
        chi_lr = ft_lindhard(eta, rho0, temperature, maxp=maxp)
        chi_lr = chi_lr / kf
        kernel_table['weta'][ii] = 1.0 / chi_lr - 1.0 / chi_vw - 1.0 / chi_tf
    kernel_table['eta'] = mp.vsum(kernel_table['eta'])
    kernel_table['weta'] = mp.vsum(kernel_table['weta'])
    print("kernel table end")

    kernel_table['weta'] = coef * kernel_table['weta']
    kernel_table['rho0'] = rho0
    kernel_table['temperature'] = temperature
    return True


def _fill_kernel(rho, ke_kernel_saved: dict, rho0: float, temperature: float,
                 max_eta: float, neta: int, delta_eta: float,
                 maxp=100000, alpha=5.0 / 6.0, beta=5.0 / 6.0):
    if 'kernel_table' not in ke_kernel_saved:
        ke_kernel_saved['kernel_table'] = {}

    kernel_table_update = get_WT_kernel_table(ke_kernel_saved['kernel_table'],
                                              rho0, temperature, max_eta, neta,
                                              delta_eta, maxp=maxp, alpha=alpha,
                                              beta=beta, mp=rho.grid.mp)
    if kernel_table_update or ke_kernel_saved.get('kernel') is None:
        tkf = 2.0 * (3.0 * np.pi ** 2 * rho0) ** (1.0 / 3.0)
        q = rho.grid.get_reciprocal().q
        kernel = q / tkf
        kernel_flat = kernel.flatten()
        kernel_flat = np.interp(kernel_flat,
                                ke_kernel_saved['kernel_table']['eta'],
                                ke_kernel_saved['kernel_table']['weta'])
        kernel = kernel_flat.reshape(kernel.shape)
        if q[0, 0, 0] < 1e-10:
            kernel[0, 0, 0] = 0.0
        ke_kernel_saved['kernel'] = kernel
    else:
        kernel = ke_kernel_saved['kernel']
    return kernel


def get_WT_stress_kernel_table(kernel_table: dict, rho0: float,
                               temperature: float, delta_rho=1e-5, ft_dx=1000,
                               mp=None):
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
        get_WT_kernel_table(kernel_table_col[i], rho_h[i], temperature,
                            kernel_table['max_eta'], kernel_table['neta'],
                            kernel_table['delta_eta'], kernel_table['maxp'],
                            kernel_table['alpha'], kernel_table['beta'], mp=mp)
    kernel_table['s_wetaprho'] = dfdx_5p(kernel_table_col[0]['weta'],
                                         kernel_table_col[1]['weta'],
                                         kernel_table_col[2]['weta'],
                                         kernel_table_col[3]['weta'], rho_1k)
    kernel_table['s_weta'] = dfdr(kernel_table['neta'],
                                  kernel_table['delta_eta'],
                                  kernel_table['weta'])
    return True


def _check_stess_kernel_talbe(ke_kernel_saved: dict, rho0: float,
                              temperature: float, mp=None):
    if 'kernel_table' not in ke_kernel_saved:
        raise RuntimeError("error kernel_table")

    need_update = not check_kernel_table(ke_kernel_saved['kernel_table'],
                                         rho0, temperature)
    if need_update or 's_weta' not in ke_kernel_saved['kernel_table']:
        get_WT_stress_kernel_table(ke_kernel_saved['kernel_table'], rho0,
                                   temperature, mp=mp)
    return True
