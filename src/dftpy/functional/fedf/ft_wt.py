# Collection of finite temperature Thomas Fermi functional
from IPython.terminal.shortcuts.auto_suggest import accept_and_keep_cursor
import numpy as np
from dftpy.functional.fedf.ft_lindhard import *
from dftpy.field import DirectField
from dftpy.functional.functional_output import FunctionalOutput

__all__ = ['FT_WT', 'FT_WT']


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


def FT_WTStress(rho, x=1.0, temperature=1e-3, **kwargs):
    """
    Finite Temperature WT Stress
    """
    stress = np.zeros((3, 3))
    stress_ii = 0.0
    for i in range(3):
        stress[i, i] = stress_ii
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
    OutFunctional = FunctionalOutput(name="FT_TF")
    rho0 = np.mean(rho)
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
                        maxp=100000, alpha=5.0 / 6.0, beta=5.0 / 6.0) -> bool:
    if check_kernel_table(kernel_table, rho0, temperature): return False
    init_kernel_table(kernel_table, max_eta, neta, delta_eta)
    kf = (3.0 * np.pi ** 2 * rho0) ** (1.0 / 3.0)
    coef = np.pi ** 2.0 / (2.0 * beta * alpha) / rho0 ** (
            beta + alpha - 2.0) / kf
    chem_pot = get_chemical_potential(rho0, temperature)
    chi_tf = fermi__1_2_elegent(chem_pot / temperature, maxp=maxp)
    chi_tf = 0.5 * (2.0 * temperature) ** (0.5) * chi_tf / kf
    kernel_table['eta'] = np.zeros(neta)
    kernel_table['weta'] = np.zeros(neta)
    print("kernel table begin")
    for ii in range(0, neta):
        eta = ii * delta_eta
        if (eta < 1e-20):
            kernel_table['weta'][ii] = 0.0
            continue
        kernel_table['eta'][ii] = eta
        chi_vw = 4.0 / 3.0 / (2.0 * eta) ** 2
        chi_lr = ft_lindhard(eta, rho0, temperature, maxp=maxp)
        kernel_table['weta'][ii] = 1.0 / chi_lr - 1.0 / chi_vw - 1.0 / chi_tf
        print("ii", ii, chi_tf, chi_vw, chi_lr, kernel_table['weta'][ii])
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
                                              rho0, temperature,
                                              max_eta, neta, delta_eta,
                                              maxp=maxp, alpha=alpha,
                                              beta=beta)
    if kernel_table_update or (ke_kernel_saved['kernel'] is None):
        tkf = 2.0 * (3.0 * np.pi ** 2 * rho0) ** (1.0 / 3.0)
        q = rho.grid.get_reciprocal().q
        kernel = q / tkf
        kernel_flat = kernel.flatten()
        kernel_flat = np.interp(kernel_flat,
                                ke_kernel_saved['kernel_table']['eta'],
                                ke_kernel_saved['kernel_table']['weta'])
        print(kernel[6, 6, 6])
        kernel = kernel_flat.reshape(kernel.shape)
        if q[0, 0, 0] < 1e-20:
            kernel[0, 0, 0] = 0.0
        ke_kernel_saved['kernel'] = kernel
    else:
        kernel = ke_kernel_saved['kernel']
    return kernel
