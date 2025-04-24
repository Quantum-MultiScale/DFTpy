# Collection of local and semilocal functional

import numpy as np

from dftpy.field import DirectField
from dftpy.functional.functional_output import FunctionalOutput
from dftpy.math_utils import PowerInt
from dftpy.time_data import timer

__all__ = ['TF', 'ThomasFermiStress', 'TTF']


def ThomasFermiPotential(rho):
    """
    The Thomas-Fermi Potential
    """
    factor = (3.0 / 10.0) * (5.0 / 3.0) * (3.0 * np.pi**2) ** (2.0 / 3.0)
    pot = factor * np.cbrt(rho * rho)
    # pot = rho * rho
    # pot = np.cbrt(pot, out = pot)
    # pot = np.multiply(factor, pot, out = pot)
    # pot = rho ** (2.0 / 3.0) * factor
    # return (3.0/10.0)*(5.0/3.0)*(3.0*np.pi**2)**(2.0/3.0)*np.abs(rho)**(2.0/3.0)
    return pot


def ThomasFermiEnergyDensity(rho):
    """
    The Thomas-Fermi EnergyDensity
    """
    # edens = (3.0/10.0)*(3.0*np.pi**2)**(2.0/3.0)*np.abs(rho)**(5.0/3.0)
    # edens = np.cbrt(rho * rho * rho * rho * rho)
    edens = PowerInt(rho, 5, 3)
    edens *= (3.0 / 10.0) * (3.0 * np.pi**2) ** (2.0 / 3.0)
    return edens


def ThomasFermiEnergy(rho):
    """
    The Thomas-Fermi Energy
    """
    edens = ThomasFermiEnergyDensity(rho)
    ene = edens.sum() * rho.grid.dV
    return ene


def ThomasFermiF(rho):
    ctf = 0.3 * (3.0 * np.pi * np.pi) ** (2.0 / 3.0)
    return 10.0 / 9.0 * ctf / np.cbrt(rho)


def ThomasFermiStress(rho, x=1.0, energy=None, **kwargs):
    """
    The Thomas-Fermi Stress
    """
    if energy is None:
        energy = TF(rho, x=x, calcType={"E"}).energy
    Etmp = -2.0 / 3.0 * energy / rho.grid.volume
    stress = np.zeros((3, 3))
    for i in range(3):
        stress[i, i] = Etmp
    return stress


@timer()
def TF(rho, x=1.0, calcType={"E", "V"}, split=False, **kwargs):
    OutFunctional = FunctionalOutput(name="TF")
    if "E" in calcType or "D" in calcType:
        energydensity = ThomasFermiEnergyDensity(rho)
        ene = energydensity.sum() * rho.grid.dV
        OutFunctional.energy = ene * x
        if 'D' in calcType:
            OutFunctional.energydensity = energydensity
    if "V" in calcType:
        pot = ThomasFermiPotential(rho)
        OutFunctional.potential = pot * x
    if "V2" in calcType:
        v2rho2 = ThomasFermiF(rho)
        OutFunctional.v2rho2 = v2rho2 * x
    return OutFunctional


@timer()
def TTF(rho, x=1.0, calcType={"E", "V"}, temperature=1e-3, temperature0=None, **kwargs):
    OutFunctional = FunctionalOutput(name="TTF")
    if "D" in calcType:
        raise AttributeError("Sorry TTF not support energy density")

    if temperature0 is not None:
        temperature = temperature0

    if "E" in calcType:
        if temperature0 is not None:
            OutFunctional.energy = TemperatureTFEnergy_old(rho, temperature) * x
        else:
            OutFunctional.energy = TemperatureTFEnergy(rho, temperature) * x
    if "V" in calcType:
        OutFunctional.potential = TemperatureTFPotential(rho, temperature) * x
    return OutFunctional


def TemperatureTFPotential(rho: DirectField, temperature: float) -> DirectField:
    '''
    Temperature dependent Thomas-Fermi potential
    Parameters
    ----------
    rho
    temperature: in Hartree

    Returns
    -------


    '''
    v_tf = ThomasFermiPotential(rho)
    theta = temperature / v_tf
    theta_cut = 1.36
    a = [0.016, -0.957, -0.293, 0.209]
    c = 0.752252778063675
    mask = theta >= theta_cut
    potential = np.zeros_like(theta)
    temp = np.ones_like(theta)
    for i in range(4):
        temp *= theta
        potential += a[i] * temp
    potential *= v_tf

    temp = 1.0 / PowerInt(theta, 3, 2)
    potential[mask] = temperature * (
        np.log(c * temp[mask]) + np.log(1 + c * temp[mask] / 2.0**1.5)
    )
    potential[mask] -= v_tf[mask]

    return potential


def TemperatureTFEnergy(rho: DirectField, temperature: float) -> float:
    '''
    Temperature dependent Thomas-Fermi energy
    Parameters
    ----------
    rho
    temperature: in Hartree

    Returns
    -------


    '''
    v_tf = ThomasFermiPotential(rho)
    ene_den_tf = ThomasFermiEnergyDensity(rho)
    theta = temperature / v_tf
    theta_cut = 1.36
    a = [0.016, -0.957 * 3, -0.293 * -3, -0.209]
    c = 0.752252778063675
    factor = (3.0 / 10.0) * (5.0 / 3.0) * (3.0 * np.pi**2) ** (2.0 / 3.0)
    k = temperature / factor
    mask = theta >= theta_cut
    ene_den = np.zeros_like(theta)
    temp = rho.copy()
    for i in range(4):
        ene_den += a[i] * temp
        temp *= theta
    ene_den *= temperature

    temp = 1.0 / PowerInt(theta, 3, 2)
    ene_den[mask] = (
        -2.0 * rho[mask]
        + 2**1.5 * k**1.5 * np.arctanh(c * temp[mask] / 2**1.5) / c
        + rho[mask] * np.log(1.0 + c / 2**1.5 * temp[mask])
        + rho[mask] * np.log(c * temp[mask])
        + 2**0.5
        * k
        * np.sqrt(theta[mask])
        * PowerInt(rho, 1, 3)[mask]
        * np.log(8.0 * k**3.0 - c**2.0 * PowerInt(rho, 2)[mask])
        / c
    ) * temperature - ene_den_tf[mask]

    energy = ene_den.integral()

    return energy


def TemperatureTFEnergy_old(rho: DirectField, temperature: float) -> float:
    '''
    Temperature dependent Thomas-Fermi energy
    Parameters
    ----------
    rho
    temperature: in Hartree

    Returns
    -------


    '''
    v_tf = ThomasFermiPotential(rho)
    theta = temperature / v_tf
    v_tf_t = TemperatureTFPotential(rho, temperature) + v_tf
    ene_den = v_tf_t * rho
    ene_den += (
        -2
        * np.sqrt(2)
        / 3
        / np.pi**2
        * temperature**2.5
        / PowerInt(theta, 5, 2)
        * np.sqrt(0.16 + PowerInt(theta, 2))
        / (1.0 - 0.14 * (np.exp(-theta) - np.exp(-3.68 * theta)))
    )
    energy = ene_den.integral()

    energy -= ThomasFermiEnergy(rho)

    return energy
