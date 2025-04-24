import numpy as np

from dftpy.functional.functional_output import FunctionalOutput
from dftpy.functional.kedf.tf import TF, ThomasFermiStress
from dftpy.functional.kedf.wt import WT, WTStress
from dftpy.time_data import timer

__all__ = ["WTE", "WTEStress"]

"""
Ref:
    Witt, William C., et al. "Random structure searching with orbital-free density functional theory." The Journal of Physical Chemistry A 125.7 (2021): 1650-1660.
"""


def WTEPotential(tf, wt):
    pot = (1 - wt.energy / tf.energy) * tf.potential + wt.potential
    pot *= np.exp(wt.energy / tf.energy)
    return pot


def WTEEnergy(tf, wt):
    energy = tf.energy * np.exp(wt.energy / tf.energy)
    return energy


def WTEStress(
    rho,
    x=1.0,
    y=1.0,
    sigma=None,
    alpha=5.0 / 6.0,
    beta=5.0 / 6.0,
    energy=None,
    ke_kernel_saved=None,
    calcType={"E"},
    rho0=None,
    **kwargs,
):
    if rho0 is None:
        rho0 = rho.amean()
    tf = TF(rho, x=x, calcType={"E"}, **kwargs)
    wt = WT(
        rho,
        x=x,
        y=y,
        sigma=sigma,
        alpha=alpha,
        beta=beta,
        rho0=rho0,
        calcType={"E"},
        ke_kernel_saved=ke_kernel_saved,
        **kwargs,
    )
    tf_stress = ThomasFermiStress(rho, x=x, energy=tf.energy, **kwargs)
    wt_stress = WTStress(
        rho,
        x=x,
        y=y,
        sigma=sigma,
        alpha=alpha,
        beta=beta,
        rho0=rho0,
        calcType={"E"},
        ke_kernel_saved=ke_kernel_saved,
        energy=wt.energy,
        **kwargs,
    )
    ex = np.exp(wt.energy / tf.energy)
    for i in range(3):
        wt_stress[i, i] += 2.0 / 3.0 * wt.energy / rho.grid.volume
    stress = wt_stress * ex + tf_stress * ex
    return stress


@timer()
def WTE(
    rho,
    x=1.0,
    y=1.0,
    sigma=None,
    alpha=5.0 / 6.0,
    beta=5.0 / 6.0,
    rho0=None,
    calcType={"E", "V"},
    split=False,
    ke_kernel_saved=None,
    **kwargs,
):

    if 'E' not in calcType:
        calcType = set(calcType)
        calcType.add('E')
    NL = FunctionalOutput(name="NL")
    tf = TF(rho, x=x, calcType=calcType, split=split, **kwargs)
    wt = WT(
        rho,
        x=x,
        y=y,
        sigma=sigma,
        alpha=alpha,
        beta=beta,
        rho0=rho0,
        calcType=calcType,
        split=split,
        ke_kernel_saved=ke_kernel_saved,
        **kwargs,
    )

    if "E" in calcType:
        NL.energy = WTEEnergy(tf, wt)

    if "V" in calcType:
        NL.potential = WTEPotential(tf, wt)

    return NL
