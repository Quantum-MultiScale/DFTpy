from dftpy.functional.kedf.wt import WT, WTStress
from dftpy.time_data import timer

"""
F. Perrot : Hydrogen-hydrogen interaction in an electron gas.
J. Phys. : Condens. Matter 6, 431 (1994).
"""

__all__ = ["FP", "FPStress"]


@timer()
def FP(
    rho,
    x=1.0,
    y=1.0,
    sigma=None,
    alpha=1.0,
    beta=1.0,
    rho0=None,
    calcType={"E", "V"},
    split=False,
    ke_kernel_saved=None,
    **kwargs,
):
    return WT(
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


def FPStress(
    rho,
    x=1.0,
    y=1.0,
    sigma=None,
    alpha=1.0,
    beta=1.0,
    energy=None,
    ke_kernel_saved=None,
    **kwargs,
):
    return WTStress(
        rho,
        x=x,
        y=y,
        sigma=sigma,
        alpha=alpha,
        beta=beta,
        energy=energy,
        ke_kernel_saved=ke_kernel_saved,
        **kwargs,
    )
