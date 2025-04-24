from dftpy.functional.kedf.wt import WT, WTStress
from dftpy.time_data import timer

"""
E. Smargiassi and P.A. Madden : Orbital-free kinetic-energy functionals for first-principles molecular dynamics.
Phys.Rev.B 49,  5220 (1994).
"""

__all__ = ["SM", "SMStress"]


@timer()
def SM(
    rho,
    x=1.0,
    y=1.0,
    sigma=None,
    alpha=0.5,
    beta=0.5,
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


def SMStress(
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
