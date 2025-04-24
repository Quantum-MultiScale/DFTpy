import numpy as np

from dftpy.constants import ZERO
from dftpy.functional.functional_output import FunctionalOutput
from dftpy.functional.kedf.kernel import LindhardDerivative, WTKernel
from dftpy.mpi import sprint
from dftpy.time_data import timer

__all__ = ["WT", "WTStress"]


def WTPotential(rho, rho0, Kernel, alpha, beta):
    alphaMinus1 = alpha - 1.0
    betaMinus1 = beta - 1.0
    mask = rho < ZERO
    rho_saved = rho[mask]
    rho[mask] = ZERO
    if abs(beta - alpha) < 1e-9:
        rhoBeta = rho**beta
        rhoAlpha1 = rhoBeta / rho
        fac = 2.0 * alpha
        pot = fac * rhoAlpha1 * (rhoBeta.fft() * Kernel).ifft(force_real=True)
    else:
        pot = (
            alpha
            * rho**alphaMinus1
            * ((rho**beta).fft() * Kernel).ifft(force_real=True)
        )
        pot += (
            beta * rho**betaMinus1 * ((rho**alpha).fft() * Kernel).ifft(force_real=True)
        )
    rho[mask] = rho_saved

    return pot


def WTPotentialEdens(rho, rho0, Kernel, alpha, beta):
    mask = rho < 0.0
    rho[mask] = 0.0
    edens = rho**alpha * ((rho**beta).fft() * Kernel).ifft(force_real=True)
    return edens


def WTEnergy(rho, rho0, Kernel, alpha, beta):
    energydensity = WTEnergyDensity(rho, rho0, Kernel, alpha, beta)
    energy = energydensity.sum() * rho.grid.dV
    return energy


def WTEnergyDensity(rho, rho0, Kernel, alpha, beta):
    mask = rho < ZERO
    rho_saved = rho[mask]
    rho[mask] = ZERO
    rhoBeta = rho**beta
    if abs(beta - alpha) < 1e-9:
        rhoAlpha = rhoBeta
    else:
        rhoAlpha = rho**alpha
    pot1 = (rhoBeta.fft() * Kernel).ifft(force_real=True)
    energydensity = pot1 * rhoAlpha
    rho[mask] = rho_saved
    return energydensity


def get_ke_kernel_wt(
    q,
    rho,
    rho0=None,
    x=1.0,
    y=1.0,
    alpha=5.0 / 6.0,
    beta=5.0 / 6.0,
    ke_kernel_saved=None,
    **kwargs,
):
    if not rho0:
        rho0 = rho.amean()
    if ke_kernel_saved is None:
        KE_kernel_saved = {"Kernel": None, "rho0": 0.0, "shape": None}
    else:
        KE_kernel_saved = ke_kernel_saved
    if (
        abs(KE_kernel_saved["rho0"] - rho0) > 1e-6
        or np.shape(rho) != KE_kernel_saved["shape"]
    ):
        sprint("Re-calculate KE_kernel", np.shape(rho), level=1)
        # KE_kernel = WTkernel(q, rho0, alpha=alpha, beta=beta)
        KE_kernel = WTKernel(
            q, rho0, x=x, y=1.0, alpha=alpha, beta=beta
        )  # always remove whole vW
        KE_kernel_saved["Kernel"] = KE_kernel
        KE_kernel_saved["rho0"] = rho0
        KE_kernel_saved["shape"] = np.shape(rho)
    else:
        KE_kernel = KE_kernel_saved["Kernel"]
    return KE_kernel


def WTStress(
    rho,
    x=1.0,
    y=1.0,
    sigma=None,
    alpha=5.0 / 6.0,
    beta=5.0 / 6.0,
    energy=None,
    ke_kernel_saved=None,
    **kwargs,
):
    rho0 = rho.amean()
    g = rho.grid.get_reciprocal().g
    invgg = rho.grid.get_reciprocal().invgg
    q = rho.grid.get_reciprocal().q
    if energy is None:
        KE_kernel = get_ke_kernel_wt(
            q,
            rho,
            rho0=rho0,
            x=x,
            y=y,
            alpha=alpha,
            beta=beta,
            ke_kernel_saved=ke_kernel_saved,
            **kwargs,
        )
        energy = WTEnergy(rho, rho0, KE_kernel, alpha, beta)
    mask = rho.grid.get_reciprocal().mask
    # factor = 5.0 / (9.0 * alpha * beta * rho0 ** (alpha + beta - 5.0 / 3.0))
    tkf = 2.0 * (3.0 * rho0 * np.pi**2) ** (1.0 / 3.0)
    tkf = float(tkf)
    rhoG_A = (rho**alpha).fft() / rho.grid.volume
    rhoG_B = np.conjugate((rho**beta).fft()) / rho.grid.volume
    DDrho = LindhardDerivative(q / tkf, y) * rhoG_A * rhoG_B
    stress = np.zeros((3, 3))
    mask = rho.grid.get_reciprocal().mask
    for i in range(3):
        for j in range(i, 3):
            if i == j:
                fac = 1.0 / 3.0
            else:
                fac = 0.0
            den = (g[i][mask] * g[j][mask] * invgg[mask] - fac) * DDrho[mask]
            stress[i, j] = stress[j, i] = (np.einsum("i->", den)).real
    stress *= np.pi**2 / (alpha * beta * rho0 ** (alpha + beta - 2) * tkf / 2.0)
    for i in range(3):
        stress[i, i] -= 2.0 / 3.0 * energy / rho.grid.volume

    return stress


@timer()
def WT(
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
    q = rho.grid.get_reciprocal().q
    if rho0 is None:
        rho0 = rho.amean()

    KE_kernel = get_ke_kernel_wt(
        q,
        rho,
        rho0=rho0,
        x=x,
        y=y,
        alpha=alpha,
        beta=beta,
        ke_kernel_saved=ke_kernel_saved,
        **kwargs,
    )

    NL = FunctionalOutput(name="NL")

    if "V" in calcType:
        pot = WTPotential(rho, rho0, KE_kernel, alpha, beta)
        NL.potential = pot

    if "E" in calcType or 'D' in calcType:
        if abs(beta - alpha) < 1e-9 and "V" in calcType:
            energydensity = pot * rho / (2 * alpha)
        else:
            energydensity = WTEnergyDensity(rho, rho0, KE_kernel, alpha, beta)
        if 'D' in calcType:
            NL.energydensity = energydensity
        NL.energy = energydensity.sum() * rho.grid.dV

    return NL
