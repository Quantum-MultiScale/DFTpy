import numpy as np

from dftpy.functional.functional_output import FunctionalOutput
from dftpy.functional.kedf.kernel import MGPKernel, MGPOmegaE
from dftpy.functional.kedf.wt import WTEnergyDensity, WTPotential
from dftpy.mpi import sprint
from dftpy.time_data import timer

__all__ = ["MGP", "MGPStress", "MGPA", "MGPG", "MGP0"]


def MGPStress(
    rho, x=1.0, y=1.0, sigma=None, alpha=5.0 / 6.0, beta=5.0 / 6.0, calcType={"E", "V"}
):
    pass


@timer()
def MGP(
    rho,
    x=1.0,
    y=1.0,
    sigma=None,
    alpha=5.0 / 6.0,
    beta=5.0 / 6.0,
    lumpfactor=0.2,
    maxpoint=1000,
    symmetrization=None,
    calcType={"E", "V"},
    split=False,
    ke_kernel_saved=None,
    revke=True,
    **kwargs,
):
    q = rho.grid.get_reciprocal().q
    rho0 = rho.amean()
    if ke_kernel_saved is None:
        KE_kernel_saved = {"Kernel": None, "rho0": 0.0, "shape": None}
    else:
        KE_kernel_saved = ke_kernel_saved
    if (
        abs(KE_kernel_saved['rho0'] - rho0) > 1e-6
        or np.shape(rho) != KE_kernel_saved['shape']
    ):
        sprint("Re-calculate KE_kernel", comm=rho.mp.comm, level=1)
        KE_kernel = MGPKernel(
            q, rho0, maxpoints=maxpoint, symmetrization=symmetrization
        )
        if lumpfactor is not None:
            Ne = rho0 * rho.grid.volume
            KE_kernel += MGPOmegaE(q, Ne, lumpfactor, revke=revke)
        # -----------------------------------------------------------------------
        # rh0 = 0.03;lumpfactor = 0.0;q = np.linspace(1E-3, 8, 10000).reshape((1, 1, 1, -1))
        # mgp = MGPKernel(q,rho0,  maxpoints = maxpoint, symmetrization = None, KernelTable = None)
        # mgpa = MGPKernel(q,rho0, maxpoints = maxpoint, symmetrization = 'Arithmetic', KernelTable = None)
        # mgpg = MGPKernel(q,rho0, maxpoints = maxpoint, symmetrization = 'Geometric', KernelTable = None)
        # np.savetxt('mgp.dat', np.c_[q.ravel()/2.0, mgp.ravel(), mgpa.ravel(), mgpg.ravel()])
        # stop
        # -----------------------------------------------------------------------
        KE_kernel_saved["Kernel"] = KE_kernel
        KE_kernel_saved["rho0"] = rho0
        KE_kernel_saved["shape"] = np.shape(rho)
    else:
        KE_kernel = KE_kernel_saved["Kernel"]

    NL = FunctionalOutput(name="NL")
    if "E" in calcType or 'D' in calcType:
        energydensity = WTEnergyDensity(rho, rho0, KE_kernel, alpha, beta)
        NL.energy = energydensity.sum() * rho.grid.dV
        if 'D' in calcType:
            NL.energydensity = energydensity
    if "V" in calcType:
        pot = WTPotential(rho, rho0, KE_kernel, alpha, beta)
        NL.potential = pot
    return NL


def MGPA(
    rho,
    x=1.0,
    y=1.0,
    sigma=None,
    alpha=5.0 / 6.0,
    beta=5.0 / 6.0,
    lumpfactor=0.2,
    maxpoint=1000,
    symmetrization="Arithmetic",
    calcType={"E", "V"},
    split=False,
    **kwargs,
):
    return MGP(
        rho,
        x,
        y,
        sigma,
        alpha,
        beta,
        lumpfactor,
        maxpoint,
        "Arithmetic",
        calcType,
        split,
        **kwargs,
    )


def MGPG(
    rho,
    x=1.0,
    y=1.0,
    sigma=None,
    alpha=5.0 / 6.0,
    beta=5.0 / 6.0,
    lumpfactor=0.2,
    maxpoint=1000,
    symmetrization="Geometric",
    calcType={"E", "V"},
    split=False,
    **kwargs,
):
    return MGP(
        rho,
        x,
        y,
        sigma,
        alpha,
        beta,
        lumpfactor,
        maxpoint,
        "Geometric",
        calcType,
        split,
        **kwargs,
    )


def MGP0(
    rho,
    x=1.0,
    y=1.0,
    sigma=None,
    alpha=5.0 / 6.0,
    beta=5.0 / 6.0,
    lumpfactor=0.2,
    maxpoint=1000,
    symmetrization=None,
    calcType={"E", "V"},
    split=False,
    revke=False,
    **kwargs,
):
    return MGP(
        rho,
        x,
        y,
        sigma,
        alpha,
        beta,
        lumpfactor,
        maxpoint,
        symmetrization,
        calcType,
        split,
        revke=False,
        **kwargs,
    )
