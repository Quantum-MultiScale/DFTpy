import numpy as np
from scipy.interpolate import splev, splrep

from dftpy.field import DirectField
from dftpy.functional.functional_output import FunctionalOutput
from dftpy.functional.kedf.kernel import (
    LWTKernelKf,
    MGPKernelTable,
    MGPOmegaE,
    WTKernelTable,
)
from dftpy.mpi import sprint
from dftpy.time_data import timer

__all__ = ["LWT", "LWTStress", "LMGP", "LMGPA", "LMGPG"]


def LWTStress(
    rho,
    x=1.0,
    y=1.0,
    sigma=None,
    interp="linear",
    kerneltype="WT",
    symmetrization=None,
    lumpfactor=None,
    alpha=5.0 / 6.0,
    beta=5.0 / 6.0,
    etamax=50.0,
    neta=50000,
    order=3,
    calcType={"E", "V"},
    energy=None,
    ke_kernel_saved=None,
    **kwargs,
):
    pass


def guess_kf_bound(kf, kfmin=None, kfmax=None, kftol=1e-3, ke_kernel_saved=None):
    if ke_kernel_saved is not None:
        kfmin_prev = ke_kernel_saved['kfmin']
        kfmax_prev = ke_kernel_saved['kfmax']
    else:
        kfmin_prev = None
        kfmax_prev = None

    if kfmin is not None and kfmax is not None:
        return [kfmin, kfmax]

    kf_l = kf.amin()
    kf_r = kf.amax()

    if kfmin_prev is None:
        kfmin_prev = 10
        if kfmin is None:
            kfmin = kf_l

    if kfmax_prev is None:
        kfmax_prev = 1.0
        if kfmax is None:
            kfmax = kf_r

    if kfmin is None or kfmin > kfmin_prev:
        kfmin = kfmin_prev
    if kfmax is None or kfmax < kfmax_prev:
        kfmax = kfmax_prev

    if kfmin > kf_l:
        kfl = [1e-5, 1e-4, 5e-3, 1e-3, 5e-3, 1e-2, 5e-2, 0.1, 0.5, 1.0]
        ratio = kf_l / kfmin
        for i in range(len(kfl) - 1, 0, -1):
            if ratio > kfl[i]:
                kfmin *= kfl[i]
                break
    if kfmin < kftol:
        kfmin = kftol

    if kfmax < kf_r:
        dk = kf_r - kfmax
        kfmax += (np.round(dk / 0.5) + 1) * 0.5

    if ke_kernel_saved is not None:
        ke_kernel_saved['kfmin'] = kfmin
        ke_kernel_saved['kfmax'] = kfmax

    kfBound = [kfmin, kfmax]
    return kfBound


def LWTPotentialEnergy(
    rho,
    etamax=100.0,
    ratio=1.15,
    nsp=None,
    delta=None,
    kdd=3,
    alpha=5.0 / 6.0,
    beta=5.0 / 6.0,
    interp="linear",
    calcType={"E", "V"},
    kfmin=None,
    kfmax=None,
    ldw=None,
    ke_kernel_saved=None,
    **kwargs,
):
    """
    ldw : local density weight
    """

    KE_kernel_saved = ke_kernel_saved
    savetol = 1e-16
    q = rho.grid.get_reciprocal().q
    rho0 = rho.amean()
    kf0 = 2.0 * np.cbrt(3.0 * np.pi**2 * rho0)
    kf = 2.0 * np.cbrt(3.0 * np.pi**2 * rho)
    kf = DirectField(
        grid=rho.grid, memo=rho.memo, rank=rho.rank, griddata_3d=kf, cplx=rho.cplx
    )
    # gamma = 1.0
    # rhomod = (0.5 * (rho **gamma + rho0 ** gamma)) ** (1.0/gamma)
    # kf = 2.0 * (3.0 * np.pi ** 2 * rhomod) ** (1.0 / 3.0)

    ### HEG
    if abs(kf.amax() - kf.amin()) < 1e-8:
        NL = FunctionalOutput(name="NL", energy=0.0)
        if 'D' in calcType:
            energydensity = DirectField(
                grid=rho.grid, memo=rho.memo, rank=rho.rank, cplx=rho.cplx
            )
            NL.energydensity = energydensity
        if 'V' in calcType:
            pot = DirectField(
                grid=rho.grid, memo=rho.memo, rank=rho.rank, cplx=rho.cplx
            )
            NL.potential = pot
        return NL

    kfmin, kfmax = guess_kf_bound(kf, kfmin, kfmax, ke_kernel_saved=ke_kernel_saved)
    kfBound = [kfmin, kfmax]

    if 'V' in calcType:
        vcalc = True
    else:
        vcalc = False
        kdd = 1

    # ----------------Test WT------------------------------------------------
    # kfBound = [kf0, kf0]
    # nsp = 2
    # -----------------------------------------------------------------------
    # ignore the vacuum density contribution
    # mask1 = kf < kfBound[0]
    # kf[mask1] = kfBound[0]
    # -----------------------------------------------------------------------
    mask2 = kf > kfBound[1]
    kf[mask2] = kfBound[1]
    if kfmin is None:
        kfMin = max(kf.amin(), kfBound[0])
    else:
        kfMin = kfmin

    if kfmax is None:
        kfMax = kf.amax()
    else:
        kfMax = kfmax

    if nsp is not None:
        # kflists = kfMin + (kfMax - kfMin)/(nsp - 1) * np.arange(nsp)
        if kfMax - kfMin < 1e-3:
            kflists = [kfMin, kfMax]
            nsp = 2
        else:
            kflists = kfMin + (kfMax - kfMin) / (nsp - 1) * np.arange(nsp)
        kflists = np.asarray(kflists)
    elif delta is not None:  # delta = 0.10
        nsp = int(np.ceil((kfMax - kfMin) / delta)) + 1
        kflists = np.linspace(kfMin, kfMax, nsp)
    else:
        nsp = int(np.ceil(np.log(kfMax / kfMin) / np.log(ratio))) + 1
        kflists = kfMin * ratio ** np.arange(nsp)
        # kflists = np.geomspace(kfMin, kfMax, nsp)
    kflists[0] -= savetol  # for numerical safe
    kflists[-1] += savetol  # for numerical safe
    sprint(
        'nsp',
        nsp,
        kfMax,
        kfMin,
        kf0,
        np.max(kflists),
        np.min(kflists),
        comm=rho.mp.comm,
        level=1,
    )
    # -----------------------------------------------------------------------
    kernel0 = np.empty_like(q)
    kernel1 = np.empty_like(q)
    kernelDeriv0 = np.empty_like(q)
    kernelDeriv1 = np.empty_like(q)
    Rmask = np.empty_like(rho)
    pot1 = np.zeros_like(rho)
    pot2G = None
    pot3 = np.zeros_like(rho)
    # pot4 = np.zeros_like(rho)
    # -----------------------------------------------------------------------
    mask = rho > 0
    mask2 = np.invert(mask)
    rho_saved = rho[mask2]
    rho[mask2] = 1e-30

    rhoAlpha = rho**alpha
    rhoAlpha1 = rhoAlpha / rho
    if abs(alpha - beta) < 1e-8:
        rhoBeta = rhoAlpha
        rhoBeta1 = rhoAlpha1
    else:
        rhoBeta = rho**beta
        rhoBeta1 = rhoBeta / rho
    rhoBetaG = rhoBeta.fft()
    # if abs(alpha - beta) < 1e-8:
    # rhoAlphaG = rhoBetaG
    # else:
    # rhoAlphaG = rhoAlpha.fft()

    rho[mask2] = rho_saved
    # -----------------------------------------------------------------------

    KernelTable = KE_kernel_saved["KernelTable"]
    KernelDeriv = KE_kernel_saved["KernelDeriv"]
    MGPKernelE = KE_kernel_saved["MGPKernelE"]
    if kdd > 1 or interp == "hermite":
        mcalc = True
    else:
        mcalc = False
    # -----------------------------------------------------------------------
    for i in range(nsp - 1):
        if i == 0:
            kernel0 = LWTKernelKf(
                q, kflists[i], KernelTable, etamax=etamax, out=kernel0
            )
            p0 = (kernel0 * rhoBetaG).ifft(force_real=True)
            if mcalc:
                kernelDeriv0 = (
                    LWTKernelKf(
                        q, kflists[i], KernelDeriv, etamax=etamax, out=kernelDeriv0
                    )
                    / kflists[i]
                )
                m0 = (kernelDeriv0 * rhoBetaG).ifft(force_real=True)
        else:
            p0, p1 = p1, p0
            kernel0, kernel1 = kernel1, kernel0
            if mcalc:
                m0, m1 = m1, m0
                kernelDeriv0, kernelDeriv1 = kernelDeriv1, kernelDeriv0

        kernel1 = LWTKernelKf(
            q, kflists[i + 1], KernelTable, etamax=etamax, out=kernel1
        )
        p1 = (kernel1 * rhoBetaG).ifft(force_real=True)
        if mcalc:
            kernelDeriv1 = (
                LWTKernelKf(
                    q, kflists[i + 1], KernelDeriv, etamax=etamax, out=kernelDeriv1
                )
                / kflists[i + 1]
            )
            m1 = (kernelDeriv1 * rhoBetaG).ifft(force_real=True)

        mask = np.logical_and(
            kf > kflists[i], kf < kflists[i + 1] + 1e-18
        )  # 1E-18 for numerical errors, must be very small
        # -----------------------------------------------------------------------
        if i == 0:
            small = kf < kflists[0] + 1e-18
            if len(small) > 0:
                Dkf = kflists[i]
                t = (kf - kflists[i]) / Dkf
                pot1[small] = p0[small] * t[small]
                if kdd > 1:
                    Rmask[:] = 0.0
                    Rmask[small] = 1.0
                    rhoU = rhoAlpha * Rmask
                    if pot2G is None:
                        pot2G = kernel0 * (rhoU * t).fft()
                    if kdd == 3:
                        pot3[mask] = m0[mask] * t[mask]
        # -----------------------------------------------------------------------
        Rmask[:] = 0.0
        Rmask[mask] = 1.0
        rhoU = rhoAlpha * Rmask
        Dkf = kflists[i + 1] - kflists[i]
        t = (kf - kflists[i]) / Dkf
        if interp == "newton" or interp == "linear":
            t0 = 1 - t
            pot1[mask] = p0[mask] * t0[mask] + p1[mask] * t[mask]
            if kdd > 1:
                if pot2G is None:
                    pot2G = kernel0 * (rhoU * t0).fft() + kernel1 * (rhoU * t).fft()
                else:
                    pot2G += kernel0 * (rhoU * t0).fft() + kernel1 * (rhoU * t).fft()
                if kdd == 3:
                    pot3[mask] = m0[mask] * t0[mask] + m1[mask] * t[mask]

        elif interp == "hermite":
            t2 = t * t
            t3 = t2 * t
            h00 = 2.0 * t3 - 3.0 * t2 + 1.0
            h10 = t3 - 2.0 * t2 + t
            h01 = 1.0 - h00
            h11 = t3 - t2
            pot1[mask] = (
                h00[mask] * p0[mask]
                + h01[mask] * p1[mask]
                + Dkf * (h10[mask] * m0[mask] + h11[mask] * m1[mask])
            )

            if kdd > 1:
                pG = (
                    kernel0 * (rhoU * h00).fft()
                    + kernel1 * (rhoU * h01).fft()
                    + Dkf
                    * (
                        kernelDeriv0 * (rhoU * h10).fft()
                        + kernelDeriv1 * (rhoU * h11).fft()
                    )
                )
                if pot2G is None:
                    pot2G = pG
                else:
                    pot2G += pG

                if kdd == 3:
                    t = t[mask]
                    t2 = t2[mask]
                    t3 = t3[mask]
                    h00D = 6.0 * t2 - 6 * t
                    h10D = 3.0 * t2 - 4 * t + 1.0
                    h01D = -h00D
                    h11D = 3.0 * t2 - 2 * t
                    pot3[mask] = (
                        (h00D * p0[mask] + h01D * p1[mask]) / Dkf
                        + h10D * m0[mask]
                        + h11D * m1[mask]
                    )

    if MGPKernelE is not None:
        mgpe = (MGPKernelE * rhoBetaG).ifft(force_real=True)
        pot1 += mgpe

    if vcalc:
        if pot2G is not None:
            pot2 = pot2G.ifft(force_real=True)
            if MGPKernelE is not None:
                pot2 += mgpe
        else:
            pot2 = pot1.copy()
    # -----------------------------------------------------------------------
    if ldw is None:
        ldw = 1.0 / 6.0
    factor = np.ones_like(rho)
    rhov = rho.amax()
    mask = rho < 1e-6
    ld = max(0.1, ldw)
    factor[mask] = np.abs(rho[mask]) ** ld / (rhov**ld)
    factor[rho < 0] = 0.0
    pot1 *= factor
    # -----------------------------------------------------------------------
    NL = FunctionalOutput(name="NL")
    # if 'E' in calcType or 'D' in calcType :
    energydensity = rhoAlpha * pot1
    NL.energy = energydensity.sum() * rho.grid.dV
    if 'D' in calcType:
        NL.energydensity = energydensity
    # -----------------------------------------------------------------------
    if vcalc:
        pot2 *= factor
        pot3 *= factor

        pot1 *= alpha * rhoAlpha1
        pot2 *= beta * rhoBeta1
        pot3 *= (kf / 3.0) * rhoAlpha1
        pot1 += pot2 + pot3
        pot = pot1
        sprint('lwt', NL.energy, pot.amin(), pot.amax(), comm=rho.mp.comm, level=1)
        NL.potential = pot

    return NL


def LWTLineIntegral(
    rho,
    etamax=100.0,
    ratio=1.15,
    nsp=None,
    delta=None,
    kdd=3,
    alpha=5.0 / 6.0,
    beta=5.0 / 6.0,
    nt=500,
    interp="linear",
    calcType={"E", "V"},
    kfmin=None,
    kfmax=None,
    ke_kernel_saved=None,
    **kwargs,
):
    KE_kernel_saved = ke_kernel_saved
    q = rho.grid.get_reciprocal().q

    dt = 1.0 / nt
    tlists = np.linspace(dt, 1.0, nt)

    trho = tlists[:, None, None, None] * rho
    # kf0 = 2.0 * (3.0 * np.pi ** 2 * np.mean(rho)) ** (1.0 / 3.0)
    kfLI = 2.0 * (3.0 * np.pi**2 * trho) ** (1.0 / 3.0)

    trhoAlpha1 = trho ** (alpha - 1)

    if kfmin is None or kfmax is None:
        kfBound = [1e-3, 100.0]
    else:
        kfBound = [kfmin, kfmax]
    mask2 = kfLI > kfBound[1]
    kfLI[mask2] = kfBound[1]

    if kfmin is None:
        kfMin = max(np.min(kfLI), kfBound[0])
    else:
        kfMin = kfmin

    if kfmax is None:
        kfMax = np.max(kfLI)
    else:
        kfMax = kfmax

    ### HEG
    if abs(kfMax - kfMin) < 1e-8:
        NL = FunctionalOutput(name="NL", energy=0.0)
        if 'D' in calcType:
            energydensity = DirectField(
                grid=rho.grid, memo=rho.memo, rank=rho.rank, cplx=rho.cplx
            )
            NL.energydensity = energydensity
        if 'V' in calcType:
            pot = DirectField(
                grid=rho.grid, memo=rho.memo, rank=rho.rank, cplx=rho.cplx
            )
            NL.potential = pot
        return NL

    if nsp is not None:
        # kflists = kfMin + (kfMax - kfMin)/(nsp - 1) * np.arange(nsp)
        if kfMax - kfMin < 1e-3:
            kflists = [kfMin, kfMax]
            nsp = 2
        else:
            kflists = kfMin + (kfMax - kfMin) / (nsp - 1) * np.arange(nsp)
        kflists = np.asarray(kflists)
    elif delta is not None:  # delta = 0.10
        nsp = int(np.ceil((kfMax - kfMin) / delta)) + 1
        kflists = np.linspace(kfMin, kfMax, nsp)
    else:
        nsp = int(np.ceil(np.log(kfMax / kfMin) / np.log(ratio))) + 1
        kflists = kfMin * ratio ** np.arange(nsp)
        # kflists = np.geomspace(kfMin, kfMax, nsp)
    kflists[0] -= 1e-16  # for numerical safe
    kflists[-1] += 1e-16  # for numerical safe
    # -----------------------------------------------------------------------
    kernel0 = np.empty_like(q)
    kernel1 = np.empty_like(q)
    kernelDeriv0 = np.empty_like(q)
    kernelDeriv1 = np.empty_like(q)
    pot = np.zeros_like(rho)
    pot1 = np.zeros_like(rho)
    rhoAlpha = rho**alpha
    if abs(alpha - beta) < 1e-8:
        rhoBeta = rhoAlpha
    else:
        rhoBeta = rho**beta
    rhoBetaG = rhoBeta.fft()

    KernelTable = KE_kernel_saved["KernelTable"]
    KernelDeriv = KE_kernel_saved["KernelDeriv"]
    MGPKernelE = KE_kernel_saved["MGPKernelE"]
    if kdd > 1 or interp == "hermite":
        mcalc = True
    else:
        mcalc = False

    for i in range(nsp - 1):
        if i == 0:
            kernel0 = LWTKernelKf(
                q, kflists[i], KernelTable, etamax=etamax, out=kernel0
            )
            p0 = (kernel0 * rhoBetaG).ifft(force_real=True)
            if mcalc:
                kernelDeriv0 = (
                    LWTKernelKf(
                        q, kflists[i], KernelDeriv, etamax=etamax, out=kernelDeriv0
                    )
                    / kflists[i]
                )
                m0 = (kernelDeriv0 * rhoBetaG).ifft(force_real=True)
        else:
            p0, p1 = p1, p0
            kernel0, kernel1 = kernel1, kernel0
            if mcalc:
                m0, m1 = m1, m0
                kernelDeriv0, kernelDeriv1 = kernelDeriv1, kernelDeriv0

        kernel1 = LWTKernelKf(
            q, kflists[i + 1], KernelTable, etamax=etamax, out=kernel1
        )
        p1 = (kernel1 * rhoBetaG).ifft(force_real=True)
        if mcalc:
            kernelDeriv1 = (
                LWTKernelKf(
                    q, kflists[i + 1], KernelDeriv, etamax=etamax, out=kernelDeriv1
                )
                / kflists[i + 1]
            )
            m1 = (kernelDeriv1 * rhoBetaG).ifft(force_real=True)

        for kf, trhoi in zip(kfLI, trhoAlpha1):
            mask = np.logical_and(kf > kflists[i], kf < kflists[i + 1] + 1e-18)
            Dkf = kflists[i + 1] - kflists[i]
            t = (kf - kflists[i]) / Dkf
            if interp == "newton" or interp == "linear":
                t0 = 1 - t
                potmask = p0[mask] * t0[mask] + p1[mask] * t[mask]
            elif interp == "hermite":
                t2 = t * t
                t3 = t2 * t
                h00 = 2.0 * t3 - 3.0 * t2 + 1.0
                h10 = t3 - 2.0 * t2 + t
                h01 = 1.0 - h00
                h11 = t3 - t2
                potmask = (
                    h00[mask] * p0[mask]
                    + h01[mask] * p1[mask]
                    + Dkf * (h10[mask] * m0[mask] + h11[mask] * m1[mask])
                )
            potmask *= trhoi[mask]
            pot1[mask] += potmask
        else:
            pot[mask] = potmask

    if MGPKernelE is not None:
        mgpe = (MGPKernelE * rhoBetaG).ifft(force_real=True)
        pot += mgpe

    # -----------------------------------------------------------------------
    ene = np.einsum("ijk, ijk ->", rho, pot1) * rho.grid.dV * dt * alpha
    # ene2 = np.einsum("ijk, ijk ->", rho, pot) * rho.grid.dV
    # sprint('ene', ene, ene2, level=1)
    # -----------------------------------------------------------------------
    pot *= alpha
    # pot *= alpha * rhoAlpha1
    NL = FunctionalOutput(name="NL", potential=pot, energy=ene)

    return NL


@timer()
def LWT(
    rho,
    x=1.0,
    y=1.0,
    sigma=None,
    interp="linear",
    kerneltype="WT",
    symmetrization=None,
    lumpfactor=None,
    alpha=5.0 / 6.0,
    beta=5.0 / 6.0,
    etamax=50.0,
    maxpoints=1000,
    fd=1,
    kdd=3,
    ratio=1.15,
    nsp=None,
    delta=None,
    neta=50000,
    order=3,
    calcType={"E", "V"},
    split=False,
    ke_kernel_saved=None,
    **kwargs,
):
    # Only performed once for each grid
    q = rho.grid.get_reciprocal().q
    rho0 = rho.amean()
    if ke_kernel_saved is None:
        KE_kernel_saved = {"Kernel": None, "rho0": 0.0, "shape": None}
    else:
        KE_kernel_saved = ke_kernel_saved
    # if abs(KE_kernel_saved["rho0"] - rho0) > 1e-6 or np.shape(rho) != KE_kernel_saved["shape"]:
    if tuple(rho.grid.nrR) != KE_kernel_saved["shape"]:
        sprint(
            'Re-calculate %s KernelTable ' % kerneltype,
            rho.grid.nrR,
            comm=rho.mp.comm,
            level=1,
        )
        eta = np.linspace(0, etamax, neta)
        if kerneltype == "WT":
            KernelTable = WTKernelTable(eta, x, y, alpha, beta)
        elif kerneltype == "MGP":
            KernelTable = MGPKernelTable(
                eta, maxpoints=maxpoints, symmetrization=symmetrization, mp=rho.grid.mp
            )
        elif kerneltype == "MGPA":
            KernelTable = MGPKernelTable(
                eta, maxpoints=maxpoints, symmetrization="Arithmetic", mp=rho.grid.mp
            )
        elif kerneltype == "MGPG":
            KernelTable = MGPKernelTable(
                eta, maxpoints=maxpoints, symmetrization="Geometric", mp=rho.grid.mp
            )
        # Add MGP kinetic electron
        if lumpfactor is not None:
            # sprint('Calculate MGP kinetic electron({})'.format(lumpfactor), rho.mp.comm, level=1)
            Ne = rho0 * rho.grid.volume
            MGPKernelE = MGPOmegaE(q, Ne, lumpfactor)
            KE_kernel_saved["MGPKernelE"] = MGPKernelE
        # Different method to interpolate the kernel
        if order > 1:
            KE_kernel_saved["KernelTable"] = splrep(eta, KernelTable, k=order)
            KernelDerivTable = splev(eta, KE_kernel_saved["KernelTable"], der=1) * (
                -1.0 * eta
            )
            KE_kernel_saved["KernelDeriv"] = splrep(eta, KernelDerivTable, k=order)
        else:
            tck = splrep(eta, KernelTable, k=3)
            KernelDerivTable = splev(eta, tck, der=1) * (-1.0 * eta)
            KE_kernel_saved["KernelTable"] = KernelTable
            KE_kernel_saved["KernelDeriv"] = KernelDerivTable
        KE_kernel_saved["etamax"] = etamax
        KE_kernel_saved["shape"] = tuple(rho.grid.nrR)
        KE_kernel_saved["rho0"] = rho0
    NL = LWTPotentialEnergy(
        rho,
        alpha=alpha,
        beta=beta,
        etamax=etamax,
        ratio=ratio,
        nsp=nsp,
        kdd=kdd,
        delta=delta,
        interp=interp,
        calcType=calcType,
        ke_kernel_saved=KE_kernel_saved,
        **kwargs,
    )
    # NL = LWTLineIntegral(rho, alpha=alpha, beta=beta, etamax=etamax, ratio=ratio, nsp=nsp, kdd=kdd, delta=delta, interp=interp, calcType=calcType, ke_kernel_saved = KE_kernel_saved, **kwargs)
    return NL


def LMGP(rho, kerneltype="MGP", **kwargs):
    return LWT(rho, kerneltype=kerneltype, **kwargs)


def LMGPA(rho, kerneltype="MGPA", **kwargs):
    return LWT(rho, kerneltype=kerneltype, **kwargs)


def LMGPG(rho, kerneltype="MGPG", **kwargs):
    return LWT(rho, kerneltype=kerneltype, **kwargs)
