import numpy as np
from scipy.interpolate import splev, splrep

from dftpy.constants import ZERO
from dftpy.field import DirectField
from dftpy.functional.functional_output import FunctionalOutput
from dftpy.mpi import sprint
from dftpy.time_data import timer

from .gga import GGAFs
from .kernel import HCKernelTable, HCKernelXi, MGPKernelTable, MGPOmegaE, WTKernelTable

__all__ = ["HC"]


def get_kflist(
    kf,
    kf0=None,
    ratio=1.2,
    nsp=None,
    delta=None,
    kfmin=None,
    kfmax=None,
    savetol=1e-16,
    ke_kernel_saved=None,
    **kwargs,
):
    if ke_kernel_saved is not None:
        kfmin_prev = ke_kernel_saved['kfmin']
        kfmax_prev = ke_kernel_saved['kfmax']
    else:
        kfmin_prev = 1e8
        kfmax_prev = -1e8

    if kfmin is None:
        kfmin = max(kf.amin(), 1e-3)
        n = int(np.floor(np.log(kfmin / kf0) / np.log(ratio))) - 1
        kfmin = kf0 * ratio**n

    if kfmax is None:
        kfmax = min(kf.amax(), 100)
        n = int(np.ceil(np.log(kfmax / kf0) / np.log(ratio))) + 1
        kfmax = kf0 * ratio**n

    if kfmin_prev is not None:
        if kfmin > kfmin_prev and (kfmin - kfmin_prev) < 0.2:
            kfmin = kfmin_prev
    if kfmax_prev is not None:
        if kfmax < kfmax_prev and (kfmax_prev - kfmax) < 1:
            kfmax = kfmax_prev

    if nsp is not None:
        if kfmax - kfmin < 1e-3:
            kflists = [kfmin, kfmax]
            nsp = 2
        else:
            kflists = kfmin + (kfmax - kfmin) / (nsp - 1) * np.arange(nsp)
        kflists = np.asarray(kflists)
    elif delta is not None:
        nsp = int(np.ceil((kfmax - kfmin) / delta)) + 1
        kflists = np.linspace(kfmin, kfmax, nsp)
    else:
        # nsp = int(np.ceil(np.log(kfmax / kfmin) / np.log(ratio)))
        nsp = int(np.log(kfmax / kfmin) / np.log(ratio)) + 1
        kflists = kfmin * ratio ** np.arange(nsp)
        # kflists = np.geomspace(kfmin, kfmax, nsp)

    kflists[0] -= savetol  # for numerical safe
    kflists[-1] += savetol  # for numerical safe
    sprint(
        'nsp',
        len(kflists),
        np.max(kflists),
        np.min(kflists),
        kf0,
        comm=kf.mp.comm,
        level=1,
    )
    if ke_kernel_saved is not None:
        ke_kernel_saved['kfmin'] = kfmin
        ke_kernel_saved['kfmax'] = kfmax
    return kflists


def one_point_potential_energy(
    rho,
    etamax=100.0,
    ratio=1.2,
    nsp=None,
    delta=None,
    alpha=5.0 / 6.0,
    beta=5.0 / 6.0,
    interp="linear",
    calcType=["E", "V"],
    kfmin=None,
    kfmax=None,
    ldw=None,
    rho0=None,
    ke_kernel_saved=None,
    functional='REVAPBEK',
    params=None,
    **kwargs,
):
    """
    ldw : local density weight
    """
    # -----------------------------------------------------------------------
    mask_zero = rho < ZERO
    rho_saved = rho[mask_zero]
    rho[mask_zero] = ZERO

    rhoAlpha = rho**alpha
    rhoAlpha1 = rhoAlpha / rho
    if abs(alpha - beta) < 1e-8:
        rhoBeta = rhoAlpha
        rhoBeta1 = rhoAlpha1
    else:
        rhoBeta = rho**beta
        rhoBeta1 = rhoBeta / rho
    rhoBetaG = rhoBeta.fft()
    # -----------------------------------------------------------------------
    sigma = None
    rho43 = rho ** (4.0 / 3.0)
    rho83 = rho43 * rho43
    q = rho.grid.get_reciprocal().q
    g = rho.grid.get_reciprocal().g

    rhoG = rho.fft()
    rhoGrad = []
    for i in range(3):
        if sigma is None:
            grhoG = g[i] * rhoG * 1j
        else:
            grhoG = g[i] * rhoG * np.exp(-q * (sigma) ** 2 / 4.0) * 1j
        item = (grhoG).ifft(force_real=True)
        rhoGrad.append(item)
    s = np.sqrt(rhoGrad[0] ** 2 + rhoGrad[1] ** 2 + rhoGrad[2] ** 2) / rho43
    # -----------------------------------------------------------------------
    # np.savetxt('s.dat', np.c_[rho.ravel(), s.ravel()])
    # -----------------------------------------------------------------------
    # GGAFs use s /= 2*(3*\pi^2)^{1/3}, which is 38.283120002509214 for s^2
    if functional.upper() == 'HC':
        """
        Also can reproduce with functional = 'TFVW' and  params = 1 hc_lambda*3.0/5.0*38.283120002509214
        """
        if params is None:
            hc_lambda = 0.0
        else:
            hc_lambda = params[0]
        F = 1 + hc_lambda * s * s
        dFds2 = 2 * hc_lambda
    else:
        F, dFds2 = GGAFs(
            s, functional=functional, calcType=calcType, params=params, **kwargs
        )
    # -----------------------------------------------------------------------
    q = rho.grid.get_reciprocal().q
    kf_std = np.cbrt(3.0 * np.pi**2 * rho)
    kf = F * kf_std
    kf = DirectField(
        grid=rho.grid, memo=rho.memo, rank=rho.rank, griddata_3d=kf, cplx=rho.cplx
    )

    # if rho0 is None :
    #     rho0 = ke_kernel_saved.get('rho0', None)
    # if rho0 is not None and rho0 > ZERO :
    #     kf0 = np.cbrt(3.0 * np.pi ** 2 * rho0)
    kf0 = 1.0
    kflists = get_kflist(
        kf,
        kf0,
        ratio=ratio,
        nsp=nsp,
        delta=delta,
        rho0=rho0,
        kfmin=kfmin,
        kfmax=kfmax,
        ke_kernel_saved=ke_kernel_saved,
        **kwargs,
    )
    nsp = len(kflists)

    ### HEG
    if abs(kf.amax() - kf.amin()) < 1e-8 or nsp < 3:
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

    mask = kf > kflists[-1]
    kf[mask] = kflists[-1]

    if 'V' in calcType:
        vcalc = True
    else:
        vcalc = False
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
    KernelTable = ke_kernel_saved["KernelTable"]
    KernelDeriv = ke_kernel_saved["KernelDeriv"]
    # -----------------------------------------------------------------------
    for i in range(nsp - 1):
        if i == 0:
            kernel0, kernelDeriv0 = HCKernelXi(
                q,
                kflists[i],
                KernelTable,
                KernelDeriv,
                etamax=etamax,
                out=kernel0,
                out2=kernelDeriv0,
            )
            kernelDeriv0 /= kflists[i]
            p0 = (kernel0 * rhoBetaG).ifft(force_real=True)
            m0 = (kernelDeriv0 * rhoBetaG).ifft(force_real=True)
        else:
            p0, p1 = p1, p0
            kernel0, kernel1 = kernel1, kernel0
            m0, m1 = m1, m0
            kernelDeriv0, kernelDeriv1 = kernelDeriv1, kernelDeriv0

        kernel1, kernelDeriv1 = HCKernelXi(
            q,
            kflists[i + 1],
            KernelTable,
            KernelDeriv,
            etamax=etamax,
            out=kernel1,
            out2=kernelDeriv1,
        )
        kernelDeriv1 /= kflists[i + 1]
        p1 = (kernel1 * rhoBetaG).ifft(force_real=True)
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
                Rmask[:] = 0.0
                Rmask[small] = 1.0
                rhoU = rhoAlpha * Rmask
                if pot2G is None:
                    pot2G = kernel0 * (rhoU * t).fft()
                pot3[mask] = m0[mask] * t[mask]
        # -----------------------------------------------------------------------
        Rmask[:] = 0.0
        Rmask[mask] = 1.0
        rhoU = rhoAlpha * Rmask
        Dkf = kflists[i + 1] - kflists[i]
        t = (kf - kflists[i]) / Dkf
        if interp == "newton" or interp == "linear":
            t0 = 1 - t
            pot1[mask] += p0[mask] * t0[mask] + p1[mask] * t[mask]
            if pot2G is None:
                pot2G = kernel0 * (rhoU * t0).fft() + kernel1 * (rhoU * t).fft()
            else:
                pot2G += kernel0 * (rhoU * t0).fft() + kernel1 * (rhoU * t).fft()
            pot3[mask] += m0[mask] * t0[mask] + m1[mask] * t[mask]

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

    pot2 = pot2G.ifft(force_real=True)
    NL = FunctionalOutput(name="NL")
    energydensity = rhoAlpha * pot1
    # energydensity = rhoBeta * pot2
    NL.energy = energydensity.sum() * rho.grid.dV
    if 'D' in calcType:
        NL.energydensity = energydensity
    # -----------------------------------------------------------------------
    if vcalc:
        pot1 *= alpha * rhoAlpha1
        pot2 *= beta * rhoBeta1
        pot3 *= rhoAlpha
        # -----------------------------------------------------------------------
        # pxi/pn
        pot3_1 = kf_std / (3 * rho) * (F - 4 * dFds2 * s * s) * pot3
        pot_dn = kf_std * dFds2 / rho83 * pot3
        # -----------------------------------------------------------------------
        # pxi/pdn
        p3 = []
        for i in range(3):
            item = rhoGrad[i] * pot_dn
            p3.append(item.fft())
        pot3G = g[0] * p3[0] + g[1] * p3[1] + g[2] * p3[2]
        pot3_2 = (1j * pot3G).ifft(force_real=True)
        # -----------------------------------------------------------------------
        pot1 += pot2 + pot3_1 - pot3_2
        sprint('HC', NL.energy, pot1.amin(), pot1.amax(), comm=rho.mp.comm, level=1)
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
        NL.potential = pot1
        sprint(
            'HC2',
            NL.energy,
            NL.potential.amin(),
            NL.potential.amax(),
            comm=rho.mp.comm,
            level=1,
        )
        # np.savetxt('pot.dat', np.c_[rho.ravel(), pot1.ravel()])
        # sprint('density', np.max(rho), np.min(rho), rho.integral(), comm = rho.mp.comm, level=1)
    # -----------------------------------------------------------------------
    rho[mask_zero] = rho_saved
    # -----------------------------------------------------------------------
    return NL


@timer()
def HC(
    rho,
    x=1.0,
    y=1.0,
    sigma=None,
    interp="hermite",
    kerneltype="HC",
    symmetrization=None,
    lumpfactor=None,
    alpha=2.0,
    beta=2.0 / 3.0,
    etamax=50.0,
    maxpoints=1000,
    fd=1,
    kdd=3,
    ratio=1.15,
    nsp=None,
    delta=None,
    neta=50000,
    order=3,
    calcType=["E", "V"],
    split=False,
    ke_kernel_saved=None,
    functional='HC',
    **kwargs,
):
    # Only performed once for each grid
    q = rho.grid.get_reciprocal().q
    rho0 = rho.amean()
    if ke_kernel_saved is None:
        KE_kernel_saved = {"Kernel": None, "rho0": 0.0, "shape": None}
    else:
        KE_kernel_saved = ke_kernel_saved

    kerneltype = 'HC'
    if tuple(rho.grid.nrR) != KE_kernel_saved["shape"]:
        sprint(
            'Re-calculate %s KernelTable ' % kerneltype,
            rho.grid.nrR,
            KE_kernel_saved["shape"],
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
        elif kerneltype == "HC":
            KernelTable = HCKernelTable(eta, beta=beta, x=x, y=y, mp=rho.grid.mp)
        # Add MGP kinetic electron
        if lumpfactor is not None:
            sprint(
                'Calculate MGP kinetic electron({})'.format(lumpfactor),
                comm=rho.mp.comm,
                level=1,
            )
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
    NL = one_point_potential_energy(
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
        functional=functional,
        **kwargs,
    )
    # -----------------------------------------------------------------------
    # kwargs['functional'] = 'HC'
    # kwargs['params'] = [0.01]
    # one_point_potential_energy(rho, alpha=alpha, beta=beta, etamax=etamax, ratio=ratio, nsp=nsp, kdd=kdd, delta=delta, interp=interp, calcType=calcType, ke_kernel_saved = KE_kernel_saved, **kwargs)
    # -----------------------------------------------------------------------
    return NL


def revHC(
    rho,
    x=1.0,
    y=1.0,
    sigma=None,
    interp="hermite",
    kerneltype="HC",
    symmetrization=None,
    lumpfactor=None,
    alpha=2.0,
    beta=2.0 / 3.0,
    etamax=50.0,
    maxpoints=1000,
    fd=1,
    kdd=3,
    ratio=1.15,
    nsp=None,
    delta=None,
    neta=50000,
    order=3,
    calcType=["E", "V"],
    split=False,
    ke_kernel_saved=None,
    params=[0.1, 0.45],
    functional='PBE2',
    **kwargs,
):
    return HC(
        rho,
        x=x,
        y=y,
        sigma=sigma,
        interp=interp,
        kerneltype=kerneltype,
        symmetrization=symmetrization,
        lumpfactor=lumpfactor,
        alpha=alpha,
        beta=beta,
        etamax=etamax,
        maxpoints=maxpoints,
        fd=fd,
        kdd=kdd,
        ratio=ratio,
        nsp=nsp,
        delta=delta,
        neta=neta,
        order=order,
        calcType=calcType,
        split=split,
        ke_kernel_saved=ke_kernel_saved,
        params=params,
        functional=functional,
        **kwargs,
    )
