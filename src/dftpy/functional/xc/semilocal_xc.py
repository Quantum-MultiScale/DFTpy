# Drivers for LibXC

import json
import os

import numpy as np

from dftpy.field import DirectField
from dftpy.functional.functional_output import FunctionalOutput
from dftpy.time_data import timer


def CheckLibXC(stop=True):
    import importlib.util

    islibxc = importlib.util.find_spec("pylibxc")
    found = islibxc is not None
    if not found and stop:
        raise ModuleNotFoundError("Install LibXC and pylibxc to use this functionality")
    return found


def Get_LibXC_Input(
    density,
    do_sigma=False,
    do_tau=False,
    do_lapl=False,
    sigma=None,
    lapl=None,
    tau=None,
    **kwargs,
):
    inp = {}
    if density.rank > 1:
        rhoT = density.reshape((2, -1)).T
        inp["rho"] = rhoT.ravel()
    else:
        inp["rho"] = density.ravel()
    if do_sigma:
        if sigma is None:
            sigma = density.sigma("standard")
        if density.rank > 1:
            sigma = sigma.reshape((3, -1)).T
        inp["sigma"] = sigma.ravel()
    if do_lapl:
        if lapl is None:
            lapl = density.laplacian(force_real=True, sigma=None)
        inp["lapl"] = lapl.ravel()
    if do_tau:
        if tau is None:
            raise ValueError("Please give 'tau' for MGGA functional.")
        inp["tau"] = tau.ravel()
    return inp


def Get_LibXC_Output(out, density, gradient=None, **kwargs):
    if not isinstance(out, (dict)):
        raise TypeError("LibXC output must be a dictionary")

    OutFunctional = FunctionalOutput(name="LibXC")

    rank_dict = {
        "vrho": 2,
        "v2rho2": 3,
        "v3rho3": 4,
        "v4rho4": 5,
        "vsigma": 3,
        "v2rhosigma": 6,
        "v2sigma2": 6,
    }

    for key in ["vrho", "v2rho2", "v3rho3", "v4rho4"]:
        if key in out.keys():
            if density.rank > 1:
                v = out[key].reshape((-1, rank_dict[key])).T
                v = DirectField(density.grid, rank=rank_dict[key], griddata_3d=v)
            else:
                v = DirectField(density.grid, rank=1, griddata_3d=out[key])
            if key == "vrho":
                OutFunctional.potential = v
            else:
                setattr(OutFunctional, key, v)

    vsigmas = {}
    for key in ["vsigma", "v2rhosigma", "v2sigma2"]:
        if key in out.keys():
            if density.rank > 1:
                vsigmas[key] = out[key].reshape((-1, rank_dict[key])).T
                vsigmas[key] = DirectField(
                    density.grid, rank=rank_dict[key], griddata_3d=vsigmas[key]
                )
            else:
                vsigmas[key] = DirectField(
                    density.grid, griddata_3d=out[key].reshape(np.shape(density))
                )

    if vsigmas:
        if gradient is not None:
            if density.rank > 1:
                grhoU = gradient[0]
                grhoD = gradient[1]
            else:
                grho = gradient
        else:
            if density.rank > 1:
                grhoU = density[0].gradient(flag="standard")
                grhoD = density[1].gradient(flag="standard")
            else:
                grho = density.gradient(flag="standard")

        if hasattr(OutFunctional, 'potential'):
            if density.rank > 1:
                prodotto = vsigmas['vsigma'][0] * grhoU
                v00 = prodotto.divergence(flag="standard")
                prodotto = vsigmas['vsigma'][1] * grhoD
                v01 = prodotto.divergence(flag="standard")
                prodotto = vsigmas['vsigma'][1] * grhoU
                v10 = prodotto.divergence(flag="standard")
                prodotto = vsigmas['vsigma'][2] * grhoD
                v11 = prodotto.divergence(flag="standard")
                OutFunctional.potential[0] -= 2 * v00 + v01
                OutFunctional.potential[1] -= 2 * v11 + v10
            else:
                prodotto = vsigmas['vsigma'] * grho
                vsigma_last = prodotto.divergence(flag="standard")
                OutFunctional.potential -= 2 * vsigma_last

        if hasattr(OutFunctional, 'v2rho2'):
            if density.rank > 1:
                prodotto = -vsigmas['v2rhosigma'][0] * grhoU
                v2rhosigma00 = prodotto.divergence(flag="standard")
                prodotto = -vsigmas['v2rhosigma'][1] * grhoD * 0.5
                v2rhosigma01 = prodotto.divergence(flag="standard")
                prodotto = -vsigmas['v2rhosigma'][1] * grhoU * 0.5
                v2rhosigma11 = prodotto.divergence(flag="standard")
                prodotto = -vsigmas['v2rhosigma'][2] * grhoD
                v2rhosigma12 = prodotto.divergence(flag="standard")
                prodotto = -vsigmas['v2rhosigma'][3] * grhoU
                v2rhosigma13 = prodotto.divergence(flag="standard")
                prodotto = -vsigmas['v2rhosigma'][4] * grhoD * 0.5
                v2rhosigma14 = prodotto.divergence(flag="standard")
                prodotto = -vsigmas['v2rhosigma'][4] * grhoU * 0.5
                v2rhosigma24 = prodotto.divergence(flag="standard")
                prodotto = -vsigmas['v2rhosigma'][5] * grhoD
                v2rhosigma25 = prodotto.divergence(flag="standard")
                prolapto = vsigmas['v2sigma2'][0] * grhoU.dot(grhoU) * 2.0
                v2sigma200 = prolapto.laplacian(force_real=True)
                prolapto = vsigmas['v2sigma2'][1] * grhoU.dot(grhoD)
                v2sigma201 = prolapto.laplacian(force_real=True)
                prolapto = vsigmas['v2sigma2'][1] * grhoU.dot(grhoU)
                v2sigma211 = prolapto.laplacian(force_real=True)
                prolapto = vsigmas['v2sigma2'][2] * grhoU.dot(grhoD) * 2.0
                v2sigma212 = prolapto.laplacian(force_real=True)
                prolapto = vsigmas['v2sigma2'][3] * grhoD.dot(grhoD) * 0.5
                v2sigma203 = prolapto.laplacian(force_real=True)
                prolapto = vsigmas['v2sigma2'][3] * grhoU.dot(grhoD)
                v2sigma213 = prolapto.laplacian(force_real=True)
                prolapto = vsigmas['v2sigma2'][3] * grhoU.dot(grhoU) * 0.5
                v2sigma223 = prolapto.laplacian(force_real=True)
                prolapto = vsigmas['v2sigma2'][4] * grhoD.dot(grhoD)
                v2sigma214 = prolapto.laplacian(force_real=True)
                prolapto = vsigmas['v2sigma2'][4] * grhoU.dot(grhoD)
                v2sigma224 = prolapto.laplacian(force_real=True)
                prolapto = vsigmas['v2sigma2'][5] * grhoD.dot(grhoD) * 2.0
                v2sigma225 = prolapto.laplacian(force_real=True)
                OutFunctional.v2rho2[0] = (
                    OutFunctional.v2rho2[0]
                    + v2rhosigma00
                    + v2rhosigma01
                    + v2sigma200
                    + v2sigma201
                    + v2sigma203
                )
                OutFunctional.v2rho2[1] = (
                    OutFunctional.v2rho2[1]
                    + v2rhosigma11
                    + v2rhosigma12
                    + v2rhosigma13
                    + v2rhosigma14
                    + v2sigma211
                    + v2sigma212
                    + v2sigma213
                    + v2sigma214
                )
                OutFunctional.v2rho2[2] = (
                    OutFunctional.v2rho2[2]
                    + v2rhosigma24
                    + v2rhosigma25
                    + v2sigma223
                    + v2sigma224
                    + v2sigma225
                )
            else:
                prodotto = -vsigmas['v2rhosigma'] * grho
                v2rhosigma = prodotto.divergence(flag="standard")
                prolapto = vsigmas['v2sigma2'] * grho.dot(grho) * 2.0
                v2sigma2 = prolapto.laplacian(force_real=True)
                OutFunctional.v2rho2 = OutFunctional.v2rho2 + v2rhosigma + v2sigma2

        if hasattr(OutFunctional, 'v3rho3') or hasattr(OutFunctional, 'v4rho4'):
            raise Exception(
                '3rd and higher order derivative for GGA functionals has not implemented yet.'
            )

    if "zk" in out.keys():
        if density.rank > 1:
            rho = np.sum(density, axis=0)
        else:
            rho = density
        edens = rho * out["zk"].reshape(np.shape(rho))
        ene = edens.sum() * density.grid.dV
        OutFunctional.energy = ene
        OutFunctional.energydensity = edens

    return OutFunctional


@timer()
def LibXC(
    density,
    libxc=None,
    calcType={"E", "V"},
    core_density=None,
    sigma=None,
    lapl=None,
    tau=None,
    flag='standard',
    **kwargs,
):
    """
    Output:
       - out_functional: a functional evaluated with LibXC
    Input:
       - density: a DirectField (rank=1)
       - libxc: strings like "gga_k_lc94", "gga_x_pbe" and "gga_c_pbe"
    """
    if CheckLibXC():
        from pylibxc.functional import LibXCFunctional

    do_sigma = False
    do_lapl = False
    do_tau = False
    # -----------------------------------------------------------------------
    libxc = get_libxc_names(libxc=libxc, **kwargs)
    # -----------------------------------------------------------------------
    density = add_core_density(density, core_density)

    if not libxc:
        raise AttributeError("Please give a short name 'xc' or a list 'libxc'.")

    if not isinstance(density, (DirectField)):
        raise TypeError("Density should be a DirectField")
    if density.rank == 1:
        polarization = "unpolarized"
    elif density.rank == 2:
        polarization = "polarized"
    else:
        raise AttributeError("Only support nspin=1 or 2.")

    for value in libxc:
        func = LibXCFunctional(value, polarization)
        if value.startswith('hyb'):
            raise AttributeError(
                'Hybrid and Meta-GGA functionals have not been implemented yet'
            )
        if value.startswith('gga'):
            do_sigma = True
        elif value.startswith('mgga'):
            do_sigma = True
            do_tau = True
            if func._needs_laplacian:
                do_lapl = True
    if do_sigma:
        gradient = density.gradient(flag=flag)
        sigma = density.sigma(flag=flag, gradient=gradient)
    else:
        gradient = None
        sigma = None

    inp = Get_LibXC_Input(
        density,
        do_sigma=do_sigma,
        do_tau=do_tau,
        do_lapl=do_lapl,
        sigma=sigma,
        lapl=lapl,
        tau=tau,
        **kwargs,
    )
    kargs = {'do_exc': False, 'do_vxc': False}
    if 'E' in calcType or 'D' in calcType:
        kargs.update({'do_exc': True})
    if 'V' in calcType or 'S' in calcType:
        kargs.update({'do_vxc': True})
    if 'V2' in calcType:
        kargs.update({'do_fxc': True})
    if 'V3' in calcType:
        kargs.update({'do_kxc': True})
    if 'V4' in calcType:
        kargs.update({'do_lxc': True})

    out_functional = None
    for value in libxc:
        func = LibXCFunctional(value, polarization)
        out = func.compute(inp, **kargs)
        if out_functional is not None:
            sa = Get_LibXC_Output(out, density, gradient=gradient, **kwargs)
            out_functional += sa
            out_functional.name += "_" + value
        else:
            sa = Get_LibXC_Output(out, density, gradient=gradient, **kwargs)
            out_functional = sa
            out_functional.name = value
        if 'S' in calcType:
            if not hasattr(out_functional, 'stress'):
                out_functional.stress = np.zeros((3, 3))
            if 'E' in calcType:
                ene = sa.energy
            else:
                ene = 0.0
            if value.startswith('lda'):
                out_functional.stress += _LDAStress(density, sa.potential, energy=ene)
            elif value.startswith('gga'):
                vsigma = DirectField(density.grid, data=out['vsigma'])
                out_functional.stress += _GGAStress(
                    density,
                    sa.potential,
                    energy=ene,
                    gradient=gradient,
                    sigma=sigma,
                    vsigma=vsigma,
                )
            else:
                raise AttributeError(
                    'Hybrid and Meta-GGA functionals have not been implemented yet'
                )
    return out_functional


def PBE(density, calcType={"E", "V"}):
    return LibXC(density=density, libxc=["gga_x_pbe", "gga_c_pbe"], calcType=calcType)


def LDA_XC(density, calcType={"E", "V"}):
    return LibXC(density=density, libxc=["lda_x", "lda_c_pz"], calcType=calcType)


@timer()
def LDA(rho, calcType={"E", "V"}, **kwargs):
    if rho.rank > 1:
        return LDA_XC(rho, calcType)
    OutFunctional = FunctionalOutput(name="XC")
    a = (0.0311, 0.01555)
    b = (-0.048, -0.0269)
    c = (0.0020, 0.0007)
    d = (-0.0116, -0.0048)
    gamma = (-0.1423, -0.0843)
    beta1 = (1.0529, 1.3981)
    beta2 = (0.3334, 0.2611)

    rho_cbrt = np.cbrt(rho)
    rho_cbrt[rho_cbrt < 1e-30] = 1e-30  # for safe
    Rs = np.cbrt(3.0 / (4.0 * np.pi)) / rho_cbrt
    rs1 = Rs < 1
    rs2 = Rs >= 1
    Rs2sqrt = np.sqrt(Rs[rs2])

    if "E" in calcType:
        ExRho = -3.0 / 4.0 * np.cbrt(3.0 / np.pi) * rho_cbrt
        ExRho[rs1] += (
            a[0] * np.log(Rs[rs1])
            + b[0]
            + c[0] * Rs[rs1] * np.log(Rs[rs1])
            + d[0] * Rs[rs1]
        )
        ExRho[rs2] += gamma[0] / (1.0 + beta1[0] * Rs2sqrt + beta2[0] * Rs[rs2])
        ene = np.einsum("ijk, ijk->", ExRho, rho) * rho.grid.dV
        OutFunctional.energy = ene
    if "V" in calcType or "S" in calcType:
        pot = np.cbrt(-3.0 / np.pi) * rho_cbrt
        pot[rs1] += (
            np.log(Rs[rs1]) * (a[0] + 2.0 / 3 * c[0] * Rs[rs1])
            + b[0]
            - 1.0 / 3 * a[0]
            + 1.0 / 3 * (2 * d[0] - c[0]) * Rs[rs1]
        )
        pot[rs2] += (
            gamma[0]
            + (7.0 / 6.0 * gamma[0] * beta1[0]) * Rs2sqrt
            + (4.0 / 3.0 * gamma[0] * beta2[0] * Rs[rs2])
        ) / (1.0 + beta1[0] * Rs2sqrt + beta2[0] * Rs[rs2]) ** 2
        OutFunctional.potential = pot
    if "V2" in calcType:
        fx = -np.cbrt(3.0 / np.pi) / 3.0 * np.cbrt(rho) / rho

        fc = np.empty(np.shape(rho))
        fc[rs1] = (
            -a[0] / 3.0
            - (c[0] / 9.0 * (np.log(Rs[rs1]) * 2.0 + 1.0) + d[0] * 2.0 / 9.0) * Rs[rs1]
        )
        tmpa = beta1[0] * Rs2sqrt
        tmpb = beta2[0] * Rs[rs2]
        deno = 1.0 + tmpa + tmpb
        fc[rs2] = (
            gamma[0]
            / 36.0
            * (
                5.0 * tmpa
                + 7.0 * tmpa * tmpa
                + 8.0 * tmpb
                + 16.0 * tmpb * tmpb
                + 21.0 * tmpa * tmpb
            )
            / deno
            / deno
            / deno
        )
        fc /= rho

        OutFunctional.v2rho2 = fx + fc

    if "S" in calcType:
        if 'E' in calcType:
            ene = OutFunctional.energy
        else:
            ene = 0.0
        OutFunctional.stress = _LDAStress(rho, OutFunctional.potential, energy=ene)
    return OutFunctional


def _LDAStress(density, potential, energy=0.0, **kwargs):
    P = energy - np.sum(potential * density) * density.grid.dV
    stress = np.eye(3) * P
    return stress / density.grid.volume


def _GGAStress(
    density, potential, energy=0.0, gradient=None, sigma=None, vsigma=None, **kwargs
):
    P = energy - np.sum(potential * density) * density.grid.dV
    stress = np.eye(3) * P
    nspin = density.rank
    for i in range(3):
        for j in range(i, 3):
            if nspin > 1:
                stress[i, j] -= (
                    2.0
                    * np.einsum(
                        "ijk, ijk, ijk -> ", gradient[0][i], gradient[0][j], vsigma[0]
                    )
                    * density.grid.dV
                )
                stress[i, j] -= (
                    2.0
                    * np.einsum(
                        "ijk, ijk, ijk -> ", gradient[0][i], gradient[1][j], vsigma[1]
                    )
                    * density.grid.dV
                )
                stress[i, j] -= (
                    2.0
                    * np.einsum(
                        "ijk, ijk, ijk -> ", gradient[1][i], gradient[1][j], vsigma[2]
                    )
                    * density.grid.dV
                )
            else:
                stress[i, j] -= (
                    2.0
                    * np.einsum("ijk, ijk, ijk -> ", gradient[i], gradient[j], vsigma)
                    * density.grid.dV
                )
            stress[j, i] = stress[i, j]
    return stress / density.grid.volume


xc_json_file = os.path.join(os.path.dirname(__file__), 'xc.json')
with open(xc_json_file) as f:
    xcformats = json.load(f)


def get_short_xc_name(libxc=None, xc=None, code=None, **kwargs):
    name = None
    if xc:
        alias = xcformats.get(xc.lower(), {}).get('alias', {}).get(code, [])
        if alias:
            name = alias[0]
    else:
        for name, value in xcformats.items():
            libxc_strs = value.get('libxc', [])
            if len(libxc) == len(libxc_strs):
                for a, b in zip(libxc, libxc_strs):
                    if a.lower() != b:
                        break
                else:
                    if code:
                        alias = value.get('alias', {}).get(code, [])
                        if alias:
                            name = alias[0]
                    break
    return name


def get_libxc_names(xc=None, libxc=None, name=None, code=None, **kwargs):
    xc = xc or name
    if xc:
        xc = xc.lower()
        if code:
            for name, value in xcformats.items():
                alias = value.get('alias', {}).get(code, [])
                if xc in alias:
                    v = value.get('libxc', None)
                    if v:
                        libxc = v
                    break
        else:
            v = xcformats.get(xc, {}).get('libxc', None)
            if v:
                libxc = v
    elif isinstance(libxc, str):
        libxc = libxc.split()

    # compatible with older version
    libxc_old = [
        v
        for k, v in kwargs.items()
        if k in ["k_str", "x_str", "c_str", "functional"] and v is not None
    ]
    if len(libxc_old) > 0:
        # print('libxc', libxc, libxc_old)
        # warnings.warn(FutureWarning("'*_str' are deprecated; please use 'libxc' or 'xc'"))
        libxc = libxc_old
    if libxc:
        libxc = [x.lower() for x in libxc]
    return libxc


def add_core_density(density, core_density=None):
    if core_density is None:
        new_density = density
    elif density.rank == core_density.rank:
        new_density = density + core_density
    elif density.rank == 2 and core_density.rank == 1:
        new_density = density + 0.5 * core_density
    else:
        raise ValueError('Not support!')
    return new_density
