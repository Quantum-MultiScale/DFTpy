# Collection of local and semilocal functional

import numpy as np

from dftpy.field import DirectField, ReciprocalField
from dftpy.functional.functional_output import FunctionalOutput
from dftpy.functional.kedf.tf import TF
from dftpy.time_data import timer

__all__ = ['vW', 'vonWeizsackerStress']


def vonWeizsackerPotentialCplx(wav, grid, sigma=0.025):
    """
    The von Weizsacker Potential for complex pseudo-wavefunction
    """
    if not isinstance(sigma, (np.generic, int, float)):
        print("Bad type for sigma")
        return Exception
    wav = DirectField(grid=grid, griddata_3d=wav, cplx=True)
    gg = grid.get_reciprocal().ggF
    potG = wav.fft() * np.exp(-gg * (sigma) ** 2 / 4.0) * gg
    potG = ReciprocalField(grid=grid, griddata_3d=wav, cplx=True)
    a = potG.ifft(force_real=True)
    np.multiply(0.5, a, out=a)
    return DirectField(grid=grid, griddata_3d=a)


def vonWeizsackerPotential(rho, sigma=None, phi=None, lphi=False, **kwargs):
    """
    The von Weizsacker Potential
    """
    # -----------------------------------------------------------------------
    tol = 1e-30
    rhom = rho.copy()
    mask = rho > 0
    mask2 = np.invert(mask)
    rhom[mask2] = tol
    # -----------------------------------------------------------------------
    gg = rho.grid.get_reciprocal().gg
    if lphi and phi is not None:
        sq_dens = phi
    else:
        sq_dens = np.sqrt(rhom)
    if sigma is None:
        n2_sq_dens = sq_dens.fft() * gg
    else:
        n2_sq_dens = sq_dens.fft() * np.exp(-gg * (sigma) ** 2 / 4.0) * gg
    a = n2_sq_dens.ifft(force_real=True)
    a *= 0.5
    sq_dens[np.abs(sq_dens) < tol] = tol  # for safe
    pot = DirectField(grid=rho.grid, griddata_3d=np.divide(a, sq_dens, out=a))
    return pot


# def vonWeizsackerPotentialDensity(rho, sigma=None, phi=None, lphi=False, **kwargs):
#     """
#     The von Weizsacker Potential (not finish)
#     """
#     # -----------------------------------------------------------------------
#     gg = rho.grid.get_reciprocal().gg
#     g = rho.grid.get_reciprocal().g
#     q = rho.grid.get_reciprocal().q
#     rho_g = rho.fft()
#     rho_grad = []
#     for i in range(3):
#         if sigma is None:
#             grho_g = g[i] * rho_g * 1j
#         else:
#             grho_g = g[i] * rho_g * np.exp(-q * (sigma) ** 2 / 4.0) * 1j
#         item = (grho_g).ifft(force_real=True)
#         rho_grad.append(item)
#     grad = (rho_grad[0] ** 2 + rho_grad[1] ** 2 + rho_grad[2] ** 2) / (rho * rho)
#     if sigma is None:
#         lapl_g = rho_g * gg
#     else:
#         lapl_g = rho_g * np.exp(-gg * (sigma) ** 2 / 4.0) * gg
#     lapl = lapl_g.ifft(force_real=True) / rho
#     results = grad * 0.125 + lapl * 0.25
#     pot = DirectField(grid=rho.grid, griddata_3d=results)
#     return pot


def vonWeizsackerEnergy(rho, potential=None, sigma=None, **kwargs):
    """
    The von Weizsacker Energy Density
    """
    if potential is None:
        edens = vonWeizsackerPotential(rho, sigma=sigma, **kwargs)
    else:
        edens = potential
    # ene = np.einsum("ijk, ijk->", rho, edens) * rho.grid.dV
    ene = np.einsum("i, i->", rho[rho > 0], edens[rho > 0]) * rho.grid.dV
    return ene


def vonWeizsackerStress(rho, y=1.0, energy=None, **kwargs):
    """
    The von Weizsacker Stress
    """
    g = rho.grid.get_reciprocal().g
    rhoG = rho.fft()
    dRho_ij = []
    stress = np.zeros((3, 3))
    for i in range(3):
        dRho_ij.append((1j * g[i] * rhoG).ifft(force_real=True))
    for i in range(3):
        for j in range(i, 3):
            Etmp = (
                -0.25
                / rho.grid.volume
                * rho.grid.dV
                * np.einsum("ijk -> ", dRho_ij[i] * dRho_ij[j] / rho)
            )
            stress[i, j] = stress[j, i] = Etmp.real * y
    return stress


@timer()
def vW(rho, y=1.0, sigma=None, calcType={"E", "V"}, split=False, **kwargs):
    pot = vonWeizsackerPotential(rho, sigma, **kwargs)
    OutFunctional = FunctionalOutput(name="vW")
    if "E" in calcType:
        ene = vonWeizsackerEnergy(rho, pot, **kwargs)
        OutFunctional.energy = ene * y

    if "V" in calcType:
        OutFunctional.potential = pot * y

    if 'D' in calcType:
        OutFunctional.energydensity = pot * y * rho

    if split:
        return {"vW": OutFunctional}
    else:
        return OutFunctional


def x_TF_y_vW(
    rho, x=1.0, y=1.0, sigma=None, calcType={"E", "V"}, split=False, **kwargs
):
    xTF = TF(rho, x=x, calcType=calcType)
    yvW = vW(rho, y=y, sigma=sigma, calcType=calcType, **kwargs)
    OutFunctional = FunctionalOutput(name=str(x) + "_TF_" + str(y) + "_vW")

    if 'E' in calcType:
        OutFunctional.energy = xTF.energy + yvW.energy

    if 'V' in calcType:
        OutFunctional.potential = xTF.potential + yvW.potential

    if 'D' in calcType:
        OutFunctional.energydensity = xTF.energydensity + yvW.energydensity

    if split:
        return {"TF": xTF, "vW": yvW}
    else:
        return OutFunctional
