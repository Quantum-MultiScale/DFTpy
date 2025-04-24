# Collection of semilocal functionals

from functools import wraps
from typing import Dict, List, Set

import numpy as np
import scipy.special as sp

from dftpy.constants import C_TF, CBRT_TWO, TKF0
from dftpy.field import DirectField
from dftpy.functional.functional_output import FunctionalOutput
from dftpy.math_utils import PowerInt
from dftpy.time_data import timer

__all__ = ["GGA", "GGAFs", "GGA_KEDF_list"]


def s_with_tkf0(function):
    r'''
    Decorator for functionals with a defination of s=\grad\rho / tkf0 / (\rho)^(4/3)
    Parameters
    ----------
    function

    Returns
    -------


    '''

    @wraps(function)
    def wrapper(*args, **kwargs):
        s = args[0]
        ss = s / TKF0
        results = function(ss, *args[1:], **kwargs)
        if 'dFds2' in results:
            results['dFds2'] /= TKF0 * TKF0
        return results

    return wrapper


@s_with_tkf0
def LKTmVW(ss: DirectField, calcType: Set[str], params: List[float], **kwargs) -> Dict:
    r'''
    \cite{luo2018simple}
    '''
    exp_1 = np.exp(-params[0] * ss)
    exp_2 = np.exp(-2.0 * params[0] * ss)
    F = 2.0 * exp_1 / (1.0 + exp_2)
    results = {'F': F}
    if 'V' in calcType:
        dFds2 = -params[0] * (1.0 - exp_2) / (1.0 + exp_2) * F / ss
        results.update({'dFds2': dFds2})

    return results


def DK(s: DirectField, calcType: Set[str], params: List[float], **kwargs) -> Dict:
    r'''
    \cite{garcia2007kinetic} (8)
    '''
    x = s * s / (72 * C_TF)
    Fa = (
        9.0 * params[5] * x**4
        + params[2] * x**3
        + params[1] * x**2
        + params[0] * x
        + 1.0
    )
    Fb = params[5] * x**3 + params[4] * x**2 + params[3] * x + 1.0
    F = Fa / Fb
    results = {'F': F}
    if "V" in calcType:
        dFds2 = (
            36.0 * params[5] * x**3
            + 3 * params[2] * x**2
            + 2 * params[1] * x
            + params[0]
        ) / Fb - Fa / (Fb * Fb) * (
            3.0 * params[5] * x**2 + 2.0 * params[4] * x + params[3]
        )
        dFds2 /= 36.0 * C_TF
        results.update({'dFds2': dFds2})

    return results


def LLP(s: DirectField, calcType: Set[str], params: List[float], **kwargs) -> Dict:
    r'''
    \cite{garcia2007kinetic} (9)[!x]
    \cite{gotz2009performance} (18)
    '''
    bs = CBRT_TWO * s
    bs2 = bs * bs
    Fa = params[0] * bs2
    Fb = 1.0 + params[1] * bs * np.arcsinh(bs)
    F = 1.0 + Fa / Fb
    results = {'F': F}
    if "V" in calcType:
        dFds2 = 2.0 * params[0] / Fb - (
            params[0] * params[1] * bs * np.arcsinh(bs)
            + Fa * params[1] / np.sqrt(1.0 + bs2)
        ) / (Fb * Fb)
        dFds2 *= CBRT_TWO * CBRT_TWO
        results.update({'dFds2': dFds2})

    return results


def OL(s: DirectField, calcType: Set[str], params: List[float], **kwargs) -> Dict:
    r'''
    \cite{gotz2009performance} (16)
    \cite{gotz2009performance} (17)
    '''
    tol2 = 1.0e-8
    F = 1.0 + s * s / 72.0 / C_TF + params[0] / C_TF * s / (1 + params[1] * 4 * s)
    results = {'F': F}
    if "V" in calcType:
        mask = s > tol2
        dFds2 = 1.0 / 36.0 / C_TF * np.ones_like(s)
        dFds2[mask] += (
            params[0] / C_TF / PowerInt((1 + params[1] * 4 * s[mask]), 2) / s[mask]
        )
        results.update({'dFds2': dFds2})

    return results


def THAK(s: DirectField, calcType: Set[str], params: List[float], **kwargs) -> Dict:
    r'''
    \cite{garcia2007kinetic} (12),
    \cite{gotz2009performance} (22),
    \cite{hfofke} (15)[!x]
    '''
    tol2 = 1.0e-8
    bs = CBRT_TWO * s
    bs2 = bs * bs
    F = (
        1.0
        + params[0] * bs2 / (1.0 + params[1] * bs * np.arcsinh(bs))
        - params[2] * bs / (1.0 + 2 ** (5.0 / 3.0) * bs)
    )
    results = {'F': F}
    if "V" in calcType:
        mask = s > tol2
        Fb = (1.0 + params[1] * bs * np.arcsinh(bs)) ** 2
        dFds2 = (
            -(params[0] * params[1] * bs2) / np.sqrt(1 + bs2)
            + (params[0] * params[1] * bs * np.arcsinh(bs))
            + 2.0 * params[0]
        ) / Fb
        dFds2[mask] -= params[2] / (1.0 + 2 ** (5.0 / 3.0) * bs[mask]) ** 2 / bs[mask]
        dFds2 *= CBRT_TWO * CBRT_TWO
        results.update({'dFds2': dFds2})

    return results


def B86A(s: DirectField, calcType: Set[str], params: List[float], **kwargs) -> Dict:
    r'''
    \cite{garcia2007kinetic} (13)
    '''
    bs = CBRT_TWO * s
    bs2 = bs * bs
    Fa = params[0] * bs2
    Fb = 1.0 + params[1] * bs2
    F = 1.0 + Fa / Fb
    results = {'F': F}
    if "V" in calcType:
        dFds2 = 2 * params[0] / (Fb * Fb) * (CBRT_TWO * CBRT_TWO)
        results.update({'dFds2': dFds2})

    return results


def B86B(s: DirectField, calcType: Set[str], params: List[float], **kwargs) -> Dict:
    r'''
    \cite{garcia2007kinetic} (14)
    '''
    bs = CBRT_TWO * s
    bs2 = bs * bs
    Fa = params[0] * bs2
    Fb = PowerInt((1.0 + params[1] * bs2), 4, 5)
    F = 1.0 + Fa / Fb
    results = {'F': F}
    if "V" in calcType:
        dFds2 = (
            (2 * params[0] * (params[1] * bs2 + 5.0))
            / (5 * (1.0 + params[1] * bs2) * Fb)
            * (CBRT_TWO * CBRT_TWO)
        )
        results.update({'dFds2': dFds2})

    return results


def DK87(s: DirectField, calcType: Set[str], params: List[float], **kwargs) -> Dict:
    r'''
    \cite{garcia2007kinetic} (15)
    '''
    bs = CBRT_TWO * s
    bs2 = bs * bs
    Fa = params[0] * bs2 * (1 + params[1] * bs)
    Fb = 1.0 + params[2] * bs2
    F = 1.0 + Fa / Fb
    results = {'F': F}
    if "V" in calcType:
        dFds2 = (
            params[0]
            * (2.0 + 3.0 * params[1] * bs + params[1] * params[2] * bs2 * bs)
            / (Fb * Fb)
            * (CBRT_TWO * CBRT_TWO)
        )
        results.update({'dFds2': dFds2})

    return results


@s_with_tkf0
def PW86(ss: DirectField, calcType: Set[str], params: List[float], **kwargs) -> Dict:
    r'''
    \cite{gotz2009performance} (19)
    '''
    s2 = ss * ss
    s4 = s2 * s2
    s6 = s2 * s4
    Fa = 1.0 + params[0] * s2 + params[1] * s4 + params[2] * s6
    F = np.power(Fa, 1.0 / 15.0)
    results = {'F': F}
    if "V" in calcType:
        dFds2 = (
            2.0
            / 15.0
            * (params[0] + 2 * params[1] * s2 + 3 * params[2] * s4)
            / np.power(Fa, 14.0 / 15.0)
        )
        results.update({'dFds2': dFds2})

    return results


@s_with_tkf0
def PW910(ss: DirectField, calcType: Set[str], params: List[float], **kwargs) -> Dict:
    r'''
    \cite{gotz2009performance} (20)
    '''
    s2 = ss * ss
    s4 = s2 * s2
    Fa = (params[1] - params[2] * np.exp(-params[3] * s2)) * s2 - params[5] * s4
    Fb = 1.0 + params[0] * ss * np.arcsinh(params[4] * ss) + params[5] * s4
    F = 1.0 + Fa / Fb
    results = {'F': F}
    if "V" in calcType:
        Fa_s2 = (params[1] - params[2] * np.exp(-params[3] * s2)) - params[5] * s2

        dFds2 = 2.0 * (
            params[1]
            + (params[3] * s2 - 1) * params[2] * np.exp(-params[3] * s2)
            - 4.0 * params[5] * s2
        ) / Fb - Fa_s2 * ss / (Fb * Fb) * (
            (params[0] * params[4] * ss) / (params[4] ** 2 * s2 + 1)
            + params[0] * np.arcsinh(params[4] * ss)
            + 4.0 * params[5] * s2
        )
        results.update({'dFds2': dFds2})

    return results


@s_with_tkf0
def PW91(ss: DirectField, calcType: Set[str], params: List[float], **kwargs) -> Dict:
    r'''
    \cite{lacks1994tests} (16) and
    \cite{garcia2007kinetic} (17)[!x]
    '''
    tol2 = 1.0e-8
    s2 = ss * ss
    s4 = s2 * s2
    Fa = (
        1.0
        + params[0] * ss * np.arcsinh(params[4] * ss)
        + (params[1] - params[2] * np.exp(-params[3] * s2)) * s2
    )
    Fb = 1.0 + params[0] * ss * np.arcsinh(params[4] * ss) + params[5] * s4
    F = Fa / Fb
    results = {'F': F}
    if "V" in calcType:
        Fa_s = (
            params[0] * params[4] * ss / np.sqrt(params[4] ** 2 * s2 + 1)
            + params[0] * np.arcsinh(params[4] * ss)
            + 2.0
            * ss
            * (
                params[1]
                + 2.0 * params[2] * params[3] * s2 * np.exp(-params[3] * s2)
                - 2.0 * params[2] * np.exp(-params[3] * s2)
            )
        )

        dFds2 = Fa_s / Fb - Fa / (Fb * Fb) * (
            params[0] * params[4] * ss / np.sqrt(params[4] ** 2 * s2 + 1)
            + (params[0] * np.arcsinh(params[4] * ss) + 4.0 * params[5] * s2 * ss)
        )

        mask = ss > tol2
        dFds2[mask] /= ss[mask]
        results.update({'dFds2': dFds2})

    return results


@s_with_tkf0
def LG94(ss: DirectField, calcType: Set[str], params: List[float], **kwargs) -> Dict:
    r'''
    \cite{garcia2007kinetic} (18)
    '''
    s2 = ss * ss
    s4 = s2 * s2
    s6 = s4 * s2
    s8 = s4 * s4
    s10 = s4 * s6
    s12 = s6 * s6
    Fa = (
        1.0
        + params[0] * s2
        + params[1] * s4
        + params[2] * s6
        + params[3] * s8
        + params[4] * s10
        + params[5] * s12
    )
    Fb = 1.0 + 1e-8 * s2
    F = np.power((Fa / Fb), params[6])
    results = {'F': F}
    if "V" in calcType:
        dFds2 = (
            -2
            * params[6]
            * F
            * (
                1e-8 * Fa
                - Fb
                * (
                    params[0]
                    + 2 * params[1] * s2
                    + 3 * params[2] * s4
                    + 4 * params[3] * s6
                    + 5 * params[4] * s8
                    + 6 * params[5] * s10
                )
            )
            / (Fa * Fb)
        )
        results.update({'dFds2': dFds2})

    return results


@s_with_tkf0
def P92(ss: DirectField, calcType: Set[str], params: List[float], **kwargs) -> Dict:
    r'''
    \cite{gotz2009performance} (14)
    \cite{gotz2009performance} (15)
    '''
    s2 = ss * ss
    s4 = s2 * s2
    Fa = params[0] + params[1] * s2 + params[2] * s4
    Fb = params[0] + params[3] * s2
    F = Fa / Fb
    results = {'F': F}
    if "V" in calcType:
        dFds2 = (2.0 * params[1] + 4.0 * params[2] * s2) / Fb - (
            2.0 * params[3] * Fa
        ) / (Fb * Fb)
        results.update({'dFds2': dFds2})

    return results


@s_with_tkf0
def PBE(ss: DirectField, calcType: Set[str], params: List[float], **kwargs) -> Dict:
    r'''
    \cite{gotz2009performance} (23)
    \cite{hfofke} (20)
    \cite{hfofke} (32)
    \cite{hfofke} (33)
    '''
    s2 = ss * ss
    Fa = params[1] * s2
    Fb = 1.0 + params[1] / params[0] * s2
    Fb2 = Fb * Fb
    F = 1.0 + Fa / Fb
    len_params = len(params)
    if len_params > 2:
        s4 = s2 * s2
        Fb3 = Fb2 * Fb
        F += params[2] * s4 / Fb2
        if len_params > 3:
            s6 = s4 * s2
            F += params[3] * s6 / Fb3
    results = {'F': F}
    if "V" in calcType:
        dFds2 = 2.0 * params[1] / Fb2
        if len_params > 2:
            dFds2 += 4 * params[2] * s2 / Fb3
            if len_params > 3:
                dFds2 += 6 * params[3] * s4 / (Fb3 * Fb)
        results.update({'dFds2': dFds2})

    return results


@s_with_tkf0
def P82(ss: DirectField, calcType: Set[str], params: List[float], **kwargs) -> Dict:
    r'''
    \cite{hfofke} (9)
    '''
    s2 = ss * ss
    s6 = s2 * s2 * s2
    Fb = 1 + s6
    F = 1.0 + params[0] * s2 / Fb
    results = {'F': F}
    if "V" in calcType:
        dFds2 = params[0] * (2.0 / Fb - 6.0 * s6 / (Fb * Fb))
        results.update({'dFds2': dFds2})

    return results


@s_with_tkf0
def VJKS00(ss: DirectField, calcType: Set[str], params: List[float], **kwargs) -> Dict:
    r'''
    \cite{hfofke} (18)
    '''
    s2 = ss * ss
    s4 = s2 * s2
    s6 = s4 * s2
    Fa = 1.0 + params[0] * s2 - params[2] * s6
    Fb = 1.0 + params[1] * s2 + params[2] * s4
    F = Fa / Fb
    results = {'F': F}
    if "V" in calcType:
        dFds2 = (2.0 * params[0] - 6.0 * params[2] * s4) / Fb - (
            2.0 * params[1] + 4.0 * params[2] * s2
        ) * Fa / (Fb * Fb)
        results.update({'dFds2': dFds2})

    return results


@s_with_tkf0
def VT84F(ss: DirectField, calcType: Set[str], params: List[float], **kwargs) -> Dict:
    r'''
    \cite{hfofke} (33)
    '''
    tol2 = 1.0e-8
    s2 = ss * ss
    s2[s2 < tol2] = tol2
    s4 = s2 * s2
    F = (
        1.0
        + 5.0 / 3.0 * s2
        + params[0] * s2 * np.exp(-params[1] * s2) / (1 + params[0] * s2)
        + (1 - np.exp(-params[1] * s4)) * (1.0 / s2 - 1.0)
    )
    results = {'F': F}
    if "V" in calcType:
        dFds2 = (
            10.0 / 3.0
            + 2.0
            * params[0]
            * np.exp(-params[1] * s2)
            * (1.0 - params[1] * s2)
            / (1 + params[0] * s2)
            - 2.0
            * params[0]
            * s2
            * np.exp(-params[1] * s2)
            / PowerInt((1 + params[0] * s2), 2)
            + 4.0 * params[1] * s2 * (1.0 / s2 - 1.0) * np.exp(-params[1] * s4)
            - 2.0 * (1 - np.exp(-params[1] * s4)) / (s4)
        )
        results.update({'dFds2': dFds2})

    return results


@s_with_tkf0
def TFVW(ss: DirectField, calcType: Set[str], params: List[float], **kwargs) -> Dict:
    ''' '''
    s2 = ss * ss
    F = params[0] + 5.0 / 3.0 * params[1] * s2
    results = {'F': F}
    if "V" in calcType:
        dFds2 = 10.0 / 3.0 * params[1] * np.ones_like(ss)
        results.update({'dFds2': dFds2})

    return results


@s_with_tkf0
def STV(ss: DirectField, calcType: Set[str], params: List[float], **kwargs) -> Dict:
    ''' '''
    s2 = ss * ss
    Fb = params[3] + params[2] * s2
    F = params[0] + 5.0 / 3.0 * params[1] * s2 / Fb
    results = {'F': F}
    if "V" in calcType:
        dFds2 = 5.0 / 3.0 * params[1] * (2.0 / Fb - 2.0 * s2 * params[2] / (Fb * Fb))
        results.update({'dFds2': dFds2})

    return results


@s_with_tkf0
def PBE2M(ss: DirectField, calcType: Set[str], params: List[float], **kwargs) -> Dict:
    ''' '''
    s2 = ss * ss
    Fa = params[2] * s2
    Fb = params[0] + params[1] * s2
    F = 1.0 + Fa / Fb
    results = {'F': F}
    if "V" in calcType:
        dFds2 = 2.0 * params[2] / (Fb * Fb)
        results.update({'dFds2': dFds2})

    return results


@s_with_tkf0
def PG(ss: DirectField, calcType: Set[str], params: List[float], **kwargs) -> Dict:
    s2 = ss * ss
    Fa = np.exp(-params[0] * s2)
    F = Fa + 5.0 / 3.0 * params[1] * s2
    results = {'F': F}
    if "V" in calcType:
        dFds2 = -2.0 * params[0] * Fa + 10.0 / 3.0 * params[1]
        results.update({'dFds2': dFds2})

    return results


@s_with_tkf0
def GE2(
    ss: DirectField, calcType: Set[str], params: List[float], q: DirectField, **kwargs
) -> Dict:
    s2 = ss * ss
    F = params[0] + params[1] * s2 + params[2] * q
    results = {'F': F}
    if "V" in calcType:
        dFds2 = 2.0 * params[1]
        dFdq = params[2]
        results['dFds2'] = dFds2
        results['dFdq'] = dFdq

    return results


@s_with_tkf0
def GE4(
    ss: DirectField, calcType: Set[str], params: List[float], q: DirectField, **kwargs
) -> Dict:
    results = GE2.__wrapped__(ss, calcType=calcType, params=params[:3], q=q, **kwargs)
    s2 = ss * ss
    F = params[3] * q * q + params[4] * q * s2 + params[5] * s2 * s2
    results['F'] += F
    if "V" in calcType:
        dFds2 = 2.0 * params[4] * q + 4.0 * params[5] * s2
        dFdq = 2.0 * params[3] * q + params[4] * s2
        results['dFds2'] += dFds2
        results['dFdq'] += dFdq

    return results


@s_with_tkf0
def PGSL(
    ss: DirectField, calcType: Set[str], params: List[float], q: DirectField, **kwargs
) -> Dict:
    s2 = ss * ss
    Fa = np.exp(-params[1] * s2)
    F = Fa + q**2 * params[0] + 5.0 / 3.0 * params[2] * s2
    results = {'F': F}
    if "V" in calcType:
        dFds2 = -2.0 * params[1] * Fa + 10.0 / 3.0 * params[2]
        dFdq = 2.0 * params[0] * q
        results['dFds2'] = dFds2
        results['dFdq'] = dFdq

    return results


@s_with_tkf0
def PGSLr(
    ss: DirectField, calcType: Set[str], params: List[float], q: DirectField, **kwargs
) -> Dict:
    results = PGSL.__wrapped__(ss, calcType=calcType, params=params[2:], q=q, **kwargs)
    s2 = ss * ss
    F = -params[0] * q * s2 + params[1] * s2 * s2
    results['F'] += F
    if "V" in calcType:
        dFds2 = -2.0 * params[0] * q + 4.0 * params[1] * s2
        dFdq = -params[0] * s2
        results['dFds2'] += dFds2
        results['dFdq'] += dFdq
    return results


@s_with_tkf0
def LKT_legacy(
    ss: DirectField, calcType: Set[str], params: List[float], **kwargs
) -> Dict:
    s2 = ss * ss
    F = np.empty_like(ss)
    dFds2 = np.empty_like(ss)
    mask1 = ss > 100.0
    mask2 = ss < 1e-5
    mask = np.invert(np.logical_or(mask1, mask2))
    F[mask] = 1.0 / np.cosh(params[0] * ss[mask]) + 5.0 / 3.0 * (s2[mask]) * params[1]
    F[mask1] = 5.0 / 3.0 * (s2[mask1]) * params[1]
    F[mask2] = (
        1.0
        + (5.0 / 3.0 * params[1] - 0.5 * params[0] ** 2) * s2[mask2]
        + 5.0 / 24.0 * params[0] ** 4 * s2[mask2] ** 2
    )  # - 61.0/720.0 * params[0] ** 6 * s2[mask2] ** 3
    results = {'F': F}
    if "V" in calcType:
        dFds2[mask] = (
            10.0 / 3.0 * params[1]
            - params[0]
            * np.sinh(params[0] * ss[mask])
            / np.cosh(params[0] * ss[mask]) ** 2
            / ss[mask]
        )
        dFds2[mask1] = 10.0 / 3.0 * params[1]
        dFds2[mask2] = (
            10.0 / 3.0 * params[1]
            - params[0] ** 2
            + 5.0 / 6.0 * params[0] ** 4 * s2[mask2]
            - 61.0 / 120.0 * params[0] ** 6 * s2[mask2] ** 2
        )
        results.update({'dFds2': dFds2})

    return results


LLP_dict = {"function": LLP, "params": [0.0044188, 0.0253]}
OL_dict = {"function": OL, "params": [0.00677, 0.0]}
THAK_dict = {"function": THAK, "params": [0.0055, 0.0253, 0.072]}
B86_dict = {"function": B86A, "params": [0.0039, 0.004]}
PW91_dict = {
    "function": PW91,
    "params": [0.19645, 0.2747, 0.1508, 100.0, 7.7956, 0.004],
}
APBE_dict = {"function": PBE, "params": [0.804, 0.23889]}
REVAPBE_dict = {"function": PBE, "params": [1.245, 0.23889]}
TFVW_dict = {"function": TFVW, "params": [1.0, 1.0]}

GGA_KEDF_list = {
    "LKT-VW": {"function": LKTmVW, "params": [1.3]},
    "LKT": {"function": LKT_legacy, "params": [1.3, 1.0]},  # \cite{luo2018simple}
    "DK": {
        "function": DK,
        "params": [0.95, 14.28111, -19.5762, -0.05, 9.99802, 2.96085],
    },
    # \cite{garcia2007kinetic} (8)
    "LLP": LLP_dict,  # \cite{garcia2007kinetic} (9)[!x] \cite{gotz2009performance} (18)
    "LLP91": LLP_dict,  # \cite{garcia2007kinetic} (9)[!x] \cite{gotz2009performance} (18)
    "OL1": OL_dict,  # \cite{gotz2009performance} (16)
    "OL": OL_dict,  # \cite{gotz2009performance} (16)
    "OL2": {"function": OL, "params": [0.0887, 1.0]},  # \cite{gotz2009performance} (17)
    "T92": THAK_dict,
    # \cite{garcia2007kinetic} (12),\cite{gotz2009performance} (22), \cite{hfofke} (15)[!x]
    "THAK": THAK_dict,
    # \cite{garcia2007kinetic} (12),\cite{gotz2009performance} (22), \cite{hfofke} (15)[!x]
    "B86A": B86_dict,  # \cite{garcia2007kinetic} (13)
    "B86": B86_dict,  # \cite{garcia2007kinetic} (13)
    "B86B": {
        "function": B86B,
        "params": [0.00403, 0.007],
    },  # \cite{garcia2007kinetic} (14)
    "DK87": {
        "function": DK87,
        "params": [7.0 / 324.0 / (18.0 * np.pi**4) ** (1.0 / 3.0), 0.861504, 0.044286],
    },
    # \cite{garcia2007kinetic} (15)
    "PW86": {
        "function": PW86,
        "params": [1.296, 14.0, 0.2],
    },  # \cite{gotz2009performance} (19)
    "PW91O": {
        "function": PW910,
        "params": [0.093907, 0.26608, 0.0809615, 100.0, 76.320, 0.57767e-4],
    },
    # (A1, A2, A3, A4, A, B1)  # \cite{gotz2009performance} (20)
    "PW91": PW91_dict,
    # (A1, A2, A3, A4, A, B1) # \cite{lacks1994tests} (16) and \cite{garcia2007kinetic} (17)[!x]
    "PW91k": PW91_dict,
    # (A1, A2, A3, A4, A, B1) # \cite{lacks1994tests} (16) and \cite{garcia2007kinetic} (17)[!x]
    "LG94": {
        "function": LG94,
        "params": [
            (1e-8 + 0.1234) / 0.024974,
            29.790,
            22.417,
            12.119,
            1570.1,
            55.944,
            0.024974,
        ],
    },
    # a2, a4, a6, a8, a10, a12, b # \cite{garcia2007kinetic} (18)
    "E00": {
        "function": P92,
        "params": [135.0, 28.0, 5.0, 3.0],
    },  # \cite{gotz2009performance} (14)
    "P92": {
        "function": P92,
        "params": [1.0, 88.3960, 16.3683, 88.2108],
    },  # \cite{gotz2009performance} (15)
    "PBE2": {
        "function": PBE,
        "params": [2.0309 / 0.2942, 2.0309],
    },  # \cite{gotz2009performance} (23)
    "PBE3": {
        "function": PBE,
        "params": [-3.7425 / 4.1355, -3.7425, 50.258],
    },  # \cite{gotz2009performance} (23)
    "PBE4": {
        "function": PBE,
        "params": [-7.2333 / 1.7107, -7.2333, 61.645, -93.683],
    },  # \cite{gotz2009performance} (23)
    "P82": {"function": P82, "params": [5.0 / 27.0]},  # \cite{hfofke} (9)
    "TW02": {"function": PBE, "params": [0.8438, 0.2319]},  # \cite{hfofke} (20)
    "APBE": APBE_dict,  # \cite{hfofke} (32)
    "APBEK": APBE_dict,  # \cite{hfofke} (32)
    "REVAPBEK": REVAPBE_dict,  # \cite{hfofke} (33)
    "REVAPBE": REVAPBE_dict,  # \cite{hfofke} (33)
    "RPBE": {"function": PBE, "params": [1.9632, 0.01979]},
    "VJKS00": {
        "function": VJKS00,
        "params": [0.8944, 0.6511, 0.0431],
    },  # \cite{hfofke} (18) !something wrong
    "LC94": {
        "function": PW91,
        "params": [0.093907, 0.26608, 0.0809615, 100.0, 76.32, 0.000057767],
    },
    # \cite{hfofke} (16) # same as PW91
    "VT84F": {
        "function": VT84F,
        "params": [2.777028126, 2.777028126 - 40.0 / 27.0],
    },  # \cite{hfofke} (33)
    # "LKT-PADE46": [1.3],
    # "LKT-PADE46-S": [1.3, 0.01],
    # "SMP": [1.0],  # test functional
    "TF": {"function": TFVW, "params": [1.0, 0.0]},
    "VW": {"function": TFVW, "params": [0.0, 1.0]},
    "X_TF_Y_VW": TFVW_dict,
    "TFVW": TFVW_dict,
    "STV": {"function": STV, "params": [1.0, 1.0, 0.01, 1.0]},
    "PBE2M": {"function": PBE2M, "params": [1.0, 0.2942, 2.0309]},
    "PG": {"function": PG, "params": [0.75, 1.0]},
    # "TEST-TF-APBEK": [1.3, 0.23889, 1.245],
}

MGGA_KEDF_list = {
    "GE2": {"function": GE2, "params": [1.0, 5.0 / 27.0, 20.0 / 9.0]},
    "GE4": {
        "function": GE4,
        "params": [1.0, 5.0 / 27.0, 20.0 / 9.0, 8.0 / 81.0, -1.0 / 9.0, 8.0 / 243.0],
    },
    "PGSL": {"function": PGSL, "params": [0.25, 40.0 / 27.0, 1.0]},  # \beta, \mu, vw
    "PGSLR": {
        "function": PGSLr,
        "params": [0.40, 0.20, 0.25, 40.0 / 27.0, 1.0],
    },  # \lambda, \sigma, \beta, \mu, vw
}


def GGAFs(
    s, functional="LKT", calcType={"E", "V"}, params=None, gga_remove_vw=None, **kwargs
):
    r"""
    ckf = (3\pi^2)^{1/3}
    C_TF = (3/10) * (3\pi^2)^{2/3} = (3/10) * ckf^2
    bb = 2^{4/3} * ckf = 2^{1/3} * TKF0
    x = (5/27) * ss * ss

    In DFTpy, default we use following definitions :
    TKF0 = 2 * ckf
    ss = s/TKF0
    x = (5/27) * s * s / (TKF0^2) = (5/27) * s * s / (4 * ckf^2) = (5 * 3)/(27 * 10 * 4) * s * s / C_TF
    x = (5/27) * ss * ss = s*s / (72*C_TF)
    bs = bb * ss = 2^{1/3} * TKF0  * ss  = 2^{1/3} * s
    b = 2^{1/3}

    Some KEDF have passed the test compared with `PROFESS3.0`.

    I hope someone can write all these equations...

    Ref:
        @article{garcia2007kinetic,
          title={Kinetic energy density study of some representative semilocal kinetic energy functionals}}
        @article{gotz2009performance,
          title={Performance of kinetic energy functional for interaction energies in a subsystem formulation of density functional theory}}
        @article{lacks1994tests,
          title = {Tests of nonlocal kinetic energy functionals}}
        @misc{hfofke,
          url = {http://www.qtp.ufl.edu/ofdft/research/KE_refdata_27ii18/Explanatory_Post_HF_OFKE.pdf}}
        @article{xia2015single,
          title={Single-point kinetic energy density functionals: A pointwise kinetic energy density analysis and numerical convergence investigation}}
        @article{luo2018simple,
          title={A simple generalized gradient approximation for the noninteracting kinetic energy density functional},
    """
    b = 2 ** (1.0 / 3.0)
    tol2 = 1e-8  # It's a very small value for safe deal with 1/s
    F = np.empty_like(s)
    dFds2 = np.empty_like(s)  # Actually, it's 1/s*dF/ds

    functional = functional.upper()
    if functional not in GGA_KEDF_list:
        raise AttributeError("%s GGA KEDF to be implemented" % functional)

    params0 = GGA_KEDF_list[functional]['params']

    if params is None:
        pass
    elif isinstance(params, (float, int)):
        params0[0] = params
    else:
        l = min(len(params), len(params0))
        params0[:l] = params[:l]

    params = params0
    if functional == "LKT":  # \cite{luo2018simple}
        ss = s / TKF0
        s2 = ss * ss
        mask1 = ss > 100.0
        mask2 = ss < 1e-5
        mask = np.invert(np.logical_or(mask1, mask2))
        F[mask] = (
            1.0 / np.cosh(params[0] * ss[mask]) + 5.0 / 3.0 * (s2[mask]) * params[1]
        )
        F[mask1] = 5.0 / 3.0 * (s2[mask1]) * params[1]
        F[mask2] = (
            1.0
            + (5.0 / 3.0 * params[1] - 0.5 * params[0] ** 2) * s2[mask2]
            + 5.0 / 24.0 * params[0] ** 4 * s2[mask2] ** 2
        )  # - 61.0/720.0 * params[0] ** 6 * s2[mask2] ** 3
        if "V" in calcType:
            dFds2[mask] = (
                10.0 / 3.0 * params[1]
                - params[0]
                * np.sinh(params[0] * ss[mask])
                / np.cosh(params[0] * ss[mask]) ** 2
                / ss[mask]
            )
            dFds2[mask1] = 10.0 / 3.0 * params[1]
            dFds2[mask2] = (
                10.0 / 3.0 * params[1]
                - params[0] ** 2
                + 5.0 / 6.0 * params[0] ** 4 * s2[mask2]
                - 61.0 / 120.0 * params[0] ** 6 * s2[mask2] ** 2
            )
            dFds2 /= TKF0**2

    elif functional == "DK":  # \cite{garcia2007kinetic} (8)
        x = s * s / (72 * C_TF)
        Fa = (
            9.0 * params[5] * x**4
            + params[2] * x**3
            + params[1] * x**2
            + params[0] * x
            + 1.0
        )
        Fb = params[5] * x**3 + params[4] * x**2 + params[3] * x + 1.0
        F = Fa / Fb
        if "V" in calcType:
            dFds2 = (
                36.0 * params[5] * x**3
                + 3 * params[2] * x**2
                + 2 * params[1] * x
                + params[0]
            ) / Fb - Fa / (Fb * Fb) * (
                3.0 * params[5] * x**2 + 2.0 * params[4] * x + params[3]
            )
            dFds2 /= 36.0 * C_TF

    elif (
        functional == "LLP" or functional == "LLP91"
    ):  # \cite{garcia2007kinetic} (9)[!x] \cite{gotz2009performance} (18)
        bs = b * s
        bs2 = bs * bs
        Fa = params[0] * bs2
        Fb = 1.0 + params[1] * bs * np.arcsinh(bs)
        F = 1.0 + Fa / Fb
        if "V" in calcType:
            dFds2 = 2.0 * params[0] / Fb - (
                params[0] * params[1] * bs * np.arcsinh(bs)
                + Fa * params[1] / np.sqrt(1.0 + bs2)
            ) / (Fb * Fb)
            dFds2 *= b * b

    elif functional == "OL1" or functional == "OL":  # \cite{gotz2009performance} (16)
        F = 1.0 + s * s / 72.0 / C_TF + params[0] / C_TF * s
        if "V" in calcType:
            mask = s > tol2
            dFds2[:] = 1.0 / 36.0 / C_TF
            dFds2[mask] += params[0] / C_TF / s[mask]

    elif functional == "OL2":  # \cite{gotz2009performance} (17)
        F = 1.0 + s * s / 72.0 / C_TF + params[0] / C_TF * s / (1 + 4 * s)
        if "V" in calcType:
            mask = s > tol2
            dFds2[:] = 1.0 / 36.0 / C_TF
            dFds2[mask] += params[0] / C_TF / (1 + 4 * s[mask]) ** 2 / s[mask]

    elif (
        functional == "T92" or functional == "THAK"
    ):  # \cite{garcia2007kinetic} (12),\cite{gotz2009performance} (22), \cite{hfofke} (15)[!x]
        bs = b * s
        bs2 = bs * bs
        F = (
            1.0
            + params[0] * bs2 / (1.0 + params[1] * bs * np.arcsinh(bs))
            - params[2] * bs / (1.0 + 2 ** (5.0 / 3.0) * bs)
        )
        # F = 1.0 + params[0] * bs2 / (1.0 + params[1] * bs * np.arcsinh(bs)) - params[2] * bs / (1.0+ 2**2 * bs)
        if "V" in calcType:
            mask = s > tol2
            Fb = (1.0 + params[1] * bs * np.arcsinh(bs)) ** 2
            dFds2 = (
                -(params[0] * params[1] * bs2) / np.sqrt(1 + bs2)
                + (params[0] * params[1] * bs * np.arcsinh(bs))
                + 2.0 * params[0]
            ) / Fb
            dFds2[mask] -= (
                params[2] / (1.0 + 2 ** (5.0 / 3.0) * bs[mask]) ** 2 / bs[mask]
            )
            # dFds2[mask] -= params[2]/(1.0+4* bs[mask]) ** 2/bs[mask]
            dFds2 *= b * b

    elif functional == "B86A" or functional == "B86":  # \cite{garcia2007kinetic} (13)
        bs = b * s
        bs2 = bs * bs
        Fa = params[0] * bs2
        Fb = 1.0 + params[1] * bs2
        F = 1.0 + Fa / Fb
        if "V" in calcType:
            dFds2 = 2 * params[0] / (Fb * Fb) * (b * b)

    elif functional == "B86B":  # \cite{garcia2007kinetic} (14)
        bs = b * s
        bs2 = bs * bs
        Fa = params[0] * bs2
        Fb = (1.0 + params[1] * bs2) ** (4.0 / 5.0)
        F = 1.0 + Fa / Fb
        if "V" in calcType:
            dFds2 = (
                (2 * params[0] * (params[1] * bs2 + 5.0))
                / (5 * (1.0 + params[1] * bs2) * Fb)
                * (b * b)
            )

    elif functional == "DK87":  # \cite{garcia2007kinetic} (15)
        bs = b * s
        bs2 = bs * bs
        Fa = params[0] * bs2 * (1 + params[1] * bs)
        Fb = 1.0 + params[2] * bs2
        F = 1.0 + Fa / Fb
        if "V" in calcType:
            dFds2 = (
                params[0]
                * (2.0 + 3.0 * params[1] * bs + params[1] * params[2] * bs2 * bs)
                / (Fb * Fb)
                * (b * b)
            )

    elif functional == "PW86":  # \cite{gotz2009performance} (19)
        ss = s / TKF0
        s2 = ss * ss
        s4 = s2 * s2
        s6 = s2 * s4
        Fa = 1.0 + params[0] * s2 + params[1] * s4 + params[2] * s6
        F = Fa ** (1.0 / 15.0)
        if "V" in calcType:
            dFds2 = (
                2.0
                / 15.0
                * (params[0] + 2 * params[1] * s2 + 3 * params[2] * s4)
                / Fa ** (14.0 / 15.0)
            )
            dFds2 /= TKF0 * TKF0

    elif functional == "PW91O":  # \cite{gotz2009performance} (20)
        ss = s / TKF0
        s2 = ss * ss
        s4 = s2 * s2
        Fa = (params[1] - params[2] * np.exp(-params[3] * s2)) * s2 - params[5] * s4
        Fb = 1.0 + params[0] * ss * np.arcsinh(params[4] * ss) + params[5] * s4
        F = 1.0 + Fa / Fb
        if "V" in calcType:
            Fa_s2 = (params[1] - params[2] * np.exp(-params[3] * s2)) - params[5] * s2

            dFds2 = 2.0 * (
                params[1]
                + (params[3] * s2 - 1) * params[2] * np.exp(-params[3] * s2)
                - 4.0 * params[5] * s2
            ) / Fb - Fa_s2 * ss / (Fb * Fb) * (
                (params[0] * params[4] * ss) / (params[4] ** 2 * s2 + 1)
                + params[0] * np.arcsinh(params[4] * ss)
                + 4.0 * params[5] * s2
            )
            dFds2 /= TKF0 * TKF0

    elif (
        functional == "PW91" or functional == "PW91k"
    ):  # \cite{lacks1994tests} (16) and \cite{garcia2007kinetic} (17)[!x]
        ss = s / TKF0
        s2 = ss * ss
        s4 = s2 * s2
        Fa = (
            1.0
            + params[0] * ss * np.arcsinh(params[4] * ss)
            + (params[1] - params[2] * np.exp(-params[3] * s2)) * s2
        )
        Fb = 1.0 + params[0] * ss * np.arcsinh(params[4] * ss) + params[5] * s4
        F = Fa / Fb
        if "V" in calcType:
            Fa_s = (
                params[0] * params[4] * ss / np.sqrt(params[4] ** 2 * s2 + 1)
                + params[0] * np.arcsinh(params[4] * ss)
                + 2.0
                * ss
                * (
                    params[1]
                    + 2.0 * params[2] * params[3] * s2 * np.exp(-params[3] * s2)
                    - 2.0 * params[2] * np.exp(-params[3] * s2)
                )
            )

            dFds2 = Fa_s / Fb - Fa / (Fb * Fb) * (
                params[0] * params[4] * ss / np.sqrt(params[4] ** 2 * s2 + 1)
                + (params[0] * np.arcsinh(params[4] * ss) + 4.0 * params[5] * s2 * ss)
            )

            mask = s > tol2
            dFds2[mask] /= ss[mask]
            dFds2 /= TKF0 * TKF0

    elif functional == "LG94":  # \cite{garcia2007kinetic} (18)
        ss = s / TKF0
        s2 = ss * ss
        s4 = s2 * s2
        s6 = s4 * s2
        s8 = s4 * s4
        s10 = s4 * s6
        s12 = s6 * s6
        Fa = (
            1.0
            + params[0] * s2
            + params[1] * s4
            + params[2] * s6
            + params[3] * s8
            + params[4] * s10
            + params[5] * s12
        )
        Fb = 1.0 + 1e-8 * s2
        F = (Fa / Fb) ** params[6]
        if "V" in calcType:
            dFds2 = (
                -2
                * params[6]
                * F
                * (
                    1e-8 * Fa
                    - Fb
                    * (
                        params[0]
                        + 2 * params[1] * s2
                        + 3 * params[2] * s4
                        + 4 * params[3] * s6
                        + 5 * params[4] * s8
                        + 6 * params[5] * s10
                    )
                )
                / (Fa * Fb)
            )
            dFds2 /= TKF0 * TKF0

    elif functional == "E00":  # \cite{gotz2009performance} (14)
        ss = s / TKF0
        s2 = ss * ss
        s4 = s2 * s2
        Fa = params[0] + params[1] * s2 + params[2] * s4
        Fb = params[0] + params[3] * s2
        F = Fa / Fb
        if "V" in calcType:
            dFds2 = (2.0 * params[1] + 4.0 * params[2] * s2) / Fb - (
                2.0 * params[3] * Fa
            ) / (Fb * Fb)
            dFds2 /= TKF0 * TKF0

    elif functional == "P92":  # \cite{gotz2009performance} (15)
        ss = s / TKF0
        s2 = ss * ss
        s4 = s2 * s2
        Fa = params[0] + params[1] * s2 + params[2] * s4
        Fb = params[0] + params[3] * s2
        F = Fa / Fb
        if "V" in calcType:
            dFds2 = (2.0 * params[1] + 4.0 * params[2] * s2) / Fb - (
                2.0 * params[3] * Fa
            ) / (Fb * Fb)
            dFds2 /= TKF0 * TKF0

    elif functional == "PBE2":  # \cite{gotz2009performance} (23)
        ss = s / TKF0
        s2 = ss * ss
        Fa = params[1] * s2
        Fb = 1.0 + params[0] * s2
        F = 1.0 + Fa / Fb
        if "V" in calcType:
            dFds2 = 2.0 * params[1] / (Fb * Fb)
            dFds2 /= TKF0 * TKF0

    elif functional == "PBE3":  # \cite{gotz2009performance} (23)
        ss = s / TKF0
        s2 = ss * ss
        s4 = s2 * s2
        Fb = 1.0 + params[0] * s2
        Fb2 = Fb * Fb
        F = 1.0 + params[1] * s2 / Fb + params[2] * s4 / Fb2
        if "V" in calcType:
            dFds2 = 2.0 * params[1] / Fb2 + 4 * params[2] * s2 / (Fb2 * Fb)
            dFds2 /= TKF0 * TKF0

    elif functional == "PBE4":  # \cite{gotz2009performance} (23)
        ss = s / TKF0
        s2 = ss * ss
        s4 = s2 * s2
        s6 = s4 * s2
        Fb = 1.0 + params[0] * s2
        Fb2 = Fb * Fb
        Fb3 = Fb * Fb * Fb
        F = 1.0 + params[1] * s2 / Fb + params[2] * s4 / Fb2 + params[3] * s6 / Fb3
        if "V" in calcType:
            dFds2 = (
                2.0 * params[1] / Fb2
                + 4 * params[2] * s2 / (Fb3)
                + 6 * params[3] * s4 / (Fb3 * Fb)
            )
            dFds2 /= TKF0 * TKF0

    elif functional == "P82":  # \cite{hfofke} (9)
        ss = s / TKF0
        s2 = ss * ss
        s6 = s2 * s2 * s2
        Fb = 1 + s6
        F = 1.0 + params[0] * s2 / Fb
        if "V" in calcType:
            dFds2 = params[0] * (2.0 / Fb - 6.0 * s6 / (Fb * Fb))
            dFds2 /= TKF0 * TKF0

    elif functional == "TW02":  # \cite{hfofke} (20)
        ss = s / TKF0
        s2 = ss * ss
        Fa = params[1] * s2
        Fb = 1.0 + params[1] * s2
        F = 1.0 + params[0] - params[0] / Fb
        if "V" in calcType:
            dFds2 = 2.0 * params[0] * params[1] / (Fb * Fb)
            dFds2 /= TKF0 * TKF0

    elif functional == "APBE" or functional == "APBEK":  # \cite{hfofke} (32)
        ss = s / TKF0
        s2 = ss * ss
        Fa = params[0] * s2
        Fb = 1.0 + params[0] / params[1] * s2
        F = 1.0 + Fa / Fb
        if "V" in calcType:
            dFds2 = 2.0 * params[0] / (Fb * Fb)
            dFds2 /= TKF0 * TKF0

    elif functional == "REVAPBEK" or functional == "REVAPBE":  # \cite{hfofke} (33)
        ss = s / TKF0
        s2 = ss * ss
        Fa = params[0] * s2
        Fb = 1.0 + params[0] / params[1] * s2
        F = 1.0 + Fa / Fb
        if "V" in calcType:
            dFds2 = 2.0 * params[0] / (Fb * Fb)
            dFds2 /= TKF0 * TKF0

    elif functional == "VJKS00":  # \cite{hfofke} (18) !something wrong
        ss = s / TKF0
        s2 = ss * ss
        s4 = s2 * s2
        s6 = s4 * s2
        Fa = 1.0 + params[0] * s2 - params[2] * s6
        Fb = 1.0 + params[1] * s2 + params[2] * s4
        F = Fa / Fb
        if "V" in calcType:
            dFds2 = (2.0 * params[0] - 6.0 * params[2] * s4) / Fb - (
                2.0 * params[1] + 4.0 * params[2] * s2
            ) * Fa / (Fb * Fb)
            dFds2 /= TKF0 * TKF0

    elif functional == "LC94":  # \cite{hfofke} (16) # same as PW91
        ss = s / TKF0
        s2 = ss * ss
        s4 = s2 * s2
        Fa = (
            1.0
            + params[0] * ss * np.arcsinh(params[4] * ss)
            + (params[1] - params[2] * np.exp(-params[3] * s2)) * s2
        )
        Fb = 1.0 + params[0] * ss * np.arcsinh(params[4] * ss) + params[5] * s4
        F = Fa / Fb
        if "V" in calcType:
            Fa_s = (
                params[0] * params[4] * ss / np.sqrt(params[4] ** 2 * s2 + 1)
                + params[0] * np.arcsinh(params[4] * ss)
                + 2.0
                * ss
                * (
                    params[1]
                    + 2.0 * params[2] * params[3] * s2 * np.exp(-params[3] * s2)
                    - 2.0 * params[2] * np.exp(-params[3] * s2)
                )
            )

            dFds2 = Fa_s / Fb - Fa / (Fb * Fb) * (
                params[0] * params[4] * ss / np.sqrt(params[4] ** 2 * s2 + 1)
                + (params[0] * np.arcsinh(params[4] * ss) + 4.0 * params[5] * s2 * ss)
            )

            mask = s > tol2
            dFds2[mask] /= ss[mask]
            dFds2 /= TKF0 * TKF0

    elif functional == "VT84F":  # \cite{hfofke} (33)
        ss = s / TKF0
        s2 = ss * ss
        s2[s2 < tol2] = tol2
        s4 = s2 * s2
        F = (
            1.0
            + 5.0 / 3.0 * s2
            + params[0] * s2 * np.exp(-params[1] * s2) / (1 + params[0] * s2)
            + (1 - np.exp(-params[1] * s4)) * (1.0 / s2 - 1.0)
        )
        if "V" in calcType:
            dFds2 = (
                10.0 / 3.0
                + 2.0
                * params[0]
                * np.exp(-params[1] * s2)
                * (1.0 - params[1] * s2)
                / (1 + params[0] * s2)
                - 2.0
                * params[0]
                * s2
                * np.exp(-params[1] * s2)
                / (1 + params[0] * s2) ** 2
                + 4.0 * params[1] * s2 * (1.0 / s2 - 1.0) * np.exp(-params[1] * s4)
                - 2.0 * (1 - np.exp(-params[1] * s4)) / (s4)
            )

            dFds2 /= TKF0 * TKF0

    elif functional == "LKT-PADE46":
        coef = [131040, 3360, 34, 62160, 3814, 59]
        # (131040 - 3360 *x**2 + 34 *x**4)/(131040 + 62160 *x**2 + 3814 *x**4 + 59 *x**6)
        coef[1] *= params[0] ** 2
        coef[2] *= params[0] ** 4
        coef[3] *= params[0] ** 2
        coef[4] *= params[0] ** 4
        coef[5] *= params[0] ** 6

        ss = s / TKF0
        s2 = ss * ss
        s4 = s2 * s2
        s6 = s4 * s2
        Fa = coef[0] + coef[1] * s2 + coef[2] * s4
        Fb = coef[0] + coef[3] * s2 + coef[4] * s4 + coef[5] * s6
        F = Fa / Fb + 5.0 / 3.0 * s2
        if "V" in calcType:
            dFds2 = (
                (2.0 * coef[1] + 4.0 * coef[2] * s2) / Fb
                - Fa * (2 * coef[3] + 4 * coef[4] * s2 + 6 * coef[5] * s4) / (Fb * Fb)
                + 10.0 / 3.0
            )
            dFds2 /= TKF0 * TKF0

    elif functional == "LKT-PADE46-S":
        coef = [131040, 3360, 34, 62160, 3814, 59]
        coef[1] *= params[0] ** 2
        coef[2] *= params[0] ** 4
        coef[3] *= params[0] ** 2
        coef[4] *= params[0] ** 4
        coef[5] *= params[0] ** 6
        alpha = params[1]

        ss = s / TKF0
        s2 = ss * ss
        s4 = s2 * s2
        s6 = s4 * s2
        ms = 1.0 / (1.0 + alpha * s2)
        Fa = coef[0] + coef[1] * s2 + coef[2] * s4
        Fb = coef[0] + coef[3] * s2 + coef[4] * s4 + coef[5] * s6
        F = Fa / Fb + 5.0 / 3.0 * s2 * ms
        if "V" in calcType:
            dFds2 = (
                (2.0 * coef[1] + 4.0 * coef[2] * s2) / Fb
                - Fa * (2 * coef[3] + 4 * coef[4] * s2 + 6 * coef[5] * s4) / (Fb * Fb)
                + 10.0 / 3.0 * (ms - alpha * s2 * ms * ms)
            )
            dFds2 /= TKF0 * TKF0

    elif functional == "SMP":  # test functional
        ss = s / TKF0
        s2 = ss * ss
        F = 5.0 / 3.0 * s2 + sp.erfc(s2 / params[0])
        mask = ss > 1e-5
        # mask1 = np.invert(mask)
        if "V" in calcType:
            dFds2[:] = 10.0 / 3.0
            dFds2[mask] += (
                2 / np.sqrt(np.pi) * np.exp(-((s2[mask] / params[0]) ** 2)) / ss[mask]
            )
            dFds2 /= TKF0**2

    elif functional == "TF":
        F = np.ones_like(s)
        dFds2 = np.zeros_like(F)

    elif functional == "VW":
        ss = s / TKF0
        s2 = ss * ss
        F = 5.0 / 3.0 * params[0] * s2
        if "V" in calcType:
            dFds2[:] = 10.0 / 3.0 * params[0]
            dFds2 /= TKF0**2

    elif functional == "X_TF_Y_VW" or functional == "TFVW":
        ss = s / TKF0
        s2 = ss * ss
        F = params[0] + 5.0 / 3.0 * params[1] * s2
        if "V" in calcType:
            dFds2[:] = 10.0 / 3.0 * params[1]
            dFds2 /= TKF0**2

    elif functional == "STV":
        ss = s / TKF0
        s2 = ss * ss
        Fb = params[3] + params[2] * s2
        F = params[0] + 5.0 / 3.0 * params[1] * s2 / Fb
        if "V" in calcType:
            dFds2[:] = (
                5.0 / 3.0 * params[1] * (2.0 / Fb - 2.0 * s2 * params[2] / (Fb * Fb))
            )
            dFds2 /= TKF0**2

    elif functional == "PBE2M":  #
        ss = s / TKF0
        s2 = ss * ss
        Fa = params[2] * s2
        Fb = params[0] + params[1] * s2
        F = 1.0 + Fa / Fb
        if "V" in calcType:
            dFds2 = 2.0 * params[2] / (Fb * Fb)
            dFds2 /= TKF0 * TKF0

    elif functional == "PG":
        ss = s / TKF0
        s2 = ss * ss
        Fa = np.exp(-params[0] * s2)
        F = Fa + 5.0 / 3.0 * params[1] * s2
        if "V" in calcType:
            dFds2 = -2.0 * params[0] * Fa + 10.0 / 3.0 * params[1]
            dFds2 /= TKF0 * TKF0

    # elif functional == "TEST-TF-APBEK":
    #     ss = s / TKF0
    #     s2 = ss * ss
    #
    #     Fx, dFds2 = _GGAFx(ss, s2, calcType=calcType, params=params, **kwargs)
    #
    #     Fa = params[1] * s2
    #     Fb = 1.0 + params[1] / params[2] * s2
    #     F = 1.0 + Fa / Fb * (1 - Fx)
    #     if "V" in calcType:
    #         dFds2_rest = 2.0 * params[1] / (Fb * Fb)
    #
    #         dFds2 = (1.0 - Fx) * dFds2_rest - dFds2 * Fa / Fb
    #
    #         dFds2 /= TKF0 ** 2
    # -----------------------------------------------------------------------
    if gga_remove_vw is not None and gga_remove_vw:
        if isinstance(gga_remove_vw, (int, float)):
            pa = float(gga_remove_vw)
        else:
            pa = 1.0
        ss = s / TKF0
        s2 = ss * ss
        F -= 5.0 / 3.0 * s2 * pa
        if "V" in calcType:
            dFds2 -= 10.0 / 3.0 / TKF0**2 * pa

    return F, dFds2


@timer()
def GGA(
    rho: DirectField,
    functional: str = "LKT",
    calcType: Set[str] = {"E", "V"},
    split: bool = False,
    params=None,
    sigma=None,
    gga_remove_vw=None,
    mgga=False,
    **kwargs,
):
    """
    Interface to compute GGAs internally to DFTpy.
    This is the default way, even though DFTpy can generate some of the GGAs with LibXC.
    Nota Bene: gradient and divergence is done brute force here and in a non-smooth way.
               while the LibXC implementation can be numerically smoothed by changing
               flag='standard' to flag='smooth'. The results with smooth math are
               slightly different.
    """
    if 'S' in calcType:
        calcType = list(calcType)
        calcType.extend(['E', 'V'])
    functional = functional.upper()
    if mgga:
        if functional in MGGA_KEDF_list:
            Fs_dict = MGGA_KEDF_list[functional]
        else:
            raise AttributeError("%s MGGA KEDF to be implemented" % functional)
    elif functional in GGA_KEDF_list:
        Fs_dict = GGA_KEDF_list[functional]
    else:
        raise AttributeError("%s GGA KEDF to be implemented" % functional)
    #
    rhom = rho.copy()
    tol = 1e-16
    rhom[rhom < tol] = tol

    rho23 = PowerInt(rhom, 2, 3)
    rho53 = rho23 * rhom
    tf = C_TF * rho53
    rho43 = rho23 * rho23
    rho83 = rho43 * rho43

    if sigma is None:
        gradient_flag = 'standard'
    else:
        gradient_flag = 'supersmooth'
    rhoGrad = rho.gradient(flag=gradient_flag, force_real=True, sigma=sigma)
    s = (
        np.sqrt(
            PowerInt(rhoGrad[0], 2) + PowerInt(rhoGrad[1], 2) + PowerInt(rhoGrad[2], 2)
        )
        / rho43
    )
    s[s < tol] = tol

    if mgga:
        q = rho.laplacian(force_real=True, sigma=sigma) / (TKF0**2) / rho53
        dq0 = -5.0 / 3.0 * q / rhom
        dq2 = 1.0 / (TKF0**2) / rho53
    else:
        q = None

    if params is None:
        params = Fs_dict["params"]
    else:
        if isinstance(params, (float, int)):
            params = [params]
        len_default_params = len(Fs_dict["params"])
        len_params = len(params)
        if len_params > len_default_params:
            params = params[:len_default_params]
        elif len_params < len_default_params:
            params.extend(Fs_dict["params"][len_params:])

    results = Fs_dict["function"](s, calcType, params, q=q, **kwargs)

    if gga_remove_vw is not None:
        if isinstance(gga_remove_vw, (int, float)):
            pa = float(gga_remove_vw)
        else:
            pa = 1.0
        vw = TFVW(s, calcType, [0.0, pa], **kwargs)
        results['F'] -= vw['F']
        if 'V' in calcType:
            results['dFds2'] -= vw['dFds2']

    OutFunctional = FunctionalOutput(name="GGA-" + str(functional))

    if 'E' in calcType or 'D' in calcType:
        energydensity = tf * results['F']
        if 'D' in calcType:
            OutFunctional.energydensity = energydensity
        OutFunctional.energy = energydensity.sum() * rhom.grid.dV

    if 'V' in calcType:
        pot1 = 5.0 / 3.0 * C_TF * rho23 * results['F']
        pot1 += -4.0 / 3.0 * tf * results['dFds2'] * s * s / rhom

        p3 = []
        for i in range(3):
            item = tf * results['dFds2'] * rhoGrad[i] / rho83
            p3.append(item.fft())
        g = rhom.grid.get_reciprocal().g
        pot3G = g[0] * p3[0] + g[1] * p3[1] + g[2] * p3[2]
        pot = pot1 - (1j * pot3G).ifft(force_real=True)

        if mgga:
            pot += tf * results['dFdq'] * dq0
            pot_q2 = tf * results['dFdq'] * dq2
            pot += pot_q2.laplacian(force_real=True, sigma=sigma)

        OutFunctional.potential = pot

    if 'S' in calcType:
        if mgga:
            raise AttributeError("Stress not support MGGA yet")
        stress = np.zeros((3, 3))
        vkin2 = tf * results['dFds2'] / rho83
        p = OutFunctional.energy - (pot1 * rho).sum() * rho.grid.dV
        for i in range(3):
            p -= (rhoGrad[i] * rhoGrad[i] * vkin2).sum() * rho.grid.dV
        for i in range(3):
            stress[i, i] = p
            for j in range(i, 3):
                stress[i, j] -= (vkin2 * rhoGrad[i] * rhoGrad[j]).sum() * rho.grid.dV
                stress[j, i] = stress[i, j]
        OutFunctional.stress = stress / rho.grid.volume
    return OutFunctional


def MGGA(
    rho: DirectField,
    functional: str = "LKT",
    calcType: Set[str] = {"E", "V"},
    split: bool = False,
    params=None,
    sigma=None,
    gga_remove_vw=None,
    mgga=True,
    **kwargs,
):
    return GGA(
        rho,
        functional=functional,
        calcType=calcType,
        split=split,
        params=params,
        sigma=sigma,
        gga_remove_vw=gga_remove_vw,
        mgga=True,
        **kwargs,
    )


def GGAStress(rho, energy=None, calcType={'S'}, **kwargs):
    obj = GGA(rho, calcType=calcType, energy=energy, **kwargs)
    return obj.stress
