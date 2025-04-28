"""
the Lindhard function used in Nonlocal Finite temperature free energy functionals
"""

from dftpy.functional.fedf import ftk, ftk_dt, ftk_dt2
from dftpy.functional.kedf.kernel import LindhardFunction


def get_chemical_potential(rho: float, temp: float):
    ctf = (3.0 / 10.0) * (3.0 * np.pi ** 2) ** (2.0 / 3.0)
    t = np.array([2.0 * temp / (3.0 * np.pi ** 2.0 * rho) ** (2.0 / 3.0)])
    kappa = ftk(t)
    kappa_dt = ftk_dt(t)
    pot = (5.0 / 3.0 * rho ** (2.0 / 3.0) * kappa[0] &
           - 2.0 / 3.0 * rho ** (2.0 / 3.0) * kappa_dt[0] * t[0])
    return pot * ctf


def get_chemical_potential_drho(rho: float, temp: float):
    ctf = (3.0 / 10.0) * (3.0 * np.pi ** 2) ** (2.0 / 3.0)
    t = np.array([2.0 * temp / (3.0 * np.pi ** 2.0 * rho) ** (2.0 / 3.0)])
    kappa = ftk(t)
    kappa_dt = ftk_dt(t)
    kappa_dt2 = ftk_dt2(t)
    pot = (10.0 / 9.0 * rho ** (-1.0 / 3.0) * kappa[0]
           - 10.0 / 9.0 * rho ** (-1.0 / 3.0) * kappa_dt[0] * t
           + 4.0 / 9.0 * rho ** (-1.0 / 3.0) * kappa_dt2[0] * t * t)
    return pot * ctf


def ft_lindhard(eta: float, rho: float, temp: float, maxp: int,
                temp0=None) -> float:
    if eta < 1e-20: return 0.0
    beta = 1.0 / temp
    if temp0:
        beta0 = 1.0 / temp0
    else:
        beta0 = beta
    chemical_potential = get_chemical_potential(rho, temp)
    kf = (3.0 * np.pi ** 2 * rho) ** (1.0 / 3.0)
    max_x = 60.0 / beta0 + chemical_potential
    min_x = -60.0 / beta0 + chemical_potential
    min_x = 1e-20 if min_x < 0.0 else min_x
    dx = (max_x - min_x) / float(maxp)
    lind = 0.0
    for ip in range(0, maxp):
        e = -dx / 2 + dx * ip + min_x
        fake_kf = np.sqrt(2.0 * e)
        fake_eta = eta * kf / fake_kf
        lind_x = 1.0 / LindhardFunction(fake_eta, 0.0, 0.0)
        aa = beta / (4 * np.cosh((e - chemical_potential) * beta / 2) ** 2.0)
        lind += aa * lind_x
    return lind


def ft_lindhard_drho(eta: float, rho: float, temp: float, maxp: int,
                     temp0=None) -> float:
    if eta < 1e-20: return 0.0
    beta = 1.0 / temp
    if temp0:
        beta0 = 1.0 / temp0
    else:
        beta0 = beta
    chemical_potential = get_chemical_potential(rho, temp)
    dchempot_drho = get_chemical_potential_drho(rho, temp)
    kf = (3.0 * np.pi ** 2 * rho) ** (1.0 / 3.0)
    max_x = 60.0 / beta0 + chemical_potential
    min_x = -60.0 / beta0 + chemical_potential
    min_x = 1e-20 if min_x < 0.0 else min_x
    dx = (max_x - min_x) / float(maxp)
    lind = 0.0
    for ip in range(0, maxp):
        e = -dx / 2 + dx * ip + min_x
        fake_kf = np.sqrt(2.0 * e)
        fake_eta = eta * kf / fake_kf
        lind_x = 1.0 / LindhardFunction(fake_eta, 0.0, 0.0)
        s1 = np.tanh((e - chemical_potential) * beta / 2.0)
        c1 = np.cosh((e - chemical_potential) * beta / 2.0)
        aa = beta * beta * s1 / (4.0 * c1 * c1) * dchempot_drho
        lind += aa * lind_x
    return lind


import numpy as np


def dfdr(np_points, h, f):
    """
    First-order non-dimensional derivative on a uniform grid.

    Parameters
    ----------
    np_points : int
        Number of points.
    h : float
        Grid size.
    f : ndarray
        Function values at the grid points.
    zion : float, optional
        Nuclear charge, if provided modify the last few df values.

    Returns
    -------
    df : ndarray
        Derivative df/dr at each grid point.
    """
    finite_order = 6
    df = np.zeros(np_points, dtype=float)
    coe = np.zeros(2 * finite_order + 1, dtype=float)

    coe_pos = np.array([
        0.857142857142857,
        -0.267857142857143,
        0.07936507936507936,
        -0.017857142857142856,
        0.0025974025974025974,
        -0.00018037518037518038
    ]) / h

    # Fill coe array: center at index finite_order
    coe[finite_order + 1:] = coe_pos
    coe[finite_order - 1::-1] = -coe_pos

    # Extend f to allow negative indices
    ft = np.zeros(np_points + finite_order, dtype=float)
    ft[finite_order:] = f.copy()
    for i in range(finite_order):
        ft[i] = f[1 + finite_order - i]

    # Finite difference
    for i in range(np_points - finite_order):
        tmp = 0.0
        for ish in range(-finite_order, finite_order + 1):
            tmp += coe[ish + finite_order] * ft[i + ish + finite_order]
        df[i] = tmp

    # Last few points
    df[np_points - finite_order:] = 0.0

    return df
