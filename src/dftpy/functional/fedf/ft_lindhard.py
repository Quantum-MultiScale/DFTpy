"""
the Lindhard function used in Nonlocal Finite temperature free energy functionals
"""
from distutils.dir_util import remove_tree

import numpy as np
from dftpy.functional.fedf import ftk, ftk_dt, ftk_dt2
from dftpy.functional.kedf.kernel import LindhardFunction
from dftpy.time_data import timer


def fermi__1_2_elegent(mu: float, maxp: int) -> float:
    i12 = 0.0
    max_x = mu + 60.0
    min_x = mu - 60.0
    min_x = 0.0 if min_x < 0.0 else min_x
    dx = (max_x - min_x) / maxp
    for ip in range(1, maxp + 1):
        e = - dx / 2 + dx * ip + min_x
        lind_x = 2.0 * (e ** 0.5)
        AA = 1.0 / (4 * np.cosh((e - mu) / 2.0) ** 2.0)
        f1 = AA * lind_x
        i12 = i12 + dx * f1
    return i12


def fermi__2_elegent_drho(mu: float, temp, rho0, maxp: int) -> float:
    i12 = 0.0
    max_x = mu + 60.0
    min_x = mu - 60.0
    min_x = 0.0 if min_x < 0.0 else min_x
    beta = 1.0 / temp
    dx = (max_x - min_x) / maxp
    dchem_drho = get_chemical_potential_drho(rho0, temp)
    for ip in range(1, maxp + 1):
        e = - dx / 2 + dx * ip + min_x
        lind_x = 2.0 * (e ** 0.5)
        s1 = np.tanh((e - mu) / 2.0)
        c1 = np.cosh((e - mu) / 2.0)
        aa = beta * s1 / (4.0 * c1 * c1) * dchem_drho
        i12 = i12 + aa * lind_x * dx
    return i12


def get_chemical_potential(rho: float, temp: float):
    ctf = (3.0 / 10.0) * (3.0 * np.pi ** 2) ** (2.0 / 3.0)
    t = np.array([2.0 * temp / (3.0 * np.pi ** 2.0 * rho) ** (2.0 / 3.0)])
    kappa = ftk(t)
    kappa_dt = ftk_dt(t)
    pot = (5.0 / 3.0 * rho ** (2.0 / 3.0) * kappa[0]
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
    """
    old version 
    for ip in range(1, maxp + 1):
        e = -dx / 2 + dx * ip + min_x
        fake_kf = np.sqrt(2.0 * e)
        fake_eta = np.array([eta * kf / fake_kf])
        lind_x = 1.0 / LindhardFunction(fake_eta, 0.0, 0.0)
        aa = beta / (4 * np.cosh((e - chemical_potential) * beta / 2) ** 2.0)
        lind += aa * lind_x[0]
    """
    ip = np.arange(1, maxp + 1)
    e = -dx / 2 + dx * ip + min_x
    fake_kf = np.sqrt(2.0 * e)
    fake_eta = eta * kf / fake_kf
    lind_x = fake_kf / LindhardFunction(fake_eta, 0.0, 0.0)
    aa = beta / (4 * np.cosh((e - chemical_potential) * beta / 2) ** 2.0)
    lind = np.sum(aa * lind_x * dx)
    # print("CCC")
    return lind


@timer()
def ft_lindhard_drho(eta: float, rho: float, temp: float, maxp: int,
                     temp0=None) -> float:
    if eta < 1e-20: return 0.0
    beta = 1.0 / temp
    chem_pot = get_chemical_potential(rho, temp)
    if temp0:
        beta0 = 1.0 / temp0
        chem_pot0 = get_chemical_potential(rho, temp0)
    else:
        beta0 = beta
        chem_pot0 = chem_pot
    dchempot_drho = get_chemical_potential_drho(rho, temp)
    kf = (3.0 * np.pi ** 2 * rho) ** (1.0 / 3.0)
    max_x = 60.0 / beta0 + chem_pot0
    min_x = -60.0 / beta0 + chem_pot0
    min_x = 1e-20 if min_x < 0.0 else min_x
    dx = (max_x - min_x) / float(maxp)
    """
    loop 
    """
    ip = np.arange(1, maxp + 1)
    e = -dx / 2 + dx * ip + min_x
    fake_kf = np.sqrt(2.0 * e)
    fake_eta = eta * kf / fake_kf
    lind_x = fake_kf / LindhardFunction(fake_eta, 0.0, 0.0)
    s1 = np.tanh((e - chem_pot) * beta / 2.0)
    c1 = np.cosh((e - chem_pot) * beta / 2.0)
    aa = beta * beta * s1 / (4.0 * c1 * c1) * dchempot_drho
    lind = np.sum(aa * lind_x * dx)

    return lind


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


def check_kernel_table(kernel_table: dict, rho0: float,
                       temperature: float) -> bool:
    """
    check whither we should new or renew the kernel table.
    Ture --> not need renew
    False --> need renew
    """
    if not kernel_table:
        print("E1")
        return False
    if abs(kernel_table['rho0'] - rho0) > 1e-10:
        print("E2")
        print("saved", kernel_table['rho0'])
        print("gaved", rho0)
        return False
    if abs(kernel_table['temperature'] - temperature) > 1e-10:
        print("E3")
        return False
    return True


def init_kernel_table(kernel_table: dict, max_eta: float, neta: int,
                      delta_eta: float, maxp: int) -> bool:
    if not kernel_table:
        kernel_table['max_eta'] = max_eta
        kernel_table['neta'] = neta
        kernel_table['delta_eta'] = delta_eta
        kernel_table['maxp'] = maxp
    return True


def dfdx_5p(f0, f1, f2, f3, h: float):
    df = f0 - 8.0 * f1 + 8.0 * f2 - f3
    df = df / (12.0 * h)
    return df


def peta_pe(rho_0, q_norm, q, i: int, j: int):
    """
    copy this code to latex or markdown, then you will know everthing
    about this function
    \frac{\part\eta((1-\epsilon)q,det(1+\epsilon)^{-1}\rho_0)}{\part\epsilon_{ij}}
    =\frac{|(1-\epsilon)q|}{2(3\pi^2\rho_0)^{\frac{1}{3}}}det(1+\epsilon)^{\frac{1}{3}}
    \\= \frac{1}{3}\delta_{ij}\eta(q,\rho_0)-\frac{\eta_i(q,\rho_0)\eta_j(q,\rho_0)}{\eta(q,\rho_0)}\\
    \eta_i = \frac{q_i}{2(3\pi^2\rho_0)^{\frac{1}{3}}}

    i,j in 0 , 1, 2 means,x,y,z
    """
    kf = (3.0 * np.pi ** 2 * rho_0) ** (1.0 / 3.0)
    eta = q_norm / 2.0 / kf
    if i == j:
        peta = eta / 3.0
    else:
        peta = np.zeros_like(eta)

    peta = peta - q[i] * q[j] / q_norm / 2.0 / kf
    if q_norm[0, 0, 0] < 1e-10:
        peta[0, 0, 0] = 0.0
    return peta


def peta_peS(rho_0, q_norm, q, i: int, j: int):
    kf = (3.0 * np.pi ** 2 * rho_0) ** (1.0 / 3.0)
    eta = q_norm / 2.0 / kf
    peta = np.zeros_like(eta)
    peta = peta - q[i] * q[j] / q_norm / 2.0 / kf
    if q_norm[0, 0, 0] < 1e-10:
        peta[0, 0, 0] = 0.0
    return peta


def fill_kernel_via_table(rho0, q_norm, eta_table, weta_table):
    kf = (3.0 * np.pi ** 2 * rho0) ** (1.0 / 3.0)
    eta = q_norm / 2.0 / kf
    kernel_flat = np.interp(eta.flatten(),
                            eta_table,
                            weta_table)
    kernel = kernel_flat.reshape(eta.shape)
    if q_norm[0, 0, 0] < 1e-20:
        kernel[0, 0, 0] = 0.0
    return kernel
