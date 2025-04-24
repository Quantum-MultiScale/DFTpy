import numpy as np
from ase.build import make_supercell
from scipy.special import spherical_jn

from dftpy.field import DirectGrid, ReciprocalField
from dftpy.mpi import sprint


def create_basis(grid=None, ns=10, rcut=10.0, **kwargs):
    if isinstance(grid, DirectGrid):
        grid = grid.get_reciprocal()
    vg = ReciprocalField(grid)
    rc = np.arange(1, ns + 1) * np.pi / rcut
    basis = []
    for i in range(ns):
        j0 = spherical_jn(0, grid.q * rc[i])
        j0[0, 0, 0] = 0.0
        j0[grid.q > rcut] = 0.0

        vg[:] = j0
        jr = vg.ifft().real.ravel()
        basis.append(jr)
    return basis


def _eval_w(
    coefs, basis=None, veff0=None, driver=None, rho_ref=None, rho=None, conv=1e-6
):
    veff = veff0 * 1.0
    for i, c in enumerate(coefs):
        veff[:, 0] += c * basis[i]
    driver.veff = veff
    #
    driver.qepy.qepy_mod.qepy_set_effective_potential(driver.embed, veff)
    driver.qepy.qepy_mod.qepy_diagonalize(2, conv)
    driver.qepy.qepy_mod.qepy_calc_density(rho)
    w = -(driver.qepy.ener.get_eband() + driver.qepy.qepy_delta_e(veff))
    #
    rho_diff = rho_ref - rho
    w_deriv = coefs * 0
    for i in range(len(coefs)):
        w_deriv[i] = np.sum(rho_diff[:, 0] * basis[i])

    w += np.mean(veff * rho_diff) * driver.get_volume()
    d = np.abs(rho_diff).mean() * driver.get_volume() / 2
    sprint('w', w, np.abs(w_deriv).max(), d, comm=driver.comm, level=1)
    return w, w_deriv.ravel()


def split_ions(ions, rcut=10.0, order='atom-major', cutcell=False):
    latp = ions.cell.lengths()
    nmax = np.ceil(rcut / latp)
    P = np.diag(nmax)
    atoms = make_supercell(ions, P, order=order)
    split = [atoms]
    if cutcell:
        pmax = rcut / latp
        P = np.diag(pmax)
        cell = P @ ions.cell
    else:
        cell = atoms.cell

    for i in range(len(atoms)):
        a = atoms[[i]]
        if cutcell:
            a.set_cell(cell)
        split.append(a)
    return split
