import os

import numpy as np
from numpy import linalg as LA
from scipy.interpolate import splev, splrep

from dftpy.ewald import CBspline
from dftpy.field import DirectField, ReciprocalField
from dftpy.formats import io
from dftpy.functional.pseudo import ReadPseudo
from dftpy.grid import RadialGrid
from dftpy.math_utils import gaussian
from dftpy.mpi import sprint


class AtomicDensity(object):
    """
    The densities for atomic atoms.
    """

    def __init__(self, grid=None, ions=None, key=None, rcut=5.0, rtol=None, **kwargs):
        self.grid = grid
        self.ions = ions
        self.key = key
        self.rcut = rcut
        self.rtol = rtol
        self.dnr = (1.0 / self.grid.nrR).reshape((3, 1))
        border = (rcut / grid.spacings).astype(np.int32) + 1
        border = np.minimum(border, grid.nrR // 2)
        self.ixyzA = np.mgrid[
            -border[0] : border[0] + 1,
            -border[1] : border[1] + 1,
            -border[2] : border[2] + 1,
        ].reshape((3, -1))
        self.prho = np.zeros((2 * border[0] + 1, 2 * border[1] + 1, 2 * border[2] + 1))
        self.i = 0

    def distance(self):
        while self.i < self.ions.nat:
            results = self.step()
            if results is not None:
                yield results
            self.i += 1

    def step(self, iatom=None, position=None, scaled_position=None):
        if scaled_position is None:
            if position is None:
                if iatom is None:
                    iatom = self.i
                if self.ions.symbols[iatom] != self.key:
                    return None
                position = self.ions.positions[iatom]
            cell = self.grid.cell
            scaled_position = cell.scaled_positions(position)
        #
        self.prho[:] = 0.0
        atomp = np.asarray(scaled_position) * self.grid.nrR
        atomp = atomp.reshape((3, 1))
        ipoint = np.floor(atomp + 1e-8)
        px = atomp - ipoint
        l123A = np.mod(ipoint.astype(np.int32) - self.ixyzA, self.grid.nrR[:, None])
        positions = (self.ixyzA + px) * self.dnr
        positions = np.einsum("j...,kj->k...", positions, self.grid.lattice)
        dists = LA.norm(positions, axis=0).reshape(self.prho.shape)
        if self.rtol:
            index = np.logical_and(dists < self.rcut, dists > self.rtol)
        else:
            index = dists < self.rcut
        mask = self.grid.get_array_mask(l123A)
        return {
            "positions": positions,
            "dists": dists,
            "index": index,
            "l123A": l123A,
            "mask": mask,
        }


class DensityGenerator(object):
    """ """

    def __init__(
        self,
        files=None,
        pseudo=None,
        is_core=False,
        direct=False,
        r={},
        arho={},
        **kwargs,
    ):
        self._r = r
        self._arho = arho
        self.direct = direct  # reciprocal method will better
        self.is_core = is_core
        self.files = files
        self.pseudo = pseudo
        self._init_data(**kwargs)

    @property
    def r(self):
        return self._r

    @r.setter
    def r(self, value):
        self._r = value

    @property
    def arho(self):
        return self._arho

    @arho.setter
    def arho(self, value):
        self._arho = value

    def _init_data(self, **kwargs):
        readpp = None
        if self.files:
            for key, infile in self.files.items():
                if not os.path.isfile(infile):
                    raise Exception(
                        "Density file "
                        + infile
                        + " for atom type "
                        + str(key)
                        + " not found"
                    )
                else:
                    if infile[-4:].lower() == "list":
                        try:
                            self._r[key], self._arho[key] = self.read_density_list(
                                infile
                            )
                        except Exception:
                            raise Exception(
                                "density file '{}' has some problems".format(infile)
                            )
                    else:
                        readpp = ReadPseudo(self.files)
        elif self.pseudo is not None:
            if hasattr(self.pseudo, 'readpp'):
                readpp = self.pseudo.readpp
            else:
                readpp = self.pseudo

        if readpp:
            if self.direct:
                if self.is_core:
                    self._r = readpp._core_density_grid_real
                    self._arho = readpp._core_density_real
                else:
                    self._r = readpp._atomic_density_grid_real
                    self._arho = readpp._atomic_density_real
            else:
                if self.is_core:
                    self._r = readpp._core_density_grid
                    self._arho = readpp._core_density
                else:
                    self._r = readpp._atomic_density_grid
                    self._arho = readpp._atomic_density

    def read_density_list(self, infile):
        with open(infile, "r") as fr:
            lines = []
            for i, line in enumerate(fr):
                lines.append(line)

        ibegin = 0
        iend = len(lines)
        data = [line.split()[0:2] for line in lines[ibegin:iend]]
        data = np.asarray(data, dtype=float)

        r = data[:, 0]
        v = data[:, 1]
        return r, v

    def guess_rho(
        self, ions, grid=None, ncharge=None, rho=None, dtol=1e-30, nspin=1, **kwargs
    ):
        if ncharge is None and self.is_core:
            ncharge = 0
        rho_s = rho
        if rho_s is not None:
            nspin = rho.rank
            if nspin > 1:
                rho = rho_s[0]

        if grid is None:
            if rho_s is not None:
                grid = rho_s.grid
            else:
                raise AttributeError("Please give one grid.")

        rho_unspin = self.guess_rho_unspin(
            ions, grid=grid, ncharge=ncharge, rho=rho, dtol=dtol, **kwargs
        )

        if self.is_core:
            if rho_s is not None:
                rho_s[:] = rho_unspin
            else:
                rho_s = rho_unspin
        else:
            if rho_s is not None:
                rho_s[:] = rho_unspin / nspin
            else:
                if nspin > 1:
                    rho_s = DirectField(grid, rank=nspin)
                    rho_s[:] = rho_unspin / nspin
                else:
                    rho_s = rho_unspin
        return rho_s

    def guess_rho_unspin(
        self, ions, grid=None, ncharge=None, rho=None, dtol=1e-30, **kwargs
    ):
        if self.is_core:  # no core density
            if len(self._arho) == 0:
                return None
            for k, v in self._arho.items():
                if v is not None:
                    break
            else:
                return None

        if len(self._arho) == 0:
            rho = self.guess_rho_heg(ions, grid, ncharge, rho, dtol=dtol, **kwargs)
        elif self.direct:
            rho = self.guess_rho_atom(ions, grid, ncharge, rho, dtol=dtol, **kwargs)
        else:
            rho = self.get_3d_value_recipe(
                ions, grid.get_reciprocal(), ncharge, rho, dtol=dtol, **kwargs
            )

        return rho

    def guess_rho_heg(self, ions, grid, ncharge=None, rho=None, dtol=1e-30, **kwargs):
        """ """
        if ncharge is None:
            ncharge = ions.get_ncharges()
        if rho is None:
            rho = DirectField(grid=grid)
        rho[:] = ncharge / rho.grid.cell.volume
        return rho

    def guess_rho_atom(
        self, ions, grid, ncharge=None, rho=None, dtol=1e-30, rcut=None, **kwargs
    ):
        """
        Note :
            Assuming the lattices are more than double of rcut, otherwise please use `get_3d_value_recipe` instead.
        """
        if rho is None:
            rho = DirectField(grid=grid) + dtol
        else:
            rho[:] = dtol
        latp = ions.cell.lengths()
        for key in self._r:
            r = self._r[key]
            arho = self._arho[key]
            if arho is None:
                continue
            # -----------------------------------------------------------------------
            if isinstance(rcut, dict):
                rc = rcut.get(key, np.max(r))
            else:
                rc = rcut or np.max(r)
            rc = min(rc, 0.5 * np.min(latp))
            # -----------------------------------------------------------------------
            rtol = r[0]
            tck = splrep(r, arho)
            generator = AtomicDensity(grid=grid, ions=ions, key=key, rcut=rc, rtol=rtol)
            for i in range(ions.nat):
                if ions.symbols[i] != key:
                    continue
                #
                results = generator.step(position=ions.positions[i])
                l123A = results.get('l123A')
                mask = results.get('mask')
                index = results.get('index')
                dists = results.get('dists')
                prho = generator.prho
                #
                prho[index] = splev(dists[index], tck, der=0)
                rho[l123A[0][mask], l123A[1][mask], l123A[2][mask]] += prho.ravel()[
                    mask
                ]
        nc = rho.integral()
        sprint('Guess density : ', nc)
        if ncharge is None:
            ncharge = ions.get_ncharges()
        if ncharge > 1e-6:
            rho[:] *= ncharge / nc
            sprint('Guess density (Scale): ', rho.integral(), comm=grid.mp.comm)
        return rho

    def get_3d_value_recipe(
        self, ions, grid, ncharge=None, rho=None, dtol=1e-30, direct=True, **kwargs
    ):
        return get_3d_value_recipe(
            self._r, self._arho, ions, grid, ncharge, rho, dtol, direct, **kwargs
        )


def get_3d_value_recipe(
    r,
    arho,
    ions,
    grid,
    ncharge=None,
    rho=None,
    dtol=0.0,
    direct=True,
    pme=True,
    order=10,
    **kwargs,
):
    """ """
    if hasattr(grid, 'get_reciprocal'):
        reciprocal_grid = grid.get_reciprocal()
    else:
        reciprocal_grid = grid
        grid = grid.get_direct()

    rho_g = ReciprocalField(reciprocal_grid)

    radial = {}
    vlines = {}
    if pme:
        Bspline = CBspline(ions=ions, grid=grid, order=order)
        qa = np.empty(grid.nr)
        scaled_positions = ions.get_scaled_positions()
        for key in r:
            r0 = r[key]
            arho0 = arho[key]
            if arho0 is None:
                continue
            radial[key] = RadialGrid(r0, arho0, direct=False)
            vlines[key] = radial[key].to_3d_grid(reciprocal_grid.q)
            qa[:] = 0.0
            for i in range(len(ions.positions)):
                if ions.symbols[i] == key:
                    qa = Bspline.get_PME_Qarray(scaled_positions[i], qa)
            qarray = DirectField(grid=grid, data=qa)
            rho_g += vlines[key] * qarray.fft()
        rho_g *= Bspline.Barray * grid.nnrR / grid.volume
    else:
        for key in r:
            r0 = r[key]
            arho0 = arho[key]
            radial[key] = RadialGrid(r0, arho0, direct=False)
            vlines[key] = radial[key].to_3d_grid(reciprocal_grid.q)
            for i in range(ions.nat):
                if ions.symbols[i] == key:
                    strf = ions.strf(reciprocal_grid, i)
                    rho_g += vlines[key] * strf

    if direct:
        if rho is None:
            rho = rho_g.ifft(force_real=True)
        else:
            rho[:] = rho_g.ifft()
        rho[rho < dtol] = dtol
        nc = rho.integral()
        sprint('Guess density (Real): ', nc, comm=grid.mp.comm)
        if ncharge is None:
            ncharge = ions.get_ncharges()
        if ncharge > 1e-6:
            rho[:] *= ncharge / nc
            sprint('Guess density (Scale): ', rho.integral(), comm=grid.mp.comm)
    else:
        if rho is None:
            rho = rho_g
        else:
            rho[:] = rho_g
    return rho


def normalization_density(density, ncharge=None, grid=None, tol=1e-300, method='shift'):
    if grid is None:
        grid = density.grid
    if method == 'scale':
        if ncharge is not None:
            ncharge = np.sum(density) * grid.dV
        ncharge = grid.mp.asum(ncharge)
        density[density < tol] = tol
        nc = grid.mp.asum(np.sum(density) * grid.dV)
        density *= ncharge / nc
    elif method == 'shift':
        if ncharge is not None:
            nc = grid.mp.asum(np.sum(density) * grid.dV)
            density += (ncharge - nc) / grid.nnrR
    else:
        pass
    if not hasattr(density, 'grid'):
        density = DirectField(grid, data=density)
    return density


def file2density(filename, density, grid=None):
    if grid is None:
        grid = density.grid
    ext = os.path.splitext(filename)[1].lower()
    if ext == ".snpy":
        density[:] = io.read_density(filename, grid=grid)
    else:
        fstr = f'!WARN : snpy format for density initialization will better, but this file is "{filename}".'
        sprint(fstr, comm=grid.mp.comm, level=2)
        if grid.mp.comm.rank == 0:
            density_gather = io.read_density(filename)
        else:
            density_gather = np.zeros(1)
        grid.scatter(density_gather, out=density)


def gen_gaussian_density(ions, grid, options={}, density=None, deriv=0, **kwargs):
    if density is None:
        density = DirectField(grid)
    ncharge = 0.0
    for key in sorted(options):
        option = options[key]
        rcut = option.get('rcut', 5.0)
        sigma = option.get('sigma', 0.3)
        scale = option.get('scale', 0.0)
        if scale is None or abs(scale) < 1e-10:
            continue
        generator = AtomicDensity(grid=grid, ions=ions, key=key, rcut=rcut)
        for i in range(ions.nat):
            if ions.symbols[i] != key:
                continue
            density = build_pseudo_density(
                ions.positions[i],
                grid,
                scale=scale,
                sigma=sigma,
                density=density,
                add=True,
                deriv=deriv,
                generator=generator,
            )
            ncharge += scale

    return density, ncharge


def build_pseudo_density(
    pos,
    grid,
    scale=0.0,
    sigma=0.3,
    rcut=5.0,
    density=None,
    add=True,
    deriv=0,
    generator=None,
):
    """
    FWHM : 2*np.sqrt(2.0*np.log(2)) = 2.354820
    """
    if density is None:
        density = DirectField(grid)
    #
    if scale is None:
        return density
    scale = np.asarray(scale)
    if np.all(np.abs(scale)) < 1e-10:
        return density
    #
    fwhm = 2.354820
    sigma_min = np.max(grid.spacings) * 2 / fwhm
    sigma = max(sigma, sigma_min)
    #
    if generator is None:
        generator = AtomicDensity(grid=grid, rcut=rcut)
    results = generator.step(position=pos)
    l123A = results.get('l123A')
    mask = results.get('mask')
    index = results.get('index')
    dists = results.get('dists')
    positions = results.get('positions')
    prho = generator.prho
    #
    if scale.size > 1:
        if deriv != 1:
            raise AttributeError("'scale' with array only works for deriv==1")
        positions = positions.reshape((3, *prho.shape))
        for i, s in enumerate(scale):
            prho[index] += (
                gaussian(dists[index], sigma)
                * positions[i][index]
                / sigma**2
                * scale[i]
            )
    else:
        prho[index] = gaussian(dists[index], sigma, deriv=deriv) * scale
        if deriv == 1:
            prho[index] *= -1
    #
    if add:
        density[l123A[0][mask], l123A[1][mask], l123A[2][mask]] += prho.ravel()[mask]
    else:
        density[l123A[0][mask], l123A[1][mask], l123A[2][mask]] = prho.ravel()[mask]
    return density
