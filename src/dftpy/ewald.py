import numpy as np
from scipy import special as sp
from scipy.spatial.distance import cdist

from dftpy.field import DirectField, ReciprocalField

# from itertools import product
from dftpy.mpi import sprint
from dftpy.time_data import timer


class CBspline(object):
    """
    the Cardinal B-splines
    """

    def __init__(self, ions=None, grid=None, order=10, **kwargs):
        self._order = order
        self._Mn = None
        self._bm = None
        self._Barray = None
        self._PME_Qarray = None
        ixyzA = np.mgrid[: self.order, : self.order, : self.order].reshape((3, -1))
        self._BigArray = np.zeros(grid.nr)
        self._ixyzA = ixyzA
        self._mask = np.empty(ixyzA.shape[1], dtype=bool)
        self._mask1 = np.empty(ixyzA.shape[1], dtype=bool)

        self.ions = ions

        if grid is not None:
            self.grid = grid
        else:
            raise AttributeError("Must pass grid to CBspline")

        self.mp = grid.mp

    @property
    def order(self):
        return self._order

    @property
    def bm(self):
        if self._bm is None:
            self._bm = self._calc_bm()
        return self._bm

    @property
    def Barray(self):
        if self._Barray is None:
            if self._bm is None:
                self._bm = self._calc_bm()
            bm = self._bm
            array = np.einsum("i, j, k -> ijk", bm[0], bm[1], bm[2])
            self._Barray = ReciprocalField(
                self.grid.get_reciprocal(), griddata_3d=array, rank=1
            )
        return self._Barray

    @property
    def PME_Qarray(self):
        if self._PME_Qarray is None:
            self._PME_Qarray = self._calc_PME_Qarray()
        return self._PME_Qarray

    def calc_Mn(self, x, order=None):
        """
        x -> [0, 1)
        x --> u + {0, 1, ..., order}
        u -> [0, 1), [1, 2),...,[order, order + 1)
        M_n(u) = u/(n-1)*M_(n-1)(u) + (n-u)/(n-1)*M_(n-1)(u-1)
        """
        if not order:
            order = self.order

        Mn = np.zeros(self.order + 1)
        Mn[1] = x
        Mn[2] = 1.0 - x
        for i in range(3, order + 1):
            for j in range(0, i):
                n = i - j
                # Mn[n] = (x + n - 1) * Mn[n] + (i - (x + n - 1)) * Mn[n - 1]
                Mn[n] = (x + n - 1) * Mn[n] + (j + 1 - x) * Mn[n - 1]
                Mn[n] /= i - 1
        return Mn

    def _calc_bm(self):
        nrG = self.grid.nrG
        nrR = self.grid.nrR
        # nr = self.grid.nr
        # offsets = self.grid.offsets
        nr = self.grid.get_reciprocal().nr
        offsets = self.grid.get_reciprocal().offsets
        Mn = self.calc_Mn(1.0)
        bm = []
        for i in range(3):
            q = 2.0 * np.pi * np.arange(nrG[i]) / nrR[i]
            tmp = np.exp(-1j * (self.order - 1.0) * q)
            factor = np.zeros_like(tmp)
            for k in range(1, self.order):
                factor += Mn[k] * np.exp(-1j * k * q)
            tmp /= factor
            bm.append(tmp[offsets[i] : offsets[i] + nr[i]])
        return bm

    def get_Qarray_mask(self, l123A):
        # if self.mp.comm.size == 1 :
        # return slice(None)
        offsets = self.grid.offsets.reshape((3, 1))
        nr = self.grid.nr
        mask = self._mask
        mask1 = self._mask1
        # -----------------------------------------------------------------------
        l123A -= offsets
        np.logical_and(l123A[0] > -1, l123A[0] < nr[0], out=mask)
        np.logical_and(l123A[1] > -1, l123A[1] < nr[1], out=mask1)
        np.logical_and(mask, mask1, out=mask)
        np.logical_and(l123A[2] > -1, l123A[2] < nr[2], out=mask1)
        np.logical_and(mask, mask1, out=mask)
        # -----------------------------------------------------------------------
        return mask

    def check_out_cell(self, p):
        if self.mp.comm.size == 1:
            return False
        nr = self.grid.nr
        nrR = self.grid.nrR
        ixyzb = np.arange(0, self.order).reshape((1, -1))
        l123 = np.mod(
            np.floor(p).astype(np.int32).reshape((3, 1)) - ixyzb + 1,
            nrR.reshape((3, 1)),
        )
        offsets = self.grid.offsets.reshape((3, 1))
        # -----------------------------------------------------------------------
        l123 -= offsets
        for i in range(3):
            if np.all(l123[i] < 0) or np.all(l123[i] > nr[i] - 1):
                return True
        # -----------------------------------------------------------------------
        return False

    @timer()
    def _calc_PME_Qarray(self, ions=None):
        """
        Using the smooth particle mesh Ewald method to calculate structure factors.
        """
        if ions is None:
            if self.ions is None:
                raise AttributeError("Must pass ions to CBspline")
            else:
                ions = self.ions
        #
        nrR = self.grid.nrR
        Qarray = self._BigArray
        Qarray[:] = 0.0

        ## For speed
        # ixyzA = np.mgrid[1:self.order + 1, 1:self.order + 1, 1:self.order + 1].reshape((3, -1))
        # l123A = np.mod(np.floor(Up).astype(np.int32).reshape((3, 1)) - ixyzA, nr.reshape((3, 1)))
        # ixyzA = np.mgrid[:self.order, :self.order, :self.order].reshape((3, -1))
        ixyzA = self._ixyzA
        scaled_positions = ions.get_scaled_positions()
        for i in range(ions.nat):
            Up = scaled_positions[i] * nrR
            if self.check_out_cell(Up):
                continue
            l123A = np.mod(
                np.floor(Up).astype(np.int32).reshape((3, 1)) - ixyzA + 1,
                nrR.reshape((3, 1)),
            )
            mask = self.get_Qarray_mask(l123A)
            Mn = []
            for j in range(3):
                Mn.append(self.calc_Mn(Up[j] - np.floor(Up[j])))
            Mn_multi = np.einsum(
                "i, j, k -> ijk", ions.charges[i] * Mn[0][1:], Mn[1][1:], Mn[2][1:]
            )
            Qarray[l123A[0][mask], l123A[1][mask], l123A[2][mask]] += Mn_multi.ravel()[
                mask
            ]
        return DirectField(self.grid, griddata_3d=Qarray, rank=1)

    def get_PME_Qarray(self, pos, Qarray=None, scale=None):
        """
        Using the smooth particle mesh Ewald method to calculate structure factors.
        """
        nrR = self.grid.nrR
        if Qarray is None:
            Qarray = self._BigArray
            Qarray[:] = 0.0
        ixyzA = self._ixyzA
        Up = pos * nrR
        if self.check_out_cell(Up):
            return Qarray
        Mn = []
        for j in range(3):
            Mn.append(self.calc_Mn(Up[j] - np.floor(Up[j])))
        Mn_multi = np.einsum("i, j, k -> ijk", Mn[0][1:], Mn[1][1:], Mn[2][1:])
        l123A = np.mod(
            1 + np.floor(Up).astype(np.int32).reshape((3, 1)) - ixyzA,
            nrR.reshape((3, 1)),
        )
        mask = self.get_Qarray_mask(l123A)
        value = Mn_multi.ravel()[mask]
        if scale:
            value *= scale
        Qarray[l123A[0][mask], l123A[1][mask], l123A[2][mask]] += value
        # Qarray = DirectField(self.grid,griddata_3d=Qarray,rank=1)
        return Qarray


class ewald(object):
    def __init__(
        self,
        precision=1.0e-8,
        ions=None,
        rho=None,
        grid=None,
        verbose=False,
        BsplineOrder=10,
        PME=False,
        Bspline=None,
        **kwargs,
    ):
        """
        This computes Ewald contributions to the energy.
        INPUT: precision  float, should be bigger than the machine precision and
                          smaller than single precision.
               ions       Ions class array.
               verbose    optional, wanna sprint stuff?
        """

        self.precision = precision

        self.verbose = verbose

        self.grid = grid

        if ions is not None:
            self.ions = ions
        else:
            raise AttributeError("Must pass ions to Ewald")

        if self.grid is None:
            if rho is not None:
                self.grid = rho.grid
            else:
                raise AttributeError("Must pass grid or rho to Ewald")

        self.mp = self.grid.mp

        gmax = self.Get_Gmax(self.grid)
        eta = self.Get_Best_eta(self.precision, gmax, self.ions)
        # eta = 0.2
        self.eta = eta
        self.order = BsplineOrder

        self.PME = PME
        if self.PME:
            if Bspline is None:
                self.Bspline = CBspline(
                    ions=self.ions, grid=self.grid, order=self.order
                )
            else:
                self.Bspline = Bspline
        self._energy = None
        self._forces = None
        self._stress = None

    def __call__(self, *args, **kwargs):
        return self.compute(*args, **kwargs)

    def compute(self, *args, calcType={'E'}, **kwargs):
        if 'E' in calcType:
            if self._energy is None:
                self._energy = self.energy
        from dftpy.functional.functional_output import FunctionalOutput

        functional = FunctionalOutput(name='Ewald', energy=self._energy)
        return functional

    def Get_Gmax(self, grid):
        gmax = np.sqrt(grid.get_reciprocal().g2max)
        return gmax

    def Get_Best_eta(self, precision, gmax, ions):
        """
        INPUT: precision, gmax & ions
        OUTPUT: eta
        """

        # charge
        chargeSquare = np.sum(ions.charges * ions.charges)

        # eta
        eta = 1.6
        NotGoodEta = True
        while NotGoodEta:
            # upbound = 2.0 * charge**2 * np.sqrt ( eta / np.pi) * sp.erfc ( np.sqrt (gmax / 4.0 / eta) )
            upbound = (
                4.0
                * np.pi
                * ions.nat
                * chargeSquare
                * np.sqrt(eta / np.pi)
                * sp.erfc(gmax / 2.0 * np.sqrt(1.0 / eta))
            )
            if upbound < precision:
                NotGoodEta = False
            else:
                eta = eta - 0.01
        return eta

    @timer()
    def Energy_real(self):
        L = np.sqrt(np.einsum("ij->i", self.grid.lattice**2))
        prec = sp.erfcinv(self.precision / 3.0)
        rmax = prec / np.sqrt(self.eta)
        N = np.ceil(rmax / L)
        if self.verbose:
            sprint("Map of Cells = ", N)
            sprint("Lengths = ", rmax / L)
            sprint("rmax = ", rmax)
        charges = []
        positions = []
        for ix in np.arange(-N[0], N[0] + 1):
            for iy in np.arange(-N[1], N[1] + 1):
                for iz in np.arange(-N[2], N[2] + 1):
                    R = np.einsum(
                        "j,ji->i",
                        np.array([ix, iy, iz], dtype=np.float64),
                        self.grid.lattice,
                    )
                    for i in np.arange(self.ions.nat):
                        charges.append(self.ions.charges[i])
                        positions.append(self.ions.positions[i] - R)

        Esum = 0.0
        rtol = 0.001
        Rcut = rmax
        etaSqrt = np.sqrt(self.eta)
        ## for save memory
        # for item in self.ions :
        # for j in range(len(charges)):
        # rij=item.pos-positions[j]
        # dij=rij.length()
        # if dij < Rcut and dij > rtol:
        # Esum+=charges[i]*charges[j]*sp.erfc(etaSqrt*dij)/dij
        ## for speed
        charges = np.asarray(charges)
        lb, ub = self.mp.split_number(self.ions.nat)
        for i in range(lb, ub):
            dists = cdist(positions, self.ions.positions[i].reshape((1, 3))).ravel()
            index = np.logical_and(dists < Rcut, dists > rtol)
            Esum += self.ions.charges[i] * np.sum(
                charges[index] * sp.erfc(etaSqrt * dists[index]) / dists[index]
            )
        Esum /= 2.0

        return Esum

    @timer()
    def Energy_real_fast(self):
        L = np.sqrt(np.einsum("ij->i", self.grid.lattice**2))
        prec = sp.erfcinv(self.precision / 3.0)
        rmax = prec / np.sqrt(self.eta)
        N = np.ceil(rmax / L)
        charges = []
        positions = []
        for ix in np.arange(-N[0], N[0] + 1):
            for iy in np.arange(-N[1], N[1] + 1):
                for iz in np.arange(-N[2], N[2] + 1):
                    R = np.einsum(
                        "j,ji->i",
                        np.array([ix, iy, iz], dtype=np.float64),
                        self.grid.lattice,
                    )
                    for i in range(self.ions.nat):
                        charges.append(self.ions.charges[i])
                        positions.append(self.ions.positions[i] - R)

        Esum = 0.0
        rtol = 0.001
        Rcut = rmax
        etaSqrt = np.sqrt(self.eta)
        ## for save memory
        # for item in self.ions :
        # for j in range(len(charges)):
        # rij=item.pos-positions[j]
        # dij=rij.length()
        # if dij < Rcut and dij > rtol:
        # Esum+=charges[i]*charges[j]*sp.erfc(etaSqrt*dij)/dij
        ## for speed
        positions = np.asarray(positions)
        charges = np.asarray(charges)
        lb, ub = self.mp.split_number(self.ions.nat)
        for i in range(lb, ub):
            posi = self.ions.positions[i].reshape((1, 3))
            LBound = posi - Rcut
            UBound = posi + Rcut
            index1 = np.logical_and(positions > LBound, positions < UBound)
            index1 = np.all(index1, axis=1)
            dists = cdist(positions[index1], posi).ravel()
            charges_local = charges[index1]
            index = np.logical_and(dists < Rcut, dists > rtol)
            Esum += self.ions.charges[i] * np.sum(
                charges_local[index] * sp.erfc(etaSqrt * dists[index]) / dists[index]
            )
        Esum /= 2.0

        return Esum

    @timer()
    def Energy_real_fast2(self):
        L = np.sqrt(np.einsum("ij->i", self.grid.lattice**2))
        prec = sp.erfcinv(self.precision / 3.0)
        rmax = prec / np.sqrt(self.eta)
        N = np.ceil(rmax / L).astype(np.int32)
        charges = []
        positions = []
        Rpbc = np.empty((2 * N[0] + 1, 2 * N[1] + 1, 2 * N[2] + 1, 3))
        for ix in np.arange(-N[0], N[0] + 1):
            for iy in np.arange(-N[1], N[1] + 1):
                for iz in np.arange(-N[2], N[2] + 1):
                    R = np.einsum(
                        "j,ji->i",
                        np.array([ix, iy, iz], dtype=np.float64),
                        self.grid.lattice,
                    )
                    Rpbc[ix + N[0], iy + N[1], iz + N[2], :] = R
        for i in range(self.ions.nat):
            charges.append(self.ions.charges[i])
            # positions.append(self.ions.positions[i])

        Esum = 0.0
        rtol = 0.001
        Rcut = rmax
        etaSqrt = np.sqrt(self.eta)
        # positions = np.asarray(positions)
        positions = self.ions.positions[:]
        charges = np.asarray(charges)
        PBCmap = np.zeros((2, 3), dtype=np.int32)
        PBCmap[0, :] = 0
        PBCmap[1, :] = 2 * N[:] + 1
        # PBCmap[0, :] = -N[:]
        # PBCmap[1, :] = N[:]+1
        CellBound = np.empty((2, 3))
        CellBound[0, :] = np.min(self.ions.positions, axis=0)
        CellBound[1, :] = np.max(self.ions.positions, axis=0)
        lb, ub = self.mp.split_number(self.ions.nat)
        for i in range(lb, ub):
            posi = self.ions.positions[i].reshape((1, 3))
            LBound = posi - Rcut
            UBound = posi + Rcut
            for j in range(3):
                if LBound[0, j] < CellBound[0, j]:
                    PBCmap[1, j] = 2 * N[j] + 1
                else:
                    PBCmap[1, j] = N[j] + 1

                if UBound[0, j] > CellBound[1, j]:
                    PBCmap[0, j] = 0
                else:
                    PBCmap[0, j] = N[j]
            # for j in range(3):
            #    if LBound[0, j] < CellBound[0,j] :
            #        PBCmap[1, j] = N[j]+1
            #    else :
            #        PBCmap[1, j] = 1

            #    if UBound[0, j] > CellBound[1,j] :
            #        PBCmap[0, j] = -N[j]
            #    else :
            #        PBCmap[0, j] = 0
            for i0 in range(PBCmap[0, 0], PBCmap[1, 0]):
                for i1 in range(PBCmap[0, 1], PBCmap[1, 1]):
                    for i2 in range(PBCmap[0, 2], PBCmap[1, 2]):
                        # PBCpos = posi + Rpbc[i0 + N[0], i1 + N[1], i2 + N[2], :]
                        PBCpos = posi + Rpbc[i0, i1, i2, :]
                        LBound = PBCpos - Rcut
                        UBound = PBCpos + Rcut
                        index1 = np.logical_and(positions > LBound, positions < UBound)
                        index1 = np.all(index1, axis=1)
                        dists = cdist(positions[index1], PBCpos).ravel()
                        charges_local = charges[index1]
                        index = np.logical_and(dists < Rcut, dists > rtol)
                        Esum += self.ions.charges[i] * np.sum(
                            charges_local[index]
                            * sp.erfc(etaSqrt * dists[index])
                            / dists[index]
                        )
        Esum /= 2.0

        return Esum

    @timer()
    def Energy_rec(self):
        ions = self.ions
        # rec space sum
        reciprocal_grid = self.grid.get_reciprocal()
        gg = reciprocal_grid.gg
        invgg = reciprocal_grid.invgg
        strf = ions.strf(reciprocal_grid, 0) * ions.charges[0]
        for i in np.arange(1, ions.nat):
            strf += ions.strf(reciprocal_grid, i) * ions.charges[i]
        strf_sq = np.conjugate(strf) * strf
        mask = self.grid.get_reciprocal().mask
        energy = np.sum(
            strf_sq[mask] * np.exp(-gg[mask] / (4.0 * self.eta)) * invgg[mask]
        )
        # energy = np.sum(strf_sq * np.exp(-gg / (4.0 * self.eta)) * invgg) /2.0
        energy = 4.0 * np.pi * energy.real / self.grid.volume
        # energy /= self.grid.dV ** 2

        return energy

    @timer()
    def Energy_corr(self):
        # double counting term
        const = -np.sqrt(self.eta / np.pi)
        sum = 0
        sum = np.sum(self.ions.charges * self.ions.charges)
        dc_term = const * sum

        # G=0 term of local_PP - Hartree
        const = -4.0 * np.pi * (1.0 / (4.0 * self.eta * self.grid.volume) / 2.0)
        sum = self.ions.get_ncharges()
        gzero_limit = const * sum**2

        energy = dc_term + gzero_limit

        return energy

    @property
    @timer()
    def energy(self):
        if self._energy is None:
            e_corr = self.Energy_corr()
            if self.PME:
                e_real = self.Energy_real_fast2()
                e_rec = self.Energy_rec_PME()
            else:
                e_real = self.Energy_real()
                e_rec = self.Energy_rec()

            e_corr /= self.mp.comm.size

            Ewald_Energy = e_corr + e_real + e_rec

            if self.verbose:
                sprint("Ewald sum & divergent terms in the Energy:")
                sprint("eta used = ", self.eta)
                sprint("precision used = ", self.precision)
                sprint(
                    "Ewald Energy = ",
                    self.mp.asum(Ewald_Energy),
                    self.mp.asum(e_corr),
                    self.mp.asum(e_real),
                    self.mp.asum(e_rec),
                )
            self._energy = Ewald_Energy
        return self._energy

    @property
    @timer()
    def forces(self):
        if self._forces is None:
            Ewald_Forces = self.Forces_real()
            if self.PME:
                f_rec = self.Forces_rec_PME()
            else:
                f_rec = self.Forces_rec()
            Ewald_Forces += f_rec
            self._forces = Ewald_Forces
        return self._forces

    @property
    @timer()
    def stress(self):
        if self._stress is None:

            Ewald_Stress = self.Stress_real()
            if self.PME:
                s_rec = self.Stress_rec_PME()
            else:
                s_rec = self.Stress_rec()

            Ewald_Stress += s_rec

            if self.verbose:
                sprint("Ewald_Stress\n", Ewald_Stress, s_rec)

            self._stress = Ewald_Stress
        return self._stress

    @timer()
    def Forces_real(self):
        L = np.sqrt(np.einsum("ij->i", self.grid.lattice**2))
        prec = sp.erfcinv(self.precision / 3.0)
        rmax = prec / np.sqrt(self.eta)
        N = np.ceil(rmax / L)
        charges = []
        positions = []
        for ix in np.arange(-N[0], N[0] + 1):
            for iy in np.arange(-N[1], N[1] + 1):
                for iz in np.arange(-N[2], N[2] + 1):
                    R = np.einsum(
                        "j,ji->i",
                        np.array([ix, iy, iz], dtype=np.float64),
                        self.grid.lattice,
                    )
                    for i in np.arange(self.ions.nat):
                        charges.append(self.ions.charges[i])
                        positions.append(self.ions.positions[i] - R)

        rtol = 0.001
        Rcut = rmax
        etaSqrt = np.sqrt(self.eta)
        charges = np.asarray(charges)
        positions = np.asarray(positions)
        piSqrt = np.sqrt(np.pi)
        F_real = np.zeros((self.ions.nat, 3))
        lb, ub = self.mp.split_number(self.ions.nat)
        for i in range(lb, ub):
            dists = cdist(positions, self.ions.positions[i].reshape((1, 3))).ravel()
            index = np.logical_and(dists < Rcut, dists > rtol)
            dists *= etaSqrt
            F_real[i] = self.ions.charges[i] * np.einsum(
                "ij,i->j",
                (np.array(self.ions.positions[i]) - positions[index])
                * charges[index][:, np.newaxis],
                sp.erfc(dists[index]) / dists[index] ** 3
                + 2.0 / piSqrt * np.exp(-dists[index] ** 2) / dists[index] ** 2,
            )
        F_real *= etaSqrt**3
        # F_real /= self.mp.comm.size

        return F_real

    @timer()
    def Forces_rec(self):
        reciprocal_grid = self.grid.get_reciprocal()
        gg = reciprocal_grid.gg
        invgg = reciprocal_grid.invgg

        charges = self.ions.charges
        strf = self.ions.strf(reciprocal_grid, 0) * self.ions.charges[0]
        for i in np.arange(1, self.ions.nat):
            strf += self.ions.strf(reciprocal_grid, i) * self.ions.charges[i]

        mask = reciprocal_grid.mask
        F_rec = np.empty((self.ions.nat, 3))
        charges = np.asarray(charges)
        for i in range(self.ions.nat):
            Ion_strf = self.ions.strf(reciprocal_grid, i) * self.ions.charges[i]
            # F_rec[i] = np.einsum('ijkl,ijkl->l', reciprocal_grid.g, \
            # (Ion_strf.real * strf.imag - Ion_strf.imag * strf.real)* \
            # np.exp(-gg/(4.0*self.eta))*invgg )
            # F_rec[i] = np.einsum('ijkl,ijkl->l', reciprocal_grid.g, \
            F_rec[i] = np.einsum(
                "ij, j->i",
                reciprocal_grid.g[:, mask],
                (
                    Ion_strf.real[mask] * strf.imag[mask]
                    - Ion_strf.imag[mask] * strf.real[mask]
                )
                * np.exp(-gg[mask] / (4.0 * self.eta))
                * invgg[mask],
            )
        F_rec *= 8.0 * np.pi / self.grid.volume
        return F_rec

    def Stress_real(self):
        L = np.sqrt(np.einsum("ij->i", self.grid.lattice**2))
        prec = sp.erfcinv(self.precision / 3.0)
        rmax = prec / np.sqrt(self.eta)
        N = np.ceil(rmax / L)
        charges = []
        positions = []
        for ix in np.arange(-N[0], N[0] + 1):
            for iy in np.arange(-N[1], N[1] + 1):
                for iz in np.arange(-N[2], N[2] + 1):
                    R = np.einsum(
                        "j,ji->i",
                        np.array([ix, iy, iz], dtype=np.float64),
                        self.grid.lattice,
                    )
                    for i in np.arange(self.ions.nat):
                        charges.append(self.ions.charges[i])
                        positions.append(self.ions.positions[i] - R)
        rtol = 0.001
        Rcut = rmax
        etaSqrt = np.sqrt(self.eta)
        charges = np.asarray(charges)
        S_real = np.zeros((3, 3))
        piSqrt = np.sqrt(np.pi)
        positions = np.asarray(positions)

        Stmp = np.zeros(6)
        lb, ub = self.mp.split_number(self.ions.nat)
        for ia in range(lb, ub):
            dists = cdist(positions, self.ions.positions[ia].reshape((1, 3))).ravel()
            index = np.logical_and(dists < Rcut, dists > rtol)
            Rijs = np.array(self.ions.positions[ia]) - positions[index]

            # Rvv = np.einsum('ij, ik -> ijk', Rijs, Rijs)
            k = 0
            Rv = np.zeros((len(Rijs), 6))
            for i in range(3):
                for j in range(i, 3):
                    Rv[:, k] = Rijs[:, i] * Rijs[:, j] / dists[index] ** 2
                    k += 1

            Stmp += self.ions.charges[ia] * np.einsum(
                "i, ij->j",
                charges[index]
                * (
                    2 * etaSqrt / piSqrt * np.exp(-self.eta * dists[index] ** 2)
                    + sp.erfc(etaSqrt * dists[index]) / dists[index]
                ),
                Rv,
            )

        Stmp *= -0.5 / self.grid.volume
        k = 0
        for i in range(3):
            for j in range(i, 3):
                S_real[i, j] = S_real[j, i] = Stmp[k]
                k += 1
        return S_real

    def Stress_real_fast(self):
        L = np.sqrt(np.einsum("ij->i", self.grid.lattice**2))
        prec = sp.erfcinv(self.precision / 3.0)
        rmax = prec / np.sqrt(self.eta)
        N = np.ceil(rmax / L)
        charges = []
        positions = []
        for ix in np.arange(-N[0], N[0] + 1):
            for iy in np.arange(-N[1], N[1] + 1):
                for iz in np.arange(-N[2], N[2] + 1):
                    R = np.einsum(
                        "j,ji->i",
                        np.array([ix, iy, iz], dtype=np.float64),
                        self.grid.lattice,
                    )
                    for i in np.arange(self.ions.nat):
                        charges.append(self.ions.charges[i])
                        positions.append(self.ions.positions[i] - R)
        rtol = 0.001
        Rcut = rmax
        etaSqrt = np.sqrt(self.eta)
        charges = np.asarray(charges)
        S_real = np.zeros((3, 3))
        piSqrt = np.sqrt(np.pi)
        positions = np.asarray(positions)

        Stmp = np.zeros(6)
        lb, ub = self.mp.split_number(self.ions.nat)
        for ia in range(lb, ub):
            dists = cdist(positions, self.ions.positions[ia].reshape((1, 3))).ravel()
            index = np.logical_and(dists < Rcut, dists > rtol)
            Rijs = np.array(self.ions.positions[ia]) - positions[index]

            # Rvv = np.einsum('ij, ik -> ijk', Rijs, Rijs)
            k = 0
            Rv = np.zeros((len(Rijs), 6))
            for i in range(3):
                for j in range(i, 3):
                    Rv[:, k] = Rijs[:, i] * Rijs[:, j] / dists[index] ** 2
                    k += 1

            Stmp += self.ions.charges[ia] * np.einsum(
                "i, ij->j",
                charges[index]
                * (
                    2 * etaSqrt / piSqrt * np.exp(-self.eta * dists[index] ** 2)
                    + sp.erfc(etaSqrt * dists[index]) / dists[index]
                ),
                Rv,
            )

        Stmp *= -0.5 / self.grid.volume
        k = 0
        for i in range(3):
            for j in range(i, 3):
                S_real[i, j] = S_real[j, i] = Stmp[k]
                k += 1
        return S_real

    def Stress_rec(self):
        reciprocal_grid = self.grid.get_reciprocal()
        gg = reciprocal_grid.gg
        invgg = reciprocal_grid.invgg
        strf = self.ions.strf(reciprocal_grid, 0) * self.ions.charges[0]
        for i in np.arange(1, self.ions.nat):
            strf += self.ions.strf(reciprocal_grid, i) * self.ions.charges[i]
        strf_sq = np.conjugate(strf) * strf
        mask = reciprocal_grid.mask

        Stmp = np.zeros(6)
        size = 6, *reciprocal_grid.nr
        sfactor = np.zeros(size)
        k = 0
        for i in range(3):
            for j in range(i, 3):
                sfactor[k] = reciprocal_grid.g[i] * reciprocal_grid.g[j]
                sfactor[k] *= 2.0 * invgg * (1 + gg / (4.0 * self.eta))
                if i == j:
                    sfactor[k] -= 1.0
                k += 1

        # Stmp =np.einsum('ijkl, ijkl->l', strf_sq*np.exp(-gg/(4.0*self.eta))*invgg, sfactor)
        Stmp = np.einsum(
            "i, ji->j",
            strf_sq[mask] * np.exp(-gg[mask] / (4.0 * self.eta)) * invgg[mask],
            sfactor[:, mask],
        )
        Stmp = Stmp.real * 4.0 * np.pi / self.grid.volume**2
        # G = 0 term
        sum = self.ions.get_ncharges()
        S_g0 = (
            sum**2
            * 4.0
            * np.pi
            * (1.0 / (4.0 * self.eta * self.grid.volume**2) / 2.0)
            / self.mp.comm.size
        )
        k = 0
        S_rec = np.zeros((3, 3))
        for i in range(3):
            for j in range(i, 3):
                if i == j:
                    S_rec[i, i] = Stmp[k] + S_g0
                else:
                    S_rec[i, j] = S_rec[j, i] = Stmp[k]
                k += 1

        return S_rec

    def PME_Qarray_Ewald(self):
        """
        Using the smooth particle mesh Ewald method to calculate structure factors.
        """
        nr = self.grid.nr
        Qarray = np.zeros(nr)
        Bspline = self.Bspline

        ## For speed
        ixyzA = np.mgrid[
            1 : self.order + 1, 1 : self.order + 1, 1 : self.order + 1
        ].reshape((3, -1))
        scaled_positions = self.ions.get_scaled_positions()
        for i in range(self.ions.nat):
            Up = scaled_positions[i] * nr
            Mn = []
            for j in range(3):
                Mn.append(Bspline.calc_Mn(Up[j] - np.floor(Up[j])))
            Mn_multi = np.einsum(
                "i, j, k -> ijk", self.ions.charges[i] * Mn[0][1:], Mn[1][1:], Mn[2][1:]
            )
            l123A = np.mod(
                np.floor(Up).astype(np.int32).reshape((3, 1)) - ixyzA,
                nr.reshape((3, 1)),
            )
            mask = self.Bspline.get_Qarray_mask(l123A)
            Qarray[l123A[0][mask], l123A[1][mask], l123A[2][mask]] += Mn_multi.ravel()[
                mask
            ]
        return DirectField(self.grid, data=Qarray)

    @timer()
    def Energy_rec_PME(self):
        QarrayF = self.Bspline.PME_Qarray
        # bm = self.Bspline.bm
        # method 1
        strf = QarrayF.fft()
        # b123 = np.einsum('i, j, k -> ijk', bm[0], bm[1], bm[2])
        # strf *= b123
        strf *= self.Bspline.Barray
        strf_sq = np.conjugate(strf) * strf
        # method 2
        # Barray = np.einsum('i, j, k -> ijk', \
        # bm[0] * np.conjugate(bm[0]), bm[1] * np.conjugate(bm[1]), bm[2] * np.conjugate(bm[2]))
        # strf_sq =np.conjugate(strf) * Barray * strf

        gg = self.grid.get_reciprocal().gg
        invgg = self.grid.get_reciprocal().invgg
        mask = self.grid.get_reciprocal().mask
        # energy = np.real(4.0*np.pi*np.sum(strf_sq*np.exp(-gg/(4.0*self.eta))*invgg)) / 2.0 / self.grid.volume
        energy = np.sum(
            strf_sq[mask] * np.exp(-gg[mask] / (4.0 * self.eta)) * invgg[mask]
        )
        energy = 4.0 * np.pi * energy.real / self.grid.volume
        energy /= self.grid.dV**2
        return energy

    @timer()
    def Forces_rec_PME(self):
        QarrayF = self.Bspline.PME_Qarray
        strf = QarrayF.fft()
        Bspline = self.Bspline
        Barray = Bspline.Barray
        Barray = Barray * np.conjugate(Barray)
        strf *= Barray
        # bm = Bspline.bm
        # Barray = np.einsum('i, j, k -> ijk', \
        # bm[0] * np.conjugate(bm[0]), bm[1] * np.conjugate(bm[1]), bm[2] * np.conjugate(bm[2]))
        # strf *= Barray
        gg = self.grid.get_reciprocal().gg
        invgg = self.grid.get_reciprocal().invgg

        nrR = self.grid.nrR
        strf *= np.exp(-gg / (4.0 * self.eta)) * invgg
        strf = strf.ifft(force_real=True)

        F_rec = np.zeros((self.ions.nat, 3))
        cell_inv = np.linalg.inv(self.ions.cell)

        ## For speed
        ixyzA = np.mgrid[: self.order, : self.order, : self.order].reshape((3, -1))
        Q_derivativeA = np.zeros((3, self.order * self.order * self.order))
        scaled_positions = self.ions.get_scaled_positions()
        for i in range(self.ions.nat):
            Up = scaled_positions[i] * nrR
            if self.Bspline.check_out_cell(Up):
                continue
            Mn = []
            Mn_2 = []
            for j in range(3):
                Mn.append(Bspline.calc_Mn(Up[j] - np.floor(Up[j])))
                Mn_2.append(
                    Bspline.calc_Mn(Up[j] - np.floor(Up[j]), order=self.order - 1)
                )
            Q_derivativeA[0] = (
                nrR[0]
                * np.einsum(
                    "i, j, k -> ijk", Mn_2[0][1:] - Mn_2[0][:-1], Mn[1][1:], Mn[2][1:]
                ).ravel()
            )
            Q_derivativeA[1] = (
                nrR[1]
                * np.einsum(
                    "i, j, k -> ijk", Mn[0][1:], Mn_2[1][1:] - Mn_2[1][:-1], Mn[2][1:]
                ).ravel()
            )
            Q_derivativeA[2] = (
                nrR[2]
                * np.einsum(
                    "i, j, k -> ijk", Mn[0][1:], Mn[1][1:], Mn_2[2][1:] - Mn_2[2][:-1]
                ).ravel()
            )

            l123A = np.mod(
                1 + np.floor(Up).astype(np.int32).reshape((3, 1)) - ixyzA,
                nrR.reshape((3, 1)),
            )
            mask = self.Bspline.get_Qarray_mask(l123A)
            F_rec[i] -= np.sum(
                np.matmul(Q_derivativeA.T, cell_inv)[mask]
                * strf[l123A[0][mask], l123A[1][mask], l123A[2][mask]][:, np.newaxis],
                axis=0,
            )
            F_rec[i] *= self.ions.charges[i]

        F_rec *= 4.0 * np.pi / self.grid.dV

        return F_rec

    def Stress_rec_PME_full(self):
        QarrayF = self.Bspline.PME_Qarray
        # bm = self.Bspline.bm
        # method 1
        strf = QarrayF.fft()
        # b123 = np.einsum('i, j, k -> ijk', bm[0], bm[1], bm[2])
        # strf *= b123
        strf *= self.Bspline.Barray
        strf_sq = np.conjugate(strf) * strf
        # method 2
        # Barray = np.einsum('i, j, k -> ijk', \
        # bm[0] * np.conjugate(bm[0]), bm[1] * np.conjugate(bm[1]), bm[2] * np.conjugate(bm[2]))
        # strf_sq =np.conjugate(strf) * Barray * strf

        reciprocal_grid = self.grid.get_reciprocal()
        gg = reciprocal_grid.gg
        invgg = reciprocal_grid.invgg

        Stmp = np.zeros(6)
        size = 6, *reciprocal_grid.nr
        sfactor = np.zeros(size)
        k = 0
        for i in range(3):
            for j in range(i, 3):
                sfactor[k] = reciprocal_grid.g[i] * reciprocal_grid.g[j]
                sfactor[k] *= 2.0 / gg * (1 + gg / (4.0 * self.eta))
                if i == j:
                    sfactor[k] -= 1.0
                k += 1

        Stmp = np.einsum(
            "ijk, ijkl->l", strf_sq * np.exp(-gg / (4.0 * self.eta)) * invgg, sfactor
        )

        Stmp = Stmp.real * 2.0 * np.pi / self.grid.volume**2 / self.grid.dV**2
        # G = 0 term
        sum = self.ions.get_ncharges()
        S_g0 = (
            sum**2
            * 4.0
            * np.pi
            * (1.0 / (4.0 * self.eta * self.grid.volume**2) / 2.0)
            / self.mp.comm.size
        )
        k = 0
        S_rec = np.zeros((3, 3))
        for i in range(3):
            for j in range(i, 3):
                if i == j:
                    S_rec[i, i] = Stmp[k] + S_g0
                else:
                    S_rec[i, j] = S_rec[j, i] = Stmp[k]
                k += 1

        return S_rec

    @timer()
    def Stress_rec_PME(self):
        QarrayF = self.Bspline.PME_Qarray
        # bm = self.Bspline.bm
        # method 1
        strf = QarrayF.fft()
        # b123 = np.einsum('i, j, k -> ijk', bm[0], bm[1], bm[2])
        # strf *= b123
        strf *= self.Bspline.Barray
        strf_sq = np.conjugate(strf) * strf
        # method 2
        # Barray = np.einsum('i, j, k -> ijk', \
        # bm[0] * np.conjugate(bm[0]), bm[1] * np.conjugate(bm[1]), bm[2] * np.conjugate(bm[2]))
        # strf_sq =np.conjugate(strf) * Barray * strf

        reciprocal_grid = self.grid.get_reciprocal()
        gg = reciprocal_grid.gg
        invgg = reciprocal_grid.invgg
        mask = reciprocal_grid.mask

        Stmp = np.zeros(6)
        size = 6, *reciprocal_grid.nr
        sfactor = np.zeros(size)
        k = 0
        for i in range(3):
            for j in range(i, 3):
                sfactor[k] = reciprocal_grid.g[i] * reciprocal_grid.g[j]
                sfactor[k] *= 2.0 * invgg * (1 + gg / (4.0 * self.eta))
                if i == j:
                    sfactor[k] -= 1.0
                Stmp[k] = (
                    2.0
                    * np.einsum(
                        "i, i->",
                        strf_sq[mask]
                        * np.exp(-gg[mask] / (4.0 * self.eta))
                        * invgg[mask],
                        sfactor[k][mask],
                    ).real
                )
                k += 1

        # Stmp =np.einsum('ijk, ijkl->l', strf_sq*np.exp(-gg/(4.0*self.eta))*invgg, sfactor)

        Stmp = Stmp.real * 2.0 * np.pi / self.grid.volume**2 / self.grid.dV**2
        # G = 0 term
        sum = self.ions.get_ncharges()
        S_g0 = (
            sum**2
            * 4.0
            * np.pi
            * (1.0 / (4.0 * self.eta * self.grid.volume**2) / 2.0)
            / self.mp.comm.size
        )
        k = 0
        S_rec = np.zeros((3, 3))
        for i in range(3):
            for j in range(i, 3):
                if i == j:
                    S_rec[i, i] = Stmp[k] + S_g0
                else:
                    S_rec[i, j] = S_rec[j, i] = Stmp[k]
                k += 1

        return S_rec
