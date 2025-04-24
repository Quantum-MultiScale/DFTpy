import numpy as np
import scipy.special as sp
from ase.cell import Cell
from scipy.interpolate import splev, splrep

from dftpy.math_utils import ecut2nr, spacing2ecut
from dftpy.mpi import MP


class BaseGrid:
    """
    Object representing a grid (Cell (lattice) plus discretization)
    extends Cell

    Attributes
    ----------
    nr : array of numbers used for discretization

    nnr : total number of grid points

    dV : volume of a grid point

    Node:
    Virtual class, DirectGrid and ReciprocalGrid should be used in actual applications

    """

    def __init__(
        self,
        lattice,
        nr=None,
        origin=np.array([0.0, 0.0, 0.0]),
        full=True,
        direct=True,
        cplx=False,
        mp=None,
        ecut=None,
        comm=None,
        **kwargs,
    ):
        if mp is None:
            mp = MP(comm)
        self._origin = np.asarray(origin)
        if not isinstance(lattice, Cell):
            cell = Cell(lattice)
        else:
            cell = lattice
        #
        self.cplx = cplx
        self._cell = cell
        self._direct = direct
        #
        if nr is None:
            nr = ecut2nr(ecut=ecut, lattice=lattice, **kwargs)
        #
        self._nrR = np.array(nr, dtype=np.int32)
        self._nnrR = np.prod(self._nrR)
        self._dV = np.abs(self.cell.volume) / self._nnrR
        self._nrG = self._nrR.copy()
        if not full:
            self._nrG[-1] = self._nrG[-1] // 2 + 1
        self._nnrG = np.prod(self._nrG)
        self._spacings = self.cell.lengths() / self._nrR
        self._mp = mp
        if self.cplx:
            full = True
        self.local_slice(nr, direct=direct, full=full, cplx=cplx, **kwargs)
        self._nnr = np.prod(self._nr)
        # print('nr_local', self.mp.comm.rank, self._nr, direct, self.mp.comm.size, flush = True)
        self._full = full
        self._ecut = ecut

    def __eq__(self, other: 'BaseGrid') -> bool:
        if np.allclose(self.lattice, other.lattice) and np.allclose(
            self.nrR, other.nrR
        ):
            return True
        else:
            return False

    @property
    def mp(self):
        return self._mp

    @mp.setter
    def mp(self, value):
        self._mp = value

    @property
    def nr(self):
        return self._nr

    @property
    def nnr(self):
        return self._nnr

    @property
    def nrR(self):
        return self._nrR

    @property
    def nnrR(self):
        return self._nnrR

    @property
    def nrG(self):
        return self._nrG

    @property
    def nnrG(self):
        return self._nnrG

    @property
    def dV(self):
        return self._dV

    @property
    def volume(self):
        return self.cell.volume

    @property
    def spacings(self):
        return self._spacings

    @property
    def cell(self):
        return self._cell

    @property
    def full(self):
        return self._full

    @property
    def direct(self):
        return self._direct

    @property
    def origin(self):
        return self._origin

    def tile(self, reps=1):
        # it only repeat last three dimensions with same rep
        if self.mp.size > 1:
            raise ValueError("Only works for serial version.")
        try:
            tup = tuple(reps)
        except TypeError:
            tup = (reps,)
        reps = np.ones(3, dtype='int')
        for i, x in enumerate(tup):
            reps[i] = x
        lattice = self.lattice.copy()
        for i in range(3):
            lattice[i] *= reps[i]
        nr = self.nr * reps
        results = self.__class__(
            lattice,
            nr,
            origin=self.origin,
            full=self.full,
            cplx=self.cplx,
            direct=self.direct,
        )
        return results

    def create(self, lattice=None, **kwargs):
        options = {
            'origin': self.origin,
            'full': self.full,
            'cplx': self.cplx,
            'mp': self.mp,
            'ecut': self.ecut,
        }
        if lattice is None:
            lattice = self.cell
        options.update(kwargs)
        results = self.__class__(lattice, **options)
        return results

    def repeat(self, rep=1):
        # it only repeat last three dimensions with same rep
        if not isinstance(rep, int):
            raise AttributeError(
                "Grid repeat only support one integer, Please use 'tile'."
            )
        if self.rank == 1:
            reps = np.ones(3, dtype='int') * rep
        return self.tile(reps)

    def local_slice(self, nr, **kwargs):
        self._slice, self._nr, self._offsets = self.mp.get_local_fft_shape(nr, **kwargs)
        if self.mp.is_mpi:
            self.slice_all = self.mp.comm.allgather(self._slice)
            self.nr_all = self.mp.comm.allgather(self._nr)
            self.offsets_all = self.mp.comm.allgather(self._offsets)
        else:
            self.slice_all = self._slice
            self.nr_all = self._nr
            self.offsets_all = self._offsets

    @property
    def slice(self):
        return self._slice

    @property
    def offsets(self):
        return self._offsets

    def gather(self, data, nr=None, out=None, root=0, **kwargs):
        if self.mp.is_mpi:
            reqs = []
            bufs = []
            rank = 1 if getattr(data, 'ndim', 1) < 4 else data.shape[0]
            if self.mp.rank == root:
                if out is None:
                    if nr is None:
                        nr = self.nrR
                    if rank > 1:
                        nr = (rank, *nr)
                    out = np.empty(nr, dtype=data.dtype)
                for i in range(0, self.mp.comm.size):
                    if i == root:
                        buf = data
                    else:
                        shape = self.nr_all[i]
                        if rank > 1:
                            shape = (rank, *shape)
                        buf = np.empty(shape, dtype=data.dtype)
                        req = self.mp.comm.Irecv(buf, source=i, tag=i)
                        reqs.append(req)
                    bufs.append(buf)
            else:
                req = self.mp.comm.Isend(data, dest=root, tag=self.mp.rank)
                reqs.append(req)
                out = np.ones(rank)
            self.mp.MPI.Request.Waitall(reqs)
            if self.mp.rank == root:
                for i in range(0, self.mp.comm.size):
                    inds = self.slice_all[i]
                    if rank > 1:
                        inds = (slice(None), *inds)
                    out[inds] = bufs[i]
            self.mp.comm.Barrier()
        else:
            if out is None:
                out = data.copy()
            else:
                out[:] = data
        return out

    def scatter(self, data, out=None, root=0, **kwargs):
        if self.mp.is_mpi:
            reqs = []
            rank = 1 if getattr(data, 'ndim', 1) < 4 else data.shape[0]
            rank = self.mp.amax(rank)
            if out is None:
                nr = self.nr
                if rank > 1:
                    nr = (rank, *nr)
                out = np.empty(nr, dtype=data.dtype)
            if self.mp.rank == root:
                for i in range(0, self.mp.comm.size):
                    if i == root:
                        inds = self.slice_all[i]
                        if rank > 1:
                            inds = (slice(None), *inds)
                        out[:] = data[inds]
                    else:
                        shape = self.nr_all[i]
                        inds = self.slice_all[i]
                        if rank > 1:
                            shape = (rank, *shape)
                            inds = (slice(None), *inds)
                        buf = np.empty(shape, dtype=data.dtype)
                        buf[:] = data[inds]
                        req = self.mp.comm.Isend(buf, dest=i, tag=i)
                        reqs.append(req)
            else:
                req = self.mp.comm.Irecv(out, source=root, tag=self.mp.rank)
                reqs.append(req)
            self.mp.MPI.Request.Waitall(reqs)
            self.mp.comm.Barrier()
        else:
            if out is None:
                out = data.copy()
            else:
                out[:] = data
        return out

    def free(self):
        self.mp.free()

    @property
    def lattice(self):
        return self.cell.array

    @property
    def ecut(self):
        if self._ecut is None:
            if hasattr(self, 'guess_ecut'):
                ecut = self.guess_ecut()
            elif hasattr(self, 'get_direct'):
                ecut = self.get_direct().guess_ecut()
            else:
                ecut = None
        else:
            ecut = self._ecut
        return ecut

    @ecut.setter
    def ecut(self, value):
        self._ecut = value
        if hasattr(self, 'Dgrid'):
            self._qmask = None
            if hasattr(self.Dgrid, '_ecut'):
                self.Dgrid._ecut = value
        elif hasattr(self, 'RPgrid'):
            if hasattr(self.RPgrid, '_ecut'):
                self.RPgrid._ecut = value
                self.RPgrid._qmask = None


class DirectGrid(BaseGrid):
    """
    Attributes:
    ----------
    All of BaseGrid and DirectCell

    r : cartesian coordinates of each grid point

    s : crystal coordinates of each grid point
    """

    def __init__(
        self,
        lattice,
        nr=None,
        origin=np.array([0.0, 0.0, 0.0]),
        full=True,
        uppergrid=None,
        **kwargs,
    ):
        """
        Parameters
        ----------
        lattice : array_like[3,3]
            matrix containing the direct lattice vectors (as its colums)
        """
        self.init_options = locals()
        for k in ['__class__', 'self', 'kwargs', 'uppergrid']:
            self.init_options.pop(k, None)
        self.init_options.update(kwargs)
        super().__init__(
            lattice=lattice, nr=nr, origin=origin, full=full, direct=True, **kwargs
        )
        self._r = None
        self._rr = None
        self._s = None
        self.RPgrid = uppergrid
        self._Rtable = None

    def __eq__(self, other):
        """
        Implement the == operator in the DirectGrid class.
        Refer to the __eq__ method of Grid for more information.
        """
        if not isinstance(other, DirectGrid):
            raise TypeError("You can only compare a DirectGrid with another DirectGrid")
        return BaseGrid.__eq__(self, other)

    def _calc_grid_crys_points(self):
        if self._s is None:
            # s0 = np.linspace(0, 1, self.nr[0], endpoint=False)
            # s1 = np.linspace(0, 1, self.nr[1], endpoint=False)
            # s2 = np.linspace(0, 1, self.nr[2], endpoint=False)
            # S0, S1, S2 = np.meshgrid(s0, s1, s2, indexing="ij")
            # self._s = np.asarray([S0, S1, S2])
            ax = []
            for i in range(3):
                s0 = np.linspace(0, 1, self.nrR[i], endpoint=False)
                ax.append(s0)
            AX = [a[sl] for a, sl in zip(ax, self.slice)]
            S = np.meshgrid(*AX, indexing="ij")
            self._s = np.asarray(S)

    def _calc_grid_cart_points(self):
        if self._r is None:
            self._r = np.einsum("j...,jk->k...", self.s, self.lattice)

    @property
    def r(self):
        if self._r is None:
            self._calc_grid_cart_points()
        return self._r

    @property
    def rr(self):
        if self._rr is None:
            rr = np.einsum("lijk,lijk->ijk", self.r, self.r)
            # self._rr = np.reshape(rr, [self.nr[0], self.nr[1], self.nr[2], 1])
            self._rr = rr
        return self._rr

    @property
    def s(self):
        if self._s is None:
            self._calc_grid_crys_points()
        return self._s

    @property
    def full(self):
        return self._full

    @full.setter
    def full(self, value):
        if self._full != value:
            '''
            Clean stored information of reciprocal grid.
            '''
            self._full = value
            self.RPgrid = None
            self._nrG = self.nr.copy()
            if not self._full:
                self._nrG[-1] = self._nrG[-1] // 2 + 1

    def get_reciprocal(
        self, scale=None, convention: str = "physics"
    ) -> 'ReciprocalGrid':
        r"""
        Returns a new ReciprocalCell, the reciprocal cell of self
        The ReciprocalCell is scaled properly to include
        the scaled (*self.nr) reciprocal grid points
        -----------------------------
        Note1: We need to use the 'physics' convention where bg^T = 2 \pi * at^{-1}
        physics convention defines the reciprocal lattice to be
        exp^{i G \cdot R} = 1
        Now we have the following "crystallographer's" definition ('crystallograph')
        which comes from defining the reciprocal lattice to be
        e^{2\pi i G \cdot R} =1
        In this case bg^T = at^{-1}
        -----------------------------
        Note2: We have to use 'Bohr' units to avoid changing hbar value
        """
        # TODO define in constants module hbar value for all units allowed
        if self.RPgrid is None or scale is not None:
            if scale is None:
                scale = [1.0, 1.0, 1.0]
            scale = np.array(scale)
            fac = 1.0
            if convention == "physics" or convention == "p":
                fac = 2 * np.pi
            fac = 2 * np.pi
            bg = fac * np.linalg.inv(self.lattice)
            bg = bg.T
            reciprocal_lat = np.einsum("ij,i->ij", bg, scale)

            self.RPgrid = ReciprocalGrid(
                lattice=reciprocal_lat,
                nr=self.nrR,
                full=self.full,
                uppergrid=self,
                cplx=self.cplx,
                mp=self.mp,
                ecut=self.ecut,
            )
        return self.RPgrid

    def get_Rtable(self, rcut=10):
        '''Only support for serial'''
        if self._Rtable is None:
            self._Rtable = {}
            metric = np.dot(self.lattice, self.lattice.T)
            latticeConstants = np.sqrt(np.diag(metric))
            gaps = latticeConstants / self.nr
            Nmax = np.ceil(rcut / gaps).astype(np.int32) + 1
            # print('lc', latticeConstants)
            # print('gaps', gaps)
            # print(Nmax)
            # mgrid = np.mgrid[0:Nmax[0], 0:Nmax[0], 0:Nmax[0]].reshape((3, -1))
            # array = np.einsum('jk,ij->ik',gridpos,self.lattice)
            # dists = np.einsum('ij,ij->j', array, array)
            # index = np.arange(0, Nmax[0] * Nmax[1] * Nmax[2]).reshape(Nmax)
            # mgrid = np.mgrid[0:Nmax[0], 0:Nmax[1], 0:Nmax[2]].astype(np.float64)
            mgrid = np.mgrid[
                1 - Nmax[0] : Nmax[0], 1 - Nmax[1] : Nmax[1], 1 - Nmax[2] : Nmax[2]
            ].astype(np.float64)
            mgrid[0] /= self.nr[0]
            mgrid[1] /= self.nr[1]
            mgrid[2] /= self.nr[2]
            gridpos = mgrid.astype(np.float64)
            array = np.einsum("jklm,ji->iklm", gridpos, self.lattice)
            dists = np.sqrt(np.einsum("ijkl,ijkl->jkl", array, array))
            self._Rtable["Nmax"] = Nmax
            self._Rtable["table"] = dists
        return self._Rtable

    def gather(self, data, out=None, **kwargs):
        value = super().gather(data, self.nrR, out=out, **kwargs)
        return value

    def get_array_mask(self, xyz):
        if self.mp.comm.size == 1:
            return slice(None)
        offsets = self.offsets.reshape((3, 1))
        nr = self.nr
        # -----------------------------------------------------------------------
        xyz -= offsets
        mask = np.logical_and(xyz[0] > -1, xyz[0] < nr[0])
        mask1 = np.logical_and(xyz[1] > -1, xyz[1] < nr[1])
        np.logical_and(mask, mask1, out=mask)
        np.logical_and(xyz[2] > -1, xyz[2] < nr[2], out=mask1)
        np.logical_and(mask, mask1, out=mask)
        # -----------------------------------------------------------------------
        return mask

    def guess_ecut(self):
        spacings2 = self.cell.lengths() / (self._nrR - 1)
        spacings = 0.5 * (self.spacings + spacings2)
        return spacing2ecut(spacings.max())


class ReciprocalGrid(BaseGrid):
    """
    Attributes:
    ----------
    All of BaseGrid and DirectCell

    g : coordinates of each point in the reciprocal cell

    gg : square of each g vector
    """

    def __init__(
        self,
        lattice,
        nr=None,
        origin=np.array([0.0, 0.0, 0.0]),
        full=True,
        uppergrid=None,
        **kwargs,
    ):
        """
        Parameters
        ----------
        lattice : array_like[3,3]
            matrix containing the direct lattice vectors (as its colums)
        """
        self.init_options = locals()
        for k in ['__class__', 'self', 'kwargs', 'uppergrid']:
            self.init_options.pop(k, None)
        self.init_options.update(kwargs)
        super().__init__(
            lattice=lattice, nr=nr, origin=origin, full=full, direct=False, **kwargs
        )
        self._g = None
        self._gg = None
        self.Dgrid = uppergrid
        self._q = None
        self._mask = None
        self._gF = None
        self._ggF = None
        self._invgg = None
        self._invq = None
        self._gmask = None

    def __eq__(self, other):
        """
        Implement the == operator in the ReciprocalGrid class.
        Refer to the __eq__ method of Grid for more information.
        """
        if not isinstance(other, ReciprocalGrid):
            raise TypeError(
                "You can only compare a ReciprocalGrid with another ReciprocalGrid"
            )
        return BaseGrid.__eq__(self, other)

    @property
    def g(self):
        if self._g is None:
            self._g = self._calc_grid_points()
        return self._g

    @property
    def q(self):
        if self._q is None:
            self._q = np.sqrt(self.gg)
        return self._q

    @property
    def gg(self):
        if self._gg is None:
            if self._g is None:
                self._g = self._calc_grid_points()
            gg = np.einsum("lijk,lijk->ijk", self._g, self._g)
            self._gg = gg
        return self._gg

    @property
    def invgg(self):
        if self._invgg is None:
            if self.mp.is_root:
                self.gg[0, 0, 0] = 1.0
            invgg = 1.0 / self.gg
            if self.mp.is_root:
                self.gg[0, 0, 0] = 0.0
                invgg[0, 0, 0] = 0.0
            self._invgg = invgg
        return self._invgg

    @property
    def invq(self):
        if self._invq is None:
            if self.mp.is_root:
                self.q[0, 0, 0] = 1.0
            invq = 1.0 / self.q
            if self.mp.is_root:
                self.q[0, 0, 0] = 0.0
                invq[0, 0, 0] = 0.0
            self._invq = invq
        return self._invq

    def get_direct(self, scale=None, convention="physics"):
        r"""
        Returns a new DirectCell, the direct cell of self
        The DirectCell is scaled properly to include
        the scaled (*self.nr) reciprocal grid points
        -----------------------------
        Note1: We need to use the 'physics' convention where bg^T = 2 \pi * at^{-1}
        physics convention defines the reciprocal lattice to be
        exp^{i G \cdot R} = 1
        Now we have the following "crystallographer's" definition ('crystallograph')
        which comes from defining the reciprocal lattice to be
        e^{2\pi i G \cdot R} =1
        In this case bg^T = at^{-1}
        -----------------------------
        """
        # TODO define in constants module hbar value for all units allowed
        if self.Dgrid is None or scale is not None:
            if scale is None:
                scale = [1.0, 1.0, 1.0]
            scale = np.array(scale)
            fac = 1.0
            if convention == "physics" or convention == "p":
                fac = 1.0 / (2 * np.pi)
            at = np.linalg.inv(self.lattice.T * fac)
            direct_lat = np.einsum("ij,i->ij", at, 1.0 / scale)
            self.Dgrid = DirectGrid(
                lattice=direct_lat,
                nr=self.nrR,
                full=self.full,
                uppergrid=self,
                cplx=self.cplx,
                mp=self.mp,
                ecut=self._ecut,
            )
        return self.Dgrid

    def _calc_grid_points(self, full=None):
        ax = []
        for i in range(3):
            # use fftfreq function so we don't have to
            # worry about odd or even number of points
            # dd: this choice of "spacing" is due to the
            # definition of real and reciprocal space for
            # a grid (which is not exactly a conventional
            # lattice), specifically:
            #    1) the real-space points go from 0 to 1 in
            #       crystal coords in n steps of length 1/n
            #    2) thus the reciprocal space (g-space)
            #       crystal coords go from 0 to n in n steps
            #    3) the "physicists" 2*np.pi factor is
            #       included in the definition of reciprocal
            #       lattice vectors in the "grid" class and
            #       is applied with s2r in going from crystal
            #       to Cartesian g-space
            dd = 1 / self.nrR[i]
            if full is None:
                full = self.full
            if i == 2 and not full:
                ax.append(np.fft.rfftfreq(self.nrR[i], d=dd))
            else:
                freq = np.fft.fftfreq(self.nrR[i], d=dd)
                # if freq.size % 2 == 0 :
                # freq[freq.size//2] *= -1
                # ax.append(freq)
                # else :
                # ax.append(freq)
                ax.append(freq)
        AX = [a[sl] for a, sl in zip(ax, self.slice)]
        S = np.meshgrid(*AX, indexing="ij")
        S_cart = np.asarray(S)
        S_cart = np.einsum("j...,jk->k...", S_cart, self.lattice)

        return S_cart

    @property
    def mask_serial(self):
        if self._mask is None:
            nrR = self.nrR[:3]
            # Dnr = nr[:3]//2
            # Dmod = nr[:3]%2
            # mask = np.ones((nr[0], nr[1], Dnr[2]+1), dtype = bool)
            Dnr = nrR[:3] // 2
            Dmod = nrR[:3] % 2
            mask = np.ones(self.nr[:3], dtype=bool)
            if np.all(self.nr == self.nrR):
                mask[:, :, Dnr[2] + 1 :] = False

            mask[0, Dnr[1] + 1 :, 0] = False
            mask[Dnr[0] + 1 :, :, 0] = False
            if Dmod[2] == 0:
                mask[0, 0, Dnr[2]] = False
                mask[0, Dnr[1] + 1 :, Dnr[2]] = False
                mask[Dnr[0] + 1 :, :, Dnr[2]] = False
                if Dmod[1] == 0:
                    mask[0, Dnr[1], Dnr[2]] = False
                if Dmod[0] == 0:
                    mask[Dnr[0], 0, Dnr[2]] = False
                    mask[Dnr[0], Dnr[1] + 1 :, Dnr[2]] = False
            if Dmod[0] == 0:
                mask[Dnr[0], Dnr[1] + 1 :, 0] = False
                if Dmod[1] == 0:
                    mask[Dnr[0], Dnr[1], 0] = False
            if Dmod[1] == 0:
                mask[0, Dnr[1], 0] = False
            if all(Dmod == 0):
                mask[Dnr[0], Dnr[1], Dnr[2]] = False
            self._mask = mask
        return self._mask

    @property
    def mask(self):
        if self._mask is None:
            nrR = self.nrR[:3]
            Dnr = nrR[:3] // 2 - self.offsets
            mask = np.ones(self.nr[:3], dtype=bool)
            if np.all(self.nrG == self.nrR):
                if Dnr[2] >= 0:
                    mask[:, :, Dnr[2] + 1 :] = False
                else:
                    mask[:, :, :] = False
            Dnr = np.where(Dnr > 0, Dnr, 0)
            Dmod = nrR[:3] % 2
            if self.offsets[0] == self.offsets[2] == 0:
                mask[0, Dnr[1] + 1 :, 0] = False
            if self.offsets[2] == 0:
                mask[Dnr[0] + 1 :, :, 0] = False
            if Dmod[2] == 0:
                if self.offsets[0] == 0:
                    if self.offsets[1] == 0:
                        mask[0, 0, Dnr[2] : Dnr[2] + 1] = False
                    mask[0, Dnr[1] + 1 :, Dnr[2] : Dnr[2] + 1] = False
                mask[Dnr[0] + 1 :, :, Dnr[2] : Dnr[2] + 1] = False
                if Dmod[1] == 0 and self.offsets[0] == 0:
                    mask[0, Dnr[1] : Dnr[1] + 1, Dnr[2] : Dnr[2] + 1] = False
                if Dmod[0] == 0:
                    if self.offsets[1] == 0:
                        mask[Dnr[0] : Dnr[0] + 1, 0, Dnr[2] : Dnr[2] + 1] = False
                    mask[Dnr[0] : Dnr[0] + 1, Dnr[1] + 1 :, Dnr[2] : Dnr[2] + 1] = False
            if Dmod[0] == 0 and self.offsets[2] == 0:
                mask[Dnr[0] : Dnr[0] + 1, Dnr[1] + 1 :, 0] = False
                if Dmod[1] == 0:
                    mask[Dnr[0] : Dnr[0] + 1, Dnr[1] : Dnr[1] + 1, 0] = False
            if Dmod[1] == 0 and self.offsets[2] == 0:
                mask[0, Dnr[1] : Dnr[1] + 1, 0] = False
            if all(Dmod == 0):
                mask[Dnr[0] : Dnr[0] + 1, Dnr[1] : Dnr[1] + 1, Dnr[2] : Dnr[2] + 1] = (
                    False
                )
            self._mask = mask
        return self._mask

    @property
    def gF(self):
        if self._gF is None:
            self._gF = self._calc_grid_points(full=True)
        return self._gF

    @property
    def ggF(self):
        if self._ggF is None:
            if self._gF is None:
                self._gF = self._calc_grid_points(full=True)
            ggF = np.einsum("lijk,lijk->ijk", self._gF, self._gF)
            self._ggF = ggF
        return self._ggF

    @property
    def full(self):
        return self._full

    @full.setter
    def full(self, value):
        if self._full != value:
            self._full = value

    @property
    def g2max(self):
        return 2.0 * self.ecut

    def get_gmask(self, g2max=None):
        if g2max is None:
            return self.gmask
        gmask = self.gg <= g2max
        return gmask

    @property
    def gmask(self):
        if self._gmask is None:
            self._gmask = self.get_gmask(self.g2max)
        return self._gmask


class RadialGrid(object):
    def __init__(self, r=None, v=None, direct=True, vr=None, **kwargs):
        self._r = r
        self._v = v
        self._vr = vr
        self._v_interp = None
        self.direct = direct

    @property
    def r(self):
        return self._r

    @r.setter
    def r(self, r):
        self._r = r

    @property
    def v(self):
        return self._v

    @v.setter
    def v(self, v):
        self._v = v

    @property
    def vr(self):
        return self._vr

    @property
    def v_interp(self):
        if self._v_interp is None:
            self._v_interp = splrep(self.r, self.v)
        return self._v_interp

    def to_3d_grid(self, dist, direct=None, out=None):
        if out is None:
            results = np.zeros_like(dist)
        else:
            results = out
        mask = dist < self._r[-1]
        if np.count_nonzero(mask) > 0:
            results[mask] = splev(dist[mask], self.v_interp, der=0, ext=1)
        return results

    def _ft(self, x, method='simpson', comm=None, mp=None, **kwargs):
        v = self.v
        r = self.r
        if mp is None:
            mp = MP(comm=comm)
        vp = np.zeros_like(x)

        if method == 'simpson':
            from scipy.integrate import simpson as integrate
        elif method == 'trapezoid':
            from scipy.integrate import trapezoid as integrate

        if self.vr:
            vr = v * r
        else:
            vr = v * r * r

        lb, ub = mp.split_number(len(x))

        for k in range(lb, ub):
            y = sp.spherical_jn(0, x[k] * r) * vr
            vp[k] = integrate(y, x=r)

        vp = mp.vsum(vp)

        return vp

    def ft(self, x, method='simpson', mp=None, **kwargs):
        y = self._ft(x, method=method, mp=mp, **kwargs)
        if self.direct:
            y *= 4.0 * np.pi
        else:
            y *= 0.5 / np.pi**2
        return y
