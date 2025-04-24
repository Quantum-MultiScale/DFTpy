import numpy as np

from dftpy.grid import DirectGrid
from dftpy.math_utils import bestFFTsize


class System(object):
    def __init__(self, ions, grid, density=None, index=None, **kwargs):
        self._ions = ions
        self._grid = grid
        self._density = density
        self._index = index
        self.options = kwargs
        #
        self._info_type = None

    @property
    def info_type(self):
        if self._info_type is None:
            self._info_type = np.dtype(
                [
                    ('index', self.index.dtype, len(self.index)),
                    ('shift', self.grid.shift.dtype, (3)),
                    ('global_nr', self.grid.nrR.dtype, (3)),
                    ('global_cell', self.grid.lattice.dtype, (3, 3)),
                ]
            )
        return self._info_type

    def get_info(self):
        global_grid = self.options.get('global_grid', self.grid)
        return np.array(
            (self.index, self.grid.shift, global_grid.nrR, global_grid.lattice),
            dtype=self.info_type,
        )

    @property
    def density(self):
        return self._density

    @density.setter
    def density(self, value):
        self._density = value

    @property
    def grid(self):
        return self._grid

    @property
    def ions(self):
        return self._ions

    @ions.setter
    def ions(self, value):
        self._ions = value

    @property
    def index(self):
        if self._index is None:
            self._index = slice(None)
        if isinstance(self._index, slice):
            self._index = np.arange(0, len(self.ions))[self._index]
        return self._index

    @property
    def shift(self):
        if hasattr(self.grid, 'shift'):
            return self.grid.shift
        else:
            return np.zeros(3, dtype=np.int64)

    @property
    def comm(self):
        return self.grid.mp.comm

    def get_subsystem(self, index=None, grid_sub=None, wrap=True, **kwargs):
        grid_sub = self.gen_grid_sub(
            self.ions, self.grid, index=index, grid_sub=grid_sub, **kwargs
        )

        if index is None:
            index = slice(None)
        origin = grid_sub.shift / self.grid.nrR
        pos_cry = self.ions.get_scaled_positions()[index] - origin
        if wrap:
            pos_cry %= 1.0
        ions_sub = self.ions[index]
        ions_sub.set_scaled_positions(pos_cry)
        ions_sub.set_cell(grid_sub.lattice)
        return self.__class__(
            ions=ions_sub,
            grid=grid_sub,
            index=index,
            global_grid=self.grid,
            global_ions=self.ions,
        )

    @staticmethod
    def gen_grid_sub(
        ions,
        grid,
        index=None,
        cellcut=[0.0, 0.0, 0.0],
        cellsplit=None,
        grid_sub=None,
        nr=None,
        mp=None,
        **kwargs,
    ):
        tol = 1e-8
        cell = ions.cell
        lattice_sub = cell.copy()
        latp = cell.cellpar()[:3]
        if index is None:
            index = slice(None)
        if isinstance(cellcut, (int, float)) or len(cellcut) == 1:
            cellcut = np.ones(3) * cellcut
        if cellsplit is not None:
            if isinstance(cellsplit, (int, float)) or len(cellsplit) == 1:
                cellsplit = np.ones(3) * cellsplit
        spacings = grid.spacings.copy()
        shift = np.zeros(3, dtype='int')
        origin = np.zeros(3)

        if grid_sub is not None:
            nr = grid_sub.nrR

        pos_cry = ions.get_scaled_positions()[index]
        cs = np.min(pos_cry, axis=0)
        pos_cry -= cs
        pos_cry -= np.rint(pos_cry)
        pos = ions.cell.cartesian_positions(pos_cry)
        # -----------------------------------------------------------------------
        cell_size = np.ptp(pos, axis=0)
        if nr is None:
            nr = grid.nr.copy()
            for i in range(3):
                if cellsplit is not None:
                    cell_size[i] = cellsplit[i] * latp[i]
                elif cellcut[i] > tol:
                    cell_size[i] += cellcut[i] * 2.0
                else:
                    cell_size[i] = latp[i]
                nr[i] = int(cell_size[i] / spacings[i])
                nr[i] = bestFFTsize(nr[i], **kwargs)
        for i in range(3):
            if nr[i] < grid.nrR[i]:
                lattice_sub[i] *= (nr[i] * spacings[i]) / latp[i]
                origin[i] = 0.5
            else:
                nr[i] = grid.nrR[i]
                origin[i] = 0.0

        c1 = lattice_sub.cartesian_positions(origin)

        c0 = np.mean(pos, axis=0)
        center = cell.scaled_positions(c0) + cs
        center[origin < tol] = 0.0
        c0 = cell.cartesian_positions(center)

        origin = np.array(c0) - np.array(c1)
        shift[:] = np.array(cell.scaled_positions(origin)) * grid.nrR

        if grid_sub is None:
            grid_sub = DirectGrid(
                lattice=lattice_sub, nr=nr, origin=origin, mp=mp, **kwargs
            )
        grid_sub.shift = shift
        return grid_sub

    def free(self):
        self.grid.free()
