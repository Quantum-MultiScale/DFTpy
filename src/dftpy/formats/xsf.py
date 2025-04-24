import numpy as np

from dftpy.constants import Units
from dftpy.field import DirectField
from dftpy.grid import DirectGrid
from dftpy.ions import Ions


def xsf_readline(fr):
    for line in fr:
        line = line.strip()
        if len(line) == 0 or line.startswith("#"):
            continue
        else:
            yield line


def read_xsf(
    infile,
    kind="all",
    full=False,
    pbc=True,
    units='angstrom',
    data_type='density',
    **kwargs,
):
    # http ://www.xcrysden.org/doc/XSF.html
    with open(infile, "r") as fr:
        fh = xsf_readline(fr)
        celltype = next(fh).upper()
        if celltype != 'CRYSTAL':
            raise AttributeError("Only support crystal xsf.")
        lattice = []
        line = next(fh).upper()
        if line.startswith("PRIMVEC"):
            for i in range(3):
                l = list(map(float, fr.readline().split()))
                lattice.append(l)
            lattice = np.asarray(lattice)
            line = next(fh).upper()

        label = []
        pos = []
        if line.startswith("CONVVEC"):
            for i in range(3):
                line = next(fh)
            line = next(fh).upper()

        if line.startswith("PRIMCOORD"):
            natom = int(fr.readline().split()[0])
            for i in range(natom):
                line = next(fh).split()
                label.append(line[0])
                p = list(map(float, line[1:4]))
                pos.append(p)
            pos = np.asarray(pos)
            line = next(fh).upper()

        if units.lower() == 'angstrom':
            ions_units = 'ase'
        else:
            ions_units = 'au'
        ions = Ions(symbols=label, positions=pos, cell=lattice, units=ions_units)

        if kind == "ions":
            return ions
        else:
            if line.startswith("BEGIN_BLOCK_DATAGRID_"):
                line = next(fh)
            blocks = []
            for line in fh:
                data = []
                line = line.upper()
                if line.startswith("BEGIN_DATAGRID"):
                    nrx = np.ones(3, dtype=int)
                    npbc = 3
                    if line.startswith("BEGIN_DATAGRID_3D"):
                        nrx[0], nrx[1], nrx[2] = map(int, next(fh).split())
                        npbc = 3
                    elif line.startswith("BEGIN_DATAGRID_2D"):
                        nrx[0], nrx[1] = map(int, next(fh).split())
                        npbc = 2
                    elif line.startswith("BEGIN_DATAGRID_1D"):
                        nrx[0] = map(int, next(fh).split())
                        npbc = 1
                    next(fh)  # read the origin
                    vlat = np.zeros((3, 3))
                    for i in range(npbc):
                        l = list(map(float, next(fh).split()))
                        vlat[i] = np.asarray(l)
                    if npbc == 1:
                        for i in range(3):
                            if abs(vlat[0][i]) > 1e-4:
                                j = i - 1
                                vlat[1][j] = vlat[0][i]
                                vlat[1][i] = -vlat[0][j]
                                vlat[1] = vlat[1] / np.sqrt(np.dot(vlat[1], vlat[1]))
                                break
                        vlat[2] = np.cross(vlat[0], vlat[1])
                        vlat[2] = vlat[2] / np.sqrt(np.dot(vlat[2], vlat[2]))
                    elif npbc == 2:
                        vlat[2] = np.cross(vlat[0], vlat[1])
                        vlat[2] = vlat[2] / np.sqrt(np.dot(vlat[2], vlat[2]))
                    if units.lower() == 'angstrom':
                        data_lat = vlat / Units.Bohr
                    else:
                        data_lat = vlat
                    for line in fh:
                        if line[0] == "E":
                            break
                        else:
                            line = line.split()
                            l = list(map(float, line))
                            data.extend(l)
                    data = np.asarray(data)
                    if np.size(data) > np.prod(nrx):  # double xsf grid data
                        data = data[: np.prod(nrx)]
                    data = np.reshape(data, nrx, order="F")
                    blocks.append(data)
                if line.strip().startswith('END_BLOCK_DATAGRID'):
                    break

            if not blocks:
                raise AttributeError("!!!ERROR : XSF file have some problem")

            nrx_prev = None
            rank = len(blocks)
            for spin in range(rank):
                data = blocks[spin]
                nrx = np.array(data.shape)
                if nrx_prev is not None:
                    if not np.all(nrx_prev == nrx):
                        raise AttributeError("All DATAGRID should have same shape.")
                else:
                    nrx_prev = nrx.copy()
                if pbc:
                    for i in range(len(nrx)):
                        if nrx[i] > 1:
                            nrx[i] -= 1
                    data = data[: nrx[0], : nrx[1], : nrx[2]]
                #
                if units.lower() == 'angstrom':
                    data *= Units.Bohr**3
                else:
                    data_lat = vlat
                blocks[spin] = data

            grid = DirectGrid(lattice=data_lat, nr=nrx, full=full)
            field = DirectField(grid=grid, griddata_3d=blocks, rank=rank)
            if data_type == 'potential':
                field /= Units.Ha
    if kind == 'data':
        return field
    else:
        return ions, field, None


def write_xsf(filexsf, ions=None, data=None, **kwargs):
    return XSF(filexsf).write(ions, data, **kwargs)


class XSF(object):

    def __init__(self, filexsf):
        self.filexsf = filexsf
        self.cutoffvars = {}

    def write(
        self,
        ions=None,
        data=None,
        data_type='density',
        units="angstrom",
        title='DFTpy',
        **kwargs,
    ):
        """
        Write a ions and data into an xsf file.
        Not all specifications of the xsf file format are implemented, they will
        be added as needed.
        So far it can:
            - write the ions
            - write the 1D/2D/3D grid data
        """

        with open(self.filexsf, "w") as fileout:
            if units.lower() == 'angstrom':
                ions = ions.to_ase()
            self._write_header(fileout, title)
            self._write_cell(fileout, ions.cell)
            self._write_coord(fileout, ions)
            self._write_datagrid(fileout, data, data_type=data_type, units=units)

        return

    def read(self, kind="all", full=False, **kwargs):
        return read_xsf(self.filexsf, kind=kind, full=full, **kwargs)

    def _write_header(self, fileout, title):
        mywrite(fileout, ("# ", title))
        mywrite(fileout, "CRYSTAL \n", True)

    def _write_cell(self, fileout, cell):
        mywrite(fileout, "PRIMVEC", True)
        for ilat in range(3):
            latt = cell[ilat]
            mywrite(fileout, latt, True)

    def _write_coord(self, fileout, ions):
        mywrite(fileout, "PRIMCOORD", True)
        mywrite(fileout, (len(ions.positions), 1), True)
        for i in range(len(ions.positions)):
            mywrite(fileout, (ions.symbols[i], ions.positions[i]), True)

    def _write_datagrid(
        self, fileout, plot, data_type='density', units='angstrom', **kwargs
    ):
        ndim = plot.span  # 2D or 3D grid?
        if ndim < 2:
            return  # XSF format doesn't support one data grids
        val_per_line = 5
        rank = plot.rank
        grid = plot.grid
        if rank == 1:
            plot = [plot]
        data = []
        for p in plot:
            values = p.get_values_flatarray(pad=1, order="F")
            if units == 'angstrom':
                values /= Units.Bohr**3
            if data_type == 'potential':
                values = values * Units.Ha
            data.append(values)

        mywrite(fileout, "BEGIN_BLOCK_DATAGRID_{}D".format(ndim), True)
        mywrite(fileout, "{}d_datagrid_{}".format(ndim, data_type), True)
        for i, values in enumerate(data):
            mywrite(fileout, "BEGIN_DATAGRID_{}D#{}".format(ndim, i), True)
            origin = grid.origin * Units.Bohr
            if ndim == 3:
                mywrite(fileout, (grid.nr[0] + 1, grid.nr[1] + 1, grid.nr[2] + 1), True)
            elif ndim == 2:
                mywrite(fileout, (grid.nr[0] + 1, grid.nr[1] + 1), True)
            mywrite(
                fileout, origin, True
            )  # TODO, there might be an actual origin if we're dealing with a custom cut of the grid
            for ilat in range(ndim):
                latt = grid.lattice[ilat]
                if units == 'angstrom':
                    latt = latt * Units.Bohr
                mywrite(fileout, latt, True)

            nnr = len(values)
            nlines = nnr // val_per_line
            for iline in range(nlines):
                igrid = iline * val_per_line
                mywrite(fileout, values[igrid : igrid + val_per_line], True)
            igrid = nlines * val_per_line
            if igrid < nnr:
                mywrite(fileout, values[igrid:nnr], True)
            mywrite(fileout, "END_DATAGRID_{}D".format(ndim), True)

        mywrite(fileout, "END_BLOCK_DATAGRID_{}D".format(ndim), True)


def mywrite(fileobj, iterable, newline=False):
    if newline:
        fileobj.write("\n  ")
    if isinstance(iterable, (np.ndarray, list, tuple)):
        for ele in iterable:
            mywrite(fileobj, ele)
            # fileobj.write(str(ele)+'    ')
    else:
        fileobj.write(str(iterable) + "    ")
