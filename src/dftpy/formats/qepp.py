import numpy as np

from dftpy.field import DirectField
from dftpy.grid import DirectGrid
from dftpy.ions import Ions


class PP(object):
    def __init__(self, filepp):
        self.filepp = filepp
        self.title = "DFTpy"
        self.cutoffvars = {
            'gcutm': 1.0,
            'dual': 4.0,
            'ecut': 1.0,
        }

    def read(self, full=True, data_type='density', kind='all', **kwargs):

        with open(self.filepp) as filepp:
            # title
            self.title = filepp.readline()

            # nr1x, nr2x, nr3x, nr1, nr2, nr3, nat, ntyp
            nrx = np.empty(3, dtype=int)
            nr = np.empty(3, dtype=int)
            nrx[0], nrx[1], nrx[2], nr[0], nr[1], nr[2], nat, ntyp = (
                int(x) for x in filepp.readline().split()
            )

            # ibrav, celldm
            celldm = np.zeros(6, dtype=float)
            linesplt = filepp.readline().split()
            ibrav = int(linesplt[0])
            celldm = np.asarray(linesplt[1:], dtype=float)

            # at(i) three times
            if ibrav == 0:
                at = np.zeros((3, 3), dtype=float)
                for ilat in range(3):
                    linesplt = filepp.readline().split()
                    at[ilat] = np.asarray(linesplt, dtype=float)
                at *= celldm[0]
            else:
                at = self.celldm2at(ibrav, celldm)

            # gcutm, dual, ecut, plot_num
            gcutm, dual, ecut, plot_num = (float(x) for x in filepp.readline().split())
            plot_num = int(plot_num)
            # filepp.readline()
            # gcutm, dual, ecut, plot_num = 1.0, 1.0, 1.0, 0
            self.cutoffvars["ibrav"] = ibrav
            self.cutoffvars["celldm"] = celldm
            self.cutoffvars["gcutm"] = gcutm
            self.cutoffvars["dual"] = dual
            self.cutoffvars["ecut"] = ecut
            self.cutoffvars["plot_num"] = plot_num
            grid = DirectGrid(lattice=at, nr=nrx, full=full)

            # ntyp
            atm = []
            zv = np.empty(ntyp, dtype=float)
            zval = {}
            for ity in range(ntyp):
                linesplt = filepp.readline().split()
                atm.append(linesplt[1])
                zv[ity] = float(linesplt[2])
                zval[linesplt[1]] = float(linesplt[2])
            # tau
            # tau = np.zeros((nat,3), dtype=float)
            # tau = np.zeros(3, dtype=float)
            # ityp = np.zeros(nat, dtype=int)
            label = []
            pos = []
            for iat in range(nat):
                linesplt = filepp.readline().split()
                # tau[iat,:] = np.asarray(linesplt[1:4],dtype=float)
                # ityp[iat] = int(linesplt[4]) -1
                tau = np.asarray(linesplt[1:4], dtype=float)
                ity = int(linesplt[4]) - 1
                label.append(atm[ity])
                pos.append(tau)
            pos = np.asarray(pos)
            if ibrav == 0:
                pos *= celldm[0]
                ions = Ions(symbols=label, positions=pos, cell=grid.cell, units='au')
            else:
                ions = Ions(
                    symbols=label, scaled_positions=pos, cell=grid.cell, units='au'
                )

            ions.set_charges(zval)

            if kind == 'ions':
                return ions

            # plot
            blocksize = 1024 * 8
            strings = ""
            while True:
                line = filepp.read(blocksize)
                if not line:
                    break
                strings += line
            ppgrid = np.fromstring(strings, dtype=float, sep=" ")

            # igrid = 0
            # nnr = nrx[0] * nrx[1] * nrx[2]
            # ppgrid = np.zeros(nnr, dtype=float)
            # for line in filepp:
            # line = line.split()
            # npts = len(line)
            # ppgrid[igrid:igrid + npts] = np.asarray(line, dtype=float)
            # igrid += npts

            plot = DirectField(grid=grid, griddata_F=ppgrid, rank=1)

            if data_type == 'potential':
                plot *= 0.5  # Ry to Hartree

            if kind == 'data':
                return plot

            return ions, plot, self.cutoffvars

    def writepp(self, ions, data, **kwargs):
        self.write(ions, data, **kwargs)

    def celldm2at(self, ibrav, celldm):

        at = np.zeros((3, 3), dtype=float)

        if ibrav == 1:
            at = celldm[0] * np.identity(3)
        elif ibrav == 2:
            at[:, 0] = 0.5 * celldm[0] * np.array([-1.0, 0.0, 1.0])
            at[:, 1] = 0.5 * celldm[0] * np.array([0.0, 1.0, 1.0])
            at[:, 2] = 0.5 * celldm[0] * np.array([-1.0, 1.0, 0.0])
        else:
            # implement all the other Bravais lattices
            raise NotImplementedError(
                "celldm2at is only implemented for ibrav = 0 and ibrav = 1"
            )

        return at

    def write(
        self, ions, data, data_type='density', header=None, information=None, **kwargs
    ):
        info = self.cutoffvars.copy()
        if information:
            info.update(information)
        fmt = "%22.15e"
        with open(self.filepp, "w") as filepp:
            val_per_line = 5
            grid = data.grid

            # title
            header = header or self.title
            filepp.write(header)

            # nr1x, nr2x, nr3x, nr1, nr2, nr3, nat, ntyp
            mywrite(filepp, grid.nrR, True)
            mywrite(filepp, grid.nrR, False)
            mywrite(filepp, [ions.nat, len(ions.symbols_uniq)], False)

            # ibrav, celldm
            ibrav = info.get("ibrav", 0)
            celldm = info.get("celldm", None)
            if celldm is None:
                celldm = np.zeros(6)
                celldm[0] = ions.cell.lengths()[0]
            mywrite(filepp, ibrav, True)
            mywrite(filepp, celldm, False)
            if ibrav == 0:
                for ilat in range(3):
                    mywrite(filepp, ions.cell[ilat] / celldm[0], True)
            # gcutm, dual, ecut, plot_num
            mywrite(filepp, info["gcutm"], True)
            mywrite(filepp, info["dual"], False)
            mywrite(filepp, info["ecut"], False)
            if data_type == 'potential':
                plot_num = 1
            else:
                plot_num = 0
            mywrite(filepp, info.get('plot_num', plot_num), False)

            # ntyp
            for ity, spc in enumerate(ions.symbols_uniq):
                mywrite(filepp, [ity + 1, spc, ions.zval.get(spc, 0.0)], True)

            # tau
            tau = ions.positions / celldm[0]
            for iat in range(ions.nat):
                mywrite(filepp, iat + 1, True)
                mywrite(filepp, tau[iat], False)
                mywrite(
                    filepp,
                    np.where(ions.symbols_uniq == ions.symbols[iat])[0][0] + 1,
                    False,
                )

            # plot
            filepp.write('\n')
            nlines = data.grid.nnr // val_per_line
            grid_pp = data.get_values_flatarray(order="F")
            if data_type == 'potential':
                grid_pp = grid_pp * 2.0  # Hartree to Ry
            for iline in range(nlines):
                igrid = iline * val_per_line
                grid_pp[igrid : igrid + val_per_line].tofile(
                    filepp, sep=" ", format=fmt
                )
                filepp.write('\n')
            igrid = (iline + 1) * val_per_line
            grid_pp[igrid : grid.nnr].tofile(filepp, sep=" ", format=fmt)


def mywrite(fileobj, iterable, newline):
    if newline:
        fileobj.write("\n  ")
    # if len(iterable) > 1 :
    try:
        for ele in iterable:
            fileobj.write(str(ele) + "    ")
    except Exception:
        fileobj.write(str(iterable) + "    ")


def read_qepp(infile, **kwargs):
    return PP(infile).read(**kwargs)


def write_qepp(infile, ions=None, data=None, **kwargs):
    PP(infile).write(ions=ions, data=data, **kwargs)
    return
