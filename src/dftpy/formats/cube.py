import numpy as np

from dftpy.field import DirectField
from dftpy.grid import DirectGrid
from dftpy.ions import Ions


def read_cube(infile, kind="all", full=False, pbc=True, data_type='density', **kwargs):
    # http://gaussian.com/cubegen/
    # http://www.ks.uiuc.edu/Research/vmd/plugins/molfile/cubeplugin.html
    with open(infile, "r") as fh:
        fh.readline()
        fh.readline()
        line = fh.readline().split()
        natom = int(line[0])
        origin = np.asarray(list(map(float, line[1:4])))
        if len(line) > 4:
            rank = int(line[4])
        else:
            rank = 1
        nr = []
        lattice = []
        for i in range(3):
            line = fh.readline().split()
            nr.append(int(line[0]))
            lattice.append(list(map(float, line[1:])))
        nr = np.asarray(nr)
        lattice = np.asarray(lattice)
        for i in range(3):
            lattice[i] *= nr[i]

        numbers = []
        pos = []
        for i in range(natom):
            line = fh.readline().split()
            numbers.append(int(line[0]))
            pos.append(list(map(float, line[2:])))

        ions = Ions(numbers=numbers, positions=pos, cell=lattice, units='au')

        if kind == "ions":
            values = ions
        else:
            data = []
            for line in fh:
                line = line.split()
                l = list(map(float, line))
                data.extend(l)
            data = np.asarray(data)
            grid = DirectGrid(lattice=lattice, nr=nr, full=full, origin=origin)
            rank = data.size // grid.nnrR
            if rank > 1:
                data = data.reshape((-1, rank)).ravel('F')
            plot = DirectField(grid=grid, data=data, rank=rank)
            values = (ions, plot, None)
    if kind == 'data':
        return plot
    else:
        return values


def write_cube(
    filename,
    ions,
    data=None,
    data_type='density',
    header=None,
    origin=None,
    long=True,
    **kwargs,
):
    if long:
        fmt = "{0:15}{1:22.15e}{2:22.15e}{3:22.15e}"
        fmt2 = fmt + "{4:22.15e}"
        fmt3 = "%22.15e"
    else:
        fmt = "{0:5}{1:12.6f}{2:12.6f}{3:12.6f}"  # original
        fmt2 = fmt + "{4:12.6f}"
        fmt3 = "%13.5e"
    fh = open(filename, "w")
    if header is None:
        header = 'DFTpy'
    header += "\nOUTER LOOP: X, MIDDLE LOOP: Y, INNER LOOP: Z\n"
    fh.write(header)

    if origin is None:
        origin = np.zeros(3)
    else:
        origin = np.asarray(origin)

    if data.ndim == 4:
        rank = data.shape[0]
        shape = data.shape[1:]
        data = data.reshape((rank, -1)).ravel('F')
    else:
        rank = 1
        shape = data.shape
        data = data.ravel()

    fh.write(fmt.format(ions.nat, *origin))
    if rank > 1:
        fh.write('{0:8d}'.format(rank))
    fh.write('\n')
    lattice = ions.cell

    for i in range(3):
        v = lattice[i] / shape[i]
        fh.write(fmt.format(shape[i], *v) + '\n')

    charges = ions.get_charges()
    for i in range(ions.nat):
        z = ions.numbers[i]
        c = charges[i]
        p = ions.positions[i]
        fh.write(fmt2.format(z, c, *p) + '\n')

    val_per_line = 6
    nnr = data.size
    nlines = nnr // val_per_line
    for iline in range(nlines):
        i = iline * val_per_line
        data[i : i + val_per_line].tofile(fh, sep=" ", format=fmt3)
        fh.write('\n')
    i = nlines * val_per_line
    if i < nnr:
        data[i:nnr].tofile(fh, sep=" ", format=fmt3)
    fh.close()
