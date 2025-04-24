"""
IO of snpy file

SNPY format
===========
snpy format is just contains some numpy NPY files, but has a definite order which contains structure information.
 - description of snpy (1d integer array)
   1. lattice matrix  (3x3 float array)
   2. numbers of ions (1d integer array)
   3. positions of ions (Nx3 float array)
   4. volumetric data (3d float array)
   5. other data
 - repeat

The first item of snpy is the description of all items of this frame. snpy format can contains multiframe, and each
frame is start with the description.

Notes :
    snpy format also can be directly replace with numpy npz format, but it's need parallel compress and decompress
"""

import numpy as np

from dftpy.field import DirectField
from dftpy.formats import npy
from dftpy.grid import DirectGrid
from dftpy.ions import Ions
from dftpy.mpi import MP, MPIFile, sprint

MAGIC_PREFIX = b'\x93DFTPY'


def write(
    fname,
    ions=None,
    data=None,
    information=None,
    kind='all',
    desc=None,
    mp=None,
    **kwargs,
):
    if mp is None:
        mp = data.grid.mp
    if hasattr(fname, 'close'):
        fh = fname
    else:
        if mp.size > 1:
            fh = MPIFile(fname, mp, amode=mp.MPI.MODE_CREATE | mp.MPI.MODE_WRONLY)
        else:
            fh = open(fname, "wb")

    if desc is None:
        if kind == 'ions':
            desc = np.arange(1, 4)
        elif information is not None:
            desc = np.arange(1, 6)
        else:
            desc = np.arange(1, 5)

    if mp.rank == 0:  # write description
        npy.write(fh, desc, single=True)

    for key in desc:
        if key == 1:  # write cell
            if mp.rank == 0:
                npy.write(fh, ions.cell, single=True)
        elif key == 2:  # write numbers
            if mp.rank == 0:
                npy.write(fh, ions.numbers, single=True)
        elif key == 3:  # write coordinates
            if mp.rank == 0:
                npy.write(fh, ions.positions, single=True)
        elif key == 4:  # write volumetric data
            npy.write(fh, data)
        elif key == 5:  # write information
            if mp.rank == 0:
                npy.write(fh, information, single=True)

    if isinstance(fname, str):
        fh.close()
    return


def read(
    fname,
    mp=None,
    grid=None,
    kind="all",
    full=False,
    datarep='native',
    desc=None,
    **kwargs,
):
    """
    Notes :
        Only support DirectField
    """
    if mp is None:
        if grid is None:
            mp = MP()
        else:
            mp = grid.mp
    if isinstance(fname, str):
        if mp.size > 1:
            # fh = mp.MPI.File.Open(mp.comm, fname, amode = mp.MPI.MODE_RDONLY)
            fh = MPIFile(fname, mp, amode=mp.MPI.MODE_RDONLY)
        else:
            fh = open(fname, "rb")
    else:
        fh = fname

    ions = None
    information = None
    if desc is None:
        # read description
        desc = npy.read(fh, single=True)
    else:
        sprint(
            'WARN : You set the description by yourself {}'.format(desc),
            comm=mp.comm,
            level=3,
        )

    for key in desc:
        if key == 1:  # read cell
            lattice = npy.read(fh, single=True)
        elif key == 2:  # read numbers
            numbers = npy.read(fh, single=True)
        elif key == 3:  # read coordinates
            pos = npy.read(fh, single=True)
            ions = Ions(numbers=numbers, positions=pos, cell=lattice)
            if kind == 'ions':
                break
        elif key == 5:  # read information
            information = npy.read(fh, single=True)
        elif key == 4:  # read volumetric data
            if mp.size == 1:
                data = npy.read(fh, single=True)
                shape = data.shape
                if len(shape) == 4:
                    rank = shape[0]
                    shape = shape[1:]
                else:
                    rank = 1
                if grid is None:
                    grid = DirectGrid(lattice=lattice, nr=shape, full=full, mp=mp)
                data = DirectField(grid=grid, griddata_3d=data, rank=rank)
            else:
                shape, fortran_order, dtype = npy._read_header(fh)
                if len(shape) == 4:
                    rank = shape[0]
                    shape = shape[1:]
                else:
                    rank = 1
                # if fortran_order :
                #     raise AttributeError("Not support Fortran order")
                if grid is None:
                    grid = DirectGrid(lattice=lattice, nr=shape, full=full, mp=mp)
                elif not (np.all(shape == grid.nrR) or np.all(shape == grid.nrG)):
                    raise AttributeError(
                        "The shape {} is not match with grid {} (or {})".format(
                            shape, grid.nrR, grid.nrG
                        )
                    )
                order = 'F' if fortran_order else 'C'
                data = DirectField(grid=grid, rank=rank, order=order)
                npy._read_value(fh, data, datarep=datarep, fortran_order=fortran_order)

    if isinstance(fname, str):
        fh.close()
    if kind == 'ions':
        return ions
    elif kind == 'data':
        return data
    else:
        return ions, data, information


def read_snpy(fname, **kwargs):
    return read(fname, **kwargs)


def write_snpy(fname, *args, **kwargs):
    return write(fname, *args, **kwargs)
