"""
MPI-IO of npy file

Notes:
    In order to support write/read multi-data in one file, not guarantee read and write for same file handle.
"""

import numpy as np
from numpy.lib import format as npyf

# from dftpy.mpi import sprint


def write(fh, data, grid=None, single=False, version=(1, 0), datarep='native'):
    """
    Write npy file

    Args:
        fh: Name of output file or a file descriptor
        data: the value to output
        single: parallel version but run one processor
        version: the version of npy format, only support (1, 0), (2, 0), (3, 0)
        datarep: Data representation

    Notes:
        For datarep, ``native`` fastest, but ``external32`` most portable
        For easy to use and faster, if 'single = True' only rank == 0 will write.
    """
    if hasattr(data, 'grid'):
        grid = data.grid

    npyf._check_version(version)

    if not isinstance(data, np.ndarray):
        data = np.asanyarray(data)

    if single or (not hasattr(fh, 'Get_position')):
        return _write_single(fh, data, version)

    if grid.mp.rank == 0:
        fp = _write_header(fh, data, version=version, grid=grid)
        fp = fh.Get_position()
    else:
        fp = 0
    fp = grid.mp.comm.bcast(fp, root=0)
    _write_value(fh, data, fp=fp, grid=grid, datarep=datarep)


def _write_single(fh, data, version=(1, 0)):
    if not hasattr(fh, 'Get_position'):
        # serial version use numpy.save
        return np.save(fh, data)
    if fh.mp.rank > 0:
        return
    _write_header(fh, data, version)
    dtype = data.dtype
    if dtype.isbuiltin == 0:
        fh.Write(data.tobytes())
    else:
        fh.Write(data)


def _write_header(fh, data, version=(1, 0), grid=None):
    if hasattr(data, 'grid'):
        grid = data.grid
    header = npyf.header_data_from_array_1_0(data)
    if grid is None:
        shape = data.shape
    else:
        shape = grid.nrR
        if data.ndim == 4:
            shape = data.shape[0], *shape
        # if header['fortran_order'] and grid.mp.size > 1:
        #     raise AttributeError("Not support Fortran order")
    header['shape'] = tuple(shape)
    npyf._write_array_header(fh, header, version)
    return


def _write_value(fh, data, fp=None, grid=None, datarep='native'):
    if hasattr(data, 'grid'):
        grid = data.grid

    if fp is None:
        fp = fh.Get_byte_offset(fh.Get_position())
        fp = grid.mp.comm.bcast(fp, root=0)
    MPI = grid.mp.MPI
    etype = MPI._typedict[data.dtype.char]
    if data.flags.f_contiguous:
        order = MPI.ORDER_F
    else:
        order = MPI.ORDER_C
    nrR, nr, offsets = grid.nrR, grid.nr, grid.offsets
    if data.ndim == 4:
        rank = data.shape[0]
        nrR = np.insert(nrR, 0, rank)
        nr = np.insert(nr, 0, rank)
        offsets = np.insert(offsets, 0, 0)
    filetype = etype.Create_subarray(nrR, nr, offsets, order=order)
    filetype.Commit()
    fh.Set_view(fp, etype, filetype, datarep=datarep)
    fh.Write_all(data)
    filetype.Free()
    # fp += grid.nnrR*etype.size
    fp = fh.Get_byte_offset(fh.Get_position())
    fp = grid.mp.comm.bcast(fp, root=0)
    fh.Set_view(0, MPI.BYTE, MPI.BYTE, datarep=datarep)
    fh.Seek(fp)
    return


def read(fh, data=None, grid=None, single=False, datarep='native', allow_pickle=False):
    """
    Read npy file

    Args:
        fh: Name of output file or a file descriptor
        data: stored the read data
        single: parallel version but run one processor
        grid: grid of the field
        datarep: Data representation

    Notes:
        For safe, please make sure everytime with 'single = True' always on rank == 0.
    """
    if single or (not hasattr(fh, 'Get_position')):
        return _read_single(fh, data, allow_pickle=allow_pickle)

    if hasattr(data, 'grid'):
        grid = data.grid

    shape, fortran_order, dtype = _read_header(fh)

    # if fortran_order and grid.mp.size > 1 :
    #     raise AttributeError("Not support Fortran order")

    if not (np.all(shape == grid.nrR) or np.all(shape == grid.nrG)):
        raise AttributeError(
            "The shape {} is not match with grid {} ({})".format(
                shape, grid.nrR, grid.nrG
            )
        )
    if data is None:
        data = np.empty(grid.nr, dtype=dtype, order='C')

    data = _read_value(
        fh, data, grid=grid, datarep=datarep, fortran_order=fortran_order
    )
    return data


def _read_single(fh, data=None, **kwargs):
    if not hasattr(fh, 'Get_position'):
        # serial version use numpy.save
        return np.load(fh, **kwargs)
    shape, fortran_order, dtype = _read_header(fh)
    if dtype.isbuiltin == 0:
        buf = fh.read(dtype.itemsize)
        data = np.frombuffer(buf, dtype=dtype)
        if len(shape) == 0:
            data = data[0]
    else:
        if data is None:
            order = 'F' if fortran_order else 'C'
            data = np.empty(shape, dtype=dtype, order=order)
        fh.Read(data)
    return data


def _read_header(fh):
    version = npyf.read_magic(fh)
    npyf._check_version(version)
    return npyf._read_array_header(fh, version)


def _read_value(fh, data, fp=None, grid=None, datarep='native', fortran_order=False):
    if hasattr(data, 'grid'):
        grid = data.grid
    MPI = grid.mp.MPI
    if fp is None:
        fp = fh.Get_byte_offset(fh.Get_position())
        fp = grid.mp.comm.bcast(fp, root=0)
    etype = MPI._typedict[data.dtype.char]
    if fortran_order:
        order = MPI.ORDER_F
    else:
        order = MPI.ORDER_C
    nrR, nr, offsets = grid.nrR, grid.nr, grid.offsets
    if data.ndim == 4:
        rank = data.shape[0]
        nrR = np.insert(nrR, 0, rank)
        nr = np.insert(nr, 0, rank)
        offsets = np.insert(offsets, 0, 0)
    filetype = etype.Create_subarray(nrR, nr, offsets, order=order)
    filetype.Commit()
    fh.Set_view(fp, etype, filetype, datarep=datarep)
    fh.Read_all(data)
    filetype.Free()
    fp = fh.Get_byte_offset(fh.Get_position())
    fp = grid.mp.comm.bcast(fp, root=0)
    fh.Set_view(0, MPI.BYTE, MPI.BYTE, datarep=datarep)
    fh.Seek(fp)
    return data
