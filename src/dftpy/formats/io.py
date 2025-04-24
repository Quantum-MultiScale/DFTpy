import os
from collections import namedtuple
from importlib import import_module

from dftpy.field import DirectField
from dftpy.grid import DirectGrid
from dftpy.ions import Ions
from dftpy.mpi import MP, SerialComm

ioformat = namedtuple('ioformat', ['format', 'module', 'read', 'write', 'kind'])
iounkeys = ['mp', 'comm', 'kind', 'data_type', 'grid', 'names']

IOFormats = {
    "snpy": ioformat(
        'snpy', 'dftpy.formats.snpy', 'read_snpy', 'write_snpy', ['ions', 'data', 'all']
    ),
    "xsf": ioformat(
        'xsf', 'dftpy.formats.xsf', 'read_xsf', 'write_xsf', ['ions', 'data', 'all']
    ),
    "qepp": ioformat(
        'qepp', 'dftpy.formats.qepp', 'read_qepp', 'write_qepp', ['ions', 'data', 'all']
    ),
    "cube": ioformat(
        'cube', 'dftpy.formats.cube', 'read_cube', 'write_cube', ['ions', 'data', 'all']
    ),
    "chg": ioformat(
        'chg', 'dftpy.formats.chg', 'read_chg', 'write_chg', ['ions', 'data', 'all']
    ),
    "chgcar": ioformat(
        'chgcar',
        'dftpy.formats.chg',
        'read_chgcar',
        'write_chgcar',
        ['ions', 'data', 'all'],
    ),
    "den": ioformat('den', 'dftpy.formats.den', 'read_den', 'write_den', ['data']),
    "xyz": ioformat('xyz', 'dftpy.formats.xyz', 'read_xyz', 'write_xyz', ['ions']),
    "vasp": ioformat('vasp', 'dftpy.formats.vasp', 'read_vasp', 'write_vasp', ['ions']),
    "ase": ioformat('ase', 'dftpy.formats.ase_io', 'read_ase', 'write_ase', ['ions']),
    "pmg": ioformat('pmg', 'dftpy.formats.pmg_io', 'read_pmg', 'write_pmg', ['ions']),
}

IOFormats['pp'] = IOFormats['qepp']
IOFormats['extxyz'] = IOFormats['xyz']
IOFormats['poscar'] = IOFormats['vasp']
IOFormats['contcar'] = IOFormats['vasp']
IOFormats['pymatgen'] = IOFormats['pmg']


def guessType(infile, **kwargs):
    return guess_format(infile, **kwargs)


def guess_format(infile, **kwargs):
    basename = os.path.basename(infile).lower()
    ext = os.path.splitext(infile)[1].lower()
    format = IOFormats.get(basename, None)
    if format is None and len(ext) > 1:
        format = IOFormats.get(ext[1:], None)

    if format is None:
        format = ext[1:] if len(ext) > 1 else basename
    else:
        format = format.format
    return format


def get_io_driver(infile, format=None, mode='r'):
    if format is None:
        format = guess_format(infile)
    iof = IOFormats.get(format, None)

    if iof is None:
        raise AttributeError("%s format not support yet" % format)

    try:
        module = import_module(iof.module)
        if 'r' in mode:
            func = getattr(module, iof.read, None)
            if not iof.read or func is None:
                raise AttributeError("%s format not support read" % iof.format)
            iof = iof._replace(read=func)
        if 'w' in mode or 'a' in mode:
            func = getattr(module, iof.write, None)
            if not iof.write or func is None:
                raise AttributeError("%s format not support write" % iof.format)
            iof = iof._replace(write=func)
    except Exception as e:
        raise e

    return iof


def read(
    infile,
    format=None,
    kind='ions',
    ecut=None,
    driver=None,
    mp=None,
    grid=None,
    **kwargs,
):
    if driver is None:
        driver = get_io_driver(infile, format, mode='r')
    elif isinstance(driver, str):
        for key in iounkeys:
            if key in kwargs:
                kwargs.pop(key)
        driver = get_io_driver(infile, driver, mode='r')
    #
    if mp is None:
        if grid is None:
            mp = MP()
        else:
            mp = grid.mp

    if hasattr(driver, 'read'):
        if mp.is_root or driver.format == "snpy":  # only snpy format support MPI-IO
            if driver.format in ['ase', 'pmg']:
                values = driver.read(infile, format=format, **kwargs)
            else:
                values = driver.read(infile, kind=kind, mp=mp, grid=grid, **kwargs)
        else:
            values = [
                None,
            ] * 3
    else:
        raise AttributeError(f"Sorry, not support {driver} driver")

    ions = data = info = None
    if kind == 'ions':
        ions = values
    elif kind == 'data':
        data = values
    else:
        if 'ions' not in driver.kind:
            data = values
        elif 'data' not in driver.kind:
            ions = values
        elif len(values) == 2:
            ions, data = values
        elif len(values) == 3:
            ions, data, info = values
        else:
            raise AttributeError(
                f"Sorry, the {driver} driver should only return 3 values."
            )
    # -----------------------------------------------------------------------
    if mp.size > 1 and driver.format != "snpy":
        ions_options = getattr(ions, 'init_options', 0)
        ions_options = mp.comm.bcast(ions_options)
        if ions_options:
            ions = Ions(**ions_options)
        grid_options = getattr(getattr(data, 'grid', 0), 'init_options', 0)
        grid_options = mp.comm.bcast(grid_options)
        if grid_options:
            grid_options['mp'] = mp
            grid = DirectGrid(**grid_options)
            data_options = getattr(data, 'init_options', 0)
            data_options = mp.comm.bcast(data_options)
            data2 = DirectField(grid=grid, **data_options)
            grid.scatter(data, out=data2)
            data = data2
        info = mp.comm.bcast(info)
    # -----------------------------------------------------------------------
    if ecut and hasattr(data, 'grid'):
        data.grid.ecut = ecut
    if kind == 'ions':
        values = ions
    elif kind == 'data':
        values = data
    else:
        values = (ions, data, info)
    return values


def write(
    outfile,
    ions=None,
    data=None,
    information=None,
    format=None,
    comm=None,
    driver=None,
    **kwargs,
):
    if isinstance(data, Ions):
        if ions is None or isinstance(ions, DirectField):
            ions, data = data, ions
        else:
            raise AttributeError("Please check the input data")

    if comm is None:
        if hasattr(data, 'mp'):
            comm = data.mp.comm
        else:
            comm = SerialComm()
    if driver is not None:
        if comm.size > 1 and data is not None:
            data = data.gather()
        if isinstance(driver, str):
            for key in iounkeys:
                if key in kwargs:
                    kwargs.pop(key)
            driver = get_io_driver(outfile, driver, mode='w')
            if comm.rank == 0:
                driver.write(outfile, ions=ions, data=data, **kwargs)
        elif hasattr(driver, 'write'):
            if comm.rank == 0:
                driver.write(outfile, ions=ions, data=data, **kwargs)
        else:
            raise AttributeError(f"Sorry, not support {driver} driver")
    else:
        driver = get_io_driver(outfile, format, mode='w')
        if driver.format == "snpy":  # only snpy format support MPI-IO
            return driver.write(
                outfile, ions=ions, data=data, information=information, **kwargs
            )
        else:  # only rank==0 write
            if comm.size > 1 and data is not None:
                data = data.gather()
            if comm.rank == 0:
                driver.write(
                    outfile, ions=ions, data=data, information=information, **kwargs
                )

            comm.Barrier()
    return


def read_data(infile, format=None, **kwargs):
    return read(infile, format=format, kind='data', **kwargs)


def read_all(infile, format=None, **kwargs):
    return read(infile, format=format, kind='all', **kwargs)


def read_density(infile, format=None, **kwargs):
    kwargs.pop('data_type', None)
    return read(infile, format=format, kind='data', data_type='density', **kwargs)


def read_potential(infile, format=None, **kwargs):
    kwargs.pop('data_type', None)
    return read(infile, format=format, kind='data', data_type='potential', **kwargs)


def write_all(outfile, ions=None, data=None, information=None, format=None, **kwargs):
    return write(
        outfile, ions=ions, data=data, information=information, format=format, **kwargs
    )


def write_density(outfile, data=None, ions=None, format=None, **kwargs):
    kwargs.pop('data_type', None)
    return write(
        outfile, data=data, ions=ions, format=format, data_type='density', **kwargs
    )


def write_potential(outfile, data=None, ions=None, format=None, **kwargs):
    kwargs.pop('data_type', None)
    return write(
        outfile, data=data, ions=ions, format=format, data_type='potential', **kwargs
    )
