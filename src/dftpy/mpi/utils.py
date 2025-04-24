import os
import sys

from dftpy.constants import environ

from .mpi import MP, PMI

__all__ = ["mp", "sprint"]

mp = MP()
pmi = PMI()


def sprint(
    *args, comm=None, lprint=False, debug=False, level=2, fileobj=None, **kwargs
):
    kwargs['flush'] = True
    if comm is not None:
        if comm.rank == 0:
            lprint = True
    elif mp.is_root:
        lprint = True

    if level < environ['LOGLEVEL']:
        lprint = False
    elif level == 0:
        debug = True

    if not fileobj:
        fileobj = kwargs.get('file', environ.get('STDOUT', sys.stdout))

    args = funcs2args(*args)

    if debug:
        comm = comm or mp.comm
        nodename = os.uname().nodename
        header = f'{nodename[:30]}->{comm.rank:<7d} -> '
        comm.Barrier()
        print(header, *args, file=fileobj, **kwargs)
        comm.Barrier()
    elif lprint:
        print(*args, file=fileobj, **kwargs)


def funcs2args(*args, **kwargs):
    newargs = None
    for i, item in enumerate(args):
        if hasattr(item, '__call__'):
            if not newargs:
                newargs = list(args)
            newargs[i] = item()
    if not newargs:
        newargs = args
    return newargs
