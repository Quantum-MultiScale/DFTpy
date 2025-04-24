from collections import OrderedDict

import ase.io
import numpy as np

from dftpy.ions import Ions


def ase_read(infile, index=None, format=None, **kwargs):
    if isinstance(infile, ase.Atoms):
        atoms = infile
    else:
        atoms = ase.io.read(infile, index=index, format=format, **kwargs)
    ions = ase2ions(atoms)
    return ions


def ase_write(outfile, ions, format=None, pbc=None, **kwargs):
    atoms = ions.to_ase()
    ase.io.write(outfile, atoms, format=format, **kwargs)
    return


def ase2ions(ase_atoms, wrap=True):
    ions = Ions.from_ase(ase_atoms)
    if wrap:
        ions.wrap()
    return ions


def ions2ase(ions):
    return ions.to_ase()


def read_ase(infile, **kwargs):
    return ase_read(infile, **kwargs)


def write_ase(outfile, ions, data=None, **kwargs):
    ase_write(outfile, ions, **kwargs)


def sort_ase_atoms(atoms, tags=None):
    symbols = atoms.symbols
    if tags is None:
        tags = (OrderedDict.fromkeys(symbols)).keys()
    index = []
    for s in tags:
        ind = np.where(symbols == s)[0]
        index.extend(ind.tolist())
    return atoms[index]


def subtract_ase_atoms(a1, a2, tol=0.1):
    from ase.neighborlist import neighbor_list

    atoms = a1 + a2
    inda, indb = neighbor_list('ij', atoms, tol)
    index = []
    for i in range(len(atoms)):
        firsts = np.where(inda == i)[0]
        if len(firsts) > 0:
            neibors = indb[firsts]
            index.append(i)
            index.extend(neibors)
    del atoms[index]
    return atoms
