"""
API for pymatgen

Notes :
    From v2021.3.4, pymatgen removed the root imports.
"""

from dftpy.ions import Ions


def pmg_read(infile, index=None, format=None, **kwargs):
    from pymatgen.core import Structure

    struct = Structure.from_file(infile)
    ions = pmg2ions(struct, **kwargs)
    return ions


def pmg_write(outfile, ions, format=None, pbc=None, **kwargs):
    struct = ions2pmg(ions)
    struct.to(filename=outfile)
    return


def pmg2ions(pmg_atoms, wrap=True, **kwargs):
    from pymatgen.io.ase import AseAtomsAdaptor

    atoms = AseAtomsAdaptor.get_atoms(pmg_atoms, **kwargs)
    ions = Ions.from_ase(atoms)
    return ions


def ions2pmg(ions):
    from pymatgen.io.ase import AseAtomsAdaptor

    atoms = ions.to_ase()
    struct = AseAtomsAdaptor.get_structure(atoms)
    return struct


def read_pmg(infile, **kwargs):
    return pmg_read(infile, **kwargs)


def write_pmg(outfile, ions, data=None, **kwargs):
    pmg_write(outfile, ions, **kwargs)
