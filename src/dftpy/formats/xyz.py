import re

import numpy as np

from dftpy.ions import Ions

"""
Ref :
    http://atomsk.univ-lille1.fr/doc/en/format_xyz.html
"""


def read_xyz(infile, **kwargs):
    if hasattr(infile, 'close'):
        fh = infile
    else:
        fh = open(infile, "r")

    line = fh.readline()
    natom = int(line)
    line = fh.readline().strip()
    p = re.compile(r'(Lattice)' + r'\s*=\s*["\{]([^"\{\}]+)["\}]\s*')
    m = p.match(line)
    if m is None:
        lattice = np.zeros((3, 3))
    else:
        lattice = np.fromstring(m.group(2), dtype=float, sep=" ")
        lattice = np.asarray(lattice).reshape((3, 3))
    symbols = []
    pos = []
    for i in range(natom):
        line = fh.readline().split()
        symbols.append(line[0])
        pos.append(list(map(float, line[1:4])))
    pos = np.asarray(pos)

    ions = Ions(symbols=symbols, positions=pos, cell=lattice, units='ase')

    if not hasattr(infile, 'close'):
        fh.close()
    return ions


def write_xyz(infile, ions=None, comment=None, fmt='%22.15f', **kwargs):
    if hasattr(infile, 'close'):
        fh = infile
    else:
        fh = open(infile, "w")

    atoms = ions.to_ase()

    if comment is None:
        comment = (
            'Lattice="'
            + " ".join(map(str, (atoms.cell.T).ravel(order='F').tolist()))
            + '"'
        )
        comment += ' Properties=species:S:1:pos:R:3'

    comment = comment.rstrip()
    natoms = len(atoms.positions)
    fh.write(f'{natoms}\n{comment}\n')
    pos = atoms.positions
    for s, ps in zip(atoms.symbols, pos):
        fh.write(f'{s:3s} {fmt%ps[0]} {fmt%ps[1]} {fmt%ps[2]}\n')

    if not hasattr(infile, 'close'):
        fh.close()
    return
