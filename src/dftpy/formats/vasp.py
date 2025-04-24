import numpy as np

from dftpy.ions import Ions


def read_POSCAR(infile, names=None, **kwargs):
    if hasattr(infile, 'close'):
        fh = infile
    else:
        fh = open(infile, "r")

    title = fh.readline()
    scale = list(map(float, fh.readline().split()))
    if len(scale) == 1:
        scale = np.ones(3) * scale
    elif len(scale) == 3:
        scale = np.asarray(scale)
    lat = []
    for i in range(2, 5):
        lat.append(list(map(float, fh.readline().split())))
    lat = np.asarray(lat)
    for i in range(3):
        lat[i] *= scale[i]
    lineL = fh.readline().split()
    if lineL[0].isdigit():
        typ = list(map(int, lineL))
    else:
        names = lineL
        typ = list(map(int, fh.readline().split()))
    if names is None:
        raise AttributeError("Must input the ions names")
    Format = fh.readline().strip()[0]

    nat = sum(typ)
    pos = []
    i = 0
    for line in fh:
        i += 1
        if i > nat:
            break
        else:
            pos.append(list(map(float, line.split()[:3])))
    pos = np.asarray(pos)

    if Format.lower() in ['c', 'k']:
        positions = pos
        scaled_positions = None
    else:
        positions = None
        scaled_positions = pos

    symbols = []
    for i in range(len(names)):
        symbols.extend([names[i]] * typ[i])

    ions = Ions(
        symbols=symbols,
        positions=positions,
        scaled_positions=scaled_positions,
        cell=lat,
        units='ase',
    )

    if not hasattr(infile, 'close'):
        fh.close()
    return ions


def read_vasp(infile, **kwargs):
    ions = read_POSCAR(infile, **kwargs)
    return ions


def write_vasp(infile, ions, direct=False, fmt='%22.15f', header='DFTpy', **kwargs):
    if hasattr(infile, 'close'):
        fh = infile
    else:
        fh = open(infile, "w")

    ions = ions.to_ase()

    fh.write(header + '\n')
    fh.write('1.0\n')
    for vec in ions.cell:
        for x in vec:
            fh.write(fmt % x)
        fh.write('\n')

    if direct:
        pos = ions.get_scaled_positions()
    else:
        pos = ions.get_positions()

    names, indices, counts = np.unique(
        ions.symbols, return_inverse=True, return_counts=True
    )

    for x in names:
        fh.write(f' {x:<3s}')
    fh.write('\n')
    for x in counts:
        fh.write(f' {x:<3d}')
    fh.write('\n')

    ind = np.argsort(indices)
    pos = pos[ind]

    if direct:
        fh.write('Direct\n')
    else:
        fh.write('Cartesian\n')

    for i, ps in enumerate(pos):
        fh.write(f'{fmt%ps[0]} {fmt%ps[1]} {fmt%ps[2]}\n')

    if not hasattr(infile, 'close'):
        fh.close()
    return
