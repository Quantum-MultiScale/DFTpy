import numpy as np


def read_data_den(infile, order="F", **kwargs):
    with open(infile, "r") as fr:
        line = fr.readline()
        nr0 = list(map(int, line.split()))
        blocksize = 1024 * 8
        strings = ""
        while True:
            line = fr.read(blocksize)
            if not line:
                break
            strings += line
    density = np.fromstring(strings, dtype=float, sep=" ")
    density = density.reshape(nr0, order=order)
    return density


def write_data_den(outfile, density, order="F", **kwargs):
    with open(outfile, "w") as fw:
        nr = density.shape
        if len(nr) == 3:
            fw.write("{0[0]:10d} {0[1]:10d} {0[2]:10d}\n".format(nr))
        elif len(nr) == 4:
            fw.write("{0[0]:10d} {0[1]:10d} {0[2]:10d} {0[3]:10d}\n".format(nr))
        size = np.size(density)
        nl = size // 3
        outrho = density.ravel(order="F")
        for line in outrho[: nl * 3].reshape(-1, 3):
            fw.write("{0[0]:22.15E} {0[1]:22.15E} {0[2]:22.15E}\n".format(line))
        for line in outrho[nl * 3 :]:
            fw.write("{0:22.15E}".format(line))


def read_den(infile, **kwargs):
    data = read_data_den(infile, **kwargs)
    return data


def write_den(outfile, ions=None, data=None, **kwargs):
    return write_data_den(outfile, data, **kwargs)
