import numpy as np

from dftpy.constants import Units
from dftpy.functional.pseudo.abstract_pseudo import BasePseudo


class RECPOT(BasePseudo):
    def __init__(self, fname, direct=False, **kwargs):
        super().__init__(fname, direct=direct, **kwargs)

    def read(self, fname):
        """Reads CASTEP-like recpot PP file
        Returns tuple (g, v)"""
        HARTREE2EV = Units.Ha
        BOHR2ANG = Units.Bohr
        with open(fname, "r") as outfil:
            lines = outfil.readlines()

        comment = ''
        ibegin = 0
        for i in range(0, len(lines)):
            line = lines[i]
            if "END COMMENT" in line:
                ibegin = i + 3
            elif ibegin > 1 and (line.strip() == "1000" or len(line.strip()) == 1):
                iend = i
                break
            elif ibegin < 1:
                comment += line

        line = " ".join([line.strip() for line in lines[ibegin:iend]])

        if "1000" in lines[iend] or len(lines[iend].strip()) == 1:
            pass
        else:
            raise AttributeError("Error : Check the PP file : {}".format(fname))
        gmax = float(lines[ibegin - 1].strip()) * BOHR2ANG
        self.v = np.array(line.split()).astype(np.float64) / HARTREE2EV / BOHR2ANG**3
        self.r = np.linspace(0, gmax, num=len(self.v))
        self.info = {'comment': comment}
        self._zval = round((self.v[0] - self.v[1]) * (self.r[1] ** 2) / (4.0 * np.pi))
        # self._zval = (self.v[0] - self.v[1]) * (self.r[-1] / (self.r.size - 1)) ** 2 / (4.0 * np.pi)

    def write(self, fname, header='DFTpy', **kwargs):
        HARTREE2EV = Units.Ha
        BOHR2ANG = Units.Bohr
        gmax = self.r / BOHR2ANG
        v = self.v * HARTREE2EV * BOHR2ANG**3
        fw = open(fname, 'w')
        fw.write('START COMMENT\n' + header + '\nEND COMMENT\n')
        fw.write('3     5\n')  # Now only support 5
        fw.write(str(gmax) + '\n')
        nl = v.size // 3
        # seq = ' ' * 5
        for line in v[: nl * 3].reshape(-1, 3):
            fw.write(
                "     {0[0]:22.15E}     {0[1]:22.15E}     {0[2]:22.15E}\n".format(line)
            )
        for line in v[nl * 3 :]:
            fw.write("{0:22.15E}".format(line))
        if len(v[nl * 3 :]) > 0:
            fw.write("\n")
        fw.write("1000\n")
        fw.close()
