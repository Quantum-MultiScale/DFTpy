import numpy as np

from dftpy.constants import Units
from dftpy.functional.pseudo.abstract_pseudo import BasePseudo


class USP(BasePseudo):
    def __init__(self, fname, direct=False, **kwargs):
        super().__init__(fname, direct=direct, **kwargs)

    def read(self, fname):
        """
        !!! NOT FULLY TEST !!!
        Reads CASTEP-like usp PP file
        """
        if fname.lower().endswith(("usp", "uspcc")):
            ext = "usp"
        elif fname.lower().endswith('uspso'):
            ext = "uspso"
        else:
            raise AttributeError("Pseudopotential not supported : '{}'".format(fname))

        HARTREE2EV = Units.Ha
        BOHR2ANG = Units.Bohr
        with open(fname, "r") as outfil:
            lines = outfil.readlines()

        comment = ''
        ibegin = 0
        for i in range(0, len(lines)):
            line = lines[i]
            if ext == 'usp':
                if "END COMMENT" in line:
                    ibegin = i + 4
                elif (
                    ibegin > 1
                    and (line.strip() == "1000" or len(line.strip()) == 1)
                    and i - ibegin > 4
                ):
                    iend = i
                    break
                elif ibegin < 1:
                    comment += line
            elif ext == 'uspso':
                if "END COMMENT" in line:
                    ibegin = i + 5
                elif (
                    ibegin > 1
                    and (line.strip() == "1000" or len(line.strip()) == 5)
                    and i - ibegin > 4
                ):
                    iend = i
                    break
                elif ibegin < 1:
                    comment += line

        line = " ".join([line.strip() for line in lines[ibegin:iend]])

        zval = float(lines[ibegin - 2].strip())

        if (
            "1000" in lines[iend]
            or len(lines[iend].strip()) == 1
            or len(lines[iend].strip()) == 5
        ):
            pass
        else:
            raise AttributeError("Error : Check the PP file : {}".format(fname))
        gmax = float(lines[ibegin - 1].split()[0]) * BOHR2ANG

        # v = np.array(line.split()).astype(np.float64) / (HARTREE2EV*BOHR2ANG ** 3 * 4.0 * np.pi)
        self.v = np.array(line.split()).astype(np.float64) / (HARTREE2EV * BOHR2ANG**3)
        self.r = np.linspace(0, gmax, num=len(self.v))
        self.v[1:] -= zval * 4.0 * np.pi / self.r[1:] ** 2
        self.info = {'comment': comment}
        # -----------------------------------------------------------------------
        nlcc = int(lines[ibegin - 1].split()[1])
        if nlcc == 2 and ext == 'usp':
            # num_projectors
            for i in range(iend, len(lines)):
                l = lines[i].split()
                if len(l) == 2 and all([item.isdigit() for item in l]):
                    ibegin = i + 1
                    ngrid = int(l[1])
                    break
            core_grid = []
            for i in range(ibegin, len(lines)):
                l = list(map(float, lines[i].split()))
                core_grid.extend(l)
                if len(core_grid) >= ngrid:
                    core_grid = core_grid[:ngrid]
                    break
            self._core_density_grid = np.asarray(core_grid) * BOHR2ANG
            line = " ".join([line.strip() for line in lines[ibegin:]])
            data = np.array(line.split()).astype(np.float64)
            self._core_density = data[-ngrid:]
        # -----------------------------------------------------------------------
