import numpy as np
import datetime

from dftpy.functional.pseudo.abstract_pseudo import BasePseudo

"""
Ref :
    https://docs.abinit.org/developers/psp8_info/
    https://docs.abinit.org/developers/psp6_info/
"""

class PSP(BasePseudo):
    def __init__(self, fname, direct = True, **kwargs):
        super().__init__(fname, direct = direct, **kwargs)

    def read(self, fname):
        # Only support psp8 format
        with open(fname, "r") as fr:
            lines = []
            for i, line in enumerate(fr):
                if i > 5 :
                    line = line.replace('D', 'E')
                lines.append(line)
        info = {}

        # line 2 :atomic number, pseudoion charge, date
        values = lines[1].split()
        atomicnum = int(float(values[0]))
        zval = float(values[1])
        # line 3 :pspcod,pspxc,lmax,lloc,mmax,r2well
        values = lines[2].split()
        info['atomicnum'] = atomicnum
        info['zval'] = zval
        info['pspcod'] = int(values[0])
        info['pspxc'] = int(values[1])
        info['lmax'] = int(values[2])
        info['lloc'] = int(values[3])
        info['mmax'] = int(values[4])
        info['r2well'] = int(values[5])
        # line 4 : rchrg fchrg qchrg
        values = lines[3].split()
        info['rchrg'] = float(values[0])
        info['fchrg'] = float(values[1])
        info['qchrg'] = float(values[2])
        self.info = info
        #
        if info['pspcod'] == 8:
            return self._read_psp8(lines)
        elif info['pspcod'] == 6:
            return self._read_psp6(lines)
        else:
            raise ValueError("Only support psp8/psp6 format pseudopotential with psp")

    def _read_psp8(self, lines):
        info = self.info
        info['info'] = lines[:6]
        # line 5 : nproj
        info['nproj'] = list(map(int, lines[4].split()[:5]))
        # line 6 : extension_switch
        values = lines[5].split()
        v = []
        for item in values :
            if not item.isdigit():
                break
            else :
                v.append(int(item))
        info['extension_switch'] = v
        #
        mmax = info['mmax']
        lloc = info['lloc']
        fchrg = info['fchrg']

        ibegin = 7
        iend = ibegin + mmax
        # line = " ".join([line for line in lines[ibegin:iend]])
        # data = np.fromstring(line, dtype=float, sep=" ")
        # data = np.array(line.split()).astype(np.float64) / HARTREE2EV / BOHR2ANG ** 3
        if fchrg > 0.0 and self.info['lloc'] is not None :
            data = [line.split()[1:4] for line in lines[ibegin:iend]]
            data = np.asarray(data, dtype = np.float64)
            core_density = data[:, 2]
            core_density = np.asarray(core_density, dtype = np.float64)
            core_density /= (4.0 * np.pi)
            self._core_density_grid = data[:,0]
            self._core_density = core_density
        elif fchrg > 0.0 and self.infor['lloc'] is None:
            ibegin = 6+ (mmax + 1) * lloc + mmax
            iend = ibegin + mmax
            core_density = [line.split()[1:3] for line in lines[ibegin:iend]]
            core_density = np.asarray(core_density, dtype = np.float64)
            core_density[:, 1] /= (4.0 * np.pi)
            self._core_density_grid = core_density[:, 0]
            self._core_density = core_density[:,1]

        else : 
            data = [line.split()[1:3] for line in lines[ibegin:iend]]
            data = np.asarray(data, dtype = np.float64)

        self.r = data[:, 0]
        self.v = data[:, 1]
        self._zval = self.info['zval']
        
        #if fchrg > 0.0 :
        #    ibegin = 6+ (mmax + 1) * lloc + mmax
        #    iend = ibegin + mmax
        #    core_density = [line.split()[1:3] for line in lines[ibegin:iend]]
        #    core_density = np.asarray(core_density, dtype = np.float64)
        #    core_density[:, 1] /= (4.0 * np.pi)
        #    self._core_density_grid = core_density[:, 0]
        #    self._core_density = core_density[:,1]

    def _read_psp6(self, lines):
        info = self.info
        info['info'] = lines[:18]
        mmax = info['mmax']
        lmax = info['lmax']
        fchrg = info['fchrg']
        if lmax > 0 :
            raise ValueError("Only support local PP now (psp6).")

        # line 5-18 skip
        # line 19 mmax dx skip
        ibegin = 19
        iend = ibegin + mmax
        data = [line.split()[1:4] for line in lines[ibegin:iend]]
        data = np.asarray(data, dtype = np.float64)

        self.r = data[:, 0]
        self.v = data[:, 2]
        self._zval = self.info['zval']

        if fchrg > 0.0 :
            i1 = iend
            i2 = i1 + mmax
            core_density = [line.split()[0:2] for line in lines[i1:i2]]
            core_density = np.asarray(core_density, dtype = np.float64)
            core_density[:, 1] /= (4.0 * np.pi)
            self._core_density_grid = core_density[:, 0]
            self._core_density = core_density[:,1]

    def write(self, fname, header = 'DFTpy'):
        info_default = {
                'atomicnum' : 'none',
                'zval' : 'none',
                'date' : datetime.date.today().strftime("%m%d%Y"),
                'pspcod' : 8,
                'pspxc' : 2,
                'lmax' : 0,
                'lloc' : 0,
                'mmax' : 'none',
                'r2well' : 0,
                'rchrg' : 0,
                'fchrg' : -1,
                'qchrg' : 0,
                'nproj' : [0, 0, 0, 0, 0],
                'extension_switch' : [0],
                }
        comments = [
            "zatom,zion,pspd",
            "pspcod,pspxc,lmax,lloc,mmax,r2well",
            "rchrg fchrg qchrg",
            "nproj",
            "extension_switch"]
        info_default.update(self.info)
        info = info_default
        info['mmax'] = len(self.r)
        if self._zval is not None : info['zval'] = self._zval
        for k, v in info.items():
            if v == 'none' :
                raise AttributeError(f"Missing value of {k}")
        with open(fname, 'w') as fh:
            sp = ' '*4
            fh.write(header + '\n')
            fh.write(f"{info['atomicnum']:.4f}{sp}")
            fh.write(f"{info['zval']:.4f}{sp}")
            fh.write(f"{info['date']:s}{sp}")
            fh.write(f"{comments[0]}\n")
            fh.write(f"{info['pspcod']:d}{sp}")
            fh.write(f"{info['pspxc']:d}{sp}")
            fh.write(f"{info['lmax']:d}{sp}")
            fh.write(f"{info['lloc']:d}{sp}")
            fh.write(f"{info['mmax']:d}{sp}")
            fh.write(f"{info['r2well']:d}{sp}")
            fh.write(f"{comments[1]}\n")
            fh.write(f"{info['rchrg']:6f}{sp}")
            fh.write(f"{info['fchrg']:6f}{sp}")
            fh.write(f"{info['qchrg']:6f}{sp}")
            fh.write(f"{comments[2]}\n")
            for item in info['nproj'] :
                fh.write(f"{item:d}{sp}")
            fh.write(f"{comments[3]}\n")
            for item in info['extension_switch'] :
                fh.write(f"{item:d}{sp}")
            fh.write(f"{comments[4]}\n")
            fh.write('4\n')
                        
            for i, (r, v, core_density) in enumerate(zip(self.r, self.v, self.core_density)):
                fh.write(f"{i+1:<4d}  {r:.13e} {v:.13e} {core_density:.13}\n")
