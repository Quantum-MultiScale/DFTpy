import os
import importlib.util
import re
import numpy as np
from dftpy.constants import Units
from dftpy.functional.pseudo.abstract_pseudo import BasePseudo

class PP(BasePseudo):
    def __init__(self, fname, direct = False, **kwargs):
        super().__init__(fname, direct = direct, **kwargs)

    def read(self, fname):
        engine = self._guess_format(fname)
        engine(fname)
        
    def _guess_format(self, fname):
        PPEngines = {
            "recpot" : self.read_recpot,
            "usp"    : self.read_usp,
            "uspcc"  : self.read_usp,
            "uspso"  : self.read_usp,
            "upf"    : self.read_upf,
            "psp"    : self.read_psp,
            "psp8"   : self.read_psp,
            "lps"    : self.read_psp,
            "psp6"   : self.read_psp,
            "fhi"    : self.read_psp,
            "cpi"    : self.read_psp,
            "xml"    : self.read_pawxml,
        }
        suffix = os.path.splitext(fname)[1][1:].lower()
        if suffix in PPEngines:
            return PPEngines[suffix]
        else:
            raise AttributeError("Pseudopotential '{}' is not supported".format(fname))
        
    def read_upf(self, fname, direct=True):
        self._direct = direct
        try:
            self._upf_loader_xml2dict(fname)
        except Exception:
            if importlib.util.find_spec("upf_to_json"):
                try:
                    self._upf_loader_upf2json(fname)
                except Exception:
                    raise ModuleNotFoundError("Please use a standard 'UPF' file.")
            else:
                raise ModuleNotFoundError("Please install 'upf_to_json' or use a standard 'UPF' file.")

    def read_psp(self, fname, direct=True):
        self._direct = direct
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
            return self._psp8_loader(lines)
        elif info['pspcod'] == 6:
            return self._psp6_loader(lines)
        else:
            raise ValueError("Only support psp8/psp6 format pseudopotential with psp")

    def read_recpot(self, fname, direct=False):
        self._direct = direct      
        HARTREE2EV = Units.Ha
        BOHR2ANG = Units.Bohr
        with open(fname, "r") as outfil:
            lines = outfil.readlines()
            
        comment = ''
        ibegin = 0
        for i, line in enumerate(lines):
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
        self.v = np.array(line.split()).astype(np.float64) / HARTREE2EV / BOHR2ANG ** 3
        self.r = np.linspace(0, gmax, num=len(self.v))
        self.info = {'comment': comment}
        self._zval = round((self.v[0] - self.v[1]) * (self.r[1] ** 2) / (4.0 * np.pi))
        if len(lines) > iend + 1 and len(lines[iend + 1].strip()) != 0:
            self._core_density = np.array(line.split()).astype(np.float64) * BOHR2ANG
        else:
            self._core_density = None
            
    def read_pawxml(self, fname, direct=True):
        self._direct = direct
        import xml.etree.ElementTree as ET
        tree = ET.iterparse(fname,events=['start', 'end'])
        for event, elem in tree:
            if event == 'end':
                if elem.tag in ['radial_grid'] :
                    self.r = self._get_xml_radical_grid(elem.attrib)
                elif elem.tag in ['zero_potential']:
                    self.v = np.fromstring(elem.text, dtype=float, sep=" ")
                elif elem.tag in ['atom'] :
                    self._zval = float(elem.attrib['valence'])
                elif elem.tag in ['pseudo_valence_density'] :
                    self._atomic_density = np.fromstring(elem.text, dtype=float, sep=" ")
                elif elem.tag in ['pseudo_core_density'] :
                    self._core_density = np.fromstring(elem.text, dtype=float, sep=" ")

    def read_usp(self, fname, direct=False):
        """
        !!! NOT FULLY TEST !!!
        Reads CASTEP-like usp PP file
        """
        self._direct = direct
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
                elif ibegin > 1 and (line.strip() == "1000" or len(line.strip()) == 1) and i - ibegin > 4:
                    iend = i
                    break
                elif ibegin<1 :
                    comment += line
            elif ext == 'uspso':
                if "END COMMENT" in line:
                    ibegin = i + 5
                elif ibegin > 1 and (line.strip() == "1000" or len(line.strip()) == 5) and i - ibegin > 4:
                    iend = i
                    break
                elif ibegin<1 :
                    comment += line

        line = " ".join([line.strip() for line in lines[ibegin:iend]])

        zval = float(lines[ibegin - 2].strip())

        if "1000" in lines[iend] or len(lines[iend].strip()) == 1 or len(lines[iend].strip()) == 5:
            pass
        else:
            raise AttributeError("Error : Check the PP file : {}".format(fname))
        gmax = float(lines[ibegin - 1].split()[0]) * BOHR2ANG

        # v = np.array(line.split()).astype(np.float64) / (HARTREE2EV*BOHR2ANG ** 3 * 4.0 * np.pi)
        self.v = np.array(line.split()).astype(np.float64) / (HARTREE2EV * BOHR2ANG ** 3)
        self.r = np.linspace(0, gmax, num=len(self.v))
        self.v[1:] -= zval * 4.0 * np.pi / self.r[1:] ** 2
        self.info = {'comment' : comment}
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
        
    def _upf_loader_xml2dict(self, fname):
        if importlib.util.find_spec("xmltodict"):
            import xmltodict
        else:
            raise ImportError("Please install xmltodict to read UPF files.")
            
        with open(fname) as fd:
            string = fd.read()
        doc = xmltodict.parse(string, attr_prefix='')
        info = doc[next(iter(doc.keys()))]
        
        def get_array(attr):
            pattern = re.compile(r'\s+')
            if isinstance(attr, dict): attr = attr['#text']
            if attr is None or len(attr) == 0: return None
            value = pattern.split(attr)
            return np.array(value, dtype=np.float64)
        
        r = get_array(info['PP_MESH']['PP_R'])
        v = get_array(info['PP_LOCAL']) * 0.5  # Ry to a.u.
        self.r = r
        self.v = v
        self.info = info
        self._zval = float(self.info["PP_HEADER"]["z_valence"])
        if 'PP_NLCC' in self.info:
            self._core_density = get_array(self.info['PP_NLCC'])
        if 'PP_RHOATOM' in self.info:
            rho = get_array(self.info['PP_RHOATOM'])[:self.r.size]
            if self.r[0] > 1E-10 :
                rho[:] /= (4*np.pi*self.r[:]**2)
            else :
                rho[1:] /= (4*np.pi*self.r[1:]**2)
            self._atomic_density = rho
            
    def _upf_loader_upf2json(self, fname):
        if importlib.util.find_spec("upf_to_json"):
            from upf_to_json import upf_to_json
        else:
            raise ImportError("Please install upf_to_json to read UPF files.")
        
        with open(fname, "r") as outfil:
            info = upf_to_json(upf_str=outfil.read(), fname=fname)["pseudo_potential"]
        r = np.array(info["radial_grid"], dtype=np.float64)
        v = np.array(info["local_potential"], dtype=np.float64)
        self.r = r
        self.v = v
        self.info = info
        self._zval = self.info["header"]["z_valence"]
        if 'core_charge_density' in self.info:
            self._core_density = np.array(self.info["core_charge_density"], dtype=np.float64)
        if 'total_charge_density' in self.info:
            rho = np.array(self.info["total_charge_density"], dtype=np.float64)[:self.r.size]
            if self.r[0] > 1E-10 :
                rho[:] /= (4*np.pi*self.r[:]**2)
            else :
                rho[1:] /= (4*np.pi*self.r[1:]**2)
            self._atomic_density = rho
            
    def _psp8_loader(self, lines):
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
        data = [line.split()[1:3] for line in lines[ibegin:iend]]
        data = np.asarray(data, dtype = np.float64)

        self.r = data[:, 0]
        self.v = data[:, 1]
        self._zval = self.info['zval']

        if fchrg > 0.0 :
            ibegin = 6+ (mmax + 1) * lloc + mmax
            iend = ibegin + mmax
            core_density = [line.split()[1:3] for line in lines[ibegin:iend]]
            core_density = np.asarray(core_density, dtype = np.float64)
            core_density[:, 1] /= (4.0 * np.pi)
            self._core_density_grid = core_density[:, 0]
            self._core_density = core_density[:,1]
            
    def _psp6_loader(self, lines):        
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
            
    def _get_xml_radical_grid(self, dict):
        istart = int(dicts['istart'])
        iend = int(dicts['iend'])
        x = np.arange(istart, iend + 1, dtype = 'float')
        eq = dicts['eq']
        if eq == 'r=d*i':
            d = float(dicts['d'])
            r = d * x
        elif eq == 'r=a*exp(d*i)':
            a = float(dicts['a'])
            d = int(dicts['d'])
            r = a * np.exp(d * x)
        elif eq == 'r=a*(exp(d*i)-1)':
            a = float(dicts['a'])
            d = float(dicts['d'])
            r = a * (np.exp(d * x) - 1)
        elif eq == 'r=a*i/(1-b*i)':
            a = float(dicts['a'])
            b = float(dicts['b'])
            r = a * x / (1 - b * x)
        elif eq == 'r=a*i/(n-i)':
            a = float(dicts['a'])
            n = int(dicts['n'])
            r = a * x / (n - x)
        elif eq == 'r=(i/n+a)^5/a-a^4':
            a = float(dicts['a'])
            n = int(dicts['n'])
            r = (x/n + a) ** 5/a - a ** 4
        else :
            raise AttributeError("!ERROR : not support eq= ", eq)
        return r