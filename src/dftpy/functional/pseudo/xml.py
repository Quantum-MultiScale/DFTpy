import numpy as np

from dftpy.functional.pseudo.abstract_pseudo import BasePseudo


class GPAWXML(BasePseudo):
    '''
    Use the `gpaw` module to read xml-format pseudopotential
    '''

    def __init__(self, fname, direct=True, **kwargs):
        super().__init__(fname, direct=direct, **kwargs)

    def read(self, fname, symbol='Al', xctype='LDA', name='paw'):
        from gpaw.setup_data import PAWXMLParser, SetupData

        self.info = SetupData(symbol, xctype, name=name, readxml=False)
        fd = open(fname, 'rb')
        PAWXMLParser(self.info).parse(source=fd.read(), world=None)
        fd.close()
        nj = len(self.info.l_j)
        self.info.e_kin_jj.shape = (nj, nj)
        self.r = self.info.rgd.r_g
        self.v = self.vbar_g
        self._zval = self.info.Nv
        self._core_density = self.info.nct_g
        self._atomic_density = self.info.nvt_g


class PAWXML(BasePseudo):
    '''
    Ref : https://wiki.fysik.dtu.dk/gpaw/setups/pawxml.html
    '''

    def __init__(self, fname, direct=True, **kwargs):
        super().__init__(fname, direct=direct, **kwargs)

    def read(self, fname):
        import xml.etree.ElementTree as ET

        tree = ET.iterparse(fname, events=['start', 'end'])
        for event, elem in tree:
            if event == 'end':
                if elem.tag in ['radial_grid']:
                    self.r = self.get_radial_grid(elem.attrib)
                elif elem.tag in ['zero_potential']:
                    self.v = np.fromstring(elem.text, dtype=float, sep=" ")
                elif elem.tag in ['atom']:
                    self._zval = float(elem.attrib['valence'])
                elif elem.tag in ['pseudo_valence_density']:
                    self._atomic_density = np.fromstring(
                        elem.text, dtype=float, sep=" "
                    )
                elif elem.tag in ['pseudo_core_density']:
                    self._core_density = np.fromstring(elem.text, dtype=float, sep=" ")

    def get_radial_grid(self, dicts):
        istart = int(dicts['istart'])
        iend = int(dicts['iend'])
        x = np.arange(istart, iend + 1, dtype='float')
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
            r = (x / n + a) ** 5 / a - a**4
        else:
            raise AttributeError("!ERROR : not support eq= ", eq)
        return r
