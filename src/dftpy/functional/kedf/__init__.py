# Collection of Kinetic Energy Density Functionals
import copy

import numpy as np

from dftpy.functional.abstract_functional import AbstractFunctional
from dftpy.functional.functional_output import FunctionalOutput, ZeroFunctional
from dftpy.functional.kedf.gga import GGA, GGA_KEDF_list, GGAFs, MGGA, \
    MGGA_KEDF_list, GGAStress
from dftpy.functional.kedf.hc import HC, revHC
from dftpy.functional.kedf.lwt import LWT, LMGP, LMGPA, LMGPG
from dftpy.functional.kedf.mgp import MGP, MGPA, MGPG, MGP0
from dftpy.functional.kedf.tf import TF, TTF, ThomasFermiStress
from dftpy.functional.kedf.vw import vW, vonWeizsackerStress
from dftpy.functional.kedf.wt import WT, WTStress
from dftpy.functional.kedf.sm import SM, SMStress
from dftpy.functional.kedf.fp import FP, FPStress
from dftpy.functional.kedf.wte import WTE, WTEStress
from dftpy.functional.semilocal_xc import LibXC
from dftpy.functional.fedf.ft_tf import FT_TF, FT_TFStress
from dftpy.functional.fedf.ft_vW import FT_vW, FT_vWStress
from dftpy.functional.fedf.ft_gga import FT_GGA, FT_GGAStress
from dftpy.functional.fedf.ft_wt import FT_WT
from dftpy.mpi import sprint
from dftpy.utils import name2functions
from dftpy.functional.kedf.lkt import LKT

__all__ = ["KEDF", "MIXGGAS", "NLGGA",
           "kedf2mixkedf", "kedf2nlgga",
           "KEDFEngines", "KEDFEngines_Stress"]

KEDFEngines = {
    "NONE": ZeroFunctional,
    "TF": TF,
    "VW": vW,
    "LKT": LKT,
    "GGA": GGA,
    "MGGA": MGGA,
    "WT-NL": WT,
    "SM-NL": SM,
    "FP-NL": FP,
    "MGP-NL": MGP,
    "MGPA-NL": MGPA,
    "MGPG-NL": MGPG,
    "LWT-NL": LWT,
    "LMGP-NL": LMGP,
    "LMGPA-NL": LMGPA,
    "LMGPG-NL": LMGPG,
    "HC-NL": HC,
    "REVHC-NL": revHC,
    "DTTF": TTF,
    "LIBXC": LibXC,
    "WTE-NL": WTE,
    #
    "X_TF_Y_VW": ("TF", "VW"),
    "XTFYVW": ("TF", "VW"),
    "TFVW": ("TF", "VW"),
    "WT": ("TF", "VW", "WT-NL"),
    "SM": ("TF", "VW", "SM-NL"),
    "FP": ("TF", "VW", "FP-NL"),
    "MGP": ("TF", "VW", "MGP-NL"),
    "MGPA": ("TF", "VW", "MGPA-NL"),
    "MGPG": ("TF", "VW", "MGPG-NL"),
    "LWT": ("TF", "VW", "LWT-NL"),
    "LMGP": ("TF", "VW", "LMGP-NL"),
    "LMGPA": ("TF", "VW", "LMGPA-NL"),
    "LMGPG": ("TF", "VW", "LMGPG-NL"),
    "HC": ("TF", "VW", "HC-NL"),
    "REVHC": ("TF", "VW", "REVHC-NL"),
    "TTF": TF,
    "MGP0-NL": MGP0,
    "MGP0": ("TF", "VW", "MGP0-NL"),
    "WTE": ("VW", "WTE-NL"),
    ## finite temperature
    "FT_TF": FT_TF,
    "FT_VW": FT_vW,
    "FT_TFVW": ("FT_TF", "FT_VW"),
    "FT_GGA": FT_GGA,
    "FT_WTNL": FT_WT,
    "FT_WT": ("FT_TF", "VW", "FT_WTNL"),
}


def LibXCStress(density, energy=0, **kwargs):
    stress = np.eye(3) * energy / density.grid.volume
    stress += LibXC(density, calcType=["S"], **kwargs).stress
    return stress


KEDFEngines_Stress = {
    "TF": ThomasFermiStress,
    "VW": vonWeizsackerStress,
    "WT-NL": WTStress,
    "SM-NL": SMStress,
    "FP-NL": FPStress,
    "WTE-NL": WTEStress,
    "GGA": GGAStress,
    "LIBXC": LibXCStress,
    "X_TF_Y_VW": ("TF", "VW"),
    "XTFYVW": ("TF", "VW"),
    "TFVW": ("TF", "VW"),
    "WT": ("TF", "VW", "WT-NL"),
    "SM": ("TF", "VW", "SM-NL"),
    "FP": ("TF", "VW", "FP-NL"),
    "WTE": ("VW", "WTE-NL"),
    ## finite temperature 
    "FT_TF": FT_TFStress,
    "FT_VW": FT_vWStress,
    "FT_TFVW": ("FT_TF", "FT_VW"),
    "FT_GGA": FT_GGAStress
}


class KEDF(AbstractFunctional):
    def __init__(self, name="WT", kedf=None, **kwargs):
        self.type = 'KEDF'
        self.name = kedf or name
        self.options = kwargs
        self.options['kedf'] = self.name
        self.ke_kernel_saved = {
            "Kernel": None,
            "rho0": 0.0,
            "shape": None,
            "KernelTable": None,
            "etamax": None,
            "KernelDeriv": None,
            "MGPKernelE": None,
            "kfmin": None,
            "kfmax": None,
        }
        self.energies = {}

    def __new__(cls, name="WT", **kwargs):
        if name == 'SMP21':
            name = kwargs['kedf'] = 'MIX_TF+GGA_REVAPBEK'
        if name.startswith('STV+GGA+'):
            return kedf2nlgga(name, **kwargs)
        elif name.startswith('MIX_'):
            return kedf2mixkedf(name, **kwargs)
        else:
            return super(KEDF, cls).__new__(cls)

    def compute(self, density, calcType={"E", "V"}, name=None, kedf=None,
                split=False, **kwargs):
        if kedf: name = kedf
        if name is None: name = self.name
        name = name.upper()
        options = copy.deepcopy(self.options)
        options.update(kwargs)
        # -----------------------------------------------------------------------
        k_str = options.pop('k_str', None)  # For GGA functional
        if name.startswith('LIBXC'):
            options['libxc'] = list(k_str.split())
            name = 'LIBXC'
        elif name.startswith('GGA_'):
            k_str = name[4:]
            options['mgga'] = False
            name = 'GGA'
        elif name.startswith('MGGA_'):
            k_str = name[5:]
            options['mgga'] = True
            name = 'MGGA'
        options['functional'] = k_str
        options = {k: v for k, v in options.items() if v is not None}
        # -----------------------------------------------------------------------
        functional = {}
        if density.ndim > 3:
            nspin = density.rank
            rhos = density * nspin
        else:
            nspin = 1
            rhos = [density]
        names = name2functions(name, KEDFEngines)
        # -----------------------------------------------------------------------
        temperature = options.get('temperature0', None) or options.get(
            'temperature', None)
        if not name.startswith('FT_'):
            if temperature: names = {**names, 'DTTF': TTF}
        # -----------------------------------------------------------------------
        for rho in rhos:
            for key in names:
                func = KEDFEngines.get(key, None)
                if func is None:
                    raise AttributeError("%s KEDF to be implemented" % name)

                out = func(rho, calcType=calcType,
                           ke_kernel_saved=self.ke_kernel_saved, **options)

                if not split: key = 'KEDF'
                if key not in functional:
                    functional[key] = out
                else:
                    functional[key] += out
        # -----------------------------------------------------------------------
        if nspin > 1:
            """
            Note :
                For polarization case, same potential for both.
            """
            for key, out in functional.items():
                out = out / nspin
                if 'V' in calcType:
                    out.potential = out.potential.tile((nspin, 1, 1, 1))
                if 'D' in calcType:
                    out.energydensity = out.energydensity.tile((nspin, 1, 1, 1))
                functional[key] = out
        # -----------------------------------------------------------------------
        if split:  # Save energies for stress
            if 'E' in calcType:
                for key, value in functional.items():
                    self.energies[key] = value.energy
        else:
            functional = functional[key]
        return functional

    def stress(self, density, name=None, kedf=None, split=False, **kwargs):
        if kedf: name = kedf
        if name is None: name = self.name
        name = name.upper()
        # -----------------------------------------------------------------------
        options = copy.deepcopy(self.options)
        options.update(kwargs)
        k_str = options.pop('k_str', None)  # For GGA functional
        if name.startswith('LIBXC'):
            options['libxc'] = list(k_str.split())
            name = 'LIBXC'
        elif name.startswith('GGA_'):
            k_str = name[4:]
            options['mgga'] = False
            name = 'GGA'
        elif name.startswith('MGGA_'):
            k_str = name[5:]
            options['mgga'] = True
            name = 'MGGA'
        options['functional'] = k_str
        options = {k: v for k, v in options.items() if v is not None}
        # -----------------------------------------------------------------------
        funcs = name2functions(name, KEDFEngines_Stress)
        if len(self.energies) < len(funcs):
            self.compute(density, calcType={"E"}, name=name, split=True,
                         **options)
        if density.ndim > 3:
            nspin = density.rank
            rhol = density * nspin
        else:
            nspin = 1
            rhol = [density]

        stress = np.zeros((3, 3))
        out = {}
        for k, func in funcs.items():
            if split: stress = np.zeros((3, 3))
            energy = self.energies[k]
            for i in range(0, nspin):
                stress += func(rhol[i], energy=energy, **options) / nspin
            if split: out[k] = stress
        if split: stress = out
        return stress


def KEDFStress(rho, name="WT", energy=None, **kwargs):
    """
    KEDF stress interface (deprecated)
    """
    KEDF_Stress_Dict = {
        "TF": ThomasFermiStress,
        "VW": vonWeizsackerStress,
        "WT": WTStress,
    }

    if name in KEDF_Stress_Dict:
        func = KEDF_Stress_Dict[name]

    stress = np.zeros((3, 3))
    if rho.ndim > 3:
        nspin = rho.rank
        rhol = rho
    else:
        nspin = 1
        rhol = [rho]

    for i in range(0, nspin):
        stress += func(rhol[i] * nspin, energy=energy, **kwargs) / nspin

    return stress


class NLGGA(AbstractFunctional):
    def __init__(self, stv=None, gga=None, nl=None, rhomax=None,
                 name='STV+GGA+LMGPA'):
        self.type = 'KEDF'
        self.stv = stv
        self.gga = gga
        self.nl = nl
        self.rhomax = rhomax
        self.level = 3  # if smaller than 3 only use gga to guess the rhomax
        self.name = name.upper()

    def compute(self, density, calcType={"E", "V"}, **kwargs):
        calcType = {"E", "V"}
        # T_{s}[n]=\int \epsilon[n](\br) d\br=\int W[n](\br)\left[\epsilon_{NL} [n] + \epsilon_{STV}(n)\right] + \bigg(1-W[n](\br)\bigg)\epsilon_{GGA} d\br
        if 'V' in calcType:
            calc = {'D', 'V'}
        elif 'E' in calcType:
            calc = {'D'}

        split = kwargs.get('split', False)
        kwargs['split'] = False

        rhomax_w = density.amax()

        nmax = kwargs.get('rhomax', None) or self.rhomax
        if not nmax: nmax = rhomax_w
        kfmax = 2.0 * np.cbrt(3.0 * np.pi ** 2 * nmax)
        kwargs['kfmax'] = kfmax

        func_gga = self.gga(density, calcType=calc, **kwargs)

        # truncate the density higher than rhomax for NL
        if rhomax_w > nmax:
            sprint('!WARN : some density large than rhomax', rhomax_w, nmax,
                   comm=density.mp.comm, level=1)

        mask = density > nmax
        if np.count_nonzero(mask) > 0:
            density_trunc = density.copy()
            density_trunc[mask] = nmax
        else:
            density_trunc = density

        if self.level > 2:
            func_stv = self.stv(density, calcType=calc, **kwargs)
            func_nl = self.nl(density_trunc, calcType=calc, **kwargs)
            kdd = kwargs.get('kdd', self.nl.options.get('kdd', 1))
            if kdd < 3:
                func_nl.energydensity = 3.0 / 8.0 * density_trunc * func_nl.potential

        wn = density_trunc / nmax

        obj = FunctionalOutput(name='NLGGA')

        if 'V' in calcType:
            # V_{STV} = W[n]v_{STV} + \frac{\epsilon_{STV}}{n_{max}}
            # V_{GGA} = \left(1-W[n]\right)v_{GGA} -\frac{\epsilon_{GGA}}{n_{max}}
            # V_{NL} =W[n]v_{NL} +\frac{\epsilon_{NL}}{n_{max}}
            if self.level > 2:
                pot = wn * func_stv.potential + func_stv.energydensity / nmax
                pot += (
                                   1 - wn) * func_gga.potential - func_gga.energydensity / nmax
                pot += wn * func_nl.potential + func_nl.energydensity / nmax
            else:
                pot = func_gga.potential
            obj.potential = pot

        if 'E' in calcType:
            # E_{STV} =\int  W[n(\br)] \epsilon_{STV}(\br) d\br
            # E_{GGA} =\int (1-W[n](\br))\epsilon_{GGA}(\br)
            # E_{NL} =\int \epsilon_{NL} W[n](\br) d\br .
            if self.level > 2:
                energydensity = wn * func_stv.energydensity
                energydensity += (1 - wn) * func_gga.energydensity
                energydensity += wn * func_nl.energydensity
            else:
                energydensity = func_gga.energydensity
            energy = energydensity.sum() * density.grid.dV
            obj.energy = energy

        if split: obj = {'NG': obj}
        return obj


def kedf2nlgga(name='STV+GGA+LMGPA', **kwargs):
    """
    Base on the input generate NLKEDF

    Args:
        name: the name of NLGGA

    Raises:
        AttributeError: Must contain three parts in the name

    """

    kedf = kwargs.get('kedf', None)
    if kedf:
        name = kedf
        del kwargs['kedf']
    name = name.upper()

    if not name.startswith('STV+GGA+'):
        raise AttributeError(
            "The name of NLGGA is not correct : {}".format(name))
    names = name.split('+')

    # Only for STV
    params = kwargs.get("params", None)
    if params is not None: del kwargs["params"]

    # Only for GGA
    k_str = kwargs.pop("k_str", None)
    if not k_str: k_str = "REVAPBEK"

    # Remove TF and vW from NL
    for key in ['x', 'y']:
        kwargs[key] = 0.0

    sigma = kwargs.get("sigma", None)
    rhomax = kwargs.get("rhomax", None)

    if k_str.upper() in GGA_KEDF_list:
        names[1] = 'GGA'
        k_str = k_str.upper()
    else:
        names[1] = 'LIBXC'
        k_str = k_str.lower()

    stv = KEDF(name='GGA', k_str='STV', params=params, sigma=sigma)
    gga = KEDF(name=names[1], k_str=k_str, sigma=sigma)
    # -----------------------------------------------------------------------
    if names[2] == 'HC':
        kwargs['k_str'] = 'PBE2'
        kwargs['params'] = [0.1, 0.45]
        kwargs['delta'] = 0.3
    # -----------------------------------------------------------------------
    nl = KEDF(name=names[2], **kwargs)
    obj = NLGGA(stv, gga, nl, rhomax=rhomax, name=name)

    return obj


def kedf2mixkedf(name='MIX_TF+GGA', first_high=True, **kwargs):
    """
    Base on the input generate MIXKEDF

    Args:
        name: the name of MIXKEDF

    Raises:
        AttributeError: The name must startswith 'MIX_'

    """

    kedf = kwargs.get('kedf', None)
    if kedf:
        name = kedf
        del kwargs['kedf']
    name = name.upper()

    if not name.startswith('MIX_'):
        raise AttributeError(
            "The name of MIXKEDF is not correct : {}".format(name))
    names = name[4:].split('+')

    # Only for second(GGA)
    k_str = kwargs.pop("k_str", None)
    if not k_str: k_str = "REVAPBEK"

    sigma = kwargs.get("sigma", None)
    rhomax = kwargs.get("rhomax", None)

    if k_str.upper() in GGA_KEDF_list:
        names[1] = 'GGA'
        k_str = k_str.upper()
    else:
        names[1] = 'LIBXC'
        k_str = k_str.lower()

    stv = KEDF(name=names[0], **kwargs)
    gga = KEDF(name=names[1], k_str=k_str, sigma=sigma)
    obj = MIXGGAS(stv, gga, rhomax=rhomax, name=name)

    return obj


class MIXGGAS(AbstractFunctional):
    def __init__(self, stv=None, gga=None, rhomax=None, name='MIX_TF+GGA'):
        self.type = 'KEDF'
        self.stv = stv
        self.gga = gga
        self.rhomax = rhomax
        self.name = name.upper()

    def compute(self, density, calcType={"E", "V"}, **kwargs):
        calcType = {"E", "V"}
        if 'V' in calcType:
            calc = {'D', 'V'}
        elif 'E' in calcType:
            calc = {'D'}

        split = kwargs.get('split', False)
        kwargs['split'] = False

        func_stv = self.stv(density, calcType=calc, **kwargs)
        func_gga = self.gga(density, calcType=calc, **kwargs)

        obj = FunctionalOutput(name='MIXGGAS')

        self.interpfunc(density, calcType=calcType, **kwargs)
        interpolate_f = self.interpolate_f
        interpolate_df = self.interpolate_df

        if 'V' in calcType:
            pot = interpolate_f * func_stv.potential + interpolate_df * func_stv.energydensity
            pot += (
                               1 - interpolate_f) * func_gga.potential - interpolate_df * func_gga.energydensity
            obj.potential = pot

        if 'E' in calcType:
            energydensity = interpolate_f * func_stv.energydensity
            energydensity += (1 - interpolate_f) * func_gga.energydensity
            energy = energydensity.sum() * density.grid.dV
            obj.energy = energy

        if split: obj = {'MG': obj}
        return obj

    def interpfunc(self, rho, calcType={"E", "V"}, func='tanh', **kwargs):
        if self.rhomax is None or self.rhomax < 1E-30:
            self.interpolate_f = 1.0
            self.interpolate_df = 0.0
        elif self.rhomax > 100:
            self.interpolate_f = 0.0
            self.interpolate_df = 0.0
        else:
            if func == 'tanh':
                self.interp_tanh(rho, calcType=calcType, **kwargs)
            else:
                self.interp_linear(rho, calcType=calcType, **kwargs)
        return

    def interp_tanh(self, rho, calcType={"E", "V"}, **kwargs):
        # x = -2 * np.abs(rho/self.rhomax)
        # ex = np.exp(x)
        # self.interpolate_f = (1 - ex)/(1 + ex)
        x = np.abs(rho / self.rhomax)
        self.interpolate_f = np.tanh(x)
        self.interpolate_df = 1.0 / (np.cosh(x) ** 2 * self.rhomax)
        return

    def interp_linear(self, rho, calcType={"E", "V"}, **kwargs):
        self.interpolate_f = np.abs(rho / self.rhomax)
        mask = self.interpolate_f > 1.0
        self.interpolate_f[mask] = 1.0
        self.interpolate_df = np.ones_like(self.interpolate_f) / self.rhomax
        self.interpolate_df[mask] = 0.0
        return

    def _interp_lkt(self, rho, calcType={"E", "V"}, **kwargs):
        sigma = kwargs.get('sigma', None)
        rhom = rho.copy()
        tol = 1e-16
        rhom[rhom < tol] = tol

        rho23 = rhom ** (2.0 / 3.0)
        rho43 = rho23 * rho23
        rho83 = rho43 * rho43
        q = rho.grid.get_reciprocal().q
        g = rho.grid.get_reciprocal().g

        rhoG = rho.fft()
        rhoGrad = []
        for i in range(3):
            if sigma is None:
                grhoG = g[i] * rhoG * 1j
            else:
                grhoG = g[i] * rhoG * np.exp(-q * (sigma) ** 2 / 4.0) * 1j
            item = (grhoG).ifft(force_real=True)
            rhoGrad.append(item)
        s = np.sqrt(rhoGrad[0] ** 2 + rhoGrad[1] ** 2 + rhoGrad[2] ** 2) / rho43
        F, dFds2 = GGAFs(s, functional='LKT', calcType=calcType,
                         gga_remove_vw=True, **kwargs)

        if 'V' in calcType:
            dFdn = -4.0 / 3.0 * dFds2 * s * s / rhom
            p3 = []
            for i in range(3):
                item = dFds2 * rhoGrad[i] / rho83
                p3.append(item.fft())
            pot3G = g[0] * p3[0] + g[1] * p3[1] + g[2] * p3[2]
            dFdn -= (1j * pot3G).ifft(force_real=True)

        self.interpolate_s = s
        self.interpolate_f = F
        self.interpolate_df = dFdn
