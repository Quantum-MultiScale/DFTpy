import os
import sys
from ase.units import create_units
try:
    import pyfftw
    FFTLIB = "pyfftw"
except Exception:
    FFTLIB = "numpy"

codata_options = ['1986', '1998', '2002', '2006', '2010', '2014', '2018', '2022']
__codata_version__ = "2014"
Units = create_units(__codata_version__)
# _Grav = Units._Grav  #gravitational constant
# _Nav = Units._Nav    #Avogadro's number
# _amu = Units._amu    #atomic mass unit
# _auf = Units._auf    #atomic unit of force
# _aup = Units._aup    #atomic unit of pressure
# _aut = Units._aut    #atomic unit of time
# _auv = Units._auv    #atomic unit of volume
# _c = Units._c        #speed of light
# _e = Units._e        #elementary charge
# _eps0 = Units._eps0  #vacuum permittivity
# _hbar = Units._hbar  #Planck constant / 2pi, J s
# _hplanck = Units._hplanck  # Planck constant, J s
# _k = Units._k      Boltzmann constant, J/K
# _me = Units._me    #electron mass
# _mp = Units._mp    #proton mass
# _mu0 = Units._mu0  #vacuum permeability
# alpha = Units.alpha  #fine structure constant
# eV = Units.eV   #electron volt
# fs = Units.fs   #femtosecond
# invcm = Units.invcm  #inverse cm
# kB = Units.kB   Boltzmann constant, eV/K
# kJ = Units.kJ   #kilojoule
# kcal = Units.kcal  #kilocalorie
# kg = Units.kg   #kilogram
# m = Units.m     #meter
# mol = Units.mol  #mole
# nm = Units.nm   #nanometer
# s = Units.s     #second
# second = Units.second #second
# A = Units.A     #Ampere
# AUT = Units.AUT  #atomic unit of time
# Ang = Units.Ang  
# Angstrom = Units.Angstrom
# Bohr = Units.Bohr
# C = Units.C
# Debye = Units.Debye
# GPa = Units.GPa
# Ha = Units.Ha
# Hartree = Units.Hartree
# J = Units.J
# Pascal = Units.Pascal
# bar = Units.bar
# Ry = Units.Ry
# Rydberg = Units.Rydberg

def conv2conv(conv, base = None):
    if base is None : base = list(conv.keys())[0]
    conv[base][base] = 1.0
    ref = conv[base]
    for key in ref :
        if key == base : continue
        conv[key] = {}
        for key2 in ref :
            conv[key][key2] = ref[key2]/ref[key]
    return conv

###
LEN_CONV={"Angstrom" : {"Bohr": 1.0/Units.Bohr, "nm": 1.0e-1, "m": 1.0e-10}}
LEN_CONV = conv2conv(LEN_CONV)

ENERGY_CONV= {"eV": {"Hartree": 1/Units.Ha, "kJ": 1/Units.kJ, "kcal": 1/Units.kcal, "Ry": 2/Units.Ha, "J": 1/Units.J}}
ENERGY_CONV = conv2conv(ENERGY_CONV)

FORCE_CONV = {"eV/A": {"Hartree/Bohr" : Units.Bohr/Units.Ha, "eV/Bohr": 1/Units.Ha, "Ha/A": Units.Bohr}}
FORCE_CONV = conv2conv(FORCE_CONV)

STRESS_CONV = {"eV/A3" : {"GPa": 1.0/Units.GPa, "Ha/Bohr3" : Units.Bohr ** 3 / Units.Ha, "Pascal": 1.0/Units.Pascal, "bar": 1.0/Units.bar, "Ha/A3": 1.0 / Units.Ha}}
STRESS_CONV = conv2conv(STRESS_CONV)

TIME_CONV = {"ase" : {'s' : 1/Units.s, 'fs' : 1/Units.fs, 'au' : 1 / Units.AUT}}
TIME_CONV = conv2conv(TIME_CONV)

VELOCITY_CONV = {"ase": {"Bohr/au": Units.AUT/Units.Bohr, "m/s": 1.0e-10/Units.s}}
VELOCITY_CONV = conv2conv(VELOCITY_CONV)

CHARGE_CONV = {"C" : {"e": Units.C}}
CHARGE_CONV = conv2conv(CHARGE_CONV)

SPEED_OF_LIGHT = 1.0/Units.alpha
C_TF = 2.87123400018819181594
TKF0 = 6.18733545256027186194
CBRT_TWO = 1.25992104989487316477
ZERO = 1E-30

environ = {} # You can change it anytime you want
environ['STDOUT'] = sys.stdout # file descriptor of sprint
environ['LOGLEVEL'] = int(os.environ.get('DFTPY_LOGLEVEL', 2)) # The level of sprint
"""
    0 : all
    1 : debug
    2 : info
    3 : warning
    4 : error
"""
environ['FFTLIB'] = os.environ.get('DFTPY_FFTLIB', FFTLIB)
# DFTpy old units
# Units.Bohr = 0.5291772106712
# Units.Ha = 27.2113834279111
