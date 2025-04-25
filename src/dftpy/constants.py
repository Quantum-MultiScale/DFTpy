import os
import sys
from ase import units as Units
try:
    import pyfftw
    FFTLIB = "pyfftw"
except Exception:
    FFTLIB = "numpy"
# _Grav = Units._Grav
# _Nav = Units._Nav
# _amu = Units._amu
# _auf = Units._auf
# _aup = Units._aup
# _aut = Units._aut
# _auv = Units._auv
# _c = Units._c
# _e = Units._e
# _eps0 = Units._eps0
# _hbar = Units._hbar
# _hplanck = Units._hplanck
# _k = Units._k
# _me = Units._me
# _mp = Units._mp
# _mu0 = Units._mu0
# alpha = Units.alpha
# eV = Units.eV
# fs = Units.fs
# invcm = Units.invcm
# kB = Units.kB
# kJ = Units.kJ
# kcal = Units.kcal
# kg = Units.kg
# m = Units.m
# mol = Units.mol
# nm = Units.nm
# s = Units.s
# second = Units.second
# A = Units.A
# AUT = Units.AUT
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


LEN_CONV={"Angstrom" : {"Bohr": 1.0/Units.Bohr, "nm": 1.0e-1, "m": 1.0e-10}}
LEN_CONV = conv2conv(LEN_CONV)

ENERGY_CONV= {"Hartree": {"eV": Units.Ha}}
ENERGY_CONV = conv2conv(ENERGY_CONV)

FORCE_CONV = {"Ha/Bohr": {"eV/A" : Units.Ha/Units.Bohr}}
FORCE_CONV = conv2conv(FORCE_CONV)

STRESS_CONV = {"eV/A3" : {"GPa": 1.0/Units.GPa, "Ha/Bohr3" : Units.Bohr ** 3 / Units.Ha}}
STRESS_CONV = conv2conv(STRESS_CONV)

TIME_CONV = {"au" : {'s' : Units.AUT/Units.s, 'fs' : Units.AUT/Units.fs}}
TIME_CONV = conv2conv(TIME_CONV)

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
