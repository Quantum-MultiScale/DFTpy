import numpy as np
from ase import units as ase_units
from dftd4.ase import DFTD4

from dftpy.constants import FORCE_CONV, STRESS_CONV
from dftpy.formats.ase_io import ions2ase
from dftpy.functional.functional_output import FunctionalOutput
from dftpy.functional.semilocal_xc import get_libxc_names, get_short_xc_name


class VDWDFTD4(object):
    def __init__(self, dftd4=None, ions=None, mp=None, **kwargs):
        xc = self.get_short_xc_name(dftd4=dftd4, **kwargs)
        self.dftd4calculator = DFTD4(method=xc.upper())
        self.restart(ions)
        if mp is None:
            from dftpy.mpi import MP

            mp = MP()
        self.mp = mp

    def restart(self, ions=None):
        self._ions = None
        self._energy = None
        self._forces = None
        self._stress = None
        if ions is not None:
            self.ions = ions  # initial set ions

    @property
    def ions(self):
        return self._ions

    @ions.setter
    def ions(self, value):
        self._ions = None
        self.check_restart(value)

    @staticmethod
    def get_short_xc_name(dftd4=None, **kwargs):
        if isinstance(dftd4, str) and dftd4.upper() != 'SAME':
            xc = dftd4
        else:
            libxc = get_libxc_names(**kwargs)
            xc = get_short_xc_name(libxc=libxc, code='dftd4')
        return xc

    def check_restart(self, ions=None):
        if self.ions and ions and ions == self.ions:
            return False
        else:
            if ions:
                self._ions = ions.copy()
                self.ase_atoms = ions2ase(ions)
            elif self._ions is None:
                raise AttributeError("Please set the ions.")
            return True

    def __call__(self, *args, **kwargs):
        return self.compute(*args, **kwargs)

    def compute(self, *args, calcType={'E'}, **kwargs):
        if 'E' in calcType:
            if self._energy is None:
                self._energy = self.get_energy(ions=self.ions)
        functional = FunctionalOutput(name='DFTD4', energy=self._energy)
        return functional

    def get_energy(self, ions=None, **kwargs):
        self.check_restart(ions)
        if self.mp.rank == 0:
            energy = self.dftd4calculator.get_potential_energy(self.ase_atoms)
        else:
            energy = 0.0
        return energy / ase_units.Hartree

    def get_forces(self, ions=None, **kwargs):
        self.check_restart(ions)
        if self.mp.rank == 0:
            forces = self.dftd4calculator.get_forces(self.ase_atoms)
        else:
            forces = np.zeros_like(self.ase_atoms.positions)
        return forces / (ase_units.Hartree / ase_units.Bohr)

    def get_stress(self, ions=None, **kwargs):
        self.check_restart(ions)
        if self.mp.rank == 0:
            vec = self.dftd4calculator.get_stress(self.ase_atoms)
        else:
            vec = np.zeros(6)
        s1, s2, s3, s4, s5, s6 = vec  # xx, yy, zz, yz, xz, xy
        stress = np.asarray([[s1, s6, s5], [s6, s2, s4], [s5, s4, s3]])
        stress /= self.mp.size
        return stress / (ase_units.Hartree / ase_units.Bohr**3)

    @property
    def energy(self):
        if self._energy is None:
            self._energy = self.get_energy(ions=self.ions)
        return self._energy

    @property
    def forces(self):
        if self._forces is None:
            self._forces = self.get_forces(ions=self.ions)
        return self._forces

    @property
    def stress(self):
        if self._stress is None:
            self._stress = self.get_stress(ions=self.ions)
        return self._stress
