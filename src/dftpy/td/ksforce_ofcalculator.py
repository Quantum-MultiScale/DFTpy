import numpy as np

from ase.calculators.calculator import Calculator, all_changes

from dftpy.constants import ENERGY_CONV, FORCE_CONV
from dftpy.field import DirectField
from dftpy.functional import Functional
from dftpy.ions import Ions
from dftpy.mpi import sprint
from dftpy.td.hamiltonian import Hamiltonian
from dftpy.td.predictor_corrector import PredictorCorrector
from dftpy.td.propagator import Propagator
from dftpy.utils.utils import calc_rho, calc_j


class KS_Forces():
    def __init__(self, qepy_calculator, atoms):
        self.qepy_calculator = qepy_calculator
        self.atoms = atoms

    def get_forces(self):
        self.qepy_calculator.update_atoms(self.atoms)
        self.atoms.calc = qepy_calculator
        forces = self.atoms.get_forces()
        return forces 


