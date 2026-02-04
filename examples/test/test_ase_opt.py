#!/usr/bin/env python3
import unittest
import numpy as np

from dftpy.config import DefaultOption, OptionFormat
from dftpy.api.api4ase import DFTpyCalculator
from common import dftpy_data_path

class Test(unittest.TestCase):
    def test_opt(self):
        from ase.optimize import BFGS, LBFGS, FIRE
        from ase.optimize.sciopt import SciPyFminBFGS, SciPyFminCG
        from ase.filters import StrainFilter, UnitCellFilter
        from ase.io.trajectory import Trajectory
        from ase import units
        from ase.io import read, write
        conf = DefaultOption()
        conf['PATH']['pppath'] = dftpy_data_path
        conf['PP']['Al'] = 'al.lda.recpot'
        conf['JOB']['calctype'] = 'Energy Force Stress'
        conf['OPT']['method'] = 'TN'
        conf['OUTPUT']['stress'] = False
        conf['EXC']['xc'] = 'LDA'
        conf = OptionFormat(conf)
        path = dftpy_data_path
        atoms = read(path / 'fcc.vasp')
        calc = DFTpyCalculator(config = conf)
        atoms.calc = calc
        af = StrainFilter(atoms)
        opt = SciPyFminCG(af)
        opt.run(fmax = 0.001)
        atoms_final = read(filename=dftpy_data_path / 'ase_opt.traj',format='traj')
        self.assertTrue(np.isclose(atoms.positions, atoms_final.positions, atol=1.e-3).all())


if __name__ == "__main__":
    unittest.main()
