import unittest
import numpy as np
import dftpy.formats.io as dftpy_io
from dftpy.optimization import Optimization
from dftpy.functional import Functional
from dftpy.functional.total_functional import TotalFunctional
from dftpy.grid import DirectGrid
from dftpy.field import DirectField
from dftpy.math_utils import ecut2nr
from dftpy.functional.pseudo import LocalPseudo
from common import dftpy_data_path

class Test(unittest.TestCase):
    def test_lpps(self):
        from dftpy.ions import Ions
        from dftpy.field import DirectField
        from dftpy.grid import DirectGrid
        from dftpy.functional import LocalPseudo, Functional, TotalFunctional
        from dftpy.formats import io
        from dftpy.optimization import Optimization
        from dftpy.mpi import sprint
        from dftpy.functional.pseudo.psp import PSP
        from dftpy.constants import environ
        from scipy.optimize import minimize
        ions, rho_target, _ = io.read_all(dftpy_data_path / 'rho_target.xsf')
        grid = rho_target.grid
        PP_list = {'Au': dftpy_data_path / 'Au_pgbrv02.psp8'}
        MaxPoints=1000 
        PSEUDO = LocalPseudo(grid = grid, ions=ions, PP_list=PP_list, MaxPoints=MaxPoints)
        rho_ini = rho_target.copy()
        KE = Functional(type='KEDF',name='TFvW', y=0.2)
        XC = Functional(type='XC',name='LDA', libxc=False)
        HARTREE = Functional(type='HARTREE')
        evaluator = TotalFunctional(KE=KE, XC=XC, HARTREE=HARTREE, PSEUDO=PSEUDO)
        optimization_options = {'econv' : 1e-6*ions.nat}
        opt = Optimization(EnergyEvaluator=evaluator, optimization_options = optimization_options, optimization_method = 'TN')
        rho = opt.optimize_rho(guess_rho=rho_ini)
        diff = 0.5 * (np.abs(rho - rho_target)).integral()
        self.assertTrue(np.isclose(diff, 0.117, atol=1.e-2))
        
if __name__ == "__main__":
    unittest.main()
