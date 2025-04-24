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
    def test_nspin(self):
        from ase.io.trajectory import Trajectory
        from ase.build import bulk
        from ase import Atoms
        import matplotlib.pyplot as plt
        from dftpy.formats import io
        from dftpy.api.api4ase import DFTpyCalculator
        from dftpy.config import DefaultOption, OptionFormat
        from dftpy.functional import LocalPseudo, Functional, TotalFunctional
        from dftpy.optimization import Optimization
        from dftpy.ions import Ions
        from dftpy.field import DirectField
        from dftpy.grid import DirectGrid
        from dftpy.math_utils import ecut2nr
        
        def scale_density(rho, m):
            if rho.rank != 2:
                raise Exception("Rho must be rank 2")
            nelec = rho.integral()
            nnelec = nelec + np.array([m/2.0,-m/2.0])
            rho[0] *=  nnelec[0]/nelec[0]
            rho[1] *=  nnelec[1]/nelec[1]
            return rho
        
        atoms = bulk('Al', 'fcc', a=4.05)
        ions = Ions.from_ase(atoms)
        
        XC = Functional(type='XC',name='LDA', libxc=False)
        HARTREE = Functional(type='HARTREE')
        TF = Functional(type='KEDF', name='TFvW', y=1)
        opt_options = {'econv' : 1e-7*ions.nat, 'maxiter': 50} 
        
        PP_list = {'Al':dftpy_data_path / 'al.lda.recpot'}
        
        nr = ecut2nr(ecut=90, lattice=ions.cell)
        grid = DirectGrid(lattice=ions.cell, nr=nr)
        PSEUDO = LocalPseudo(grid=grid, ions=ions, PP_list=PP_list)
        rho = DirectField(grid=grid,rank=2)
        rho[:] = ions.get_ncharges() / ions.cell.volume / rho.rank
        
        rho = scale_density(rho,0.1)
        
        evaluator = TotalFunctional(KE=TF, XC=XC, HARTREE=HARTREE, PSEUDO=PSEUDO)
        opt = Optimization(EnergyEvaluator=evaluator, optimization_method='TN', optimization_options=opt_options)
        
        rho = opt.optimize_rho(guess_rho=rho)
        ene = evaluator.Energy(rho)

        self.assertTrue(np.isclose(ene, -2.1114251, atol=1.e-3))

if __name__ == "__main__":
    unittest.main()
