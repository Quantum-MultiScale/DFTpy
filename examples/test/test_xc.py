#!/usr/bin/env python3
import unittest
import numpy as np
import pytest

from dftpy.functional.xc import XC
from dftpy.functional.pseudo import LocalPseudo
from dftpy.formats import io
from common import dftpy_data_path


class Test(unittest.TestCase):
    def test_xc_cv(self):
        pytest.importorskip("pylibxc")
        print("*" * 50)
        print("Testing loading pseudopotentials")
        ions, rho, _ = io.read_all(dftpy_data_path / "ti.xsf")
        PP_list = {'Ti':dftpy_data_path / "ti_pbe_v1.4.uspp.F.UPF"}
        grid = rho.grid

        pseudo = LocalPseudo(grid=grid, ions=ions, PP_list=PP_list, PME=False)
        ie_energy = pseudo(rho).energy
        ie_forces = pseudo.forces(rho)
        np.set_printoptions(precision=8, suppress=True)

        print(ie_energy*2)
        print(ie_forces[0]*2)
        ie_energy_qe = -154.480261581237
        ie_forces_qe = np.array([0.00059925, 0.00050441, -0.00021724])
        self.assertTrue(np.isclose(ie_energy*2, ie_energy_qe, atol=1.E-3))
        self.assertTrue(np.allclose(ie_forces[0]*2, ie_forces_qe, atol=1.E-4))

        xc = XC(xc = 'PBE', pseudo = pseudo)
        xc_energy = xc(rho, calcType = ['E']).energy
        xc_forces = xc.forces(rho)

        print(xc_energy*2)
        print(xc_forces[0]*2)
        xc_energy_qe = -36.0578154357786
        xc_forces_qe = np.array([-0.00001616, -0.00001221, 0.00000047])
        self.assertTrue(np.isclose(xc_energy*2, xc_energy_qe, atol=1.E-4))
        self.assertTrue(np.allclose(xc_forces[0]*2, xc_forces_qe, rtol=2.E-1))


if __name__ == "__main__":
    unittest.main()
