import os
import numpy as np
from dftpy.mpi import pmi, sprint, mp
from dftpy.optimization import Optimization
from dftpy.functional import Functional, TotalFunctional
from dftpy.grid import DirectGrid
from dftpy.field import DirectField
from dftpy.ions import Ions
#-----------------------------------------------------------------------
if pmi.size > 0:
    from mpi4py import MPI
    mp.comm = MPI.COMM_WORLD
#-----------------------------------------------------------------------
from ase.lattice.cubic import FaceCenteredCubic
from ase.md.npt import NPT
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.io.trajectory import Trajectory
from ase import units

from dftpy.api.api4ase import DFTpyCalculator
import pathlib
dftpy_data_path = pathlib.Path(__file__).resolve().parents[1] / 'DATA'
MD_STEPS = int(os.environ.get("DFTPY_MD_STEPS", "5"))
np.random.seed(8888)

############################## initial config ##############################

"""
Ref :
    https://wiki.fysik.dtu.dk/ase/ase/md.html#module-ase.md
    https://wiki.fysik.dtu.dk/ase/_modules/ase/md/npt.html#NPT
"""

size = 3
a = 4.24068463425528
T = 1023  # Kelvin
atoms = FaceCenteredCubic(
    directions=[[1, 0, 0], [0, 1, 0], [0, 0, 1]], latticeconstant=a, symbol="Al", size=(size, size, size), pbc=True
)

# -----------------------------------------------------------------------
ions = Ions.from_ase(atoms)
path = str(dftpy_data_path)
PP_list = {'Al': path+os.sep+'al.lda.recpot'}
grid = DirectGrid(ecut=20, lattice=ions.cell, mp=mp, full=False)
rho = DirectField(grid=grid)
#
PSEUDO = Functional(type='PSEUDO', grid=grid, ions=ions, PP_list=PP_list)
KE = Functional(type='KEDF', name='WT')
XC = Functional(type='XC', name='LDA')
HARTREE = Functional(type='HARTREE')
#
funcDict = {'KE' :KE, 'XC' :XC, 'HARTREE' :HARTREE, 'PSEUDO' :PSEUDO}
EnergyEvaluator = TotalFunctional(**funcDict)
#
rho[:] = ions.get_ncharges() / ions.cell.volume
optimizer = Optimization(EnergyEvaluator=EnergyEvaluator, optimization_method='TN')

calc = DFTpyCalculator(optimizer = optimizer, evaluator = EnergyEvaluator, rho = rho)
# -----------------------------------------------------------------------

atoms.set_calculator(calc)

MaxwellBoltzmannDistribution(atoms, temperature_K = T, force_temp=True)

externalstress = 0.0
ttime = 10 * units.fs
# pfactor = 75 **2 * B
pfactor = 0.6

dyn = NPT(atoms, 2 * units.fs, externalstress=externalstress, ttime=ttime, pfactor=pfactor, temperature_K = T)

step = 0
interval = 1


def printenergy(a=atoms):
    global step, interval
    epot = a.get_potential_energy() / len(a)
    ekin = a.get_kinetic_energy() / len(a)
    spot = a.get_stress()
    press = np.sum(spot[:3]) / 3.0
    sprint(
        "Step={:<8d} Epot={:.5f} Ekin={:.5f} T={:.3f} Etot={:.5f} Press={:.5f}".format(
            step, epot, ekin, ekin / (1.5 * units.kB), epot + ekin, press / units.GPa
        )
    )
    step += interval

def check_stop():
    if os.path.isfile('dftpy_stopfile'): exit()


traj = Trajectory("md.traj", "w", atoms)

dyn.attach(check_stop, interval=1)
dyn.attach(printenergy, interval=1)
dyn.attach(traj.write, interval=1)

dyn.run(MD_STEPS)
