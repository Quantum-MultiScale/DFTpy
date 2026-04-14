import os
import pathlib
import numpy as np
from ase.lattice.cubic import FaceCenteredCubic
from ase.md.langevin import Langevin
from ase.md.verlet import VelocityVerlet
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.io.trajectory import Trajectory
from ase import units
from ase.md.npt import NPT

from dftpy.config import DefaultOption, OptionFormat
from dftpy.interface import OptimizeDensityConf
from dftpy.api.api4ase import DFTpyCalculator
from dftpy.mpi import MP, sprint
# MPI / parallel setup 
mp = MP(parallel=False) # Set parallel=True to run in parallel
############################## initial config ##############################
conf = DefaultOption()
conf['PATH']['pppath'] = './'
conf['PP']['Al'] = 'Al_lda.oe01.recpot'
conf['OPT']['method'] = 'TN'
conf['KEDF']['kedf'] = 'WT'
conf['JOB']['calctype'] = 'Energy Force'
conf = OptionFormat(conf)
#-----------------------------------------------------------------------

size = 3
a = 4.24068463425528
T = 1023  # Kelvin
T *= units.kB
atoms = FaceCenteredCubic(directions=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                          latticeconstant = a,
                          symbol="Al",
                          size=(size, size, size),
                          pbc=True)

calc = DFTpyCalculator(config = conf) # DFTpy can be run in parallel using the MPI interface, as follows: DFTpyCalculator(config = conf, mp=mp) 
atoms.set_calculator(calc)

MaxwellBoltzmannDistribution(atoms, T, force_temp = True)

dyn = Langevin(atoms, 2 * units.fs, T, 0.1)

step = 0
interval = 1
def printenergy(a=atoms):  
    global step, interval
    epot = a.get_potential_energy() / len(a)
    ekin = a.get_kinetic_energy() / len(a)
    print('Step={:<8d} Epot={:.5f} Ekin={:.5f} T={:.3f} Etot={:.5f}'.format(step, epot, ekin, ekin / (1.5 * units.kB), epot + ekin))
    step += interval
    
dyn.attach(printenergy, interval=1)

traj = Trajectory('md.traj', 'w', atoms)
dyn.attach(traj.write, interval=5)

dyn.run(100)
