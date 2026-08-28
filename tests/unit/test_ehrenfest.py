import numpy as np
import pathlib

from ase.io import read
from ase.md.verlet import VelocityVerlet
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.io.trajectory import Trajectory
from ase import units
from pathlib import Path

from dftpy.field import DirectField
from dftpy.functional import Functional, TotalFunctional
from dftpy.grid import DirectGrid
from dftpy.ions import Ions
from dftpy.optimization import Optimization
from dftpy.td.ehrenfest import EhrenfestCalculator
import pytest

def test_total_energy(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    data = Path(__file__).resolve().parents[2] / "examples" / "DATA"

    atoms = read(str(data / 'fcc.vasp'))
    ions = Ions.from_ase(atoms)
    grid = DirectGrid(lattice=ions.cell, spacing=0.25, full=True)

    pseudo = Functional(type='PSEUDO', grid=grid, ions=ions,
                    PP_list={'Al': str(data / 'al.lda.recpot')})
    functionals = TotalFunctional(
    KineticEnergyFunctional=Functional(type='KEDF', name='TFvW'),
    XCFunctional=Functional(type='XC', name='LDA'),
    HARTREE=Functional(type='HARTREE'),
    PSEUDO=pseudo)

    rho0 = DirectField(grid=grid)
    rho0[:] = ions.get_ncharges() / ions.cell.volume
    rho0 = Optimization(EnergyEvaluator=functionals,
                    optimization_options={'econv': 1e-8, 'maxiter': 100},
                    optimization_method='CG').optimize_rho(guess_rho=rho0)

    dt_elec = 0.05          # a.u. - resolves ELECTRON motion
    dt_ion = 5.0            # a.u. - resolves ION motion  -> nsub = 100

    calc = EhrenfestCalculator(evaluator=functionals, rho=rho0,
                           dt_elec=dt_elec, dt_ion=dt_ion, max_pc=100)
    atoms.calc = calc

    MaxwellBoltzmannDistribution(atoms, temperature_K=300)
    # dt_ion_ase converts the a.u. timestep into ASE's units for you
    dyn = VelocityVerlet(atoms, timestep=calc.dt_ion_ase)
    dyn.attach(Trajectory(str(tmp_path / 'ehrenfest.traj'), 'w', atoms).write, interval=1)
    
    # Energy must be conserved

    energy_history = []
    def report(a=atoms):
        energy_history.append(atoms.get_total_energy())
    dyn.attach(report, interval=1)
    dyn.run(5)
    diff = np.abs(energy_history[0]-energy_history[-1])
    assert diff < 1e-2


def test_clock_consistency_is_enforced():
    """dt_ion must be an integer multiple of dt_elec.

    This is the bug that motivated the check: nsub and dt_ion used to be
    independent inputs, and a run went into production with the ions advancing
    200x further than the electrons per step. Nothing failed - it simply was not
    Ehrenfest dynamics any more. nsub is now derived, so the only remaining bad
    case is a non-integer ratio, which must raise.
    """
    with pytest.raises(ValueError, match="clock mismatch"):
        EhrenfestCalculator(evaluator=None, rho=None,
                            dt_elec=0.0124, dt_ion=2.5)      # ratio 201.6


def test_nsub_is_derived_from_the_two_timesteps():
    grid = DirectGrid(lattice=np.eye(3) * 10.0, nr=[16] * 3, full=True)
    rho = DirectField(grid=grid)
    rho[:] = 0.05
    functionals = TotalFunctional(
        KineticEnergyFunctional=Functional(type='KEDF', name='TFvW'),
        HARTREE=Functional(type='HARTREE'))

    calc = EhrenfestCalculator(evaluator=functionals, rho=rho,
                               dt_elec=0.05, dt_ion=0.5)
    assert calc.nsub == 10
    # and the ASE-unit conversion of the ionic timestep
    assert np.isclose(calc.dt_ion_ase / calc.dt_ion, 0.02418884326 * units.fs)


def test_current_kinetic_energy_matches_the_analytic_value():
    """T_j = int |j|^2/(2 rho).

    For psi = sqrt(n) exp(i k z) on a uniform density the vW term vanishes, so
    the whole kinetic energy is carried by the current and T_j = (1/2) k^2 N.
    The previous expression, sum(j**2)/2, had no 1/rho and no volume element,
    and its error even depended on the grid spacing.
    """
    from dftpy.td.ehrenfest import current_kinetic_energy
    from dftpy.utils.utils import calc_j

    L, ng, n = 10.0, 24, 0.05
    grid = DirectGrid(lattice=np.eye(3) * L, nr=[ng] * 3, full=True)
    k = 2.0 * np.pi * 2 / L
    psi = DirectField(grid, rank=1, cplx=True,
                      griddata_3d=np.sqrt(n) * np.exp(1j * k * np.asarray(grid.r[2])))
    rho = (psi.conj() * psi).real

    got = current_kinetic_energy(calc_j(psi), rho)
    assert np.isclose(got, 0.5 * k * k * rho.integral(), rtol=1e-10)


def test_from_config_builds_the_calculator():
    """EhrenfestCalculator.from_config mirrors Projectile.from_config."""
    from dftpy.config import DefaultOption, OptionFormat

    grid = DirectGrid(lattice=np.eye(3) * 10.0, nr=[16] * 3, full=True)
    rho = DirectField(grid=grid)
    rho[:] = 0.05
    functionals = TotalFunctional(
        KineticEnergyFunctional=Functional(type='KEDF', name='TFvW'),
        HARTREE=Functional(type='HARTREE'))

    config = OptionFormat(DefaultOption())
    config["EHRENFEST"]["dt_elec"] = 0.02
    config["EHRENFEST"]["dt_ion"] = 0.2

    calc = EhrenfestCalculator.from_config(config, rho, functionals)
    assert calc.nsub == 10
    # an ordinary Ehrenfest run must NOT silently acquire a projectile
    assert calc.projectile is None


def test_electron_number_is_conserved(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    """N(t) drift is the most sensitive indicator of propagator trouble."""
    grid = DirectGrid(lattice=np.eye(3) * 10.0, nr=[16] * 3, full=True)
    rho = DirectField(grid=grid)
    rho[:] = 0.05
    functionals = TotalFunctional(
        KineticEnergyFunctional=Functional(type='KEDF', name='TFvW'),
        HARTREE=Functional(type='HARTREE'))

    calc = EhrenfestCalculator(evaluator=functionals, rho=rho,
                               dt_elec=0.02, dt_ion=0.1, max_pc=50)
    n0 = calc.N0
    calc.propagate(5)
    assert np.isclose(calc.rho.integral(), n0, rtol=1e-8)
    assert calc.n_electron_steps == 5
