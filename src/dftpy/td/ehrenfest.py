"""
Ehrenfest dynamics: ions move classically under forces from the *instantaneous*
electron density, while the electrons are propagated in real time.

This is the time-dependent counterpart of :class:`dftpy.api.api4ase.DFTpyCalculator`.
That calculator re-optimises the density at every ionic step (Born-Oppenheimer
MD); :class:`EhrenfestCalculator` propagates it instead, so the electrons are
allowed to lag behind the ions and to carry a current. Everything else about the
interface is the same, which means the ASE integrators and thermostats work
unchanged:

>>> from ase.md.verlet import VelocityVerlet
>>> from dftpy.td.ehrenfest import EhrenfestCalculator
>>> calc = EhrenfestCalculator(evaluator=functionals, rho=rho0,
...                            dt_elec=0.01, dt_ion=1.0)      # a.u.
>>> atoms.calc = calc
>>> dyn = VelocityVerlet(atoms, timestep=calc.dt_ion_ase)
>>> dyn.run(1000)

Two things this module is strict about, because both are silent errors
-----------------------------------------------------------------------
**The two clocks must agree.** Ions and electrons have to cover the same
physical time per MD step, ``dt_ion == nsub * dt_elec``. Nothing in ASE or
DFTpy enforces this, and getting it wrong does not raise: the run simply stops
being Ehrenfest dynamics. ``nsub`` is therefore *derived* from the two
timesteps and cross-checked, never passed independently.

**The von Weizsacker term must appear once, not twice.** With :math:`\\psi=\\sqrt{\\rho}`
the ``-1/2 \\nabla^2`` in the Hamiltonian already supplies vW exactly, so the
KEDF used for propagation must have it removed. The calculator does this for
you (setting ``y = 0``) and reports it; the vW energy is added back explicitly
when the total energy is assembled.
"""

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

__all__ = ['EhrenfestCalculator', 'current_kinetic_energy']

# ASE keeps time in its own units; 1 a.u. of time in ASE units
from ase.units import fs as _ASE_FS
AU_TIME_IN_ASE = 0.02418884326 * _ASE_FS      # 1 a.u. = 24.188843 as


def current_kinetic_energy(j, rho, floor=1e-12):
    """
    Kinetic energy carried by the current, :math:`\\int |j|^2 / (2\\rho)\\,dr`.

    Note this is *not* ``sum(j**2)/2``: it needs the ``1/rho`` weight and a
    volume-weighted integral. In TD-OF-DFT with :math:`\\psi=\\sqrt{\\rho}` the
    kinetic energy splits as :math:`T = T_{vW}[\\rho] + T_j`, so this term is
    what makes the total energy conserve during propagation.

    ``floor`` guards the vacuum, where :math:`|j|^2` and :math:`\\rho` both go to
    zero and their ratio is numerically unstable.
    """
    j2 = np.asarray(j.dot(j))
    r = np.asarray(rho)
    out = np.zeros_like(r, dtype=float)
    mask = r > floor
    out[mask] = j2[mask] / (2.0 * r[mask])
    return DirectField(rho.grid, rank=1, griddata_3d=out).integral()


class EhrenfestCalculator(Calculator):
    """
    ASE calculator running Ehrenfest dynamics with orbital-free DFT.

    Parameters
    ----------
    evaluator: TotalFunctional
        Total functional for the electrons. Its KEDF has vW removed internally
        for the propagation (see the module docstring).
    rho: DirectField
        Converged ground-state density at the initial ion positions.
    dt_elec: float
        Electronic timestep in a.u. Resolves electron motion, so it is small
        (order 0.01 a.u.).
    dt_ion: float
        Ionic timestep in a.u. Resolves ion motion, so it is large (order
        1-10 a.u.). Must be an integer multiple of ``dt_elec``.
    propagator: str
        Name of the DFTpy propagator. Default ``crank-nicholson``.
    max_pc, tol_pc: int, float
        Predictor-corrector controls. ``max_pc`` should be generous (~100);
        too few iterations silently degrades the propagation.
    norm_tol: float
        Warn once if the electron number drifts by more than this (relative).
        Drift is the most sensitive indicator of propagator trouble.
    projectile: Projectile, optional
        A charge flying through the system, for stopping-power runs with mobile
        ions. See :mod:`dftpy.td.projectile`.

    Attributes
    ----------
    nsub: int
        Electronic steps per ionic step, derived as ``dt_ion / dt_elec``.
    dt_ion_ase: float
        The ionic timestep converted to ASE units, to hand to an ASE integrator.
    """

    implemented_properties = ['energy', 'forces']

    def __init__(self, evaluator, rho, dt_elec, dt_ion, propagator='crank-nicholson',
                 max_pc=100, tol_pc=1.0e-8, norm_tol=1.0e-8, projectile=None,
                 mp=None, **kwargs):
        Calculator.__init__(self, **kwargs)

        # ---- clock consistency, checked rather than assumed ---------------
        nsub = int(round(dt_ion / dt_elec))
        if nsub < 1 or abs(dt_ion - nsub * dt_elec) > 1e-10 * max(abs(dt_ion), 1.0):
            raise ValueError(
                "Ehrenfest clock mismatch: the ionic timestep must be an "
                "integer multiple of the electronic one.\n"
                f"    dt_ion  = {dt_ion:.6e} a.u.\n"
                f"    dt_elec = {dt_elec:.6e} a.u.\n"
                f"    ratio   = {dt_ion / dt_elec:.6f} (needs to be a whole number)\n"
                "Pick dt_ion = nsub * dt_elec for an integer nsub.")
        self.dt_elec = float(dt_elec)
        self.dt_ion = float(dt_ion)
        self.nsub = nsub
        self.dt_ion_ase = self.dt_ion * AU_TIME_IN_ASE

        self.evaluator = evaluator
        self.mp = mp if mp is not None else rho.grid.mp
        self.projectile = projectile
        self.norm_tol = norm_tol
        self._norm_warned = False
        self.time = 0.0

        # ---- vW belongs to the Hamiltonian, not the KEDF -------------------
        self._vw = Functional(type='KEDF', name='vW')
        ke = getattr(evaluator, 'KineticEnergyFunctional', None)
        if ke is None:
            ke = evaluator.funcDict.get('KineticEnergyFunctional', None)
        if ke is not None and hasattr(ke, 'options'):
            ke.options.update({'y': 0})
            sprint("EhrenfestCalculator: removed vW from the KEDF "
                   "(it is supplied by the laplacian in the Hamiltonian).")

        # ---- electronic state ---------------------------------------------
        self.rho0 = rho
        self.rho = rho
        self.N0 = rho.integral()
        self.psi = np.sqrt(rho).astype(complex)
        nrm = (self.psi.conj() * self.psi).integral()
        self.psi *= np.sqrt(self.N0 / nrm)
        self.j = calc_j(self.psi)

        self.max_pc = max_pc
        self.tol_pc = tol_pc
        pot = self._potential()
        self.propagator = Propagator(Hamiltonian(v=pot), self.dt_elec,
                                     name=propagator)

        self._first_call = True
        self.n_electron_steps = 0

        sprint(f"EhrenfestCalculator: dt_elec = {self.dt_elec:.6e} a.u., "
               f"nsub = {self.nsub}, dt_ion = {self.dt_ion:.6e} a.u. "
               f"({self.dt_ion_ase:.6e} ASE units)")

    @classmethod
    def from_config(cls, config, rho, evaluator, projectile=None, **kwargs):
        """
        Build from a DFTpy config, mirroring :meth:`Projectile.from_config`.

        Reads the two time steps from ``[EHRENFEST]``; ``nsub`` is still derived
        and checked in ``__init__``. A projectile is picked up from
        ``[PROJECTILE]`` only when the task asks for one, so ordinary Ehrenfest
        MD does not silently acquire a charge flying through it.

        Parameters
        ----------
        config: dict
            Needs [EHRENFEST]; [PROJECTILE] is optional.
        rho: DirectField
            Converged ground-state density at the initial ion positions.
        evaluator: TotalFunctional
            Total functional. NOTE: its KEDF is modified in place (vW removed).
        projectile: Projectile, optional
            Overrides the [PROJECTILE] section.
        **kwargs
            Passed through to ``__init__`` (max_pc, tol_pc, norm_tol, ...).

        Returns
        -------
        EhrenfestCalculator
        """
        c = config["EHRENFEST"]
        if projectile is None and "Stopping" in config.get("JOB", {}).get("task", ""):
            from dftpy.td.projectile import Projectile
            projectile = Projectile.from_config(
                config, np.diag(np.asarray(rho.grid.lattice)))
        return cls(evaluator=evaluator, rho=rho,
                   dt_elec=c["dt_elec"], dt_ion=c["dt_ion"],
                   projectile=projectile, **kwargs)

    # ------------------------------------------------------------- internals
    def _potential(self, t=None):
        """Effective potential, including the projectile if there is one."""
        pot = self.evaluator(self.rho, current=self.j, calcType=['V']).potential
        if self.projectile is not None:
            tt = self.time if t is None else t
            pot = pot + self.projectile.potential(self.rho.grid, tt)
        return pot

    def propagate(self, nsteps=None):
        """Advance the electrons by ``nsteps`` electronic steps."""
        nsteps = self.nsub if nsteps is None else nsteps
        for _ in range(nsteps):
            if self.projectile is not None:
                # midpoint keeps Crank-Nicolson second order
                self.propagator.hamiltonian.v = \
                    self._potential(self.time + 0.5 * self.dt_elec)
            pc = PredictorCorrector(self.psi, propagator=self.propagator,
                                    max_steps=self.max_pc, tol=self.tol_pc,
                                    functionals=self.evaluator)
            pc()
            self.psi = pc.psi_pred
            self.rho = calc_rho(self.psi)
            self.j = pc.j_pred
            self.time += self.dt_elec
            self.n_electron_steps += 1
            self.propagator.hamiltonian.v = self._potential()
        self._check_norm()

    def _check_norm(self):
        N = self.rho.integral()
        drift = (N - self.N0) / self.N0 if self.N0 else 0.0
        if abs(drift) > self.norm_tol and not self._norm_warned:
            self._norm_warned = True
            sprint(f"WARNING: electron number drifted by {drift:.3e} "
                   f"(tol {self.norm_tol:.1e}) at t = {self.time:.6f} a.u. "
                   f"Reduce dt_elec, raise max_pc, or check the Hamiltonian.")

    # ------------------------------------------------------------ ASE hooks
    def calculate(self, atoms=None, properties=['energy'], system_changes=all_changes):
        Calculator.calculate(self, atoms, properties, system_changes)
        ions = Ions.from_ase(atoms)

        # the ions have moved: rebuild the ionic potential on the same grid
        for key, func in self.evaluator.funcDict.items():
            if getattr(func, 'type', None) == 'PSEUDO':
                func.restart(grid=self.rho.grid, ions=ions)

        # Propagate only when the geometry actually changed. ASE may call
        # calculate() more than once for the same positions, and propagating on
        # each call would decouple the electronic clock from the ionic one.
        if self._first_call:
            self._first_call = False
        elif 'positions' in system_changes or 'cell' in system_changes:
            self.propagate(self.nsub)

        ene = self.evaluator(self.rho, current=self.j, calcType=['E']).energy
        e_vw = self._vw(self.rho).energy
        e_j = current_kinetic_energy(self.j, self.rho)
        energy = (ene + e_vw + e_j) * ENERGY_CONV["Hartree"]["eV"]

        forces = self.evaluator.get_forces(self.rho, ions=ions)
        forces = np.asarray(forces) * FORCE_CONV["Ha/Bohr"]["eV/A"]

        self.results['energy'] = energy
        self.results['forces'] = forces
        atoms.info['dftpy_time_au'] = self.time
        atoms.info['dftpy_nelec'] = float(self.rho.integral())
        atoms.info['dftpy_e_vw'] = e_vw
        atoms.info['dftpy_e_current'] = e_j
