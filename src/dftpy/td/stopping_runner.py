"""
Electronic stopping power from real-time propagation.

:class:`StoppingRunner` subclasses :class:`dftpy.td.real_time_runner.RealTimeRunner`
so it inherits config parsing, the predictor-corrector loop, restart, observers,
MPI handling and the standard output files. It adds exactly three things:

1. a :class:`dftpy.td.projectile.Projectile` moving at constant velocity,
2. that projectile's potential injected into the functional through
   :class:`dftpy.td.projectile.ExternalPotential`, evaluated at the midpoint of
   each step so Crank-Nicolson stays second order,
3. a ``Stopping_Power.data`` output file.

The output columns match the SHRED plane-wave code, so the same post-processing
reads either:

    1 time | 2-4 R | 5-7 P | 8 S(t) | 9 <S> | 10 rho(R) | 11 rho/<rho> | 12-14 F

Usage from a config file
------------------------
Set ``task = Stopping`` in ``[JOB]`` and fill in ``[PROJECTILE]``::

    [JOB]
    task = Stopping

    [PROJECTILE]
    charge = 2.0
    position = 0.5 0.5 0.0
    velocity = 0.0 0.0 5.0
    sigma = 0.3
    mass = 4.0026

Usage from a script
-------------------
No config file is needed; the run is described by plain arguments.

>>> from dftpy.td.projectile import Projectile
>>> from dftpy.td.stopping_runner import StoppingRunner
>>> alpha = Projectile(Z=2.0, R0=[4, 4, 0], velocity=[0, 0, 5], sigma=0.3)
>>> runner = StoppingRunner(rho0, functionals, projectile=alpha,
...                         timestep=0.01, tmax=5.0)
>>> runner()
>>> print(runner.stopping_power_average)

With a config file, use the classmethod instead, which reads [TD],
[PROPAGATOR] and [PROJECTILE]:

>>> runner = StoppingRunner.from_config(config, rho0, functionals)
"""

import numpy as np

from dftpy.mpi import sprint, mp
from dftpy.td.projectile import Projectile, ExternalPotential
from dftpy.td.real_time_runner import RealTimeRunner

__all__ = ['StoppingRunner']

HA_BOHR_TO_EV_ANG = 51.42208619083232


class StoppingRunner(RealTimeRunner):
    """
    Real-time propagation with a projectile, for electronic stopping power.

    Parameters
    ----------
    rho0: DirectField
        Ground-state density of the host system.
    config: dict
        DFTpy config. ``[PROJECTILE]`` is read when ``projectile`` is None.
    functionals: AbstractFunctional
        Total functional for the host system.
    projectile: Projectile, optional
        Overrides the ``[PROJECTILE]`` config section.
    v_static: DirectField, optional
        Extra time-independent potential, e.g. frozen ions built outside the
        PSEUDO functional.
    outfile: str, optional
        Name of the stopping output file. Default ``Stopping_Power.data``.

    Notes
    -----
    The host ions are frozen: this measures the *electronic* stopping power, so
    the energy has to go into electrons rather than into recoiling ions. The
    initial kick from ``[TD]`` is not applied - the projectile is the
    perturbation - so ``strength`` is ignored.
    """

    def __init__(self, rho0, functionals, projectile=None, timestep=0.01,
                 tmax=1.0, max_pc=100, tol_pc=1.0e-8, atol_pc=1.0e-8,
                 propagator='crank-nicholson', v_static=None,
                 outfile='Stopping_Power.data', config=None):

        if config is None:
            # No config file: build a minimal one from DFTpy's own defaults so
            # RealTimeRunner still gets everything it expects. Only the keys the
            # caller can reasonably care about are exposed as arguments.
            if projectile is None:
                raise ValueError(
                    "StoppingRunner needs a projectile. Either pass "
                    "projectile=Projectile(...), or use "
                    "StoppingRunner.from_config(config, rho0, functionals) "
                    "to read one from a [PROJECTILE] config section.")
            config = self._build_config(timestep=timestep, tmax=tmax,
                                        max_pc=max_pc, tol_pc=tol_pc,
                                        atol_pc=atol_pc, propagator=propagator)
        elif projectile is None:
            cell_lengths = np.diag(np.asarray(rho0.grid.lattice))
            projectile = Projectile.from_config(config, cell_lengths)
        self.projectile = projectile

        # The projectile has to be seen by every internal recompute of the
        # potential, so it is injected at the functional level, not into the
        # Hamiltonian (PredictorCorrector would overwrite that).
        wrapped = ExternalPotential(functionals, v_static=v_static)
        wrapped.v_ext = projectile.potential(rho0.grid, 0.0)

        # No initial kick: the projectile is the perturbation.
        config = _copy_config_without_kick(config)

        super().__init__(rho0, config, wrapped)

        self.wrapped_functionals = wrapped
        self.stopping_outfile = outfile
        self.mean_rho = float(self.N0 / rho0.grid.volume)
        self.stopping_history = []
        self._sp_integral = 0.0
        self._last_time = 0.0
        self.stopping_power_average = 0.0

        self._sp_file = None
        if mp.is_root:
            self._sp_file = open(self.stopping_outfile, 'w')
            self._sp_file.write(
                " 1:Time 2-4:R(a.u.) 5-7:P(a.u.)[3] 8:SP(t)(a.u.) "
                "9:<SP> (a.u) 10:rho(a.u.) 11:rho/<rho> 12:F(a.u.)\n")

        sprint(self.projectile.info(rho0.grid))
        self.attach(self.calc_stopping_power, before_log=True)

    @staticmethod
    def _build_config(timestep, tmax, max_pc, tol_pc, atol_pc, propagator):
        """
        A minimal but complete DFTpy config, for use without an input file.

        RealTimeRunner reads a dozen keys from [TD] and [PROPAGATOR]; rather than
        enumerate them here we start from DFTpy's own defaults (so this stays
        correct if new keys are added upstream) and override only what matters.
        """
        from dftpy.config import DefaultOption, OptionFormat

        config = OptionFormat(DefaultOption())
        config["TD"]["timestep"] = timestep
        config["TD"]["tmax"] = tmax
        config["TD"]["max_pc"] = max_pc
        config["TD"]["tol_pc"] = tol_pc
        config["TD"]["atol_pc"] = atol_pc
        config["TD"]["strength"] = 0.0        # the projectile is the perturbation
        config["PROPAGATOR"]["propagator"] = propagator
        return config

    @classmethod
    def from_config(cls, config, rho0, functionals, projectile=None, v_static=None):
        """
        Build from a DFTpy config, mirroring :meth:`Projectile.from_config`.

        Every argument that can come from the config file does; anything passed
        explicitly overrides it. This is the entry point used by
        :func:`dftpy.td.interface.StoppingPowerRunner`.

        Parameters
        ----------
        config: dict
            Needs [TD], [PROPAGATOR] and [PROJECTILE].
        rho0: DirectField
            Converged ground-state density of the host system.
        functionals: AbstractFunctional
            Total functional.
        projectile: Projectile, optional
            Overrides the [PROJECTILE] section.
        v_static: DirectField, optional
            Extra time-independent potential, e.g. frozen ions supplied outside
            the PSEUDO functional.

        Returns
        -------
        StoppingRunner
        """
        if projectile is None:
            cell_lengths = np.diag(np.asarray(rho0.grid.lattice))
            projectile = Projectile.from_config(config, cell_lengths)
        outfile = config.get("PROJECTILE", {}).get("outfile", "Stopping_Power.data")
        return cls(rho0, functionals, projectile=projectile,
                   v_static=v_static, outfile=outfile, config=config)

    # ------------------------------------------------------------------ time
    @property
    def time(self):
        """Elapsed propagation time in a.u."""
        return self.nsteps * self.int_t

    def step(self):
        """
        One electronic step with the projectile at the midpoint position.

        Crank-Nicolson is second order when the Hamiltonian is evaluated at
        t + dt/2, so the projectile is placed there for the step itself and
        moved to t + dt afterwards for the observables.
        """
        t_mid = self.time + 0.5 * self.int_t
        self.wrapped_functionals.v_ext = \
            self.projectile.potential(self.rho.grid, t_mid)
        super().step()
        self.wrapped_functionals.v_ext = \
            self.projectile.potential(self.rho.grid, self.time)
        self.update_hamiltonian()

    # ----------------------------------------------------------- observables
    def calc_stopping_power(self):
        """Force on the projectile, S(t), and the running average."""
        t = self.time
        p = self.projectile
        R = p.position(t)
        F = p.force(self.rho, t)
        S = float(-np.dot(F, p.vhat)) if p.speed else 0.0

        dt = t - self._last_time
        self._sp_integral += S * dt
        self._last_time = t
        self.stopping_power_average = self._sp_integral / t if t > 0 else 0.0

        rho_at = self._density_at(R)
        self.stopping_history.append(
            dict(t=t, R=R.copy(), F=F.copy(), S=S,
                 S_avg=self.stopping_power_average))

        if self._sp_file is not None:
            vals = ([t] + list(R) + list(p.momentum())
                    + [S, self.stopping_power_average, rho_at,
                       rho_at / self.mean_rho if self.mean_rho else 0.0]
                    + list(F))
            self._sp_file.write("".join(f"{v:20.11E}" for v in vals) + "\n")
            self._sp_file.flush()

    def _density_at(self, R):
        """Electron density at R, nearest grid point. MPI-aware."""
        d = self.projectile._min_image(self.rho.grid, R)
        d2 = d[0] ** 2 + d[1] ** 2 + d[2] ** 2
        local_min = float(np.min(d2))
        idx = np.unravel_index(np.argmin(d2), d2.shape)
        local_val = float(np.asarray(self.rho)[idx])
        if not mp.is_mpi:
            return local_val
        # the owning rank is the one holding the closest point
        allmin = mp.comm.allgather((local_min, local_val))
        return min(allmin, key=lambda x: x[0])[1]

    def stop(self):
        """Close the stopping output file."""
        if self._sp_file is not None:
            self._sp_file.close()
            self._sp_file = None

    def run(self):
        try:
            return super().run()
        finally:
            self.stop()
            sprint(f"Final <S> = {self.stopping_power_average:.6f} Ha/bohr = "
                   f"{self.stopping_power_average * HA_BOHR_TO_EV_ANG:.4f} eV/Ang")


def _copy_config_without_kick(config):
    """Return a shallow copy of config with the initial TD kick disabled."""
    new = dict(config)
    td = dict(config["TD"])
    td["strength"] = 0.0
    new["TD"] = td
    return new
