"""Unit tests for the moving projectile and electronic stopping power."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from dftpy.field import DirectField
from dftpy.functional import Functional, TotalFunctional
from dftpy.grid import DirectGrid
from dftpy.td.projectile import ExternalPotential, Projectile
from dftpy.td.stopping_runner import StoppingRunner

L = 12.0
NG = 20
N_DENS = 0.05
Z = 2.0
SIGMA = 0.8


def _grid(n=NG, box=L):
    return DirectGrid(lattice=np.eye(3) * box, nr=[n] * 3, full=True)


def _uniform_rho(grid, n=N_DENS):
    rho = DirectField(grid=grid)
    rho[:] = n
    return rho


def _functionals():
    return TotalFunctional(
        KineticEnergyFunctional=Functional(type='KEDF', name='TFvW'),
        HARTREE=Functional(type='HARTREE'))


def _projectile(**kw):
    kw.setdefault('Z', Z)
    kw.setdefault('R0', [L / 2, L / 2, 0.0])
    kw.setdefault('velocity', [0.0, 0.0, 3.0])
    kw.setdefault('sigma', SIGMA)
    kw.setdefault('mass_amu', 4.0026)
    return Projectile(**kw)


# --------------------------------------------------------------- Projectile

def test_charge_is_normalised():
    """The smeared charge must integrate to exactly Z, at any resolution."""
    grid = _grid()
    p = _projectile()
    assert_allclose(p.charge_density(grid, 0.0).integral(), Z, rtol=1e-12)


def test_potential_satisfies_poisson():
    """lap(V) = 4 pi (rho - <rho>).

    Exact in a periodic cell and free of any FFT-normalisation assumption. The
    mean is subtracted because the G=0 component of V is dropped, which is the
    neutralising background.
    """
    grid = _grid()
    p = _projectile()
    V = p.potential(grid, 0.0)
    rho_p = p.charge_density(grid, 0.0)

    lap = np.asarray(V.laplacian(force_real=True, sigma=0.0))
    src = 4.0 * np.pi * (np.asarray(rho_p)
                         - float(rho_p.integral()) / grid.volume)
    assert_allclose(lap, src, atol=1e-10 * np.abs(src).max())


def test_uniform_gas_gives_zero_force():
    """By symmetry a projectile in a uniform gas feels no force."""
    grid = _grid()
    p = _projectile()
    F = p.force(_uniform_rho(grid), 0.0)
    assert np.linalg.norm(F) < 1e-10


def test_force_equals_minus_dE_dR():
    """F = -dE/dR, checked against the force's own definition.

    Comparing with the isolated Z*Ne/d^2 would be wrong here: the cell is
    periodic and carries a neutralising background, so a finite difference of
    the interaction energy is the only self-consistent reference.
    """
    grid = _grid()
    R = np.array([L / 2, L / 2, L / 2])
    p = _projectile(R0=R, velocity=[0, 0, 0])

    # a compact blob of electrons displaced along +z
    dz, ne, sb = 3.0, 1.0, 0.7
    d = p._min_image(grid, R + np.array([0.0, 0.0, dz]))
    blob = np.exp(-(d[0] ** 2 + d[1] ** 2 + d[2] ** 2) / (2 * sb ** 2))
    rho = DirectField(grid, rank=1, griddata_3d=blob)
    rho = rho * (ne / rho.integral())

    F = p.force(rho, 0.0)

    def energy_at(pos):
        return Projectile(Z=Z, R0=pos, velocity=[0, 0, 0],
                          sigma=SIGMA).interaction_energy(rho, 0.0)

    delta = 0.05
    F_fd = np.empty(3)
    for i in range(3):
        e = np.zeros(3)
        e[i] = delta
        F_fd[i] = -(energy_at(R + e) - energy_at(R - e)) / (2 * delta)

    assert_allclose(F, F_fd, atol=1e-3 * np.abs(F_fd).max())
    assert F[2] > 0.0                      # pulled toward the electrons


def test_zero_velocity_gives_zero_stopping_power():
    grid = _grid()
    p = _projectile(velocity=[0.0, 0.0, 0.0])
    assert p.stopping_power(_uniform_rho(grid), 0.0) == 0.0


def test_trajectory_is_analytic():
    p = _projectile(R0=[1.0, 2.0, 3.0], velocity=[0.0, 0.0, 5.0])
    assert_allclose(p.position(0.4), [1.0, 2.0, 5.0])


# ---------------------------------------------------------- ExternalPotential

def test_external_potential_adds_v_ext_and_delegates():
    """The wrapper must add v_ext and otherwise behave like the functional."""
    grid = _grid()
    rho = _uniform_rho(grid)
    wrapped = ExternalPotential(_functionals())

    v0 = np.asarray(wrapped(rho, calcType=['V']).potential).copy()
    wrapped.v_ext = _projectile().potential(grid, 0.0)
    v1 = np.asarray(wrapped(rho, calcType=['V']).potential)

    assert_allclose(v1 - v0, np.asarray(wrapped.v_ext), atol=1e-12)
    # attribute access falls through to the wrapped functional
    assert hasattr(wrapped, 'KineticEnergyFunctional')


def test_external_potential_adds_energy():
    """Needed so the same wrapper can drive an SCF (calcType=['E','V']).

    The density is modulated on purpose: V_proj has zero mean, so against a
    uniform gas int(rho V) would vanish and the test would pass on nothing.
    """
    grid = _grid()
    rho = _uniform_rho(grid)
    rho[:] = N_DENS * (1.0 + 0.3 * np.cos(2 * np.pi * np.asarray(grid.r[2]) / L))

    wrapped = ExternalPotential(_functionals())
    e0 = wrapped(rho, calcType=['E']).energy
    wrapped.v_ext = _projectile().potential(grid, 0.0)
    e1 = wrapped(rho, calcType=['E']).energy

    e_ext = float((rho * wrapped.v_ext).integral())
    assert abs(e_ext) > 1e-6                       # the check is not vacuous
    assert_allclose(e1 - e0, e_ext, rtol=1e-10)


# --------------------------------------------------------------- the runner

def test_runner_without_a_config_file(tmp_path, monkeypatch):
    """StoppingRunner must be usable with plain arguments, no config needed."""
    monkeypatch.chdir(tmp_path)
    grid = _grid()
    rho0 = _uniform_rho(grid)
    dt, nsteps = 0.05, 4

    runner = StoppingRunner(rho0, _functionals(), projectile=_projectile(),
                            timestep=dt, tmax=dt * nsteps, max_pc=20,
                            outfile=str(tmp_path / 'stopping.data'))
    runner()
    runner.stop()

    # one record for the initial state plus one per step
    assert len(runner.stopping_history) == nsteps + 1
    assert runner.stopping_history[0]['t'] == 0.0
    # the projectile advances exactly v * t: its trajectory is analytic
    last = runner.stopping_history[-1]
    assert_allclose(last['R'][2], 3.0 * last['t'], rtol=1e-12)
    # electron number is conserved by the propagation
    assert_allclose(runner.rho.integral(), runner.N0, rtol=1e-8)


def test_runner_requires_a_projectile_without_config():
    with pytest.raises(ValueError, match="needs a projectile"):
        StoppingRunner(_uniform_rho(_grid()), _functionals())


def test_from_config_reads_the_projectile_section(tmp_path, monkeypatch):
    """from_config must build the projectile and honour PROJECTILE.outfile."""
    monkeypatch.chdir(tmp_path)
    from dftpy.config import DefaultOption, OptionFormat

    grid = _grid()
    rho0 = _uniform_rho(grid)
    outfile = str(tmp_path / 'from_config.data')

    config = OptionFormat(DefaultOption())
    config["TD"]["timestep"] = 0.05
    config["TD"]["tmax"] = 0.05 * 3
    config["TD"]["max_pc"] = 20
    config["PROJECTILE"]["charge"] = Z
    config["PROJECTILE"]["position"] = [0.5, 0.5, 0.0]     # FRACTIONAL
    config["PROJECTILE"]["velocity"] = [0.0, 0.0, 3.0]
    config["PROJECTILE"]["sigma"] = SIGMA
    config["PROJECTILE"]["outfile"] = outfile

    runner = StoppingRunner.from_config(config, rho0, _functionals())
    assert runner.projectile.Z == Z
    assert runner.stopping_outfile == outfile
    # fractional -> cartesian conversion used the cell
    assert_allclose(runner.projectile.R0, [L / 2, L / 2, 0.0])
    runner.stop()


def test_output_columns_match_shred_layout(tmp_path, monkeypatch):
    """14 columns, so the same post-processing reads DFTpy and SHRED output."""
    monkeypatch.chdir(tmp_path)
    grid = _grid()
    dt, nsteps = 0.05, 3
    out = str(tmp_path / 'columns.data')
    runner = StoppingRunner(_uniform_rho(grid), _functionals(),
                            projectile=_projectile(), timestep=dt,
                            tmax=dt * nsteps, max_pc=20, outfile=out)
    runner()
    runner.stop()
    data = np.loadtxt(out, skiprows=1)
    assert data.ndim == 2 and data.shape[1] == 14
    assert len(data) == nsteps + 1
    assert_allclose(data[:, 0], np.arange(nsteps + 1) * dt)    # col 1 is time
