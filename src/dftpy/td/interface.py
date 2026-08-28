import os
import os.path

import numpy as np

from dftpy.formats import io
from dftpy.mpi import sprint
from dftpy.td.casida import Casida
from dftpy.td.hamiltonian import Hamiltonian


def CasidaRunner(config, rho0, E_v_Evaluator):
    numeig = config["CASIDA"]["numeig"]
    outfile = config["TD"]["outfile"]
    diagonalize = config["CASIDA"]["diagonalize"]
    tda = config["CASIDA"]["tda"]

    if diagonalize:
        potential = E_v_Evaluator(rho0, calcType={'V'}).potential
        hamiltonian = Hamiltonian(potential)
        sprint('Start diagonalizing Hamiltonian.')
        eigs, psi_list = hamiltonian.diagonalize(numeig)
        sprint('Diagonalizing Hamiltonian done.')
    else:
        raise Exception("diagonalize must be true.")

    E_v_Evaluator.UpdateFunctional(keysToRemove=['HARTREE', 'PSEUDO'])
    casida = Casida(rho0, E_v_Evaluator)

    sprint('Start building matrix.')
    casida.build_matrix(numeig, eigs, psi_list, build_ab=tda)
    sprint('Building matrix done.')

    if tda:
        omega, f = casida.tda()
    else:
        omega, f, x_minus_y_list = casida()

    with open(outfile, 'w') as fw:
        for i in range(len(omega)):
            fw.write('{0:15.8e} {1:15.8e}\n'.format(omega[i], f[i]))

    # if save_eigenvectors:
    #    if not os.isdir(ev_path):
    #        os.mkdir(ev_path)
    #    i = 0
    #    for x_minus_y in x_minus_y_list:
    #        with open('{0:s}/x_minus_y{1:d}',format(ev_path, i), 'w') as fw:


def DiagonalizeRunner(config, field, ions, E_v_Evaluator):
    numeig = config["CASIDA"]["numeig"]
    eigfile = config["TD"]["outfile"]
    direct_to_psi = './xsf'

    potential = E_v_Evaluator(field, calcType={'V'}).potential
    hamiltonian = Hamiltonian(potential)
    sprint('Start diagonalizing Hamiltonian.')
    eigs, psi_list = hamiltonian.diagonalize(numeig)
    sprint('Diagonalizing Hamiltonian done.')

    np.savetxt(eigfile, eigs, fmt='%15.8e')

    if not os.path.isdir(direct_to_psi):
        os.mkdir(direct_to_psi)
    for i in range(len(eigs)):
        io.write('{0:s}/psi{1:d}.xsf'.format(direct_to_psi, i), ions, psi_list[i])

def StoppingPowerRunner(config, rho0, E_v_Evaluator):
    """
    Electronic stopping power: fly a charged projectile through the system.

    Entry point for ``task = Stopping``. Thin wrapper around
    :class:`dftpy.td.stopping_runner.StoppingRunner`, which subclasses
    :class:`dftpy.td.real_time_runner.RealTimeRunner` and adds the projectile.

    The projectile is defined by the ``[PROJECTILE]`` config section. Ions are
    held frozen: this is the ELECTRONIC stopping power, so the energy must go
    into the electrons rather than into recoiling ions.

    Parameters
    ----------
    config: dict
        Needs [TD], [PROPAGATOR] and [PROJECTILE].
    rho0: DirectField
        Converged ground-state density of the host system.
    E_v_Evaluator: AbstractFunctional
        Total functional.

    Returns
    -------
    StoppingRunner
        The runner, so ``.stopping_power_average`` and ``.stopping_history``
        remain accessible after the run.
    """
    from dftpy.td.stopping_runner import StoppingRunner

    runner = StoppingRunner.from_config(config, rho0, E_v_Evaluator)
    sprint('Start stopping power propagation.')
    runner()
    sprint('Stopping power propagation done.')
    return runner


def EhrenfestRunner(config, ions, rho0, E_v_Evaluator):
    """
    Ehrenfest molecular dynamics: ions move while the electrons are propagated.

    Entry point for ``task = Ehrenfest``. Unlike Born-Oppenheimer MD the density
    is *not* re-minimised at each ionic step, so the electrons may lag behind the
    ions and carry a current - which is what allows electronic excitation.

    Driven by the ``[EHRENFEST]`` config section. The two time steps must satisfy
    ``dt_ion = nsub * dt_elec``; ``nsub`` is derived and an inconsistent pair
    raises, because getting this wrong does not fail loudly - it silently stops
    being Ehrenfest dynamics.

    Parameters
    ----------
    config: dict
        Needs [EHRENFEST]; [PROJECTILE] is optional (a projectile with mobile ions).
    ions: Ions
        The ionic configuration.
    rho0: DirectField
        Converged ground-state density at the initial ion positions.
    E_v_Evaluator: AbstractFunctional
        Total functional. NOTE: its KEDF is modified in place (vW removed, since
        the laplacian in the Hamiltonian supplies it). Do not reuse the same
        object for a Born-Oppenheimer run.

    Returns
    -------
    EhrenfestCalculator
        The calculator, so ``.time``, ``.rho`` and ``.n_electron_steps`` remain
        accessible after the run.
    """
    from ase.io.trajectory import Trajectory
    from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
    from ase.md.verlet import VelocityVerlet

    from dftpy.td.ehrenfest import EhrenfestCalculator

    c = config["EHRENFEST"]
    atoms = ions.to_ase() if hasattr(ions, 'to_ase') else ions

    calc = EhrenfestCalculator.from_config(config, rho0, E_v_Evaluator)
    atoms.calc = calc

    if c["temperature"] > 0:
        MaxwellBoltzmannDistribution(atoms, temperature_K=c["temperature"])

    dyn = VelocityVerlet(atoms, timestep=calc.dt_ion_ase) # We could use other ASE MD
    if c["trajfile"]:
        dyn.attach(Trajectory(c["trajfile"], 'w', atoms).write, interval=1)

    fh = open(c["logfile"], 'w') if c["logfile"] else None
    if fh:
        fh.write("# time(au)  Epot(eV)  Ekin(eV)  Etot(eV)  N_elec\n")

    def _log(a=atoms):
        epot = a.get_potential_energy()
        ekin = a.get_kinetic_energy()
        line = (f"{a.info['dftpy_time_au']:14.6f} {epot:16.8f} {ekin:14.8f} "
                f"{epot + ekin:16.8f} {a.info['dftpy_nelec']:14.8f}")
        sprint(line)
        if fh:
            fh.write(line + "\n"); fh.flush()

    dyn.attach(_log, interval=1)
    sprint('Start Ehrenfest dynamics.')
    sprint("# time(au)  Epot(eV)  Ekin(eV)  Etot(eV)  N_elec")
    dyn.run(c["nsteps"])
    sprint('Ehrenfest dynamics done.')
    if fh:
        fh.close()
    return calc


# def SternheimerRunner(config, rho0, E_v_Evaluator):
#     outfile = config["TD"]["outfile"]
#
#     sternheimer = Sternheimer(rho0, E_v_Evaluator)
#     eigs, psi_list = sternheimer.hamiltonian.diagonalize(2)
#     sternheimer.grid.full = True
#     omega = np.linspace(0.0, 0.5, 26)
#     f = sternheimer(psi_list[1], omega, 0)
#     # f = omega
#     # sternheimer(psi_list[1], 1e-4, 0.01)
#
#     with open(outfile, 'w') as fw:
#         for i in range(len(omega)):
#             fw.write('{0:15.8e} {1:15.8e}\n'.format(omega[i], f[i]))
