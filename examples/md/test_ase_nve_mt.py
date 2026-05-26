"""
NVE molecular dynamics of an Al₂ dimer with Martyna-Tuckerman screening.

Demonstrates energy conservation over 50 steps in a large vacuum cell,
verifying that the MT isolated-boundary forces are conservative.
"""
import numpy as np
from ase import Atoms
from ase.md.verlet import VelocityVerlet
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase import units

from dftpy.mpi import mp, sprint
from dftpy.config.config import DefaultOption, OptionFormat
from dftpy.api.api4ase import DFTpyCalculator
import pathlib

dftpy_data_path = pathlib.Path(__file__).resolve().parents[1] / "DATA"


def main():
    np.random.seed(42)

    conf = DefaultOption()
    conf["PATH"]["pppath"] = str(dftpy_data_path)
    conf["PP"]["Al"] = "al.lda.recpot"
    conf["OPT"]["method"] = "TN"
    conf["OPT"]["econv"] = 1e-8
    conf["KEDF"]["kedf"] = "WT"
    conf["JOB"]["calctype"] = "Energy Force"
    conf["OUTPUT"]["time"] = False
    conf["GRID"]["spacing"] = 0.35
    conf["MARTYNA_TUCKERMAN"]["enable"] = True
    conf = OptionFormat(conf)

    from dftpy.constants import LEN_CONV
    L_bohr = 20.0
    d_bohr = 3.5  # OFDFT WT equilibrium bond length for Al₂
    L_ang = L_bohr * LEN_CONV["Bohr"]["Angstrom"]
    d_ang = d_bohr * LEN_CONV["Bohr"]["Angstrom"]

    center = L_ang / 2.0
    atoms = Atoms(
        "Al2",
        positions=[
            [center - d_ang / 2, center, center],
            [center + d_ang / 2, center, center],
        ],
        cell=[L_ang, L_ang, L_ang],
        pbc=True,
    )

    calc = DFTpyCalculator(config=conf, mp=mp)
    atoms.calc = calc

    T_init = 300  # K
    MaxwellBoltzmannDistribution(atoms, temperature_K=T_init, force_temp=True)

    dt = 2.0 * units.fs
    dyn = VelocityVerlet(atoms, dt)

    nsteps = 50
    energies = []

    sprint(f"{'Step':>5s}  {'Epot (eV)':>12s}  {'Ekin (eV)':>12s}  {'Etot (eV)':>12s}  {'drift (eV)':>12s}")
    sprint("-" * 65)

    def record(a=atoms):
        epot = a.get_potential_energy()
        ekin = a.get_kinetic_energy()
        etot = epot + ekin
        energies.append(etot)
        drift = etot - energies[0] if len(energies) > 1 else 0.0
        sprint(f"{len(energies)-1:5d}  {epot:12.6f}  {ekin:12.6f}  {etot:12.6f}  {drift:12.6f}")

    record()
    dyn.attach(record, interval=1)
    dyn.run(nsteps)

    energies = np.array(energies)
    drift = np.abs(energies - energies[0])
    max_drift = drift.max()
    std_drift = np.std(energies)

    sprint("\n=== Energy conservation summary ===")
    sprint(f"  Max |E(t) - E(0)| = {max_drift:.6f} eV")
    sprint(f"  Std dev of E(t)   = {std_drift:.6f} eV")
    sprint(f"  Mean E(t)         = {np.mean(energies):.6f} eV")

    threshold = 0.002  # 2 meV
    if max_drift < threshold:
        sprint(f"\n  PASSED: energy drift ({max_drift:.6f} eV) < {threshold} eV")
    else:
        sprint(f"\n  WARNING: energy drift ({max_drift:.6f} eV) >= {threshold} eV")


if __name__ == "__main__":
    main()
