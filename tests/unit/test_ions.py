import numpy as np
import pytest
from ase import Atoms

from dftpy.constants import Units
from dftpy.ions import Ions


def test_ions_initialization():
    symbols = ["H", "He"]
    positions = [[0, 0, 0], [1, 1, 1]]
    cell = np.eye(3) * 10
    ions = Ions(symbols=symbols, positions=positions, cell=cell)

    assert ions.nat == 2
    assert ions.symbols_uniq.tolist() == ["H", "He"]
    assert np.allclose(ions.get_positions(), positions)
    assert np.allclose(ions.cell.array, cell)


def test_ions_set_charges():
    symbols = ["H", "He"]
    positions = [[0, 0, 0], [1, 1, 1]]
    ions = Ions(symbols=symbols, positions=positions)

    charges = [1.0, 2.0]
    ions.set_charges(charges)
    assert np.allclose(ions.get_charges(), charges)

    with pytest.raises(AttributeError):
        ions.set_charges({"Li": 3.0})


def test_ions_to_ase():
    symbols = ["H", "He"]
    positions = [[0, 0, 0], [1, 1, 1]]
    cell = np.eye(3) * 10
    ions = Ions(symbols=symbols, positions=positions, cell=cell)

    ase_atoms = ions.to_ase()
    assert isinstance(ase_atoms, Atoms)
    assert ase_atoms.get_chemical_symbols() == symbols
    assert np.allclose(ase_atoms.get_positions(), np.array(positions) * Units.Bohr)
    assert np.allclose(ase_atoms.cell, cell * Units.Bohr)


def test_ions_from_ase():
    symbols = ["H", "He"]
    positions = [[0, 0, 0], [1, 1, 1]]
    cell = np.eye(3) * 10
    ase_atoms = Atoms(symbols=symbols, positions=positions, cell=cell)

    ions = Ions.from_ase(ase_atoms)
    assert ions.nat == 2
    assert ions.symbols_uniq.tolist() == ["H", "He"]
    assert np.allclose(ions.get_positions(), np.array(positions) / Units.Bohr)
    assert np.allclose(ions.cell.array, cell / Units.Bohr)


def test_ions_get_ncharges():
    symbols = ["H", "He"]
    positions = [[0, 0, 0], [1, 1, 1]]
    ions = Ions(symbols=symbols, positions=positions)

    charges = [1.0, 2.0]
    ions.set_charges(charges)
    assert ions.get_ncharges() == sum(charges)


def test_ions_zval_property():
    symbols = ["H", "He"]
    positions = [[0, 0, 0], [1, 1, 1]]
    ions = Ions(symbols=symbols, positions=positions)

    charges = [1.0, 2.0]
    ions.set_charges(charges)
    zval = ions.zval
    assert zval["H"] == 1.0
    assert zval["He"] == 2.0


def test_ions_input_from_atoms():
    symbols = ["H", "He"]
    positions = [[0, 0, 0], [1, 1, 1]]
    cell = np.eye(3) * 10
    ase_atoms = Atoms(symbols=symbols, positions=positions, cell=cell)

    ions = Ions(ase_atoms)
    assert ions.nat == 2
    assert ions.symbols_uniq.tolist() == ["H", "He"]
    assert np.allclose(ions.get_positions(), np.array(positions) / Units.Bohr)
    assert np.allclose(ions.cell.array, cell / Units.Bohr)


def test_ions_invalid_methods_and_attributes():
    symbols = ["H", "He"]
    positions = [[0, 0, 0], [1, 1, 1]]
    ions = Ions(symbols=symbols, positions=positions)

    with pytest.raises(AttributeError):
        ions.unsupported_method()

    with pytest.raises(AttributeError):
        _ = ions.unsupported_attribute


def test_ions_repeat():
    symbols = ["H", "He"]
    positions = [[0, 0, 0], [1, 1, 1]]
    cell = np.eye(3) * 10
    ions = Ions(symbols=symbols, positions=positions, cell=cell)

    repeated_ions = ions.repeat((2, 1, 1))
    positions_repeated = [[0, 0, 0], [1, 1, 1], [10, 0, 0], [11, 1, 1]]
    assert repeated_ions.nat == 4
    assert np.allclose(repeated_ions.get_positions(), positions_repeated)
