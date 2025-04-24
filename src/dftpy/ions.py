import numpy as np
from ase import Atoms
from ase.atom import Atom
from ase.atoms import default
from ase.symbols import symbols2numbers
from ase.cell import Cell

from dftpy.constants import Units


class Ions(Atoms):
    """Ions object based on `ase.Atoms <https://wiki.fysik.dtu.dk/ase/ase/atoms.html>`_

    .. note::

        Only change the units of length, and others still keep the units of ASE.

             - positions : Bohr
             - cell : Bohr
             - celldisp : Bohr

    """

    allowed_methods = [
        "new_array",
        "set_cell",
        "set_celldisp",
        "get_celldisp",
        "set_tags",
        "get_tags",
        "set_array",
        "set_initial_magnetic_moments",
        "get_initial_magnetic_moments",
        "set_initial_charges",
        "get_initial_charges",
        "get_ncharges",
        "set_charges",
        "get_charges",
        "get_scaled_positions",
        "get_chemical_formula",
        "get_cell",
        "strf",
        "istrf",
        "symbols_uniq",
        "nat",
        "zval",
        "get_chemical_symbols",
        "set_pbc",
        "set_positions",
        "get_positions",
        "get_ncharges",
        "to_ase",
        "from_ase",
        "has",
        "repeat",
        "copy",
    ]
    allowed_attributes = [
        "init_options",
        "symbols",
        "positions",
        "numbers",
        "tags",
        "magmoms",
        "charges",
        "scaled_positions",
        "cell",
        "celldisp",
        "info",
        "velocities",
        "get_ncharges",
        "get_charges",
        "set_charges",
        "strf",
        "istrf",
        "symbols_uniq",
        "nat",
        "zval",
        "arrays",
        "pbc",
        "constraints", # please do not use this
    ]

    def __getattribute__(self, name):
        attr = object.__getattribute__(self, name)
        is_special = (name.startswith("__") and name.endswith("__")) or name.startswith(
            "_"
        )

        if callable(attr):
            if not is_special and name not in Ions.allowed_methods:
                raise AttributeError(
                    f"Unsupported method `{name}` in {self.__class__.__name__}. Please use 'to_ase' method to convert from ASE object."
                )
        else:
            if not is_special and name not in Ions.allowed_attributes:
                raise AttributeError(
                    f"Unsupported attribute `{name}` in {self.__class__.__name__}. Please use 'to_ase' method to convert from ASE object.")
        return attr

    def __init__(
            self,
            symbols=None,
            positions=None,
            numbers=None,
            tags=None,
            magmoms=None,
            charges=None,
            scaled_positions=None,
            cell=None,
            celldisp=None,
            info=None,
            pbc=True,
    ):
        if isinstance(symbols, Atoms):
            ase_ions = Ions.from_ase(symbols)
            self.__dict__.update(ase_ions.__dict__)
            return
        
        if hasattr(symbols, "get_positions") or (
                isinstance(symbols, (list, tuple))
                and len(symbols) > 0
                and isinstance(symbols[0], Atom)
        ):
            raise TypeError("Please use 'from_ase' method to convert from ASE object.")

        self.init_options = locals()
        for k in ["__class__", "self"]:
            self.init_options.pop(k, None)

        # Init ASE Atoms
        self._cellobj = Cell.new()
        self.arrays = {}
        if symbols is not None and numbers is not None:
            raise TypeError('Use only one of "symbols" and "numbers".')
        if symbols is not None:
            numbers = symbols2numbers(symbols)
        elif numbers is None:
            if positions is not None:
                natoms = len(positions)
            elif scaled_positions is not None:
                natoms = len(scaled_positions)
            else:
                natoms = 0
            numbers = np.zeros(natoms, int)
        self.new_array("numbers", numbers, int)

        if self.numbers.ndim != 1:
            raise ValueError('"numbers" must be 1-dimensional.')

        if cell is None:
            cell = np.zeros((3, 3))
        self.set_cell(cell)

        if celldisp is None:
            celldisp = np.zeros(shape=(3, 1))
        self.set_celldisp(celldisp)

        if positions is None:
            if scaled_positions is None:
                positions = np.zeros((len(self.arrays["numbers"]), 3))
            else:
                assert self.cell.rank == 3
                positions = np.dot(scaled_positions, self.cell)
        else:
            if scaled_positions is not None:
                raise TypeError('Use only one of "positions" and "scaled_positions".')
        self.new_array("positions", positions, float, (3,))
        self.set_tags(default(tags, 0))
        self.set_initial_magnetic_moments(default(magmoms, 0.0))
        self.set_initial_charges(default(charges, 0.0))
        self._pbc = np.array([True, True, True], dtype=bool)
        self.set_pbc(pbc)
        self._constraints = None
        self._calc = None

        if info is None:
            self.info = {}
        else:
            self.info = dict(info)

    def to_ase(self):
        atoms = Atoms(
            symbols=self.symbols,
            positions=self.get_positions() * Units.Bohr,
            # numbers=self.numbers,d
            tags=self.get_tags(),
            magmoms=self.get_initial_magnetic_moments() * (Units.A * Units.Bohr ** 2),
            charges=self.get_initial_charges(),
            cell=self.cell.array * Units.Bohr,
            celldisp=self.get_celldisp() * Units.Bohr,
            info=self.info,
            pbc=self.pbc,
        )
        return atoms

    @staticmethod
    def from_ase(atoms: Atoms):
        ions = Ions(
            symbols=atoms.symbols,
            positions=atoms.get_positions() / Units.Bohr,
            # numbers=atoms.numbers,
            tags=atoms.get_tags(),
            magmoms=atoms.get_initial_magnetic_moments() / (Units.A * Units.Bohr ** 2),
            charges=atoms.get_initial_charges(),
            cell=atoms.cell.array / Units.Bohr,
            celldisp=atoms.get_celldisp() / Units.Bohr,
            info=atoms.info,
        )
        return ions

    def get_ncharges(self):
        """Get total number of charges."""
        if not self.has("initial_charges"):
            raise AttributeError("Please call 'set_charges' before use 'charges'.")
        return self.arrays["initial_charges"].sum()

    def get_charges(self):
        """Get the atomic charges."""
        return self.get_initial_charges()

    def set_charges(self, charges=None):
        """Set the atomic charges."""
        if isinstance(charges, dict):
            values = []
            for s in self.symbols:
                if s not in charges:
                    raise AttributeError(f"{s} not in the charges")
                values.append(charges[s])
            charges = values
        elif isinstance(charges, (float, int)):
            charges = np.ones(self.nat) * charges
        self.set_initial_charges(charges=charges)

    @property
    def charges(self):
        """Get the atomic charges."""
        if not self.has("initial_charges"):
            raise AttributeError("Please call 'set_charges' before use 'charges'.")
        return self.arrays["initial_charges"]

    @charges.setter
    def charges(self, value):
        """Set the atomic charges."""
        if not self.has("initial_charges"):
            raise AttributeError("Please call 'set_charges' before use 'charges'.")
        self.arrays["initial_charges"][:] = value

    def strf(self, reciprocal_grid, iatom):
        """Returns the Structure Factor associated to i-th ion."""
        a = np.exp(
            -1j * np.einsum("lijk,l->ijk", reciprocal_grid.g, self.positions[iatom])
        )
        return a

    def istrf(self, reciprocal_grid, iatom):
        """Returns the Structure-Factor-like property associated to i-th ion."""
        a = np.exp(
            1j * np.einsum("lijk,l->ijk", reciprocal_grid.g, self.positions[iatom])
        )
        return a

    @property
    def symbols_uniq(self):
        """Unique symbols of ions"""
        return np.sort(np.unique(self.symbols))

    @property
    def nat(self):
        """Number of atoms"""
        return len(self)

    @property
    def zval(self):
        """Valance charge (atomic charge) of each atomic type"""
        zval = dict.fromkeys(self.symbols_uniq, 0)
        symbols = self.get_chemical_symbols()
        try:
            self.charges[0]
        except Exception:
            return zval

        for k in zval:
            for i in range(self.nat):
                if symbols[i] == k:
                    zval[k] = self.charges[i]
                    break
        return zval

    def set_constraint(self, constraints):
        raise AttributeError(f"Unsupported attribute `constraints` in {self.__class__.__name__}. Please use 'to_ase' method to convert from ASE object.")
