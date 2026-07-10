"""Minimal test for LocalPseudo OFPP auto-resolution."""

from pathlib import Path

from dftpy.formats.vasp import read_POSCAR
from dftpy.functional.pseudo import LocalPseudo
from dftpy.grid import DirectGrid

DATA = Path(__file__).resolve().parents[2] / "examples" / "DATA"


def test_local_pseudo_resolves_pp_without_pp_list():
    ions = read_POSCAR(DATA / "fcc.vasp")
    grid = DirectGrid(lattice=ions.cell, nr=(8, 8, 8))

    pseudo = LocalPseudo(grid=grid, ions=ions, search_paths=[DATA], offline=True)

    assert Path(pseudo.PP_list["Al"]).is_file()
    assert "Al" in Path(pseudo.PP_list["Al"]).name
