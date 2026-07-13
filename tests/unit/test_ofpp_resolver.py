"""Tests for automatic OFPP pseudopotential resolution."""

from __future__ import annotations

from pathlib import Path

import pytest
import urllib.error

from dftpy.formats.vasp import read_POSCAR
from dftpy.functional.pseudo import LocalPseudo
from dftpy.functional.pseudo.ofpp_resolver import (
    OFPPResolver,
    PPNotFoundError,
    build_pp_list,
)
from dftpy.grid import DirectGrid

DATA = Path(__file__).resolve().parents[2] / "examples" / "DATA"
POSCAR_AL = DATA / "fcc.vasp"


class _FakeIons:
    symbols_uniq = ["Al"]


def test_resolve_al_oepp_from_local_search_paths():
    resolver = OFPPResolver(
        search_paths=[DATA],
        cache_dir=Path("/tmp/dftpy_ofpp_test_cache"),
        offline=True,
    )
    path = resolver.resolve("Al")
    assert path.name == "Al_lda.oe01.recpot"
    assert path.is_file()


def test_resolve_au_pgbrv_from_local_legacy_name():
    resolver = OFPPResolver(
        families=("PGBRV0.2",),
        search_paths=[DATA],
        cache_dir=Path("/tmp/dftpy_ofpp_test_cache_au"),
        offline=True,
    )
    path = resolver.resolve("Au")
    assert path.name == "Au_pgbrv02.psp8"
    assert path.is_file()


def test_resolve_missing_element_raises():
    resolver = OFPPResolver(
        families=("OEPP",),
        search_paths=[],
        cache_dir=Path("/tmp/dftpy_ofpp_test_cache_missing"),
        offline=True,
    )
    with pytest.raises(PPNotFoundError, match="No pseudopotential for Fe"):
        resolver.resolve("Fe")


def test_build_pp_list_auto_merges_manual_and_resolved(tmp_path):
    ions = _FakeIons()
    pp_config = {"Mg": "Mg_lda.oe01.recpot"}
    pp_list = build_pp_list(
        ions=ions,
        pp_config=pp_config,
        pppath=DATA,
        ofpp_config={"auto": True},
        resolver=OFPPResolver(
            search_paths=[DATA],
            cache_dir=tmp_path / "cache",
            offline=True,
        ),
    )
    assert Path(pp_list["Mg"]).name == "Mg_lda.oe01.recpot"
    assert Path(pp_list["Al"]).name == "Al_lda.oe01.recpot"


def test_build_pp_list_manual_only():
    ions = _FakeIons()
    pp_list = build_pp_list(
        ions=ions,
        pp_config={"Al": "Al_lda.oe01.recpot"},
        pppath=DATA,
        ofpp_config={"auto": False},
    )
    assert len(pp_list) == 1
    assert Path(pp_list["Al"]).name == "Al_lda.oe01.recpot"


def test_local_pseudo_auto_resolve(tmp_path):
    if not POSCAR_AL.is_file():
        pytest.skip("missing fcc.vasp in examples/DATA")
    ions = read_POSCAR(POSCAR_AL)
    grid = DirectGrid(lattice=ions.cell, nr=(8, 8, 8))
    pseudo = LocalPseudo(
        grid=grid,
        ions=ions,
        search_paths=[DATA],
        resolver=OFPPResolver(
            search_paths=[DATA],
            cache_dir=tmp_path / "cache",
            offline=True,
        ),
    )
    assert "Al" in pseudo.PP_list
    assert Path(pseudo.PP_list["Al"]).is_file()


def test_resolve_falls_through_on_download_failure(monkeypatch, tmp_path):
    resolver = OFPPResolver(
        families=("OEPP", "PGBRV0.2"),
        search_paths=[],
        cache_dir=tmp_path / "cache",
        offline=False,
    )
    monkeypatch.setattr(
        resolver,
        "_list_github_dir",
        lambda subdir: frozenset({"Al_gbrv_new.psp8"}),
    )

    def _fake_download(url, dest):
        if "gitlab.com" in url or "OEPP" in url:
            raise urllib.error.HTTPError(url, 403, "Forbidden", hdrs=None, fp=None)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text("dummy pp")

    monkeypatch.setattr(resolver, "_download", _fake_download)
    path = resolver.resolve("Al")
    assert path.name == "Al_gbrv_new.psp8"
    assert path.is_file()


def test_custom_family_hqlpp_offline(monkeypatch, tmp_path):
    resolver = OFPPResolver(
        families=("HQLPP:recpot",),
        search_paths=[],
        cache_dir=tmp_path / "cache",
        offline=False,
    )
    monkeypatch.setattr(
        resolver,
        "_list_github_dir",
        lambda subdir: frozenset({"al_lps.pbe.recpot"}),
    )
    def _fake_download(url, dest):
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text("dummy pp")

    monkeypatch.setattr(resolver, "_download", _fake_download)
    path = resolver.resolve("Al")
    assert path.name == "al_lps.pbe.recpot"
    assert path.is_file()
