"""Resolve and cache pseudopotential files from the OFPP library."""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from dftpy.mpi import SerialComm, sprint

GITHUB_REPO = "Quantum-MultiScale/OFPP"
GITHUB_BRANCH = "main"
GITHUB_RAW = f"https://raw.githubusercontent.com/{GITHUB_REPO}/{GITHUB_BRANCH}"
OEPP_RAW = "https://gitlab.com/wenhui/OEPP/-/raw/master"

DEFAULT_FAMILIES = ("OEPP", "PGBRV0.2")
DEFAULT_OEPP_FORMATS = ("recpot", "upf")
DEFAULT_HQLPP_FORMATS = ("recpot", "upf")

PP_CONFIG_RESERVED = frozenset(
    {"auto", "families", "cache_dir", "format", "search_paths"}
)

NLPP_LIBRARIES = {
    "PGBRV0.2": ("NLPP/PGBRV0.2", "{Element}_gbrv_new.psp8"),
    "PGBRV1.0": ("NLPP/PGBRV1.0", "{Element}_gbrv_new.psp8"),
    "PPSL0.2": ("NLPP/PPSL0.2", "{Element}_psl_new.psp8"),
    "PPSL1.0": ("NLPP/PPSL1.0", "{Element}_psl_new.psp8"),
}

HQLPP_FORMATS = {
    "recpot": ("HQLPP/recpot", "{element}_lps.pbe.recpot"),
    "upf": ("HQLPP/upf", "{element}_lps.pbe.upf"),
}

FAMILY_ALIASES = {
    "PGBRV02": "PGBRV0.2",
    "PGBRV10": "PGBRV1.0",
    "PPSL02": "PPSL0.2",
    "PPSL10": "PPSL1.0",
    "OEPP_RECPOT": "OEPP:recpot",
    "OEPP_UPF": "OEPP:upf",
    "OEPP_CPI": "OEPP:cpi",
    "HQLPP_RECPOT": "HQLPP:recpot",
    "HQLPP_UPF": "HQLPP:upf",
}


class PPNotFoundError(FileNotFoundError):
    """Raised when no pseudopotential can be resolved for an element."""


@dataclass(frozen=True)
class PPRecord:
    symbol: str
    family: str
    relpath: str
    url: str
    filename: str
    urls: tuple[str, ...] = ()

    def __post_init__(self):
        if not self.urls:
            object.__setattr__(self, "urls", (self.url,))

    @property
    def cache_name(self) -> str:
        return f"{self.family.replace(':', '_')}/{self.filename}"


def _default_cache_dir() -> Path:
    env = os.environ.get("DFTPY_PP_CACHE")
    if env:
        return Path(env).expanduser()
    return Path.home() / ".cache" / "dftpy" / "ofpp"


def _bundled_catalog_path() -> Path:
    return Path(__file__).resolve().parents[2] / "data" / "ofpp_catalog.json"


def _normalize_symbol(symbol: str) -> str:
    sym = str(symbol).strip()
    if len(sym) == 1:
        return sym.upper()
    return sym[0].upper() + sym[1:].lower()


def _normalize_family(name: str) -> str:
    key = name.strip().upper().replace(" ", "")
    return FAMILY_ALIASES.get(key, name.strip())


def _parse_families(families: str | Sequence[str] | None) -> tuple[str, ...]:
    if families is None:
        return DEFAULT_FAMILIES
    if isinstance(families, str):
        parts = re.split(r"[\s,]+", families.strip())
    else:
        parts = list(families)
    return tuple(_normalize_family(part) for part in parts if part)


def _split_family(name: str) -> tuple[str, str | None]:
    if ":" in name:
        base, fmt = name.split(":", 1)
        return base, fmt.lower()
    return name, None


class OFPPResolver:
    """Locate or download OFPP pseudopotentials for chemical elements."""

    def __init__(
        self,
        families: str | Sequence[str] | None = None,
        cache_dir: Path | str | None = None,
        search_paths: Sequence[Path | str] | None = None,
        catalog_path: Path | str | None = None,
        offline: bool = False,
        comm=None,
    ):
        self.families = _parse_families(families)
        self.cache_dir = Path(cache_dir or _default_cache_dir())
        self.search_paths = [Path(p) for p in (search_paths or [])]
        self.offline = offline
        self.comm = comm or SerialComm()
        self._catalog = self._load_catalog(catalog_path)
        self._github_listings: dict[str, frozenset[str]] = {}

    def resolve(self, symbol: str) -> Path:
        symbol = _normalize_symbol(symbol)
        local = self._find_local(symbol)
        if local is not None:
            return local

        errors: list[str] = []
        for family in self.families:
            record = self._resolve_in_family(symbol, family)
            if record is None:
                continue

            dest = self.cache_dir / record.cache_name
            if dest.is_file():
                return dest

            if self.offline:
                errors.append(f"{family}: not cached ({record.url})")
                continue

            downloaded = False
            for url in record.urls:
                try:
                    self._download(url, dest)
                    downloaded = True
                    break
                except Exception as exc:
                    errors.append(f"{family} ({url}): {exc}")
                    sprint(
                        f"OFPP: failed {symbol} from {family} via {url}: {exc}",
                        comm=self.comm,
                        level=2,
                    )
            if not downloaded:
                continue

            sprint(
                f"OFPP: cached {symbol} from {record.family} -> {dest}",
                comm=self.comm,
                level=1,
            )
            return dest

        families = ", ".join(self.families)
        detail = "; ".join(errors) if errors else "no matching records"
        raise PPNotFoundError(
            f"No pseudopotential for {symbol} in OFPP families: {families} ({detail})"
        )

    def resolve_for_ions(self, ions) -> dict[str, Path]:
        symbols = [_normalize_symbol(s) for s in ions.symbols_uniq]
        return {sym: self.resolve(sym) for sym in symbols}

    def pp_list_for_ions(self, ions) -> dict[str, str]:
        return {sym: str(path) for sym, path in self.resolve_for_ions(ions).items()}

    def _load_catalog(self, catalog_path: Path | str | None) -> dict:
        path = Path(catalog_path) if catalog_path else _bundled_catalog_path()
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)

    def _find_local(self, symbol: str) -> Path | None:
        """Search ``search_paths`` only, preferring ``self.families`` order.

        The download cache is checked per-family in :meth:`resolve` so a
        lower-priority cached family cannot shadow a preferred remote one.
        """
        if not self.search_paths:
            return None
        for family in self.families:
            base, fmt = _split_family(family)
            for pattern in self._local_patterns(symbol, base, fmt):
                for root in self.search_paths:
                    if not root.is_dir():
                        continue
                    matches = sorted(root.glob(pattern))
                    if matches:
                        return matches[0]
        # Last resort: any recognizable PP in search_paths (manual drops).
        generic = (
            f"{symbol.lower()}.lda.recpot",
            f"{symbol.lower()}.lda.upf",
        )
        for root in self.search_paths:
            if not root.is_dir():
                continue
            for pattern in generic:
                matches = sorted(root.glob(pattern))
                if matches:
                    return matches[0]
        return None

    @staticmethod
    def _local_patterns(symbol: str, family: str, fmt: str | None) -> tuple[str, ...]:
        if family == "OEPP":
            if fmt == "upf":
                return (f"{symbol}_OEPP_PZ.UPF",)
            if fmt == "recpot":
                return (f"{symbol}_lda.oe*.recpot",)
            return (f"{symbol}_lda.oe*.recpot", f"{symbol}_OEPP_PZ.UPF")
        if family == "HQLPP":
            if fmt == "upf":
                return (f"{symbol.lower()}_lps.pbe.upf",)
            if fmt == "recpot":
                return (f"{symbol.lower()}_lps.pbe.recpot",)
            return (
                f"{symbol.lower()}_lps.pbe.recpot",
                f"{symbol.lower()}_lps.pbe.upf",
            )
        if family in NLPP_LIBRARIES:
            if family.startswith("PGBRV"):
                return (f"{symbol}_gbrv_new.psp8", f"{symbol}_pgbrv*.psp8")
            if family.startswith("PPSL"):
                return (f"{symbol}_psl_new.psp8",)
        return (
            f"{symbol}_lda.oe*.recpot",
            f"{symbol}_OEPP_PZ.UPF",
            f"{symbol}_gbrv_new.psp8",
            f"{symbol}_psl_new.psp8",
            f"{symbol}_pgbrv*.psp8",
            f"{symbol.lower()}_lps.pbe.recpot",
            f"{symbol.lower()}_lps.pbe.upf",
            f"{symbol.lower()}.lda.recpot",
            f"{symbol.lower()}.lda.upf",
        )

    def _resolve_remote(self, symbol: str) -> PPRecord | None:
        for family in self.families:
            record = self._resolve_in_family(symbol, family)
            if record is not None:
                return record
        return None

    def _resolve_in_family(self, symbol: str, family: str) -> PPRecord | None:
        base, fmt = _split_family(_normalize_family(family))
        if base == "OEPP":
            formats = (fmt,) if fmt else DEFAULT_OEPP_FORMATS
            return self._resolve_oepp(symbol, formats, family_label=base)
        if base == "HQLPP":
            formats = (fmt,) if fmt else DEFAULT_HQLPP_FORMATS
            return self._resolve_hqlpp(symbol, formats, family_label=base)
        if base in NLPP_LIBRARIES:
            return self._resolve_nlpp(symbol, base)
        raise ValueError(f"Unknown OFPP family: {family}")

    def _resolve_oepp(
        self, symbol: str, formats: Sequence[str], family_label: str
    ) -> PPRecord | None:
        catalog = self._catalog.get("OEPP", {})
        entry = catalog.get(symbol)
        for fmt in formats:
            if entry and fmt in entry:
                relpath = entry[fmt]
                filename = Path(relpath).name
                # Prefer OFPP GitHub mirror (catalog paths); GitLab is upstream fallback.
                gitlab_path = relpath[5:] if relpath.startswith("OEPP/") else relpath
                urls = (
                    f"{GITHUB_RAW}/{relpath}",
                    f"{OEPP_RAW}/{gitlab_path}",
                )
                return PPRecord(
                    symbol,
                    family_label,
                    relpath,
                    urls[0],
                    filename,
                    urls=urls,
                )
        return None

    def _resolve_hqlpp(
        self, symbol: str, formats: Sequence[str], family_label: str
    ) -> PPRecord | None:
        for fmt in formats:
            if fmt not in HQLPP_FORMATS:
                continue
            subdir, pattern = HQLPP_FORMATS[fmt]
            filename = pattern.format(element=symbol.lower())
            if self._github_file_exists(subdir, filename):
                relpath = f"{subdir}/{filename}"
                url = f"{GITHUB_RAW}/{relpath}"
                return PPRecord(symbol, family_label, relpath, url, filename)
        return None

    def _resolve_nlpp(self, symbol: str, family: str) -> PPRecord | None:
        subdir, pattern = NLPP_LIBRARIES[family]
        filename = pattern.format(Element=symbol)
        if not self._github_file_exists(subdir, filename):
            return None
        relpath = f"{subdir}/{filename}"
        url = f"{GITHUB_RAW}/{relpath}"
        return PPRecord(symbol, family, relpath, url, filename)

    def _github_file_exists(self, subdir: str, filename: str) -> bool:
        return filename in self._list_github_dir(subdir)

    def _list_github_dir(self, subdir: str) -> frozenset[str]:
        if subdir in self._github_listings:
            return self._github_listings[subdir]
        url = f"https://api.github.com/repos/{GITHUB_REPO}/contents/{subdir}?ref={GITHUB_BRANCH}"
        request = urllib.request.Request(url, headers={"Accept": "application/vnd.github+json"})
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                payload = json.load(response)
        except urllib.error.HTTPError:
            names = frozenset()
        else:
            names = frozenset(item["name"] for item in payload if item.get("type") == "file")
        self._github_listings[subdir] = names
        return names

    def _download(self, url: str, dest: Path) -> None:
        dest.parent.mkdir(parents=True, exist_ok=True)
        if self.comm.rank == 0:
            tmp = dest.with_suffix(dest.suffix + ".part")
            try:
                with urllib.request.urlopen(url, timeout=60) as response:
                    tmp.write_bytes(response.read())
                tmp.replace(dest)
            except Exception:
                if tmp.exists():
                    tmp.unlink()
                raise
        self.comm.Barrier()
        if not dest.is_file():
            raise PPNotFoundError(f"Failed to download pseudopotential from {url}")


def build_pp_list(
    ions,
    pp_config: dict,
    pppath: str | Path,
    ofpp_config: dict | None = None,
    resolver: OFPPResolver | None = None,
) -> dict[str, str]:
    """Build a ``PP_list`` dict from ini config, optional manual entries, and OFPP auto."""
    ofpp_config = ofpp_config or {}
    auto = _config_bool(ofpp_config.get("auto", False))
    pppath = Path(pppath)

    pp_list: dict[str, str] = {}
    for key, value in pp_config.items():
        if key.lower() in PP_CONFIG_RESERVED:
            continue
        if not value:
            continue
        symbol = _normalize_symbol(key)
        path = Path(value)
        if not path.is_absolute():
            path = pppath / value
        pp_list[symbol] = str(path)

    if not auto:
        return pp_list

    if resolver is None:
        search_paths = [pppath]
        extra = ofpp_config.get("search_paths")
        if extra:
            if isinstance(extra, str):
                search_paths.extend(Path(p) for p in re.split(r"[\s,]+", extra) if p)
            else:
                search_paths.extend(Path(p) for p in extra)
        resolver = OFPPResolver(
            families=ofpp_config.get("families"),
            cache_dir=ofpp_config.get("cache_dir"),
            search_paths=search_paths,
            offline=_config_bool(ofpp_config.get("offline", False)),
        )

    for symbol in (_normalize_symbol(s) for s in ions.symbols_uniq):
        if symbol in pp_list:
            continue
        pp_list[symbol] = str(resolver.resolve(symbol))
    return pp_list


def _config_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "on"}
