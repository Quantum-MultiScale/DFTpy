"""Explicit core-shell polarization modes for orbital-free linear response.

The d electrons sit in the core of the valence pseudopotential and carry no
orbital-free density, so this module rebuilds them as explicit dipole
oscillators coupled to the valence response: three rigid-shift (Sternheimer)
modes per atom, dn = -delta . grad n_d, scaled so that alpha = 2 D^2 / Omega
reproduces a polarizability taken from data that is not the quantity being
predicted.  D_SOURCE picks "atomic" (free-atom d -> virtual transitions) or
"bulk" (poles from a measured interband response).

K_sd and K_dd carry Hartree + f_xc ONLY -- a local kinetic approximant applied
to the d modes flips the sign of the d effect -- and K_dd is off-site only.
The d subsystem enters the RESPONSE only; the pseudopotential already carries
the static d contribution.

NO element data is hard-coded here and there is no catalog to fall back on.
The shell is in the OEPP CORE, so its parameters and radial orbitals are
properties of the ion and travel in the pseudopotential's d block, which
oepp.py writes.  Name that UPF with the upf= argument, DFTPY_D_UPF, or
D_POLES; without one, element_data() raises rather than guess.
"""

from pathlib import Path

import base64
import json
import os
import re
import zlib

import numpy as np

from dftpy.constants import ENERGY_CONV
from dftpy.grid import DirectGrid


HA = ENERGY_CONV["Hartree"]["eV"]
GAMMA_EV = 0.10                     # presentation default, not element data

_L_OF = {"s": 0, "p": 1, "d": 2, "f": 3, "g": 4}
_CACHE = {}

_DBLOCK_RE = re.compile(
    r"DFTPY_DBLOCK_BEGIN\s*(\{.*?\})\s*DFTPY_DBLOCK_END", re.S)


def _block_of(path):
    """The parsed d block of a UPF, or None."""
    try:
        txt = Path(path).read_text(errors="replace")
    except OSError:
        return None
    m = _DBLOCK_RE.search(txt)
    return json.loads(m.group(1)) if m else None


def params_from_upf(path):
    """The element_params section of a UPF's d block, or None."""
    blk = _block_of(path)
    p = (blk or {}).get("element_params")
    return dict(p, symbol=blk.get("element")) if p else None


def _resolve_upf(upf=None):
    """The pseudopotential to take the d-shell data from."""
    src = upf or os.environ.get("DFTPY_D_UPF") or os.environ.get("D_POLES")
    if src and str(src).lower().endswith(".upf") and Path(src).exists():
        return str(src)
    raise ValueError(
        "no pseudopotential to take d-shell data from. The shell lives in the "
        "OEPP core, so its parameters travel in the UPF's d block: pass "
        "upf=<Ag_OEPP_*.UPF>, or set DFTPY_D_UPF or D_POLES. Write the block "
        "with dftpy.functional.pseudo.oepp.OEPP.attach_d_block."
        + (f" (got {src!r})" if src else ""))


def element_data(upf=None):
    """d-shell parameters, read from the pseudopotential's d block.

    Adds derived fields: n_shell, l_shell and n_elec = 2(2l+1).
    """
    src = _resolve_upf(upf)
    if src in _CACHE:
        return _CACHE[src]
    el = params_from_upf(src)
    if el is None:
        raise ValueError(
            f"{src}: no element_params in its d block. Re-stamp it with "
            f"OEPP.attach_d_block(..., element_params=..., shell_data=...).")
    shell = el["shell"]
    el["n_shell"], el["l_shell"] = int(shell[:-1]), _L_OF[shell[-1]]
    el["n_elec"] = float(2 * (2 * el["l_shell"] + 1))
    el["peak_win_ev"] = tuple(el["peak_win_ev"])
    el["source"] = Path(src).name
    _CACHE[src] = el
    return el


def _decode(entry):
    b64 = entry["b64"]
    if not isinstance(b64, str):          # stored wrapped, to keep UPF lines short
        b64 = "".join(b64)
    a = np.frombuffer(zlib.decompress(base64.b64decode(b64)),
                      dtype=entry.get("dtype", "float64"))
    return a.reshape(entry["shape"]) if entry.get("shape") else a


def shell_arrays(upf=None):
    """(r, u_core, virtual_labels, virtual_ur, delta_e_Ha) from the UPF."""
    src = _resolve_upf(upf)
    key = "arrays@" + src
    if key in _CACHE:
        return _CACHE[key]
    sd = (_block_of(src) or {}).get("shell_data")
    if sd is None:
        raise ValueError(
            f"{src}: its d block carries no shell_data (the radial core and "
            f"virtual orbitals). Re-stamp it with OEPP.attach_d_block("
            f"..., shell_data=OEPP.encode_shell_data(<npz>)).")
    out = (_decode(sd["r"]), _decode(sd["u_core"]),
           [str(x).strip() for x in sd["virtual_labels"]],
           _decode(sd["virtual_ur"]), _decode(sd["delta_e_ha"]))
    _CACHE[key] = out
    return out


class Grid(DirectGrid):
    """DirectGrid plus the Poisson solve and minimum-image helper used here.

    Takes (nr, lattice) rather than DirectGrid's (lattice, nr).  Orthorhombic
    only: dist_vec wraps on the cell lengths.
    """

    def __init__(self, nr, lattice):
        lattice = np.asarray(lattice, float)
        off = lattice - np.diag(np.diag(lattice))
        if np.abs(off).max() > 1e-10 * max(np.abs(lattice).max(), 1.0):
            raise ValueError("Grid is orthorhombic only; lattice is not "
                             "diagonal")
        super().__init__(lattice=lattice, nr=tuple(int(n) for n in nr),
                         full=True)
        self.L = np.diag(lattice).copy()
        rec = self.get_reciprocal()
        self.Gx, self.Gy, self.Gz = rec.g
        self.G2, self.invG2 = rec.gg, rec.invgg
        self.X, self.Y, self.Z = self.r

    def poisson(self, rho):
        return np.fft.ifftn(4 * np.pi * np.fft.fftn(rho) * self.invG2).real

    def dist_vec(self, R):
        """Minimum-image vector from R to every grid point."""
        d = [self.X - R[0], self.Y - R[1], self.Z - R[2]]
        for k in range(3):
            d[k] = d[k] - self.L[k] * np.round(d[k] / self.L[k])
        return d


def coordination(positions, L, r_cut=None, upf=None):
    """Neighbours within r_cut of each atom.  ALL LENGTHS IN BOHR."""
    if r_cut is None:
        r_cut = element_data(upf)["coordination_r_cut"]
    n = len(positions)
    cn = np.zeros(n)
    for i in range(n):
        d = positions - positions[i]
        d = (d + L / 2.0) % L - L / 2.0
        r = np.linalg.norm(d, axis=1)
        cn[i] = float(np.sum((r > 1e-6) & (r < r_cut)))
    return cn


# "atomic" = the atomic core -> virtual transitions, "bulk" = poles from
# D_POLES (an npz from poles_from_eps.py, or a UPF carrying them in PP_INFO --
# the shell is in the OEPP CORE, so alpha_d(omega) travels with the file).
D_SOURCE = os.environ.get("D_SOURCE", "atomic").lower()
D_POLES = os.environ.get("D_POLES")
D_POLE_SET = os.environ.get("D_POLE_SET") or None


def poles_from_upf(path, which=None):
    """(omega_ha, f, meta) from the d block embedded in a UPF's PP_INFO."""
    txt = Path(path).read_text(errors="replace")
    m = _DBLOCK_RE.search(txt)
    if m is None:
        raise ValueError(f"{path}: no DFTPY_DBLOCK block in PP_INFO")
    blk = json.loads(m.group(1))
    name = which or blk.get("default")
    if name not in blk.get("pole_sets", {}):
        raise ValueError(f"{path}: pole set {name!r} not in "
                         f"{sorted(blk.get('pole_sets', {}))}")
    s = blk["pole_sets"][name]
    conv = blk.get("convention")
    meta = dict(pole_set=name,
                convention=" ".join(conv) if isinstance(conv, list) else conv,
                provenance=s.get("provenance"), pole_source=s.get("source"),
                u_eff_ev_applied=float(s.get("u_eff_ev_applied", 0.0)))
    return (np.asarray(s["omega_ha"], float), np.asarray(s["f"], float), meta)


def upf_has_dblock(path):
    """True if this UPF carries POLE SETS in its d block.

    A block holding only element_params/shell_data does not count: callers use
    this to decide whether to point D_POLES at the file.
    """
    return bool((_block_of(path) or {}).get("pole_sets"))


def load_poles(src=None, which=None):
    """(omega_ha, f, meta) from either a poles npz or a UPF.

    Resolved at CALL time: the driver points D_POLES at the pseudopotential it
    is using, possibly after this module was imported.
    """
    src = str(src or os.environ.get("D_POLES") or D_POLES)
    which = which or os.environ.get("D_POLE_SET") or D_POLE_SET
    if src.lower().endswith(".upf"):
        return poles_from_upf(src, which)
    z = np.load(src, allow_pickle=True)
    meta = dict(pole_set=Path(src).stem, convention=None, provenance="npz",
                pole_source=str(z["source"]) if "source" in z.files else None,
                u_eff_ev_applied=0.0)
    return (np.asarray(z["omega_ha"], float), np.asarray(z["f"], float), meta)


def _bulk_transitions(f_min=1e-5, upf=None):
    """(omega_j, f_j) from the bulk interband response, on the atomic shape.

    Only energies and strengths come from the bulk; the radial shape stays the
    atomic core -> dominant-virtual product, which the atomic data gets right.
    The atomic ENERGIES are what it gets wrong (lowest pole 6.82 eV against a
    real interband onset near 3.9 eV), and those are what this replaces.
    """
    el = element_data(upf)
    w_j, f_j, pole_meta = load_poles()
    r, u_core, labels, virt_ur, _ = shell_arrays(upf)
    dom = el.get("dominant_virtual")
    kd = labels.index(dom) if dom in labels else 0
    shape = u_core * virt_ur[kd] / np.maximum(r, 1e-12) ** 2

    keep = []
    for j, (de, f) in enumerate(zip(w_j, f_j)):
        if f < f_min:
            continue
        keep.append((f"bulk{j + 1}", float(de), float(f),
                     float(np.sqrt(f / (2.0 * de))), shape))
    n_t = len(keep)
    info = dict(n_trans=n_t, per_atom=3 * n_t,
                labels=[x[0] for x in keep],
                omega_ev=[x[1] * HA for x in keep],
                f=[x[2] for x in keep],
                alpha=float(sum(x[2] / x[1] ** 2 for x in keep)),
                r=r, source="bulk", poles_file=str(D_POLES),
                element=el["symbol"], shell=el["shell"], **pole_meta)
    return keep, info


def _transition_rows(n_elec=None, f_min=1e-5, upf=None):
    """(label, delta_e_Ha, f, D, radial_shape) per core -> virtual transition.

    Shared by every atomic-route builder.  The line-strength factor
    max(l_core, l_virt) / (2 l_core + 1) and the l = 1 angular shape keep only
    the dipole-active part; each mode is renormalised to D, so dropping the
    higher-l pieces of the core x nf product moves the profile, not the
    strength.
    """
    el = element_data(upf)
    l_c = el["l_shell"]
    n_elec = el["n_elec"] if n_elec is None else float(n_elec)
    r, u_core, labels, virt_ur, de_all = shell_arrays(upf)

    keep = []
    for k, lbl in enumerate(labels):
        de = float(de_all[k])
        uv = virt_ur[k]
        Rrad = float(np.trapz(u_core * r * uv, r))
        l_v = _L_OF.get(lbl[-1], 1)
        ang = max(l_c, l_v) / (2 * l_c + 1)
        f = (2.0 / 3.0) * de * Rrad * Rrad * ang * n_elec
        if f < f_min:
            continue
        keep.append((lbl, de, f, np.sqrt(f / (2.0 * de)),
                     u_core * uv / np.maximum(r, 1e-12) ** 2))
    return keep, r, el


def _info_of(keep, r, el, **extra):
    return dict(n_trans=len(keep), per_atom=3 * len(keep),
                labels=[x[0] for x in keep],
                omega_ev=[x[1] * HA for x in keep],
                f=[x[2] for x in keep],
                alpha=float(sum(x[2] / x[1] ** 2 for x in keep)),
                r=r, element=el["symbol"], shell=el["shell"], **extra)


def _explicit_transitions(n_elec=None, f_min=1e-5, upf=None):
    """The atomic transition list, with no grid work.

    Split out so the large-system path can size and index the d block without
    materialising any modes.
    """
    if D_SOURCE == "bulk":
        return _bulk_transitions(f_min=f_min, upf=upf)
    keep, r, el = _transition_rows(n_elec, f_min, upf)
    return keep, _info_of(keep, r, el)


def build_explicit_chunk(grid, pos, keep, n_t, c0, c1, r):
    """Rows c0:c1 of the explicit d-mode set, ordered A*(3*n_t) + k*3 + c."""
    out = np.zeros((c1 - c0, int(np.prod(grid.nr))))
    cache_A, cache = None, None
    for row in range(c0, c1):
        A, rem = divmod(row, 3 * n_t)
        kk, comp_i = divmod(rem, 3)
        if A != cache_A:
            dx, dy, dz = grid.dist_vec(pos[A])
            dist = np.sqrt(dx * dx + dy * dy + dz * dz)
            cache = (dx, dy, dz, dist, np.maximum(dist, 1e-12))
            cache_A = A
        dx, dy, dz, dist, dsafe = cache
        lbl, de, f, D, rad = keep[kk][:5]
        base = np.interp(np.clip(dist, r[0], r[-1]), r, rad)
        comp = (dx, dy, dz)[comp_i]
        m = base * (comp / dsafe)
        dip = float(np.sum(comp * m) * grid.dV)
        out[row - c0] = (m * (D / dip)).ravel()
    return out


def build_d_modes_local(grid, positions, cut_bohr=10.0, n_elec=None,
                        f_min=1e-5, upf=None):
    """The explicit d modes, each held on a cube around its own atom.

    Full-grid modes cost tens of GB at a few hundred atoms, but the modes are
    atom-local (the widest radial part is at 0.1% of its maximum by ~12 bohr).
    Each is renormalised to its dipole D on the cube, so truncation moves the
    shape a little and the oscillator strength not at all.  Ordering matches
    build_d_modes_explicit: row = A*(3*n_t) + k*3 + c.
    """
    keep, info = _explicit_transitions(n_elec, f_min, upf)
    r = info["r"]
    n_t, n_at = len(keep), len(positions)
    h = grid.L / grid.nr
    half = np.maximum(np.ceil(cut_bohr / h).astype(int), 1)
    off = [np.arange(-half[k], half[k] + 1) for k in range(3)]
    shape = tuple(len(o) for o in off)

    cubes, index = [], []
    for A in positions:
        centre = np.round(np.asarray(A, float) / h).astype(int)
        index.append([(centre[k] + off[k]) % grid.nr[k] for k in range(3)])
        d = [(centre[k] + off[k]) * h[k] - A[k] for k in range(3)]
        dx, dy, dz = np.meshgrid(*d, indexing="ij")
        dist = np.sqrt(dx * dx + dy * dy + dz * dz)
        dsafe = np.maximum(dist, 1e-12)
        block = np.empty((3 * n_t, int(np.prod(shape))))
        row = 0
        for lbl, de, f, D, rad in keep:
            base = np.interp(np.clip(dist, r[0], r[-1]), r, rad)
            for comp in (dx, dy, dz):
                m = base * (comp / dsafe)
                dip = float(np.sum(comp * m) * grid.dV)
                block[row] = (m * (D / dip)).ravel()
                row += 1
        cubes.append(block)

    info = dict(info)
    info.pop("r", None)
    info.update(n_atoms=n_at, cube_shape=shape, cut_bohr=float(cut_bohr))
    return dict(cubes=cubes, index=index,
                omegas=np.tile(np.repeat([k[1] for k in keep], 3), n_at),
                dipoles=np.tile(np.repeat([k[3] for k in keep], 3), n_at),
                info=info)


def build_d_modes_explicit(grid, positions, n_elec=None, f_min=1e-5,
                           upf=None):
    """Explicit d response from the atomic core -> virtual transitions.

    The ACTUAL atomic response rather than a single Lorentzian: one mode per
    (atom, transition, Cartesian direction), each at its own frequency.  Three
    modes of dipole D_k = sqrt(f_k / (2 w_k)) per transition give
        sum over 3 modes of (2/3) D^2 w = f_k            (f-sum)
        alpha_k = 2 D_k^2 / w_k         = f_k / w_k^2    (isotropic)
    which for Ag reproduces the atomic alpha = 10.70 bohr^3 with no fitted
    parameter, against 13.0 from the bulk Clausius-Mossotti route; the 18%
    shortfall is the continuum.

    Returns (modes, omegas, info), modes shape (3*n_trans*n_at, nnr), ordered
    atom-major so the on-site block of atom A is contiguous.
    """
    keep, r, el = _transition_rows(n_elec, f_min, upf)
    n_t = len(keep)
    n_at = len(positions)
    modes = np.zeros((3 * n_t * n_at, int(np.prod(grid.nr))))
    omegas = np.zeros(3 * n_t * n_at)
    row = 0
    for A in positions:
        dx, dy, dz = grid.dist_vec(A)
        dist = np.sqrt(dx * dx + dy * dy + dz * dz)
        dsafe = np.maximum(dist, 1e-12)
        for lbl, de, f, D, rad in keep:
            base = np.interp(np.clip(dist, r[0], r[-1]), r, rad)
            for comp in (dx, dy, dz):
                m = base * (comp / dsafe)                # l = 1 angular factor
                dip = float(np.sum(comp * m) * grid.dV)
                modes[row] = (m * (D / dip)).ravel()
                omegas[row] = de
                row += 1

    info = _info_of(keep, r, el)
    info.pop("r", None)
    return modes, omegas, info


def build_d_density(grid, positions, n_elec=None, upf=None):
    """Atomic core densities placed at each site, integrating to n_elec/atom.

    The d subsystem's own kernel must be evaluated at the d DENSITY, not the
    valence one: the Thomas-Fermi kernel goes as rho^(-1/3), so the small OEPP
    valence density gave an on-site coupling of 6.49 per unit dipole -- too
    large for any bare oscillator to be dressed down to 7.1 eV at alpha = 13.
    """
    el = element_data(upf)
    n_elec = el["n_elec"] if n_elec is None else float(n_elec)
    r, u, _, _, _ = shell_arrays(upf)
    n_rad = u**2 / (4.0 * np.pi * r**2)
    n_rad *= n_elec / np.trapz(u**2, r)
    out = np.zeros(grid.nr)
    for R in positions:
        dx, dy, dz = grid.dist_vec(R)
        dist = np.sqrt(dx*dx + dy*dy + dz*dz)
        out += np.interp(np.clip(dist, r[0], r[-1]), r, n_rad)
    return out


def build_d_modes(grid, positions, alpha_at, omega_d, upf=None):
    """Rigid-shift polarisation modes rho_{A,i} = -d_i n_d, scaled to dipole D.

    Returns (modes, D), modes shape (3*N_at, nnr), each satisfying
    int r_i rho_{A,i} d^3r = D and int rho = 0.
    """
    r, u, _, _, _ = shell_arrays(upf)
    with np.errstate(divide="ignore", invalid="ignore"):
        n_rad = u**2 / (4 * np.pi * r**2)          # 3D density shape
    n_rad[~np.isfinite(n_rad)] = n_rad[np.isfinite(n_rad)][0]
    n_rad /= np.trapz(u**2, r)                     # int n d^3r = 1
    rmax = float(r[-1])

    D_target = np.sqrt(alpha_at * omega_d / 2.0)   # alpha = 2 D^2 / Omega

    # FFT-differentiate the PLACED density rather than use the analytic
    # -(dn/dr)(r_i/|r|): that 1/|r| is badly behaved on a 0.49 a0 grid and made
    # the three Cartesian modes inequivalent (|m|max differed by 6x) even though
    # symmetry relates them.  FFT differentiation treats x, y, z identically.
    modes = []
    for R in positions:
        dx, dy, dz = grid.dist_vec(R)
        dist = np.sqrt(dx * dx + dy * dy + dz * dz)
        n_site = np.interp(np.clip(dist, r[0], rmax), r, n_rad)
        ng = np.fft.fftn(n_site)
        for G in (grid.Gx, grid.Gy, grid.Gz):
            m = -np.fft.ifftn(1j * G * ng).real     # -d_i n  (mean already 0)
            modes.append(m)
    modes = np.asarray(modes)

    for k, m in enumerate(modes):
        comp = k % 3
        R = positions[k // 3]
        dvec = grid.dist_vec(R)[comp]              # relative, periodic-safe
        dip = float(np.sum(dvec * m) * grid.dV)
        if abs(dip) < 1e-30:
            raise RuntimeError(f"mode {k} has vanishing dipole")
        modes[k] = m * (D_target / dip)
    return modes.reshape(len(modes), -1), D_target


def broaden(e, f, grid_ev, gamma=GAMMA_EV):
    s = np.zeros_like(grid_ev)
    for ei, fi in zip(e, f):
        if np.isfinite(ei) and np.isfinite(fi):
            s += fi * gamma**2 / ((grid_ev - ei) ** 2 + gamma**2)
    return s, grid_ev


def peak_of(e, f, win=None, gamma=GAMMA_EV, upf=None):
    """Position of the broadened maximum.

    win defaults to the pseudopotential's peak_win_ev when one is in play, and
    otherwise to the span of the data -- a search window is a presentation
    choice, so this helper stays usable with no UPF.
    """
    if win is None:
        try:
            win = element_data(upf)["peak_win_ev"]
        except (ValueError, KeyError):
            e = np.asarray(e, float)
            win = (float(np.min(e)), float(np.max(e)))
    g = np.linspace(win[0], win[1], 2000)
    s, _ = broaden(e, f, g, gamma)
    return float(g[np.argmax(s)])
