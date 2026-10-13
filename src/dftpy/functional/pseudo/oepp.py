"""
This class build OEPP pseudopotentials based on Mit etal (2016) paper. The slater average is computed from the KS psuedo wavefunctions using TM pseudopotential.

The KS pseudo-wavefunctions can be obtained from QEpy calculations

    from dftpy.functional.pseudo.oepp import OEPP

    oepp = OEPP.from_ks("qepy_U4/qepy_wfs.npz",
                        semilocal="Ag.pw-mt_fhi.UPF",
                        template="Ag_OEPP_PZ.UPF",
                        u_ks=4.0,
                        reference="qepy_U0/qepy_wfs.npz")
    oepp.write_upf("Ag_OEPP_U4_Mi2016.UPF")
    oepp.attach_d_block("Ag_OEPP_U4_Mi2016.UPF", u_applied=4.0)

An OEPP already on disk is read like any other DFTpy pseudopotential, so
``local_potential`` is in Hartree and ``radial_grid`` in bohr:

    oepp = OEPP("Ag_OEPP_U4_Mi2016.UPF")
"""

import datetime
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from dftpy.functional.pseudo.abstract_pseudo import BasePseudo

RY2HA = 0.5
HA2EV = 27.211386245988
EV2HA = 1.0 / HA2EV
RY2EV = HA2EV * RY2HA

DBLOCK_BEGIN = "DFTPY_DBLOCK_BEGIN"
DBLOCK_END = "DFTPY_DBLOCK_END"
DBLOCK_SCHEMA = "dftpy-dblock/1"
_DBLOCK_RE = re.compile(rf"\n?\s*{DBLOCK_BEGIN}\s*\{{.*?\}}\s*{DBLOCK_END}\s*", re.S)

DBLOCK_CONVENTION = ("bare polarizability; on-site K_dd resummed into chi^u; "
                     "off-site K_dd built at run time")
DBLOCK_SHAPE_SOURCE = ("examples/of_lr/ag_4d_chi_u_and_orbitals.npz, "
                       "atomic 4d->5p radial product")


def _wrap(text, width=88):
    """Long free text as a list of short lines.

    QE's Fortran UPF reader takes PP_INFO into fixed-length buffers, so no line
    of the embedded JSON may get long.  Readers join the list back up.
    """
    import textwrap
    return textwrap.wrap(str(text), width) if len(str(text)) > width else text


def _values(node):
    return np.fromstring(node.text or "", sep=" ")


def _format_upf_array(arr, columns=4):
    rows = [" ".join(f"{x: .16E}" for x in arr[i:i + columns])
            for i in range(0, len(arr), columns)]
    return "\n" + "\n".join(rows) + "\n"


class OEPP(BasePseudo):
    """A one-electron (z_valence = 1) local pseudopotential for DFTpy.

    Two ways in.  ``OEPP(fname)`` reads one that already exists.
    ``OEPP.from_ks(...)`` builds one from QEpy Kohn-Sham pseudo-wavefunctions
    by the Mi (2016) Eq. (15) Slater average over the occupied channels of a
    semilocal donor potential.
    """

    D_OCC_TOTAL = 10.0
    DEN_FRAC = 1.0e-3
    DEN_BLEND_FRAC = 1.0e-5
    SMOOTH_SIGMA_BINS = 3.0

    def __init__(self, fname=None, direct=True, **kwargs):
        self._v_ry = None
        self._u_ks = 0.0
        self._u_shift = 0.0
        self._channels_used = []
        self._build_info = {}
        super().__init__(fname=fname, direct=direct, **kwargs)

    # ------------------------------------------------------------------ read

    def read(self, fname, **kwargs):
        root = ET.parse(fname).getroot()
        header = root.find(".//PP_HEADER")
        self._zval = float(header.attrib["z_valence"])
        self.r = _values(root.find(".//PP_R"))
        self._v_ry = _values(root.find(".//PP_LOCAL"))
        self.v = self._v_ry * RY2HA
        self.info.update(dict(header.attrib))
        self.info["fname"] = str(fname)

    @property
    def local_potential_ry(self):
        """PP_LOCAL as it is stored in the UPF, i.e. Rydberg."""
        return self._v_ry

    @property
    def u_ks(self):
        """U (eV) of the Kohn-Sham run whose wavefunctions were averaged."""
        return self._u_ks

    @property
    def u_shift(self):
        """U (eV) of the analytic Dudarev shift applied to the d channel."""
        return self._u_shift

    @property
    def channels_used(self):
        """[(band, label, occupation, shift_eV)] that entered the average."""
        return list(self._channels_used)

    # ----------------------------------------------------------------- build

    @classmethod
    def from_ks(cls, ks, semilocal, template, u_ks=0.0, u_shift=None,
                reference=None, direct=True):
        """Build from QEpy wavefunctions.

        ks          npz (or dict) from run_qepy_wavefunctions.py: needs
                    ``r_radial``, ``band_density_radial``, ``occs``, ``band_l``
        semilocal   donor UPF carrying the l-resolved channels (e.g. the
                    Troullier-Martins z = 11 potential, whose 4d channel is
                    what the d electrons are averaged out of)
        template    an existing OEPP UPF: supplies the output radial grid,
                    z_valence and the -Z/r tail
        u_ks        the U (eV) used in the SCF that produced ``ks``.  Recorded
                    in the file; it does not by itself change the potential.
        u_shift     U (eV) for the analytic Dudarev shift of the d channel.
                    Defaults to ``u_ks``.  Pass 0.0 for a potential whose U
                    dependence comes only from the wavefunctions -- see the
                    note in :meth:`hubbard_shift_ry`.
        reference   if given, build differentially: take the template's
                    PP_LOCAL as the U = 0 potential and add only the change in
                    the Slater average relative to this reference npz.  This
                    is what reproduces the validated U = 0 OEPP exactly.
        """
        self = cls(direct=direct)
        self._load_donors(semilocal, template)
        self._u_ks = float(u_ks)
        self._u_shift = float(u_ks if u_shift is None else u_shift)

        v_ry, den, used = self._slater_average(cls._as_dict(ks), self._u_shift)
        self._channels_used = used

        if reference is None:
            mode = "absolute"
            v_out_ry = v_ry
        else:
            mode = "differential"
            v_ref_ry, _, _ = self._slater_average(cls._as_dict(reference), 0.0)
            v_out_ry = self._v_template_ry + (v_ry - v_ref_ry)

        self._v_ry = v_out_ry
        self.v = v_out_ry * RY2HA
        self.r = self._r_template
        self._zval = self._z_template
        self._atomic_density_grid = self._r_template
        self._atomic_density = den
        self._build_info = {
            "mode": mode,
            "u_ks_ev": self._u_ks,
            "u_shift_ev": self._u_shift,
            "semilocal": str(semilocal),
            "template": str(template),
            "channels": used,
        }
        self.info.update(self._build_info)
        return self

    @staticmethod
    def _as_dict(ks):
        if isinstance(ks, (str, Path)):
            return dict(np.load(str(ks), allow_pickle=True))
        if hasattr(ks, "files"):
            return {k: ks[k] for k in ks.files}
        return dict(ks)

    def _load_donors(self, semilocal, template):
        root = ET.parse(str(semilocal)).getroot()
        self._r_semilocal = _values(root.find(".//PP_R"))
        self._v_semilocal_local_ry = _values(root.find(".//PP_LOCAL"))
        self._l_local = int(root.find(".//PP_HEADER").attrib["l_local"])
        self._channels_ry = {}
        for node in root.iter():
            if node.tag.startswith("PP_VNL"):
                self._channels_ry[int(node.attrib["L"])] = _values(node)

        root = ET.parse(str(template)).getroot()
        self._r_template = _values(root.find(".//PP_R"))
        self._v_template_ry = _values(root.find(".//PP_LOCAL"))
        self._z_template = float(root.find(".//PP_HEADER").attrib["z_valence"])
        self._template_path = str(template)
        self._v_tail_ry = -self._z_template / self._r_template

    def _semilocal_channel_ry(self, l):
        if l == self._l_local:
            return self._v_semilocal_local_ry
        if l not in self._channels_ry:
            raise KeyError(f"no semilocal channel for l = {l} in the donor")
        return self._channels_ry[l]

    def hubbard_shift_ry(self, l, u_ev):
        """Dudarev channel shift, Delta v_d = U (1/2 - n_d / (2(2l+1))).

        For a filled 4d shell n_d = 10 and 2l+1 = 5, so this is exactly -U/2,
        the shift the atomic calculation shows for the 4d eigenvalue.

        Note this is applied ON TOP of wavefunctions that were themselves
        obtained at the same U.  The two are not independent: the analytic
        shift is a calibration of the eigenvalue, the wavefunctions carry the
        self-consistent response.  Pass ``u_shift=0`` to ``from_ks`` to keep
        only the wavefunction route.
        """
        if l != 2 or not u_ev:
            return 0.0
        n_per_spin_orbital = self.D_OCC_TOTAL / (2.0 * (2 * l + 1))
        return (u_ev * EV2HA * (0.5 - n_per_spin_orbital)) / RY2HA

    def _smooth(self, y):
        y = np.asarray(y, dtype=float)
        sigma = self.SMOOTH_SIGMA_BINS
        radius = max(1, int(np.ceil(4.0 * sigma)))
        x = np.arange(-radius, radius + 1, dtype=float)
        kernel = np.exp(-0.5 * (x / sigma) ** 2)
        kernel /= kernel.sum()
        padded = np.pad(y, radius, mode="edge")
        return np.clip(np.convolve(padded, kernel, mode="same")[radius:-radius],
                       0.0, None)

    def _tail_blend_weight(self, den, den_peak):
        hi = max(1.0e-20, self.DEN_FRAC * den_peak)
        lo = max(1.0e-30, self.DEN_BLEND_FRAC * den_peak)
        if lo >= hi:
            lo = hi * 0.01
        t = (np.log(np.maximum(den, lo)) - np.log(lo)) / (np.log(hi) - np.log(lo))
        t = np.clip(t, 0.0, 1.0)
        return t * t * (3.0 - 2.0 * t)

    def _slater_average(self, ks, u_shift):
        """Mi Eq. (15): occupation- and density-weighted average of the
        semilocal channels, blended onto the template's -Z/r tail."""
        r_ks = np.asarray(ks["r_radial"], float)
        band_density = np.asarray(ks["band_density_radial"], float)
        occs = np.asarray(ks["occs"], float)
        band_l = np.asarray(ks["band_l"], float)

        num = np.zeros_like(r_ks)
        den = np.zeros_like(r_ks)
        used = []
        for ib, (occ, l) in enumerate(zip(occs, band_l)):
            if occ <= 1.0e-6 or l < 0:
                continue
            l = int(l)
            shift_ry = self.hubbard_shift_ry(l, u_shift)
            v_l_ry = self._semilocal_channel_ry(l) + shift_ry
            v_l_ks = np.interp(r_ks, self._r_semilocal, v_l_ry)
            weight = occ * self._smooth(band_density[ib])
            num += weight * v_l_ks
            den += weight
            used.append((ib + 1, f"l={l}", float(occ), shift_ry * RY2EV))
        if not used:
            raise ValueError("no occupied channels in the KS data")

        tail_on_ks = np.interp(r_ks, self._r_template, self._v_tail_ry)
        den_peak = float(np.max(den))
        v_slater = np.divide(num, den, out=tail_on_ks.copy(),
                             where=den > 1.0e-30)

        v_on_out = np.interp(self._r_template, r_ks, v_slater,
                             left=v_slater[0], right=np.nan)
        den_on_out = np.interp(self._r_template, r_ks, den,
                               left=den[0], right=0.0)
        w = self._tail_blend_weight(den_on_out, den_peak)
        v_out = (w * np.nan_to_num(v_on_out, nan=0.0)
                 + (1.0 - w) * self._v_tail_ry)
        beyond = ~np.isfinite(v_on_out)
        v_out[beyond] = self._v_tail_ry[beyond]
        return v_out, den_on_out, used

    # ----------------------------------------------------------------- write

    def comment(self):
        b = self._build_info
        if not b:
            return "OEPP local pseudopotential"
        bits = [f"OEPP local from Mi2016 Eq.(15) ({b['mode']})",
                f"template {Path(b['template']).name}",
                f"donor {Path(b['semilocal']).name}"]
        if b["u_ks_ev"]:
            bits.append(f"U={b['u_ks_ev']:g} eV KS orbitals")
        if b["u_shift_ev"] != b["u_ks_ev"]:
            bits.append(f"d-channel shift U={b['u_shift_ev']:g} eV")
        return "; ".join(bits)

    def write_upf(self, dst, template=None, comment=None):
        """Write PP_LOCAL into a copy of the template UPF.

        Everything else in the template is kept, so the result stays a valid
        z_valence = 1 UPF that QE and DFTpy already know how to read.
        """
        src = template or getattr(self, "_template_path", None)
        if src is None:
            raise ValueError("no template to write into; pass template=")
        if self._v_ry is None:
            raise ValueError("nothing built or read yet")
        text = Path(src).read_text()
        m = re.search(r'(<PP_LOCAL[^>]*columns="(\d+)"[^>]*>)(.*?)(</PP_LOCAL>)',
                      text, re.S)
        if m is None:
            raise ValueError(f"no PP_LOCAL in {src}")
        columns = int(m.group(2))
        old = np.fromstring(m.group(3), sep=" ")
        if len(old) != len(self._v_ry):
            raise ValueError(f"PP_LOCAL length {len(old)} but potential has "
                             f"{len(self._v_ry)} points")
        block = m.group(1) + _format_upf_array(self._v_ry, columns) + m.group(4)
        text = text[:m.start()] + block + text[m.end():]
        note = comment or self.comment()
        text = re.sub(r'comment="[^"]*"', f'comment="{note}"', text, count=1)
        text = self._inherit_d_block(text, src)
        Path(dst).write_text(text)
        return Path(dst)

    def _inherit_d_block(self, text, src):
        """Keep the template's element d-shell data, drop its pole sets.

        Copying the template wholesale carries its PP_INFO across, which is how
        a derived OEPP comes out LR-ready without a separate stamping step --
        element_params and shell_data describe the ELEMENT, so they are the
        same for every Ag pseudopotential.

        Pole sets are not: each bakes in the U of the run it came from
        (u_eff_ev_applied), so inheriting one into a pseudopotential built at a
        different U would quietly misstate U.  Nothing downstream could catch
        it, because the atomic route ignores pole sets entirely while
        upf_has_dblock would still point D_POLES at the file.
        """
        m = re.search(rf"{DBLOCK_BEGIN}\s*(\{{.*?\}})\s*{DBLOCK_END}", text, re.S)
        if m is None:
            return text
        blk = json.loads(m.group(1))
        if not blk.pop("pole_sets", None):
            return text
        blk.pop("default", None)
        blk["note"] = _wrap(
            f"Element d-shell data inherited from {Path(src).name}. Pole sets "
            f"were dropped: they carry the U of their own run, so they do not "
            f"travel to a pseudopotential rebuilt at another U. Re-add with "
            f"OEPP.attach_d_block if the bulk route is wanted.")
        payload = json.dumps(blk, indent=1, ensure_ascii=True, sort_keys=False)
        return text[:m.start(1)] + payload + text[m.end(1):]

    def write_psp8(self, dst, zatom=47.0, pspxc=2, title=None):
        """ABINIT psp8, local only.  Uses this potential's own grid."""
        if self._v_ry is None:
            raise ValueError("nothing built or read yet")
        r = self.r
        v_ha = self._v_ry * RY2HA
        if len(r) != len(v_ha):
            raise ValueError(f"grid has {len(r)} points, potential "
                             f"{len(v_ha)} -- refusing to write a mismatch")
        head = title or self.comment()
        with open(dst, "w") as fh:
            fh.write(head[:132] + "\n")
            fh.write(f"{zatom:.4f}    {self._zval:.4f}    "
                     f"{datetime.date.today():%m%d%Y}    zatom,zion,pspd\n")
            fh.write(f"8    {pspxc:d}    0    0    {len(r):d}    0    "
                     f"pspcod,pspxc,lmax,lloc,mmax,r2well\n")
            fh.write("0.000000    -1.000000    0.000000    rchrg fchrg qchrg\n")
            fh.write("0    0    0    0    0    nproj\n")
            fh.write("0    extension_switch\n")
            fh.write("4\n")
            for i, (ri, vi) in enumerate(zip(r, v_ha), start=1):
                fh.write(f"{i:<4d}  {ri:.13e} {vi:.13e}\n")
        return Path(dst)

    # ------------------------------------------------------------- d block

    @staticmethod
    def _shift_poles(w, f, u_eff_ev):
        """Dudarev U_eff on a pole set, holding the dipole fixed.

        The occupied 4d drops by U_eff/2 and the radial wavefunction is
        untouched, so every d -> virtual transition rises by U_eff/2 and
        D is unchanged.  With f = 2 w D^2 that forces f to follow w:

            w' = w + U_eff/2      f' = f w'/w      D' = sqrt(f'/2w') = D
        """
        if not u_eff_ev:
            return w, f
        w2 = w + 0.5 * u_eff_ev * EV2HA
        return w2, f * (w2 / w)

    @staticmethod
    def encode_shell_data(npz, shell="4d", width=76):
        """Pack the radial core/virtual orbitals for embedding in a UPF.

        zlib+base64 float64.  float32 would halve it but underflows the small
        tails of virtual_ur to zero, so the precision is not optional.

        The base64 is stored as a LIST of short strings, not one long one: QE's
        Fortran UPF reader takes PP_INFO lines into fixed-length buffers, and a
        single 180 kB line would be truncated or rejected.  json.dumps(indent=1)
        then puts each chunk on its own ~80-column line.
        """
        import base64
        import textwrap
        import zlib
        z = np.load(str(npz), allow_pickle=True)
        key = f"selected_{shell}_ur"
        if key not in z.files:
            raise KeyError(f"{npz}: no {key!r}; has {sorted(z.files)[:8]}...")

        def enc(a):
            a = np.ascontiguousarray(a, dtype=np.float64)
            s = base64.b64encode(zlib.compress(a.tobytes(), 9)).decode("ascii")
            return {"b64": textwrap.wrap(s, width),
                    "shape": list(a.shape), "dtype": "float64"}

        return {"shell": shell,
                "r": enc(z["r"]),
                "u_core": enc(z[key]),
                "virtual_ur": enc(z["virtual_ur"]),
                "delta_e_ha": enc(z["transition_delta_e_hartree"]),
                "virtual_labels": [str(x).strip() for x in z["virtual_labels"]],
                "source": Path(npz).name}

    def d_block(self, pole_files, u_applied=0.0, default=None, element=None,
                note=None, element_params=None, shell_data=None):
        """Assemble the JSON block that dftpy.td.d_embedding reads.

        pole_files  {set_name: (path, provenance)} of npz files carrying
                    ``omega_ha`` and ``f``
        u_applied   U_eff (eV) to bake into the poles.  Recorded as
                    ``u_eff_ev_applied`` so attach_d_block refuses a run-time
                    ``d_shift_ev`` on top of it.
        element_params
                    the element's d-shell parameters (shell, peak_win_ev,
                    coordination_r_cut, ...), embedded so the UPF is
                    self-describing.  d_embedding reads these and has no
                    catalog to fall back on.
        shell_data  encode_shell_data() output: the radial core and virtual
                    orbitals, so the UPF needs no companion npz at run time.
        """
        sets = {}
        for name, item in pole_files.items():
            path, provenance = (item if isinstance(item, (tuple, list))
                                else (item, "unspecified"))
            path = Path(path)
            if not path.exists():
                continue
            z = np.load(str(path), allow_pickle=True)
            w_bare = np.asarray(z["omega_ha"], float)
            f_bare = np.asarray(z["f"], float)
            w, f = self._shift_poles(w_bare, f_bare, u_applied)
            entry = {
                "provenance": provenance,
                "source": str(z["source"]) if "source" in z.files else path.name,
                "file": path.name,
                "omega_ha": [float(x) for x in w],
                "f": [float(x) for x in f],
                "omega_ev": [round(float(x) * HA2EV, 4) for x in w],
                "sum_f": float(f.sum()),
                "u_eff_ev_applied": float(u_applied),
            }
            if u_applied:
                entry["omega_ev_bare"] = [round(float(x) * HA2EV, 4)
                                          for x in w_bare]
                entry["alpha_static_bare"] = float((f_bare / w_bare ** 2).sum())
                entry["alpha_static_total"] = float((f_bare / (w_bare * w)).sum())
            elif "alpha" in z.files:
                a = np.asarray(z["alpha"], float)
                entry["alpha_bohr3"] = [float(x) for x in a]
                entry["alpha_static_total"] = float(a.sum())
            sets[name] = entry
        if pole_files and not sets:
            raise ValueError("none of the pole files exist")
        if default is None or default not in sets:
            default = sorted(sets)[0] if sets else None
        sym = element or self.info.get("element", "Ag")
        out = {
            "schema": DBLOCK_SCHEMA,
            "element": sym,
            "z_valence": self._zval,
            "note": _wrap(note or (
                "Frequency-dependent core polarization of the filled 4d "
                "shell. The 4d electrons are in the CORE of this "
                "pseudopotential (z_valence=1), so alpha_d(omega) = "
                "sum_j f_j / (omega_j^2 - omega^2) is a core-polarization "
                "term, not a valence channel.")),
            "convention": _wrap(DBLOCK_CONVENTION),
            "shape_source": _wrap(DBLOCK_SHAPE_SOURCE),
        }
        if sets:
            out["default"] = default
            out["pole_sets"] = sets
        if element_params:
            out["element_params"] = element_params
        if shell_data:
            out["shell_data"] = shell_data
        return out

    @staticmethod
    def _read_d_block(upf):
        """The parsed d block of a UPF, or None."""
        try:
            txt = Path(upf).read_text(errors="replace")
        except OSError:
            return None
        m = re.search(rf"{DBLOCK_BEGIN}\s*(\{{.*?\}})\s*{DBLOCK_END}", txt, re.S)
        return json.loads(m.group(1)) if m else None

    def embed_shell(self, upf, npz, params=None, out=None, shell=None):
        """Make a pseudopotential self-describing for d embedding.

        Encodes the radial core and virtual orbitals from `npz` and merges
        them, with `params`, into the UPF's d block -- after which
        dftpy.td.d_embedding needs no companion data file to run linear
        response with this pseudopotential.

        element and shell are read from the npz, not passed in, and the npz's
        element is checked against the UPF's: stamping an Ag shell into a Cu
        pseudopotential would otherwise go unnoticed until the spectra came
        out wrong.  `params` carries only what the npz cannot know -- the bulk
        polarizability, the neighbour cutoff, the peak window.

        Returns (destination, info).
        """
        z = np.load(str(npz), allow_pickle=True)
        sym = str(z["element"]) if "element" in z.files else None
        mine = self.info.get("element")
        if sym and mine and sym.strip().lower() != str(mine).strip().lower():
            raise ValueError(
                f"{Path(npz).name} holds {sym} data but {Path(upf).name} is a "
                f"{mine} pseudopotential; refusing to stamp one into the other")
        if shell is None:
            shell = (params or {}).get("shell")
        if shell is None:
            lbl = [k for k in z.files
                   if k.startswith("selected_") and k.endswith("_label")]
            if not lbl:
                raise ValueError(
                    f"{npz}: cannot tell which shell this is; pass shell=")
            shell = str(z[lbl[0]]).strip()

        sd = self.encode_shell_data(npz, shell=shell)
        # merge onto whatever the file already carries, so re-embedding with
        # no params does not silently drop optional values set earlier
        prev = (self._read_d_block(upf) or {}).get("element_params") or {}
        ep = dict(prev, **dict(params or {}), shell=shell)
        before = Path(upf).stat().st_size
        dst = self.update_d_block(upf, out=out, element_params=ep,
                                  shell_data=sd)
        info = dict(element=sym or mine, shell=shell, source=Path(npz).name,
                    bytes_before=before, bytes_after=dst.stat().st_size,
                    payload_kb=sum(len("".join(v["b64"])) for v in sd.values()
                                   if isinstance(v, dict)) / 1024)
        return dst, info

    def update_d_block(self, upf, out=None, **fields):
        """Merge fields into a UPF's existing d block, creating one if absent.

        Used to add element_params/shell_data to a pseudopotential without
        re-deriving pole sets that are already baked in.
        """
        text = Path(upf).read_text(errors="replace")
        m0 = _DBLOCK_RE.search(text)
        if m0 is not None:
            inner = re.search(rf"{DBLOCK_BEGIN}\s*(\{{.*?\}})\s*{DBLOCK_END}",
                              text, re.S)
            block = json.loads(inner.group(1))
        else:
            block = self.d_block({})
        block.update(fields)
        m = re.search(r"(<PP_INFO>)(.*?)(\s*</PP_INFO>)", text, re.S)
        if m is None:
            raise ValueError(f"{upf}: no <PP_INFO> element")
        body = _DBLOCK_RE.sub("", m.group(2))
        payload = json.dumps(block, indent=1, ensure_ascii=True,
                             sort_keys=False)
        for bad in ("<", "&"):
            if bad in payload:
                raise ValueError(f"payload contains {bad!r}, illegal as bare "
                                 f"XML text")
        new = (f"{body.rstrip()}\n\n      {DBLOCK_BEGIN}\n{payload}\n"
               f"      {DBLOCK_END}\n")
        dst = Path(out or upf)
        dst.write_text(text[:m.start(2)] + new + text[m.end(2):])
        return dst

    def attach_d_block(self, upf, pole_files, u_applied=0.0, out=None,
                       default=None, element_params=None, shell_data=None):
        """Embed the d block in a UPF's PP_INFO as bare text.

        Replaces any block already there, so this is idempotent.  Element data
        already in the file is carried over when the caller does not supply
        its own: adding a pole set must not strip the shell data that the
        pseudopotential needs to do linear response at all.
        """
        text = Path(upf).read_text(errors="replace")
        if element_params is None or shell_data is None:
            prev = re.search(rf"{DBLOCK_BEGIN}\s*(\{{.*?\}})\s*{DBLOCK_END}",
                             text, re.S)
            if prev is not None:
                old = json.loads(prev.group(1))
                element_params = element_params or old.get("element_params")
                shell_data = shell_data or old.get("shell_data")
        block = self.d_block(pole_files, u_applied=u_applied, default=default,
                             element_params=element_params,
                             shell_data=shell_data)
        m = re.search(r"(<PP_INFO>)(.*?)(\s*</PP_INFO>)", text, re.S)
        if m is None:
            raise ValueError(f"{upf}: no <PP_INFO> element")
        body = _DBLOCK_RE.sub("", m.group(2))
        payload = json.dumps(block, indent=1, ensure_ascii=True,
                             sort_keys=False)
        for bad in ("<", "&"):
            if bad in payload:
                raise ValueError(f"payload contains {bad!r}, illegal as bare "
                                 f"XML text")
        new = f"{body.rstrip()}\n\n      {DBLOCK_BEGIN}\n{payload}\n      {DBLOCK_END}\n"
        dst = Path(out or upf)
        dst.write_text(text[:m.start(2)] + new + text[m.end(2):])
        return dst

    @staticmethod
    def has_d_block(upf):
        try:
            return _DBLOCK_RE.search(Path(upf).read_text(errors="replace")) is not None
        except OSError:
            return False

    @staticmethod
    def read_d_block(upf):
        m = re.search(rf"{DBLOCK_BEGIN}\s*(\{{.*?\}})\s*{DBLOCK_END}",
                      Path(upf).read_text(errors="replace"), re.S)
        if m is None:
            raise ValueError(f"{upf}: no {DBLOCK_BEGIN} block")
        return json.loads(m.group(1))

    def __repr__(self):
        b = self._build_info
        if b:
            return (f"<OEPP built {b['mode']} u_ks={b['u_ks_ev']:g} "
                    f"u_shift={b['u_shift_ev']:g} npts={len(self.r)}>")
        return f"<OEPP {self.info.get('fname', '?')} zval={self._zval}>"


def _cli(argv=None):
    """python -m dftpy.functional.pseudo.oepp --embed <npz> <upf>...

    Embeds the element's d-shell data in each pseudopotential, after which
    dftpy.td.d_embedding reads everything it needs from the UPF itself.
    element and shell come from the npz; --params adds the optional extras
    (alpha_bulk, coordination_r_cut, peak_win_ev, ...) as a JSON object or a
    path to one.
    """
    import argparse
    ap = argparse.ArgumentParser(prog="dftpy.functional.pseudo.oepp")
    ap.add_argument("--embed", metavar="NPZ", type=Path, required=True,
                    help="radial core/virtual orbitals to embed")
    ap.add_argument("upf", nargs="+", type=Path)
    ap.add_argument("--params", default=None,
                    help="JSON object, or a path to a .json file")
    ap.add_argument("--shell", default=None,
                    help="override the shell read from the npz, e.g. 4d")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)

    params = None
    if a.params:
        p = Path(a.params)
        params = json.loads(p.read_text() if p.is_file() else a.params)

    for upf in a.upf:
        if a.dry_run:
            print(f"  would embed into {upf}  "
                  f"({upf.stat().st_size / 1024:.0f} KB)")
            continue
        _, info = OEPP(str(upf)).embed_shell(upf, a.embed, params,
                                             shell=a.shell)
        print(f"  {upf.name:34s} {info['bytes_before'] / 1024:6.0f} -> "
              f"{info['bytes_after'] / 1024:6.0f} KB   "
              f"{info['element']} {info['shell']}  "
              f"({info['payload_kb']:.0f} KB orbitals)")


if __name__ == "__main__":
    _cli()
