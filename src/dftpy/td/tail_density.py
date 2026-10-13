#!/usr/bin/env python3
"""Analytic exponential replacement of the far-field density tail.

The OF excited states diagonalise H = -1/2 Lap - v_vW[rho], and for a tail
rho ~ exp(-2 kappa r) the von Weizsacker potential tends to -kappa^2/2 =
eps_HOMO -- inverted, the wall that keeps vacuum states out of the spectrum.
Beyond ~16 bohr the density instead flattens onto a ~3e-12 floor (box corners,
where periodic images overlap), so v_vW is round-off dominated and hits keV.

Do NOT regularise with rho -> rho + eps: it removes the large values and the
wall together, so vacuum states fall to zero energy and fill the basis (the Ag8
Casida baseline moved 4.63 -> 5.70 eV).  No value of eps escapes this.

Replace the tail instead -- beyond a matching radius r_m, continue each ray as

    ln rho_a(r, u) = ln rho(r_m, u) - 2 kappa (r - r_m) - 2 ln(r / r_m)

anchored on the ACTUAL density in that direction, so the angular structure
survives.  kappa comes from the shell-median profile; the quintic smoothstep
switch keeps Lap(sqrt(rho)) kink-free.
"""

import numpy as np
from scipy.ndimage import map_coordinates


def _centroid(rho_np, coords, L):
    """Density-weighted centroid, computed with periodic wrapping."""
    w = np.maximum(rho_np, 0.0)
    tot = w.sum()
    com = np.empty(3)
    for i in range(3):
        # circular mean, so a cluster straddling the boundary still works
        th = 2.0 * np.pi * coords[i] / L[i]
        com[i] = (np.arctan2((w * np.sin(th)).sum() / tot,
                             (w * np.cos(th)).sum() / tot) % (2 * np.pi)) * L[i] / (2 * np.pi)
    return com


def _smoothstep(x):
    """Quintic 0->1 on [0,1]; first and second derivatives vanish at both ends."""
    x = np.clip(x, 0.0, 1.0)
    return x * x * x * (10.0 + x * (-15.0 + 6.0 * x))


def _smoothstep_int(x):
    """Integral of _smoothstep from 0 to x (x >= 0).

    2.5x^4 - 3x^5 + x^6 for x < 1; x - 0.5 beyond, where the step saturates.
    """
    x = np.maximum(x, 0.0)
    inner = x * x * x * x * (2.5 + x * (-3.0 + x))
    return np.where(x < 1.0, inner, x - 0.5)


def shell_profile(rho_np, r, rmax, nbin=60):
    """Median density in radial shells out to rmax."""
    edges = np.linspace(0.0, rmax, nbin + 1)
    idx = np.digitize(r.ravel(), edges) - 1
    flat = rho_np.ravel()
    cen, med = [], []
    for b in range(nbin):
        m = idx == b
        if m.sum() < 50:
            continue
        cen.append(0.5 * (edges[b] + edges[b + 1]))
        med.append(np.median(flat[m]))
    return np.asarray(cen), np.asarray(med)


def fit_kappa(cen, med, lo, hi):
    """Fit ln(rho) = c - 2 kappa r - 2 ln r over the window [lo, hi].

    kappa in atomic units.  The -2 ln r term is the 1/r^2 from
    psi ~ exp(-kappa r)/r; it makes the fit stable against the window choice.
    """
    m = (cen >= lo) & (cen <= hi) & (med > 0)
    if m.sum() < 4:
        raise ValueError(f"fit window [{lo}, {hi}] has only {m.sum()} usable shells")
    x = cen[m]
    y = np.log(med[m]) + 2.0 * np.log(x)
    slope, _ = np.polyfit(x, y, 1)
    kappa = -0.5 * slope
    if not (0.2 < kappa < 1.5):
        raise ValueError(f"fitted kappa = {kappa:.3f} is outside the physical range "
                         "(0.2-1.5 a.u.); check the fit window")
    return float(kappa)


def replace_tail(rho, rho_match=1e-8, delta=4.0, fit_decades=2.5,
                 smooth=1.5, renormalise=True, verbose=True, _order=3):
    """Return a copy of `rho` whose far tail follows an analytic exponential.

    rho         DFTpy DirectField (the SCF density).
    rho_match   density at which the switch starts; r_m is where the shell
                median first drops below it.  1e-8 sits above the
                image-overlap floor (~3e-12) and inside the clean region.
    delta       width in bohr of the smoothstep switch, r_m -> r_m + delta.
    fit_decades decades of density below rho_match used for the kappa fit
                (the window ends at r_m).
    smooth      Gaussian width (bohr) de-noising ln(rho) before it is sampled
                for the continuation.  Does not affect the interior.
    renormalise rescale the result to the original electron count.

    Returns (rho_new, info_dict).
    """
    grid = rho.grid
    rho_np = np.asarray(rho, dtype=np.float64)
    L = np.diag(np.asarray(grid.lattice)).astype(float)
    nr = np.asarray(grid.nr, dtype=int)
    coords = [np.asarray(grid.r[i], dtype=np.float64) for i in range(3)]

    com = _centroid(rho_np, coords, L)

    # --- smooth periodic radial coordinate --------------------------------
    # Minimum-image |d| is only C0, and its Wigner-Seitz cusps sit in the box
    # corners where the replaced tail lives -- the source of the residual |v|
    # outliers.  The C-infinity surrogate
    #     q_i = (L_i/pi) sin(pi d_i / L_i),   R = |q|,  u = q/R
    # equals d for |d| << L and saturates in the corner, which is right: the
    # periodic images overlap there, so the density should level off.
    d = np.stack([(coords[i] - com[i] + L[i] / 2.0) % L[i] - L[i] / 2.0
                  for i in range(3)])
    q = np.stack([(L[i] / np.pi) * np.sin(np.pi * d[i] / L[i]) for i in range(3)])
    r = np.sqrt((q * q).sum(axis=0))
    r_safe = np.maximum(r, 1e-12)

    r_insc = float(L.min() / 2.0)
    cen, med = shell_profile(rho_np, r, r_insc)

    # --- matching radius: where the shell median crosses rho_match ---------
    below = np.where(med < rho_match)[0]
    if below.size == 0:
        raise ValueError(f"density never falls below rho_match={rho_match:g} "
                         f"inside the inscribed sphere (min median {med.min():.2e})")
    j = below[0]
    if j == 0:
        raise ValueError("rho_match is larger than the density at the centre")
    x0, x1 = cen[j - 1], cen[j]
    y0, y1 = np.log(med[j - 1]), np.log(med[j])
    r_m = float(x0 + (np.log(rho_match) - y0) * (x1 - x0) / (y1 - y0))

    if r_m + delta > r_insc:
        raise ValueError(f"r_m + delta = {r_m + delta:.1f} exceeds the inscribed "
                         f"radius {r_insc:.1f} bohr; box too small or rho_match too low")

    # --- kappa from the clean decade(s) just inside r_m -------------------
    lo_target = rho_match * 10.0 ** fit_decades
    above = np.where(med > lo_target)[0]
    r_lo = float(cen[above[-1]]) if above.size else float(cen[0])
    kappa = fit_kappa(cen, med, r_lo, r_m)

    # --- log-density, smoothed, as the basis for the continuation --------
    # Blending the analytic tail against the RAW field fails: corner-ward, the
    # density inside the switch shell already sits on the image-overlap floor,
    # and points 3 decades below their own shell median made steep radial
    # gradients and |v| spikes of 2 keV.  Smoothing ln(rho) on ~1.5 bohr removes
    # that grid-level junk; the genuine structure varies on much longer scales
    # out here, and the interior is unaffected because W = 0 there.
    ln_raw = np.log(np.maximum(rho_np, 1e-300))
    gg = rho.grid.get_reciprocal().gg
    lf = rho.copy(); lf[:] = ln_raw
    ln_s = np.asarray((lf.fft() * np.exp(-gg * smooth**2 / 4.0)).ifft(force_real=True))

    u = q / r_safe

    def sample(field, radius):
        pos = com[:, None, None, None] + u * radius
        frac = np.stack([pos[i] / L[i] * nr[i] for i in range(3)])
        return map_coordinates(field, frac.reshape(3, -1), order=_order,
                               mode="grid-wrap").reshape(field.shape)

    h = 1.0
    anchor = sample(ln_s, r_m)
    dlogdr = (sample(ln_s, r_m + h) - sample(ln_s, r_m - h)) / (2.0 * h)

    # rho = A exp(-2 kappa (r-r_m)) (r_m/r)^2  =>  kappa = -(1/2)(dln rho/dr + 2/r)
    kappa_loc = np.clip(-0.5 * (dlogdr + 2.0 / r_m), 0.5 * kappa, 1.5 * kappa)

    # --- decay rate relaxes kappa_loc -> kappa across the switch ----------
    # Matching the ray's own slope at r_m makes the join C1 in the exponent;
    # the quintic smoothstep is C2, so nothing kinks in Lap(sqrt(rho)).
    x = np.maximum(r - r_m, 0.0) / delta
    expo = 2.0 * delta * (kappa_loc * x + (kappa - kappa_loc) * _smoothstep_int(x))
    ln_analytic = anchor - expo - 2.0 * np.log(np.maximum(r, r_m) / r_m)

    # Two quintic switches: ln_raw -> ln_s across [r_lo, r_m] (free, the two
    # agree where the density is clean), then ln_s -> analytic across
    # [r_m, r_m+delta].  ln_raw must be gone before the tail starts: at the
    # floor rho jumps between ~1e-12 and ~1e-15 point to point, so Lap(ln rho)
    # is enormous and survives the (1-w) factor.
    w1 = _smoothstep((r - r_lo) / max(r_m - r_lo, 1e-6))
    ln_base = (1.0 - w1) * ln_raw + w1 * ln_s

    w = _smoothstep(x)
    ln_new = (1.0 - w) * ln_base + w * ln_analytic

    new_np = np.exp(ln_new)
    inner = w <= 0.0
    new_np[inner] = rho_np[inner]          # interior identical bit-for-bit

    rho_new = rho.copy()
    rho_new[:] = new_np
    n_before = float(rho.integral())
    if renormalise:
        rho_new *= n_before / float(rho_new.integral())

    grad_R = np.sqrt(sum(np.cos(np.pi * d[i] / L[i])**2 for i in range(3)) / 3.0)
    kappa_phys = float(kappa * np.median(grad_R[np.abs(r - r_m) < 1.0]))

    info = dict(com=com.tolist(), r_inscribed=r_insc, r_match=r_m, delta=delta,
                kappa=kappa, kappa_phys=kappa_phys,
                eps_homo_ev=-0.5 * kappa_phys**2 * 27.211386245988,
                fit_window=(r_lo, r_m), rho_match=rho_match,
                n_before=n_before, n_after=float(rho_new.integral()),
                frac_replaced=float((w > 0).mean()))
    if verbose:
        print(f"  tail replacement: r_m={r_m:.2f} bohr (inscribed {r_insc:.2f}), "
              f"delta={delta:.1f}, kappa={kappa:.4f} (phys {kappa_phys:.4f}) a.u.")
        print(f"    fit window {r_lo:.1f}-{r_m:.1f} bohr -> eps_HOMO = "
              f"{info['eps_homo_ev']:.3f} eV;  {100*info['frac_replaced']:.1f}% "
              f"of grid modified;  N {n_before:.6f} -> {info['n_after']:.6f}")
    return rho_new, info
