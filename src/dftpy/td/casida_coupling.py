import numpy as np

HA = 27.211386245988
PEAK_WIN = (2.0, 6.0)
GAMMA_EV = 0.10


def coupling(grid, A, B=None, kern=None):
    """Casida coupling block  K_AB = <A| v_H + f_xc |B>  on a real-space grid.

    grid    object with .poisson(rho3d), .nr and .dV (d_embedding.Grid).
    A, B    (n, nnr) flattened transition densities; B defaults to A.
    kern    (nnr,) local f_xc added to the Hartree term, or None for Hartree.

    The s and d blocks take DIFFERENT kernels deliberately: s carries XC plus
    the OF kinetic kernel (pseudo-orbitals miss the kinetic response), d
    carries XC only -- its modes have a physical polarizability, and a kinetic
    correction there flips the sign of the d effect.
    """
    B = A if B is None else B
    V = np.empty_like(B)
    for j in range(B.shape[0]):
        V[j] = grid.poisson(B[j].reshape(grid.nr)).ravel()
    if kern is None:
        return (A @ V.T) * grid.dV
    return (A @ (V + kern[None, :] * B).T) * grid.dV


def solve(K, Om, mu, label="", n_s=None):
    """Casida solve.  C = Omega^(1/2)(Omega + 2K)Omega^(1/2), eigenvalues are w^2.

    A negative w^2 means the coupled system is unstable, so that mode has no
    real excitation energy.  Those are reported rather than clamped to zero,
    which would leave w = 0 modes carrying enormous strength and a spurious
    Lorentzian background.

    With n_s, the first n_s rows of mu are the valence (s) transitions and the
    rest the d modes, and a third array is returned: the strength each mode
    carries through the s dipole alone.  The d modes are atomic-like oscillators
    at 6.8-10.3 eV forming a tall band above ~6.5 eV that is not the plasmon;
    the s projection removes it while keeping every mode energy.  The d
    screening still sets where the plasmon sits, it just stops radiating -- a
    decomposition, not a different calculation.

    Returns (energies in eV, oscillator strengths[, s-projected strengths]).
    """
    sq = np.sqrt(Om)
    C = np.diag(Om ** 2) + 2 * sq[:, None] * K * sq[None, :]
    w2, Z = np.linalg.eigh(0.5 * (C + C.T))
    n_neg = int((w2 < 0).sum())
    if n_neg:
        print(f"  !! {label} {n_neg} negative eigenvalue(s), most negative "
              f"w^2 = {w2.min():.4e} Ha^2 -- coupled system is UNSTABLE",
              flush=True)
    w = np.sqrt(np.maximum(w2, 0.0))
    D = sq[:, None] * Z
    dip = mu @ D
    f = (2.0 / 3.0) * np.sum(dip * dip, axis=0)
    if n_s is None:
        return w * HA, f
    dip_s = mu[:, :n_s] @ D[:n_s]
    return w * HA, f, (2.0 / 3.0) * np.sum(dip_s * dip_s, axis=0)


def broaden_peak(e, f, win=PEAK_WIN, g=GAMMA_EV, n=3000):
    """Lorentzian-broadened spectrum and the position of its maximum."""
    x = np.linspace(win[0], win[1], n)
    s = np.zeros_like(x)
    for ei, fi in zip(e, f):
        if np.isfinite(ei) and np.isfinite(fi):
            s += fi * g ** 2 / ((x - ei) ** 2 + g ** 2)
    return float(x[np.argmax(s)]), x, s


def couple_d_block(K_ss, w_s, mu_s, K_sd, K_dd, om_d, mu_d, label="coupled"):
    """Assemble and solve the joint s + d Casida problem.

    The s block may come from any route (the inverted von Weizsacker
    Hamiltonian or the LMGP one) as long as its transition densities use the
    same convention as the d modes.
    """
    K = np.block([[K_ss, K_sd], [K_sd.T, K_dd]])
    Om = np.hstack([w_s, om_d])
    mu = np.hstack([mu_s, mu_d])
    return solve(K, Om, mu, label=label)
