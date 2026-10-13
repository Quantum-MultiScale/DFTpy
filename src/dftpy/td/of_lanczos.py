"""Lanczos linear response for orbital-free TDDFT.

One real pseudo-orbital psi0 = sqrt(rho) makes the response two-component.
With psi = e^{-i mu t}(psi0 + x + i y) and drho = 2 psi0 x,

    xdot = D y,   ydot = -M x,
    D = P (H0 - mu) P,   M = D + 2 P psi0 K psi0 P,

H0 = -1/2 nabla^2 + v_s, K = f_Hartree + f_xc + f_Pauli, P projects off psi0.
Both are symmetric, D is positive semi-definite, and omega^2 = eig(T), T = M D.
The kick psi0 -> e^{i k r} psi0 gives

    alpha(z) = 2 <v, (T - z^2)^{-1} v>_D,  v = P r psi0,  <a,b>_D = <a|D|b>.

T is self-adjoint in <.,.>_D, so a Lanczos recursion there gives sticks

    omega_n = sqrt(lambda_n),  f_n = 2 <v,v>_D |U_{0n}|^2,  sum_n f_n = N,

the Thomas-Reiche-Kuhn sum rule.  No explicit states are needed.  Vectors are
real-space, band-limited to |G|^2/2 <= ecut; 8 FFTs per step.
"""

import numpy as np
from scipy import fft as sfft
from scipy.linalg import eigh_tridiagonal
from scipy.sparse.linalg import LinearOperator, lobpcg

from dftpy.constants import ENERGY_CONV
from dftpy.grid import DirectGrid

HA = ENERGY_CONV["Hartree"]["eV"]


class OFLanczos:
    """Operators D, M and the D-metric Lanczos recursion on one grid.

    v_s      static potential of the real-time propagation, on the grid.
    rho      ground-state density; the kernels are evaluated on it.
    f_loc    d^2E/drho^2 of the LOCAL kernel (xc + Pauli/TF).  Hartree is
             built here.
    lattice  cell in bohr, lattice vectors as rows.
    ecut     plane-wave cutoff of the basis, in Hartree.
    """

    def __init__(self, v_s, rho, f_loc, lattice, ecut=10.0, workers=-1):
        self.nr = tuple(int(n) for n in np.shape(rho))
        self.workers = workers
        self.v_s = np.ascontiguousarray(np.asarray(v_s, dtype=float))
        self.rho = np.ascontiguousarray(np.asarray(rho, dtype=float))
        self.f_loc = np.ascontiguousarray(np.asarray(f_loc, dtype=float))
        self.lattice = np.asarray(lattice, dtype=float)
        self.volume = abs(np.linalg.det(self.lattice))
        self.ntot = int(np.prod(self.nr))
        self.dV = self.volume / self.ntot

        self.grid = DirectGrid(lattice=self.lattice, nr=self.nr, full=False)
        rec = self.grid.get_reciprocal()
        g2 = rec.gg
        self.mask = (0.5 * g2 <= ecut)
        self.kin = 0.5 * g2 * self.mask
        self.coulomb = 4.0 * np.pi * rec.invgg
        self.npw = int(self.mask.sum())
        self.ecut = float(ecut)
        self.psi0 = None      # normalised to N
        self.phi0 = None      # normalised to 1
        self.mu = None
        # coupling enters as lambda * K: 0 gives the bare H0 transitions, 1 the
        # physical response.  Sweeping it separates a plasmon (moves with
        # lambda) from a single-particle line.
        self.kernel_scale = 1.0

    # ---------------- basics -------------------------------------------
    # Raw arrays, not DirectField.fft: that allocates a field per call and is
    # not threaded -- measured 6x slower on 72^3, at 8 transforms per step.
    def _fft(self, x):
        return sfft.rfftn(x, workers=self.workers)

    def _ifft(self, xk):
        return sfft.irfftn(xk, s=self.nr, workers=self.workers)

    def dot(self, a, b):
        return float(np.dot(a.ravel(), b.ravel()) * self.dV)

    def band_limit(self, x):
        return self._ifft(self.mask * self._fft(x))

    def project(self, x):
        """Remove the psi0 component: norm conservation."""
        return x - self.phi0 * self.dot(self.phi0, x)

    def hartree(self, n):
        """v_H on the grid; same kernel as dftpy.functional.hartree."""
        return self._ifft(self.coulomb * self._fft(n))

    # ---------------- ground state in the basis -------------------------
    def ground_state(self, guess, tol=1e-9, maxiter=300, nblock=4, seed=0):
        """Lowest eigenpair of the band-limited H0.

        psi0 must be an eigenvector of the SAME operator the response is built
        from, or D is not positive semi-definite on its complement.  sqrt(rho)
        already is on an LMGP density, only nearly so on an inverted KS one --
        worth several tenths of an eV in the peak.
        """
        x0 = self.band_limit(np.asarray(guess, dtype=float))
        shift = self.dot(x0, self._h0(x0)) / max(self.dot(x0, x0), 1e-300) + 1.0

        def h_shift(block):
            out = np.empty_like(block)
            for j in range(block.shape[1]):
                x = block[:, j].reshape(self.nr)
                out[:, j] = (self._h0(x) - shift * self.band_limit(x)).ravel()
            return out

        def precond(block):
            out = np.empty_like(block)
            for j in range(block.shape[1]):
                xk = self._fft(block[:, j].reshape(self.nr))
                out[:, j] = self._ifft(self.mask * xk / (self.kin + 1.0)).ravel()
            return out

        rng = np.random.default_rng(seed)
        X = np.empty((self.ntot, nblock))
        X[:, 0] = x0.ravel()
        for j in range(1, nblock):
            X[:, j] = self.band_limit(rng.standard_normal(self.nr)).ravel()
        A = LinearOperator((self.ntot, self.ntot), matmat=h_shift, dtype=float,
                           matvec=lambda z: h_shift(z.reshape(-1, 1))[:, 0])
        Tp = LinearOperator((self.ntot, self.ntot), matmat=precond, dtype=float,
                            matvec=lambda z: precond(z.reshape(-1, 1))[:, 0])
        vals, vecs = lobpcg(A, X, M=Tp, tol=tol, maxiter=maxiter, largest=False)
        k = int(np.argmin(vals))
        psi = vecs[:, k].reshape(self.nr)
        n_e = float(self.rho.sum() * self.dV)
        psi = psi / np.sqrt(self.dot(psi, psi) / n_e)
        if self.dot(psi, np.sqrt(np.maximum(self.rho, 0.0))) < 0:
            psi = -psi
        self.psi0 = psi
        self.phi0 = psi / np.sqrt(n_e)
        self.mu = float(vals[k] + shift)
        return self.psi0, self.mu

    def set_ground_state(self, psi0, mu):
        n_e = float(self.rho.sum() * self.dV)
        psi0 = self.band_limit(np.asarray(psi0, dtype=float))
        self.psi0 = psi0 / np.sqrt(self.dot(psi0, psi0) / n_e)
        self.phi0 = self.psi0 / np.sqrt(n_e)
        self.mu = float(mu)

    # ---------------- operators -----------------------------------------
    def _h0(self, x):
        """Band-limited H0 x, without the mu shift or the projection."""
        return self._ifft(self.mask * (self.kin * self._fft(x)
                                       + self._fft(self.v_s * x)))

    def apply_D(self, x):
        return self.project(self._h0(x) - self.mu * x)

    def apply_M(self, x):
        """D x + 2 lambda P psi0 K (psi0 x) P, in one pass."""
        n = self.psi0 * x
        loc = self.v_s * x + (2.0 * self.kernel_scale) * self.psi0 * (
            self.f_loc * n + self.hartree(n))
        out = self._ifft(self.mask * (self.kin * self._fft(x) + self._fft(loc)))
        return self.project(out - self.mu * x)

    def dipole_vector(self, r_component):
        """v = P r psi0, the right-hand side the kick produces."""
        return self.project(self.band_limit(np.asarray(r_component) * self.psi0))

    # ---------------- the recursion --------------------------------------
    def ritz_density(self, v0, nsteps, coeffs):
        """Transition density of one Ritz mode, without storing the basis.

        The recursion keeps three vectors, so y_n = sum_j U_jn q_j is re-run
        (deterministic) and summed on the fly: one extra recursion instead of
        nsteps stored grid fields.  Returns drho = 2 psi0 y_n.
        """
        coeffs = np.asarray(coeffs, dtype=float)
        p = self.apply_D(v0)
        norm2 = self.dot(v0, p)
        q_prev = np.zeros_like(v0)
        q = v0 / np.sqrt(norm2)
        p = p / np.sqrt(norm2)
        y = np.zeros_like(v0)
        b_prev = 0.0
        for j in range(min(nsteps, len(coeffs))):
            y += coeffs[j] * q
            u = self.apply_M(p)
            a = self.dot(p, u)
            w = u - a * q - b_prev * q_prev
            dw = self.apply_D(w)
            b = np.sqrt(max(self.dot(w, dw), 0.0))
            if b < 1e-14 * max(abs(a), 1.0):
                break
            q_prev, q, p, b_prev = q, w / b, dw / b, b
        return 2.0 * self.psi0 * y

    def ritz_density_reorth(self, v0, nsteps, target_w2):
        """Transition density of the mode nearest target_w2, reorthogonalised.

        Stores the basis (nsteps grid fields), which subdominant modes need:
        otherwise the q_j drift and the Ritz vector degrades into noise even
        where the Ritz value is still good.  Returns (drho, omega_ha, resid).
        """
        p = self.apply_D(v0)
        norm2 = self.dot(v0, p)
        q = v0 / np.sqrt(norm2)
        p = p / np.sqrt(norm2)
        basis = [q]
        alpha, beta = [], []
        q_prev = np.zeros_like(v0)
        b_prev = 0.0
        for j in range(nsteps):
            u = self.apply_M(p)
            a = self.dot(p, u)
            w = u - a * q - b_prev * q_prev
            dw = self.apply_D(w)
            # full reorthogonalisation in the D metric: <q_i, w>_D = q_i . D w
            for _ in range(2):                       # twice is enough, and is
                for qi in basis:                     # the standard safeguard
                    c = self.dot(qi, dw)
                    if c:
                        w = w - c * qi
                dw = self.apply_D(w)
            b2 = self.dot(w, dw)
            alpha.append(a)
            beta.append(np.sqrt(max(b2, 0.0)))
            if beta[-1] < 1e-12 * max(abs(a), 1.0):
                break
            q_prev = q
            q = w / beta[-1]
            p = dw / beta[-1]
            b_prev = beta[-1]
            basis.append(q)
        alpha = np.asarray(alpha)
        beta = np.asarray(beta)
        vals, vecs = self.tridiag_eig(alpha, beta[:len(alpha) - 1], full=True)
        n = int(np.argmin(np.abs(vals - target_w2)))
        col = vecs[:, n]
        y = np.zeros_like(v0)
        for j, qj in enumerate(basis[:len(col)]):
            y += col[j] * qj
        w_ha = float(np.sqrt(max(vals[n], 0.0)))
        return 2.0 * self.psi0 * y, w_ha, abs(w_ha - np.sqrt(max(target_w2, 0.0)))

    def lanczos(self, v0, nsteps, callback=None):
        """D-metric Lanczos on T = M D.  Returns (alpha, beta, <v,v>_D).

        No reorthogonalisation: lost orthogonality duplicates converged Ritz
        values but leaves the spectral measure, all the continued fraction
        needs, intact.
        """
        p = self.apply_D(v0)
        norm2 = self.dot(v0, p)
        if norm2 <= 0:
            raise RuntimeError(f"<v|D|v> = {norm2:.3e}, not positive: psi0 is "
                               "not the lowest state of this H0")
        q_prev = np.zeros_like(v0)
        q = v0 / np.sqrt(norm2)
        p = p / np.sqrt(norm2)
        alpha = np.empty(nsteps)
        beta = np.empty(nsteps)
        b_prev = 0.0
        for j in range(nsteps):
            u = self.apply_M(p)
            a = self.dot(p, u)
            w = u - a * q - b_prev * q_prev
            dw = self.apply_D(w)
            b2 = self.dot(w, dw)
            b = np.sqrt(max(b2, 0.0))
            alpha[j] = a
            beta[j] = b
            if callback is not None:
                callback(j, alpha, beta, norm2)
            if b < 1e-14 * max(abs(a), 1.0):
                alpha = alpha[:j + 1]
                beta = beta[:j + 1]
                break
            q_prev = q
            q = w / b
            p = dw / b
            b_prev = b
        return alpha, beta, norm2

    # ---------------- sticks ---------------------------------------------
    @staticmethod
    def tridiag_eig(a, b, full=False):
        """(eigenvalues, first row of eigenvectors) of the Lanczos matrix.

        Converged Ritz values repeat without reorthogonalisation, and those
        tight clusters are what MRRR (stemr, scipy's default) fails on.  Fall
        through to bisection, then to a dense solve, rather than lose them.
        """
        for driver in ("stemr", "stebz"):
            try:
                vals, vecs = eigh_tridiagonal(a, b, lapack_driver=driver)
                return vals, (vecs if full else vecs[0])
            except Exception:
                continue
        t = np.diag(a) + np.diag(b, 1) + np.diag(b, -1)
        vals, vecs = np.linalg.eigh(t)
        return vals, (vecs if full else vecs[0])

    @classmethod
    def sticks(cls, alpha, beta, norm2, nsteps=None, wmin_ev=0.02):
        """Ritz pairs -> (omega [Ha], f).  sum f = 2 <v,v>_D = N at any length."""
        k = len(alpha) if nsteps is None else min(int(nsteps), len(alpha))
        vals, first = cls.tridiag_eig(alpha[:k], beta[:k - 1])
        w2 = np.asarray(vals, dtype=float)
        f = 2.0 * norm2 * np.asarray(first) ** 2
        keep = w2 > (wmin_ev / HA) ** 2
        return np.sqrt(w2[keep]), f[keep]

    # ---------------- the d block ----------------------------------------
    def attach_d_block(self, positions, f_xc, cut_bohr=10.0, n_elec=None,
                       kdd="grid", d_shift_ev=0.0, upf=None):
        """Couple the explicit 4d oscillators to the valence response.

            D~ = [[D, 0],      M~ = [[M,    S  ],
                  [0, w_d]]          [S^T,  w_d + 2 K_dd]]

        (S q)(r) = 2 P psi0 K_d[rho_d(q)], (S^T x)_k = 2 <rho_k|K_d psi0 x>.
        T~ = M~ D~, so recursion, sticks and f-sum carry over (the f-sum now
        counting the d strength too).

        Two traps: K_d is Hartree + f_xc ONLY (a Pauli term flips the sign of
        the d effect), and K_dd drops the on-site block that alpha carries.
        """
        from dftpy.td.d_embedding import Grid as _DGrid, build_d_modes_local

        dgrid = _DGrid(self.nr, self.lattice)
        pack = build_d_modes_local(dgrid, np.asarray(positions, float),
                                   cut_bohr=cut_bohr, n_elec=n_elec,
                                   upf=upf)
        self.d_cubes = pack["cubes"]
        self.d_index = pack["index"]
        # Dudarev U_eff: poles rise by U_eff/2, dipoles unchanged.
        self.d_shift_ev = float(d_shift_ev)
        # A baked-in U (u_eff_ev_applied in the UPF d block) plus d_shift_ev
        # would apply U twice, undetectably downstream.  Refuse.
        _applied = float(pack["info"].get("u_eff_ev_applied", 0.0) or 0.0)
        if _applied and self.d_shift_ev:
            raise ValueError(
                f"the pole set already carries U_eff = {_applied} eV "
                f"(u_eff_ev_applied), so d_shift_ev = {self.d_shift_ev} eV "
                f"would double-count it. Pass d_shift_ev=0, or use a pole set "
                f"with u_eff_ev_applied = 0.")
        self.omega_d = pack["omegas"] + self.d_shift_ev / HA
        self.mu_d = pack["dipoles"]
        self.d_info = dict(pack["info"], d_shift_ev=self.d_shift_ev,
                           u_eff_ev=2.0 * self.d_shift_ev)
        if self.d_shift_ev:
            # alpha and the reported energies must follow the shift, or the
            # printout describes a block that is not the one being solved.  D
            # is fixed, so f_old = 2 de D^2 gives alpha' = sum f_old/(de de').
            de = np.asarray(self.d_info["omega_ev"], float) / HA
            fk = np.asarray(self.d_info["f"], float)
            de2 = de + self.d_shift_ev / HA
            self.d_info["alpha_bare"] = self.d_info["alpha"]
            self.d_info["omega_ev_bare"] = list(self.d_info["omega_ev"])
            self.d_info["alpha"] = float((fk / (de * de2)).sum())
            self.d_info["omega_ev"] = list(de2 * HA)
        pack["info"] = self.d_info
        self.n_d = len(self.omega_d)
        self.f_xc_d = np.ascontiguousarray(np.asarray(f_xc, dtype=float))
        per_at = self.d_info["per_atom"]

        # K_dd: Hartree between modes, on-site block removed (alpha carries
        # it).  One Poisson solve per mode.  kdd="dipole" is the cheaper
        # point-dipole tensor, valid only for non-overlapping modes.
        per_at = self.d_info["per_atom"]
        n_at = len(self.d_cubes)
        rows = np.arange(self.n_d)
        atom = rows // per_at
        same = atom[:, None] == atom[None, :]
        if kdd == "grid":
            K = np.empty((self.n_d, self.n_d))
            e = np.zeros(self.n_d)
            for k in range(self.n_d):
                e[k] = 1.0
                K[:, k] = self.d_gather(self.hartree(self.d_scatter(e)))
                e[k] = 0.0
            self.K_dd = 0.5 * (K + K.T)
        else:
            pos = np.asarray(positions, float)
            L = np.diag(self.lattice)
            t = np.zeros((3 * n_at, 3 * n_at))
            for A in range(n_at):
                for B in range(n_at):
                    if A == B:
                        continue
                    dvec = pos[A] - pos[B]
                    dvec = (dvec + L / 2) % L - L / 2
                    rr = float(np.linalg.norm(dvec))
                    u = dvec / rr
                    for i in range(3):
                        for j in range(3):
                            t[3 * A + i, 3 * B + j] = ((i == j)
                                                       - 3 * u[i] * u[j]) / rr ** 3
            comp = rows % 3
            self.K_dd = (self.mu_d[:, None] * self.mu_d[None, :]
                         * t[3 * atom[:, None] + comp[:, None],
                             3 * atom[None, :] + comp[None, :]])
        self.K_dd[same] = 0.0
        self.d_info = dict(self.d_info, kdd=kdd)
        return self.d_info

    def _d_kernel(self, n):
        return self.hartree(n) + self.f_xc_d * n

    def d_scatter(self, q):
        """rho_d(q) on the grid, from the per-atom cubes."""
        out = np.zeros(self.nr)
        per_at = self.d_info["per_atom"]
        for A, (cube, ii) in enumerate(zip(self.d_cubes, self.d_index)):
            block = q[A * per_at:(A + 1) * per_at]
            if not np.any(block):
                continue
            out[np.ix_(*ii)] += (block @ cube).reshape(self.d_info["cube_shape"])
        return out

    def d_gather(self, w):
        """<rho_k | w> for every mode."""
        per_at = self.d_info["per_atom"]
        out = np.empty(self.n_d)
        for A, (cube, ii) in enumerate(zip(self.d_cubes, self.d_index)):
            out[A * per_at:(A + 1) * per_at] = cube @ w[np.ix_(*ii)].ravel()
        return out * self.dV

    def apply_D_sd(self, x, q):
        return self.apply_D(x), self.omega_d * q

    def apply_M_sd(self, x, q):
        lam = self.kernel_scale
        rho_d = self.d_scatter(q)
        xs = self.apply_M(x) + (2.0 * lam) * self.project(
            self.band_limit(self.psi0 * self._d_kernel(rho_d)))
        xd = (self.omega_d * q + (2.0 * lam) * (self.K_dd @ q)
              + (2.0 * lam) * self.d_gather(self._d_kernel(self.psi0 * x)))
        return xs, xd

    def lanczos_sd(self, v0, q0, nsteps):
        """The same recursion on (x, q); returns (alpha, beta, <v,v>_D~)."""
        def dot(a, b):
            return self.dot(a[0], b[0]) + float(np.dot(a[1], b[1]))

        v = (v0, q0)
        p = self.apply_D_sd(*v)
        # <v_s, q_j>_D with v_s = (v0, 0): the s dipole's overlap with each
        # Lanczos vector, which is all the s projection of the sticks needs.
        # One dot product per step, no stored vectors.
        cs = np.empty(nsteps)
        norm2 = dot(v, p)
        if norm2 <= 0:
            raise RuntimeError(f"<v|D|v> = {norm2:.3e}, not positive")
        sq = np.sqrt(norm2)
        q_prev = (np.zeros_like(v0), np.zeros_like(q0))
        qv = (v[0] / sq, v[1] / sq)
        p = (p[0] / sq, p[1] / sq)
        alpha, beta = np.empty(nsteps), np.empty(nsteps)
        b_prev = 0.0
        for j in range(nsteps):
            cs[j] = self.dot(v0, p[0])
            u = self.apply_M_sd(*p)
            a = dot(p, u)
            w = (u[0] - a * qv[0] - b_prev * q_prev[0],
                 u[1] - a * qv[1] - b_prev * q_prev[1])
            dw = self.apply_D_sd(*w)
            b = np.sqrt(max(dot(w, dw), 0.0))
            alpha[j], beta[j] = a, b
            if b < 1e-14 * max(abs(a), 1.0):
                alpha, beta, cs = alpha[:j + 1], beta[:j + 1], cs[:j + 1]
                break
            q_prev = qv
            qv = (w[0] / b, w[1] / b)
            p = (dw[0] / b, dw[1] / b)
            b_prev = b
        return alpha, beta, norm2, cs

    @classmethod
    def sticks_s(cls, alpha, beta, cs, nsteps=None, wmin_ev=0.02):
        """Sticks carrying the s dipole only: (omega [Ha], f_s).

        f_s,n = 2 <v_s, y_n>_D^2; the d oscillators stop radiating, energies
        untouched.  STRENGTHS ARE UNRELIABLE -- without reorthogonalisation the
        f-sum drifts off TRK, though peaks survive.  Valence only: q0 = 0.
        """
        k = len(alpha) if nsteps is None else min(int(nsteps), len(alpha))
        vals, vecs = cls.tridiag_eig(alpha[:k], beta[:k - 1], full=True)
        w2 = np.asarray(vals, dtype=float)
        f = 2.0 * (np.asarray(cs[:k]) @ vecs) ** 2
        keep = w2 > (wmin_ev / HA) ** 2
        return np.sqrt(w2[keep]), f[keep]

    def self_test(self, seed=1):
        """Symmetry of D and M, and the f-sum, on random band-limited vectors."""
        rng = np.random.default_rng(seed)
        a = self.project(self.band_limit(rng.standard_normal(self.nr)))
        b = self.project(self.band_limit(rng.standard_normal(self.nr)))
        out = {}
        for name, op in (("D", self.apply_D), ("M", self.apply_M)):
            ab, ba = self.dot(a, op(b)), self.dot(b, op(a))
            out[f"sym_{name}"] = abs(ab - ba) / max(abs(ab), 1e-30)
        out["mu_residual_ha"] = np.sqrt(max(self.dot(
            self.apply_D(self.psi0), self.apply_D(self.psi0)), 0.0))
        return out
