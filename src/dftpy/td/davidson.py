import numpy as np


def dot_global(x, y, mp):
    return mp.vsum(np.vdot(x, y))


def norm_global(x, mp):
    return float(np.sqrt(dot_global(x, x, mp).real))


def orthonormalize(v, basis, mp, tol=1.0e-12):
    """Modified Gram-Schmidt, repeated once for numerical stability."""
    q = np.asarray(v, dtype=np.complex128).copy()
    for _ in range(2):
        for b in basis:
            q -= dot_global(b, q, mp) * b
    norm = norm_global(q, mp)
    if norm < tol:
        return None
    return q / norm


def combine(coeffs, vectors):
    out = np.zeros_like(vectors[0])
    for c, v in zip(coeffs, vectors):
        out += c * v
    return out


def lowest_diagonal_guesses(diagonal, nvec, mp):
    """Distributed unit vectors at the globally lowest diagonal entries."""
    if mp.is_mpi:
        gathered = mp.comm.allgather(np.asarray(diagonal))
    else:
        gathered = [np.asarray(diagonal)]

    modes = []
    for rank, diag in enumerate(gathered):
        modes.extend((float(value), rank, idx) for idx, value in enumerate(diag))
    modes = sorted(modes)[:nvec]

    guesses = []
    for _, owner, idx in modes:
        guess = np.zeros_like(diagonal, dtype=np.complex128)
        if mp.rank == owner:
            guess[idx] = 1.0
        guesses.append(guess)
    return guesses


def davidson(
    matvec,
    diagonal,
    numeig,
    mp,
    tol=1.0e-8,
    maxiter=80,
    max_subspace=None,
    initial_vectors=None,
    min_denom=1.0e-3,
):
    """
    Distributed Davidson eigensolver.

    Vectors are local pieces of a global vector. ``matvec`` maps one local
    vector piece to the local piece of A*v. Dot products are MPI allreduces.
    """
    diagonal = np.asarray(diagonal, dtype=np.float64)
    max_subspace = max_subspace or max(8 * numeig, numeig + 24)
    initial_vectors = initial_vectors or lowest_diagonal_guesses(diagonal, numeig, mp)

    V = []
    for guess in initial_vectors:
        q = orthonormalize(guess, V, mp)
        if q is not None:
            V.append(q)
    if len(V) < numeig:
        raise RuntimeError("Could not build enough initial Davidson vectors.")

    AV = [matvec(v) for v in V]
    theta = None
    ritz = None

    for iteration in range(maxiter):
        nsub = len(V)
        H = np.empty((nsub, nsub), dtype=np.complex128)
        for i in range(nsub):
            for j in range(i, nsub):
                value = dot_global(V[i], AV[j], mp)
                H[i, j] = value
                H[j, i] = value.conjugate()

        evals, alpha = np.linalg.eigh(H)
        order = np.argsort(evals.real)[:numeig]
        theta = evals[order].real
        alpha = alpha[:, order]

        ritz = []
        residuals = []
        corrections = []
        for k in range(numeig):
            u = combine(alpha[:, k], V)
            Au = combine(alpha[:, k], AV)
            r = Au - theta[k] * u
            res = norm_global(r, mp)
            ritz.append(u)
            residuals.append(res)
            if res < tol:
                continue

            denom = diagonal - theta[k]
            sign = np.where(denom >= 0.0, 1.0, -1.0)
            denom = np.where(np.abs(denom) < min_denom, sign * min_denom, denom)
            q = orthonormalize(-r / denom, V + corrections, mp)
            if q is not None:
                corrections.append(q)

        max_res = max(residuals)
        nconv = sum(res < tol for res in residuals)
        if mp.is_root:
            print(
                f"Davidson iter {iteration + 1:3d}: subspace={nsub:4d}  "
                f"converged={nconv:3d}/{numeig}  max residual={max_res:.3e}",
                flush=True,
            )
        if nconv == numeig:
            return theta, ritz, residuals

        if len(V) + len(corrections) > max_subspace:
            V, AV = [], []
            keep = ritz + corrections[: max(0, max_subspace - len(ritz))]
            for q in keep:
                q = orthonormalize(q, V, mp)
                if q is not None:
                    V.append(q)
                    AV.append(matvec(q))
        else:
            V.extend(corrections)
            AV.extend(matvec(q) for q in corrections)

    return theta, ritz, residuals
