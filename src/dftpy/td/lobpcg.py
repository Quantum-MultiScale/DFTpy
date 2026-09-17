import numpy as np


def dot_global(x, y, mp):
    return mp.vsum(np.vdot(x, y))


def norm_global(x, mp):
    return float(np.sqrt(dot_global(x, x, mp).real))


def stack_vectors(vectors):
    return np.column_stack(vectors)


def norms_global(vectors, mp):
    local = np.array([np.vdot(v, v).real for v in vectors], dtype=np.float64)
    total = mp.vsum(local)
    return np.sqrt(np.maximum(total, 0.0))


def orthonormalize(v, basis, mp, tol=1.0e-12):
    q = np.asarray(v, dtype=np.complex128).copy()
    if basis:
        B = stack_vectors(basis)
        for _ in range(2):
            coeffs = mp.vsum(B.conj().T @ q)
            q -= B @ coeffs
    norm = norm_global(q, mp)
    if norm < tol:
        return None
    return q / norm


def combine(coeffs, vectors):
    return stack_vectors(vectors) @ coeffs


def unstack_vectors(matrix):
    return [matrix[:, i].copy() for i in range(matrix.shape[1])]


def svqb(vectors, mp, tol=1.0e-12, max_vectors=None):
    """Block orthonormalize with one Gram-matrix reduction."""
    if not vectors:
        return [], None

    V = stack_vectors(vectors)
    gram = mp.vsum(V.conj().T @ V)
    gram = 0.5 * (gram + gram.conj().T)
    evals, evecs = np.linalg.eigh(gram)

    keep = np.flatnonzero(evals > tol * tol)
    if keep.size == 0:
        return [], None
    keep = keep[np.argsort(evals[keep])[::-1]]
    if max_vectors is not None:
        keep = keep[:max_vectors]

    transform = evecs[:, keep] / np.sqrt(evals[keep])[None, :]
    Q = V @ transform
    return unstack_vectors(Q), transform


def block_project(vectors, basis, mp, passes=2):
    """Project a block of vectors out of an existing orthonormal basis."""
    if not vectors:
        return []
    V = stack_vectors(vectors)
    if basis:
        B = stack_vectors(basis)
        for _ in range(passes):
            coeffs = mp.vsum(B.conj().T @ V)
            V -= B @ coeffs
    return unstack_vectors(V)


def block_orthonormalize(vectors, mp, basis=None, tol=1.0e-12, max_vectors=None):
    vectors = block_project(vectors, basis or [], mp)
    qvectors, _ = svqb(vectors, mp, tol=tol, max_vectors=max_vectors)
    return qvectors


def block_project_with_av(vectors, avectors, basis, abasis, mp, passes=2):
    """Project vectors and carry H*v through the same linear operation."""
    if not vectors:
        return [], []
    V = stack_vectors(vectors)
    AV = stack_vectors(avectors)
    if basis:
        B = stack_vectors(basis)
        AB = stack_vectors(abasis)
        for _ in range(passes):
            coeffs = mp.vsum(B.conj().T @ V)
            V -= B @ coeffs
            AV -= AB @ coeffs
    return unstack_vectors(V), unstack_vectors(AV)


def block_orthonormalize_with_av(vectors, avectors, mp, basis=None, abasis=None, tol=1.0e-12, max_vectors=None):
    """Block orthonormalize vectors and transform known H*v without new matvecs."""
    vectors, avectors = block_project_with_av(
        vectors,
        avectors,
        basis or [],
        abasis or [],
        mp,
    )
    qvectors, transform = svqb(vectors, mp, tol=tol, max_vectors=max_vectors)
    if transform is None:
        return [], []
    AV = stack_vectors(avectors) @ transform
    return qvectors, unstack_vectors(AV)


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


def precondition(residual, diagonal, eig, diagonal_shift=0.0, damping=0.35, min_denom=1.0e-3):
    denom = diagonal + diagonal_shift - eig
    sign = np.where(denom >= 0.0, 1.0, -1.0)
    denom = np.where(np.abs(denom) < min_denom, sign * min_denom, denom)
    return -damping * residual / denom


def apply_matvecs(matvec, vectors, matvec_block=None):
    if not vectors:
        return []
    if matvec_block is not None:
        return matvec_block(vectors)
    return [matvec(v) for v in vectors]


def rayleigh_ritz(vectors, avectors, numeig, mp):
    V = stack_vectors(vectors)
    AV = stack_vectors(avectors)
    hsub = mp.vsum(V.conj().T @ AV)
    hsub = 0.5 * (hsub + hsub.conj().T)

    evals, coeffs = np.linalg.eigh(hsub)
    order = np.argsort(evals.real)[:numeig]
    theta = evals[order].real
    coeffs = coeffs[:, order]
    ritz_mat = V @ coeffs
    aritz_mat = AV @ coeffs
    ritz = [ritz_mat[:, i].copy() for i in range(numeig)]
    aritz = [aritz_mat[:, i].copy() for i in range(numeig)]
    return theta, ritz, aritz, coeffs


def lobpcg(
    matvec,
    diagonal,
    numeig,
    mp,
    tol=1.0e-8,
    maxiter=80,
    max_subspace=None,
    initial_vectors=None,
    diagonal_shift=0.0,
    damping=0.35,
    block_size=None,
    matvec_block=None,
    preconditioner=None,
    residual_guard_factor=5.0,
    keep_best_vectors=False,
):
    """
    Small distributed LOBPCG eigensolver.

    Each vector is the local piece of a global vector. ``matvec`` maps a local
    piece to the local piece of A*v. Dot products are MPI allreduces.
    """
    diagonal = np.asarray(diagonal, dtype=np.float64)
    block_size = block_size or max(2 * numeig, numeig + 4)
    block_size = max(block_size, numeig)
    max_subspace = max_subspace or 3 * block_size
    initial_vectors = initial_vectors or lowest_diagonal_guesses(diagonal, block_size, mp)

    X = []
    for guess in initial_vectors:
        q = orthonormalize(guess, X, mp)
        if q is not None:
            X.append(q)
        if len(X) == block_size:
            break
    if len(X) < block_size:
        raise RuntimeError("Could not build enough initial LOBPCG vectors.")

    AX = apply_matvecs(matvec, X, matvec_block)
    P = []
    AP = []
    theta = None
    residuals = None
    best_max_residual = np.inf
    best_theta = None
    best_X = None
    best_AX = None

    for iteration in range(maxiter):
        theta, X, AX, _ = rayleigh_ritz(X, AX, block_size, mp)

        R = [AX[i] - theta[i] * X[i] for i in range(block_size)]
        residuals = list(norms_global(R, mp))
        requested_residuals = residuals[:numeig]
        max_requested_residual = max(requested_residuals)

        if max_requested_residual < best_max_residual:
            best_max_residual = max_requested_residual
            if keep_best_vectors:
                best_theta = theta.copy()
                best_X = [x.copy() for x in X]
                best_AX = [ax.copy() for ax in AX]
        elif max_requested_residual > residual_guard_factor * best_max_residual:
            if mp.is_root:
                print(
                    f"LOBPCG restart: residual grew from {best_max_residual:.3e} "
                    f"to {max_requested_residual:.3e}",
                    flush=True,
                )
            if keep_best_vectors and best_theta is not None:
                theta = best_theta.copy()
                X = [x.copy() for x in best_X]
                AX = [ax.copy() for ax in best_AX]
            P = []
            AP = []
            damping *= 0.5
            R = [AX[i] - theta[i] * X[i] for i in range(block_size)]
            residuals = list(norms_global(R, mp))
            requested_residuals = residuals[:numeig]
            max_requested_residual = max(requested_residuals)

        nconv_local = sum(res < tol for res in requested_residuals)
        if mp.is_mpi:
            nconv = int(mp.comm.allreduce(nconv_local, op=mp.MPI.MIN))
        else:
            nconv = nconv_local
        if mp.is_root:
            print(
                f"LOBPCG iter {iteration + 1:3d}: converged={nconv:3d}/{numeig}  "
                f"max residual={max_requested_residual:.3e}",
                flush=True,
            )
        if nconv == numeig:
            return theta[:numeig], X[:numeig], requested_residuals

        W_raw = []
        for i, res in enumerate(residuals):
            if i < numeig and res < tol:
                continue
            if preconditioner is None:
                q = precondition(
                    R[i],
                    diagonal,
                    theta[i],
                    diagonal_shift=diagonal_shift,
                    damping=damping,
                )
            else:
                q = preconditioner(R[i], theta[i])
            W_raw.append(q)

        W = block_orthonormalize(W_raw, mp, basis=X + P, max_vectors=block_size)
        if not W:
            return theta, X, residuals

        AW = apply_matvecs(matvec, W, matvec_block)
        if P and len(X) + len(W) + len(P) > max_subspace:
            P = []
            AP = []
        V_candidates = X + W + P
        AV_candidates = AX + AW + AP
        S, transform = svqb(V_candidates, mp, max_vectors=max_subspace)
        if not S:
            return theta, X, residuals
        AS = unstack_vectors(stack_vectors(AV_candidates) @ transform)

        theta, Xnew, AXnew, coeffs = rayleigh_ritz(S, AS, block_size, mp)
        original_coeffs = transform @ coeffs
        tail_coeffs = original_coeffs[len(X):, :]
        tail_vectors = W + P
        tail_avectors = AW + AP
        P_raw = unstack_vectors(stack_vectors(tail_vectors) @ tail_coeffs)
        AP_raw = unstack_vectors(stack_vectors(tail_avectors) @ tail_coeffs)
        P, AP = block_orthonormalize_with_av(
            P_raw,
            AP_raw,
            mp,
            basis=Xnew,
            abasis=AXnew,
            max_vectors=block_size,
        )
        X, AX = Xnew, AXnew

    if keep_best_vectors and best_theta is not None and best_max_residual < max(residuals[:numeig]):
        return best_theta[:numeig], best_X[:numeig], list(norms_global(
            [best_AX[i] - best_theta[i] * best_X[i] for i in range(numeig)], mp
        ))

    return theta[:numeig], X[:numeig], residuals[:numeig]
