"""PETSc/SLEPc backend for the ME-MKM spectral pipeline.

Each rank builds only the rows it owns via ``MEMKMBuilder.build_w_coo_range``
and every result (stationary distribution, eigenvectors) is gathered
to *all* ranks so the basin logic in ``coexistence`` runs identically everywhere
and the Brent search stays in lockstep across ranks.

The slow eigenvectors are found with a SLEPc shift-invert EPS targeting lambda = 0. 
The sparse LU factorization behind the shift uses the first available parallel direct 
solver in this order:
    MUMPS -> SuperLU_DIST -> PaStiX -> native PETSc 
so (hopefully) the same code runs on a laptop and on the cluster without change.
"""

import sys

import numpy as np
import slepc4py

slepc4py.init(sys.argv)

from petsc4py import PETSc
from slepc4py import SLEPc

# Preferred sparse LU backends, best-scaling first. `petsc` (the built-in
# LU) is the always-present fallback.
_FACTOR_PREFERENCE = ("mumps", "superlu_dist", "pastix", "petsc")

# Which solver a given communicator can factor with is a property of the PETSc
# build and the (serial vs parallel) matrix type, not of the matrix values, so
# it is probed once with a tiny matrix and cached per comm size.
_factor_cache = {}


def _comm(comm):
    return comm if comm is not None else PETSc.COMM_WORLD


def _probe_factor(comm):
    """First LU solver in _FACTOR_PREFERENCE that can actually factor a matrix
    on `comm`, tested by symbolically setting up a 2x2 identity."""
    T = PETSc.Mat().create(comm=comm)
    try:
        T.setSizes(((PETSc.DECIDE, 2), (PETSc.DECIDE, 2)))
        T.setType("aij")
        T.setUp()
        lo, hi = T.getOwnershipRange()
        for i in range(lo, hi):
            T.setValue(i, i, 1.0)
        T.assemble()
        chosen = "petsc"
        for st in _FACTOR_PREFERENCE:
            if st == "petsc":
                break
            ksp = PETSc.KSP().create(comm=comm)
            try:
                ksp.setOperators(T)
                ksp.setType("preonly")
                pc = ksp.getPC()
                pc.setType("lu")
                pc.setFactorSolverType(st)
                pc.setUp()  # triggers MatGetFactor. Raises if unavailable
                chosen = st
                break
            except PETSc.Error:
                pass
            finally:
                ksp.destroy()
        return chosen
    finally:
        T.destroy()


def choose_factor(A, override=None):
    """LU solver to use for A, honoring `override`, else the cached probe."""
    if override:
        return override
    size = A.getComm().getSize()
    if size not in _factor_cache:
        _factor_cache[size] = _probe_factor(A.getComm())
    return _factor_cache[size]


def build_petsc_W(builder, comm=None):
    """Assemble the dynamical generator W as a distributed PETSc AIJ matrix.

    Each rank builds only its owned rows [rstart, rend) with
    ``build_w_coo_range`` and feeds them through PETSc's COO assembly, so no
    rank ever materializes the whole matrix."""
    comm = _comm(comm)
    n = builder.n_states
    A = PETSc.Mat().create(comm=comm)
    A.setSizes(((PETSc.DECIDE, n), (PETSc.DECIDE, n)))
    A.setType("aij")
    A.setUp()
    rstart, rend = A.getOwnershipRange()
    rows, cols, vals = builder.build_w_coo_range(rstart, rend)
    i = np.asarray(rows, dtype=PETSc.IntType)
    j = np.asarray(cols, dtype=PETSc.IntType)
    v = np.asarray(vals, dtype=PETSc.ScalarType)
    A.setPreallocationCOO(i, j)
    A.setValuesCOO(v)
    return A


def rate_scale(W):
    """max_i |W_ii|, the characteristic rate used to place the shift sigma."""
    d = W.getDiagonal()
    d.abs()
    return d.max()[1]


def coverage_class_blocks(builder, A):
    """Blocks for the ASM preconditioner, one per coverage class.

    Hopping carries almost all of W's magnitude and never changes a coverage
    class, so inverting these blocks alone approximates W well. See
    docs/me_mkm_scaling_strategy.md for the numbers.

    A class spanning two ranks gets split. That costs extra GMRES iterations but
    cannot change the answer."""
    rstart, rend = A.getOwnershipRange()
    ids = np.asarray(builder.coverage_class_ids(rstart, rend))
    order = np.argsort(ids, kind="stable")
    rows = np.arange(rstart, rend, dtype=PETSc.IntType)[order]
    cuts = np.flatnonzero(np.diff(ids[order])) + 1
    return [
        PETSc.IS().createGeneral(chunk, comm=PETSc.COMM_SELF)
        for chunk in np.split(rows, cuts)
        if chunk.size
    ]


def orbit_index(builder):
    """Build the orbit index for this tile.

    The symmetry group comes from nauty rather than being assumed, because it
    depends on the actual bond offsets: at l = 16, d = 3 gives order 32 and
    d = 7 gives 4,096.

    This walks the whole state space, but the result depends only on the tile
    and the species count, so cache it and reuse it across a sweep."""
    import me_mkm.symmetry as sym

    return sym.orbit_index(builder)


def build_petsc_W_orbit(builder, index, comm=None):
    """Assemble the lumped generator W_hat as a distributed PETSc matrix.

    This is exact rather than an approximation, orbits are strictly lumpable -
    so the eigenvalues match W's. Use ``lift_orbit`` to get back to
    microstates."""
    comm = _comm(comm)
    n = index.n_orbits
    A = PETSc.Mat().create(comm=comm)
    A.setSizes(((PETSc.DECIDE, n), (PETSc.DECIDE, n)))
    A.setType("aij")
    A.setUp()
    # Split by orbit *columns*, not rows. Column A is the orbit-sum of the events
    # firing from one configuration, so it costs ~l*n_reactions. The row form has
    # to walk every member of the orbit and each member's predecessors, measured
    # 100x slower at l = 10 (61.7 s vs 0.62 s) for the same entries. The emitted
    # rows land wherever those events do, and COO assembly routes off-process
    # entries to their owners.
    # Take the comm off the Mat, not the argument: callers pass an mpi4py
    # Intracomm (Get_rank) as readily as a PETSc Comm (getRank), and PETSc
    # accepts either when creating the Mat but only returns its own type here.
    pcomm = A.getComm()
    rank, size = pcomm.getRank(), pcomm.getSize()
    cstart, cend = (n * rank) // size, (n * (rank + 1)) // size
    rows, cols, vals = builder.build_w_orbit_coo_cols(index, cstart, cend)
    i = np.asarray(rows, dtype=PETSc.IntType)
    j = np.asarray(cols, dtype=PETSc.IntType)
    v = np.asarray(vals, dtype=PETSc.ScalarType)
    A.setPreallocationCOO(i, j)
    A.setValuesCOO(v)
    return A


def orbit_rep_counts(builder, index):
    """Per-species site counts at each orbit representative.

    Shape (n_orbits, n_species). A symmetry only moves sites around, so every
    state in an orbit has the same counts and the representative speaks for all
    of them."""
    reps = np.asarray(index.reps(), dtype=np.int64)
    m, l = builder.n_species, builder.l
    counts = np.zeros((reps.size, m), dtype=np.int64)
    t = reps.copy()
    idx = np.arange(reps.size)
    for _ in range(l):
        np.add.at(counts, (idx, t % m), 1)
        t //= m
    return counts


def orbit_rep_digits(builder, index):
    """Site occupancies at each orbit representative, (n_orbits, l)."""
    reps = np.asarray(index.reps(), dtype=np.int64)
    m, l = builder.n_species, builder.l
    d = np.empty((reps.size, l), dtype=np.int8)
    t = reps.copy()
    for p in range(l - 1, -1, -1):
        d[:, p] = t % m
        t //= m
    return d


def orbit_pair_counts(builder, index, a, b):
    """Number of a-b neighbour bonds at each orbit representative.

    Counted straight from the digits. ``count_reactive_range`` would be the
    natural call, but it wants a contiguous row range and the representatives are
    scattered."""
    d = orbit_rep_digits(builder, index)
    out = np.zeros(d.shape[0], dtype=np.int32)
    for i, j in builder.neighbor_pairs():
        di, dj = d[:, i], d[:, j]
        if a == b:
            out += (di == a) & (dj == a)
        else:
            out += ((di == a) & (dj == b)) | ((di == b) & (dj == a))
    return out


def orbit_coverage(builder, index, name):
    """Fractional coverage of `name` at each orbit representative."""
    code = builder.species_names.index(name)
    return orbit_rep_counts(builder, index)[:, code] / builder.l


def orbit_class_ids(builder, index):
    """Coverage-class id of each orbit.

    A coverage class always contains whole orbits, so the representative's class
    is the orbit's class."""
    m, l = builder.n_species, builder.l
    counts = orbit_rep_counts(builder, index)
    # Class keys drop species 0 (fixed at l, sum of the rest). Encode the
    # remaining counts in mixed radix (l+1) and look the code up, so the mapping
    # is a table hit rather than a per-orbit dict probe.
    radix = (l + 1) ** np.arange(m - 1)
    keys = np.asarray(builder.coverage_class_keys(), dtype=np.int64)
    table = np.full((l + 1) ** (m - 1), -1, dtype=np.int64)
    table[keys @ radix] = np.arange(keys.shape[0])
    return table[counts[:, 1:] @ radix]


def orbit_class_blocks(builder, index, A):
    """Preconditioner blocks over A's owned orbit rows.

    The orbit-space version of ``coverage_class_blocks``. The blocks shrink with
    the matrix, which is what keeps an exact LU per block affordable."""
    rstart, rend = A.getOwnershipRange()
    ids = orbit_class_ids(builder, index)[rstart:rend]
    order = np.argsort(ids, kind="stable")
    rows = np.arange(rstart, rend, dtype=PETSc.IntType)[order]
    cuts = np.flatnonzero(np.diff(ids[order])) + 1
    return [
        PETSc.IS().createGeneral(chunk, comm=PETSc.COMM_SELF)
        for chunk in np.split(rows, cuts)
        if chunk.size
    ]


def orbit_ids(builder, index):
    """Orbit id of every microstate.

    Use it to spread an orbit-space *function* out to microstates,
    ``f[orbit_ids(...)]``. Densities go through ``lift_orbit`` instead, which
    divides by the orbit size."""
    import me_mkm.symmetry as sym

    return np.asarray(sym.orbit_ids(builder, index))


def lift_orbit(builder, index, x_hat):
    """Expand an orbit-space density back to microstates.

    Divides by the orbit size: Theta[i] = Theta_hat[orbit(i)] / |orbit(i)|. This
    is the only place orbit sizes belong, the generator itself does not need
    them. Getting it wrong still gives something that looks like a distribution,
    so check the result sums to 1."""
    import me_mkm.symmetry as sym

    return sym.lift_orbit_vector(builder, index, np.asarray(x_hat))


def _gather(vec):
    """A distributed Vec as a full numpy array replicated on every rank."""
    scatter, seq = PETSc.Scatter.toAll(vec)
    scatter.scatter(vec, seq, addv=False, mode=PETSc.ScatterMode.FORWARD)
    out = np.asarray(seq.getArray()).copy()
    scatter.destroy()
    seq.destroy()
    return out


def eigenpairs(
    A,
    k,
    sigma,
    factor=None,
    tol=1e-10,
    blocks=None,
    two_sided=False,
    inner_rtol=1e-12,
    gmres_restart=100,
    max_restart=1600,
):
    """``_eigenpairs_once``, retried when the inner GMRES stalls.

    Restarted GMRES sometimes plateaus short of `inner_rtol` and gives up. Which
    parameter points it happens at varies, so rather than picking one restart for
    the whole sweep this doubles the restart, then loosens the tolerance.

    If everything stalls it raises RuntimeError saying so. Without that you get
    an opaque SystemError from SLEPc with the real cause hidden."""
    if blocks is None:
        return _eigenpairs_once(
            A, k, sigma, factor, tol, blocks, two_sided, inner_rtol, gmres_restart
        )
    last = None
    # Escalate the restart first (free accuracy), then relax the inner tolerance.
    # PETSc measures rtol on the *preconditioned* residual, so 1e-12 there is
    # roughly 5e-10 true, so there is real headroom before the eigenvalues move
    # by more than the Brent search (xtol 1e-5) can notice.
    for rtol in (inner_rtol, inner_rtol * 1e2, inner_rtol * 1e4):
        restart = gmres_restart
        while restart <= max_restart:
            try:
                out = _eigenpairs_once(
                    A, k, sigma, factor, tol, blocks, two_sided, rtol, restart
                )
                if rtol != inner_rtol:
                    PETSc.Sys.Print(
                        f"  [memkm] inner GMRES stagnated at rtol={inner_rtol:g}; "
                        f"converged at rtol={rtol:g}, restart={restart}",
                        comm=A.getComm(),
                    )
                return out
            except (SystemError, PETSc.Error) as exc:
                last, restart = exc, restart * 2
    raise RuntimeError(
        f"inner GMRES stagnated at every restart up to {max_restart} and every "
        f"tolerance up to {inner_rtol * 1e4:g} (sigma={sigma:.3g}, "
        f"blocks={len(blocks)}). The shift-invert operator is never applied "
        f"accurately enough, so the coverage-class blocks are too weak a "
        f"preconditioner here. Use the symmetry reduction (use_orbits=True), "
        f"fall back to blocks=None for an exact LU, or reduce khop_scale."
    ) from last


def _eigenpairs_once(
    A,
    k,
    sigma,
    factor=None,
    tol=1e-10,
    blocks=None,
    two_sided=False,
    inner_rtol=1e-12,
    gmres_restart=100,
):
    """The k eigenpairs of A closest to sigma, by shift-invert.

    Returns (eigenvalues, vectors), plus left_vectors when `two_sided`.
    Eigenvalues are sorted by descending real part, so index 0 is the stationary
    one. Vectors are (n, k), gathered onto every rank. Pass A = W for right
    eigenvectors, A = W^T for left ones.

    `blocks` switches the inner solve from an exact LU of W to preconditioned
    GMRES, which is what avoids the fill-in wall. The eigenvalues come out the
    same either way. Only the iteration count changes. `blocks=None` keeps the
    exact LU, useful as a reference.

    `two_sided` gets the left eigenvectors from this solve instead of a second
    one on W^T.

    Keep `inner_rtol` well below `tol`. Krylov-Schur assumes the shift-invert
    operator is applied exactly, and a sloppy inner solve shows up as error in
    the eigenvalues."""
    E = SLEPc.EPS().create(comm=A.getComm())
    try:
        E.setOperators(A)
        E.setProblemType(SLEPc.EPS.ProblemType.NHEP)
        E.setDimensions(k, max(2 * k, k + 10))
        E.setTolerances(tol, max_it=1000)
        E.setWhichEigenpairs(SLEPc.EPS.Which.TARGET_MAGNITUDE)
        E.setTarget(sigma)
        if two_sided:
            E.setTwoSided(True)
        st = E.getST()
        st.setType("sinvert")
        st.setShift(sigma)
        ksp = st.getKSP()
        pc = ksp.getPC()
        if blocks is None:
            ksp.setType("preonly")
            pc.setType("lu")
            pc.setFactorSolverType(factor or choose_factor(A))
        else:
            ksp.setType("gmres")
            ksp.setTolerances(rtol=inner_rtol, max_it=5000)
            # The default GMRES(30) stagnates on this operator: at l = 12 it
            # stalls at a true residual of 7.7e-10 and quits with
            # DIVERGED_BREAKDOWN after 10 restarts. A longer Krylov run before
            # restarting converges and is also faster (226 its at restart=100 vs
            # 300 wasted at 30). Going much beyond that costs more than it saves.
            ksp.setGMRESRestart(gmres_restart)
            pc.setType("asm")
            # ASM with zero overlap and RESTRICT is block Jacobi over arbitrary
            # index sets. PCBJACOBI only exposes equal-sized blocks, which
            # coverage classes are not.
            pc.setASMLocalSubdomains(len(blocks), blocks)
            pc.setASMOverlap(0)
            pc.setASMType(PETSc.PC.ASMType.RESTRICT)
        E.setFromOptions()
        if blocks is not None:
            # The sub-KSPs do not exist until the PC is set up, and EPSSetUp is
            # what sets it up. Configuring them earlier fails inside EPSSolve.
            E.setUp()
            for sub in pc.getASMSubKSP():
                sub.setType("preonly")
                sub.getPC().setType("lu")
        E.solve()

        nconv = E.getConverged()
        if nconv < k:
            raise RuntimeError(
                f"SLEPc converged only {nconv}/{k} eigenpairs at sigma={sigma:.3g}; "
                "raise --n-eigs/ncv or loosen the shift."
            )
        n = A.getSize()[0]
        vals = np.empty(k, dtype=complex)
        vecs = np.empty((n, k), dtype=complex)
        lvecs = np.empty((n, k), dtype=complex) if two_sided else None
        vr = A.createVecRight()
        vi = A.createVecRight()
        wr = A.createVecRight() if two_sided else None
        wi = A.createVecRight() if two_sided else None
        try:
            for i in range(k):
                vals[i] = E.getEigenpair(i, vr, vi)
                vecs[:, i] = _gather(vr) + 1j * _gather(vi)
                if two_sided:
                    E.getLeftEigenvector(i, wr, wi)
                    lvecs[:, i] = _gather(wr) + 1j * _gather(wi)
        finally:
            for v in (vr, vi, wr, wi):
                if v is not None:
                    v.destroy()
        order = np.argsort(-vals.real)
        if two_sided:
            return vals[order], vecs[:, order], lvecs[:, order]
        return vals[order], vecs[:, order]
    finally: # Probably my first real case use of finally...
        # Make sure the matrix is properly destroyed regardless of solve.
        E.destroy()


def as_distribution(vec):
    """Turn a lambda = 0 eigenvector into a probability vector.

    Takes the real part, flips the sign if the mass came out negative, clips
    anything below zero and normalizes to sum 1."""
    theta = np.asarray(vec).real
    if theta.sum() < 0:
        theta = -theta
    theta = np.clip(theta, 0.0, None)
    return theta / theta.sum()


def stationary(W, sigma, factor=None, blocks=None):
    """Stationary distribution Theta, normalized to sum 1.

    It is the right null vector of W, the lambda = 0 eigenvector of W, not of
    W^T."""
    _, vecs = eigenpairs(W, 1, sigma, factor=factor, blocks=blocks)
    return as_distribution(vecs[:, 0])


def right_eigenpairs(W, k, sigma, factor=None, blocks=None):
    """The k slowest right eigenpairs of W, with Theta pulled out.

    Returns (Theta, eigenvalues, vectors). Mode 0 is the stationary one and Theta
    is its normalized eigenvector. The rest are the slow relaxation modes. One
    solve gives all of them, so there is no need to call ``stationary``
    separately."""
    vals, vecs = eigenpairs(W, k, sigma, factor=factor, blocks=blocks)
    return as_distribution(vecs[:, 0]), vals, vecs


def right_and_left_eigenpairs(W, k, sigma, factor=None, blocks=None):
    """The k slowest right and left eigenpairs of W, from one solve.

    Returns (Theta, eigenvalues, right_vectors, left_vectors). Doing both sides
    at once avoids rebuilding and refactorizing W^T, which used to dominate a
    sweep."""
    vals, vecs, lvecs = eigenpairs(
        W, k, sigma, factor=factor, blocks=blocks, two_sided=True
    )
    return as_distribution(vecs[:, 0]), vals, vecs, lvecs


def left_eigenpairs(W, k, sigma, factor=None, blocks=None):
    """The k slowest left eigenpairs of W, i.e. eigenpairs of W^T.

    Use ``right_and_left_eigenpairs`` if you want both sides, this makes a full
    transposed copy and solves again. `blocks` from W work here too, since the
    coverage classes partition the indices the same way either direction."""
    WT = W.transpose(PETSc.Mat())
    try:
        return eigenpairs(WT, k, sigma, factor=factor, blocks=blocks)
    finally:
        WT.destroy()


