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

from petsc4py import PETSc  # noqa: E402
from slepc4py import SLEPc  # noqa: E402

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
                pc.setUp()          # triggers MatGetFactor; raises if unavailable
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


def _gather(vec):
    """A distributed Vec as a full numpy array replicated on every rank."""
    scatter, seq = PETSc.Scatter.toAll(vec)
    scatter.scatter(vec, seq, addv=False, mode=PETSc.ScatterMode.FORWARD)
    out = np.asarray(seq.getArray()).copy()
    scatter.destroy()
    seq.destroy()
    return out


def eigenpairs(A, k, sigma, factor=None, tol=1e-10):
    """The k eigenpairs of A nearest lambda = sigma, via shift-invert.

    Returns (eigenvalues, vectors) with eigenvalues a length-k complex array
    sorted by descending real part (so index 0 is the stationary lambda ~ 0)
    and vectors an (n, k) complex array gathered to every rank, column j the
    eigenvector for eigenvalues[j]. Pass A = W for right eigenvectors, A = W^T
    for the left eigenvectors of W."""
    E = SLEPc.EPS().create(comm=A.getComm())
    try:
        E.setOperators(A)
        E.setProblemType(SLEPc.EPS.ProblemType.NHEP)
        E.setDimensions(k, max(2 * k, k + 10))
        E.setTolerances(tol, max_it=1000)
        E.setWhichEigenpairs(SLEPc.EPS.Which.TARGET_MAGNITUDE)
        E.setTarget(sigma)
        st = E.getST()
        st.setType("sinvert")
        st.setShift(sigma)
        ksp = st.getKSP()
        ksp.setType("preonly")
        pc = ksp.getPC()
        pc.setType("lu")
        pc.setFactorSolverType(factor or choose_factor(A))
        E.setFromOptions()
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
        vr = A.createVecRight()
        vi = A.createVecRight()
        try:
            for i in range(k):
                vals[i] = E.getEigenpair(i, vr, vi)
                vecs[:, i] = _gather(vr) + 1j * _gather(vi)
        finally:
            vr.destroy()
            vi.destroy()
        order = np.argsort(-vals.real)
        return vals[order], vecs[:, order]
    finally: # Probably my first real case use of finally...
        # Make sure the matrix is properly destroyed regardless of solve.
        E.destroy()


def as_distribution(vec):
    """A lambda ~ 0 right eigenvector as a probability vector: real part, sign
    fixed so the mass is positive, clipped at zero and normalized to sum 1."""
    theta = np.asarray(vec).real
    if theta.sum() < 0:
        theta = -theta
    theta = np.clip(theta, 0.0, None)
    return theta / theta.sum()


def stationary(W, sigma, factor=None):
    """Stationary distribution Theta (W Theta = 0), gathered and normalized. It
    is the right null vector of W, i.e. the lambda ~ 0 eigenvector of W itself
    (not W^T)."""
    _, vecs = eigenpairs(W, 1, sigma, factor=factor)
    return as_distribution(vecs[:, 0])


def right_eigenpairs(W, k, sigma, factor=None):
    """The k slowest right eigenpairs of W, with Theta split out.

    Returns (Theta, eigenvalues, vectors) where eigenvalues[0] ~ 0 is the
    stationary mode and Theta is its normalized eigenvector; columns 1.. are the
    nonstationary modes r_m in descending Re(lambda). One shift-invert
    factorization serves both, so the caller never pays for a separate
    ``stationary`` solve."""
    vals, vecs = eigenpairs(W, k, sigma, factor=factor)
    return as_distribution(vecs[:, 0]), vals, vecs


def left_eigenpairs(W, k, sigma, factor=None):
    """The k slowest *left* eigenpairs of W (= eigenpairs of W^T)."""
    WT = W.transpose(PETSc.Mat())
    try:
        return eigenpairs(WT, k, sigma, factor=factor)
    finally:
        WT.destroy()


