"""PETSc/SLEPc backend for the ME-MKM spectral pipeline.

Each rank builds only the rows it owns via ``MEMKMBuilder.build_w_coo_range`` 
and every result (stationary distribution, eigenvectors, committors) is gathered 
to *all* ranks so the basin logic in ``coexistence`` runs identically everywhere
and the Brent search stays in lockstep across ranks.

The slow left eigenvectors are found with a SLEPc shift-invert EPS targeting
lambda = 0. The sparse LU factorization behind the shift uses the first
available parallel direct solver (MUMPS -> SuperLU_DIST -> PaStiX -> native
PETSc), so (hopefully) the same code runs on a laptop and on the cluster without change.
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
        ksp.setOperators(T)
        ksp.setType("preonly")
        pc = ksp.getPC()
        pc.setType("lu")
        pc.setFactorSolverType(st)
        try:
            pc.setUp()          # triggers MatGetFactor; raises if unavailable
            ksp.destroy()
            chosen = st
            break
        except PETSc.Error:
            ksp.destroy()
    T.destroy()
    return chosen


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


def _scatter_in(mat, arr, left=False):
    """A full numpy array as a Vec laid out like mat's rows (left) or cols."""
    v = mat.createVecLeft() if left else mat.createVecRight()
    lo, hi = v.getOwnershipRange()
    v.setValues(np.arange(lo, hi, dtype=PETSc.IntType), np.asarray(arr)[lo:hi])
    v.assemble()
    return v


def _lu_ksp(A, factor=None):
    """A `preonly` KSP that is just a direct LU solve of A, set up eagerly."""
    ksp = PETSc.KSP().create(comm=A.getComm())
    ksp.setOperators(A)
    ksp.setType("preonly")
    pc = ksp.getPC()
    pc.setType("lu")
    pc.setFactorSolverType(factor or choose_factor(A))
    ksp.setUp()
    return ksp


def eigenpairs(A, k, sigma, factor=None, tol=1e-10):
    """The k eigenpairs of A nearest lambda = sigma, via shift-invert.

    Returns (eigenvalues, vectors) with eigenvalues a length-k complex array
    sorted by descending real part (so index 0 is the stationary lambda ~ 0)
    and vectors an (n, k) complex array gathered to every rank, column j the
    eigenvector for eigenvalues[j]. Pass A = W for right eigenvectors, A = W^T
    for the left eigenvectors of W."""
    E = SLEPc.EPS().create(comm=A.getComm())
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
    for i in range(k):
        vals[i] = E.getEigenpair(i, vr, vi)
        vecs[:, i] = _gather(vr) + 1j * _gather(vi)
    vr.destroy()
    vi.destroy()
    E.destroy()
    order = np.argsort(-vals.real)
    return vals[order], vecs[:, order]


def stationary(W, sigma, factor=None):
    """Stationary distribution Theta (W Theta = 0), gathered, normalized to
    sum 1 and made non-negative. It is the right null vector of W, i.e. the
    lambda ~ 0 eigenvector of W itself (not W^T)."""
    _, vecs = eigenpairs(W, 1, sigma, factor=factor)
    theta = vecs[:, 0].real
    if theta.sum() < 0:
        theta = -theta
    theta = np.clip(theta, 0.0, None)
    return theta / theta.sum()


def left_eigenpairs(W, k, sigma, factor=None):
    """The k slowest *left* eigenpairs of W (= eigenpairs of W^T)."""
    WT = W.transpose(PETSc.Mat())
    vals, vecs = eigenpairs(WT, k, sigma, factor=factor)
    WT.destroy()
    return vals, vecs


def committor(W, in_A, in_B, factor=None):
    """Forward committor q (q=0 on A, q=1 on B) over all microstates.

    Solves the backward equation (W^T q)_i = 0 on the interior with the basin
    values pinned as Dirichlet conditions -- imposed on the *full* n x n system
    by replacing each basin row of W^T with the identity (``MatZeroRows``) and
    setting the right-hand side to q on those rows. This keeps the solve fully
    distributed with no submatrix extraction, and is equivalent to the interior
    block solve of ``me_mkm.sparse.committor`` (Eidelson & Peters 2012 eq. 10).
    """
    in_A = np.asarray(in_A, dtype=bool)
    in_B = np.asarray(in_B, dtype=bool)
    if np.any(in_A & in_B):
        raise ValueError("basins A and B overlap")
    if not in_A.any() or not in_B.any():
        raise ValueError("both basins must be non-empty")

    WT = W.transpose(PETSc.Mat())          # backward generator W^T (full)
    rstart, rend = WT.getOwnershipRange()

    def _local(mask):
        idx = np.where(mask)[0]
        return idx[(idx >= rstart) & (idx < rend)].astype(PETSc.IntType)

    boundary = _local(in_A | in_B)
    WT.zeroRows(boundary, diag=1.0)        # basin rows -> q_i = b_i

    b = WT.createVecLeft()
    b.set(0.0)
    local_B = _local(in_B)
    b.setValues(local_B, np.ones(len(local_B)))   # q = 1 on B, 0 on A/interior RHS
    b.assemble()

    q = WT.createVecRight()
    ksp = _lu_ksp(WT, factor)
    ksp.solve(b, q)

    out = _gather(q)
    # Destroy the KSP (and its LU factor of WT) *before* WT itself: the factor
    # holds a reference to WT, so freeing WT first leaves the PC pointing at
    # released memory (a segfault under the parallel direct solvers).
    for obj in (ksp, b, q, WT):
        obj.destroy()
    return out


def committor_backward(W, in_A, in_B, theta, factor=None):
    """Backward committor q^-[i] = P(last came from A rather than B).

    q^- is the forward committor of the pi-time-reversed generator
    W* = D W^T D^{-1} (D = diag(theta)); at a driven steady state q^- != 1 - q.
    Built by diagonally scaling W^T, then reusing ``committor`` with the target
    basin swapped to A."""
    theta = np.asarray(theta, dtype=float)
    if np.any(theta <= 0.0):
        raise ValueError(
            "committor_backward needs a strictly positive theta (irreducible "
            "steady state); found non-positive entries"
        )
    WT = W.transpose(PETSc.Mat())          # W^T
    L = _scatter_in(WT, theta, left=True)  # diag(theta) on the left
    R = _scatter_in(WT, 1.0 / theta)       # diag(1/theta) on the right
    WT.diagonalScale(L, R)                 # WT <- D W^T D^{-1} = W* (reversed gen)
    # committor takes the generator and transposes it internally, so pass W*.
    q_minus = committor(WT, in_A=in_B, in_B=in_A, factor=factor)
    WT.destroy()
    L.destroy()
    R.destroy()
    return q_minus


def reactive_flux(W, theta, core_A, q_plus):
    """TPT reactive flux F = sum_{i in A} pi_i (L q+)_i with L = W^T (the
    row-convention rate matrix), computed as a distributed matvec."""
    WT = W.transpose(PETSc.Mat())          # L_ij = rate i -> j
    qp = _scatter_in(WT, q_plus)
    Lq = WT.createVecLeft()
    WT.mult(qp, Lq)
    Lq_all = _gather(Lq)
    WT.destroy()
    qp.destroy()
    Lq.destroy()
    return float(theta[core_A] @ Lq_all[core_A])
