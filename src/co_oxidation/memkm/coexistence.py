"""Locate the coexistence point x* along one designated physics axis where the two
metastable macrostates carry equal stationary weight, pi(A) = pi(B).

Basins are not hand-drawn coverage windows chosen up front. At each x the slowest
nonstationary *right* eigenvector psi_R_2 of the column generator W is computed
(through SLEPc) and every microstate is assigned by its sign,

    A(x) = {i : psi_R_2(i) <  0},
    B(x) = {i : psi_R_2(i) >= 0},

which is the approximate variational basins for a two-basin system

The finite-tile coexistence point is then defined operationally as the x at which the
two spectral macrostates carry equal stationary weight, P_A = P_B = 1/2.

The overall sign of psi_R_2 is arbitrary and only exchanges the labels A and B. That keeps
"basin A" on the same physical branch at every x, which is what makes
log pi(A)/pi(B) monotone in x so a sign change brackets x*, refined by Brent's method.

x is whichever ME-MKM physics parameter (k_o_ads, eps, temperature, k_co_ads, ...) the
caller designates as CoexistencePipeline's axis_field; it must be a continuous rate/energy
parameter, not a structural one like the tile's site count -- see
sweeps.coexistence_driver.validate_bisection_axis.
"""

import numpy as np
from me_mkm.microstates import coverage_classes, microstate_as_coverage
from scipy.optimize import brentq

from ..common import CO, EMPTY, O
from . import backend
from .model import generate_model

_COVERAGE_NAMES = (("empty", EMPTY), ("co", CO), ("o", O))
IM_RE_TOL = 1e-6


class NoCoexistingMacrostatesError(ValueError):
    """The slow mode at this point has no two-basin structure to partition on: either
    lambda_2 = 0 (the generator is reducible) or the sign partition puts all stationary
    weight on one side (the point is monostable). Callers that just want a per-step
    ME-MKM snapshot (sweeps.steps.run_memkm_for_step) can catch this and skip the step
    instead of failing the whole run; CoexistencePipeline re-raises it as a plain
    ValueError since a Brent search can't bisect through a point with no basin split."""


# --- pure, module-level helpers (no pipeline state, safe to call standalone) ---------

def species_coverage_array(builder, name):
    code = builder.species_names.index(name)
    return np.array([microstate_as_coverage(builder, i)[code]
                     for i in range(builder.n_states)])


def _reduced_coordinate(psi_R_2, theta):
    """psi_R_2 / Theta: a cheap stand-in for the slow left eigenfunction, used to
    measure how firmly a microstate belongs to its macrostate."""
    return np.divide(psi_R_2, theta, out=np.zeros_like(psi_R_2), where=theta > 0.0)


def _basin_split(psi_R_2):
    """Boolean masks (in_A, in_B) assigning every microstate by the sign of the slow
    right mode: A = {psi_R_2 < 0}, B = {psi_R_2 >= 0}."""
    in_B = np.asarray(psi_R_2) >= 0.0
    return ~in_B, in_B


def _orient(psi_R_2, theta, cov):
    """Sign-fix psi_R_2 so that A = {psi_R_2 < 0} is the ORDER_SPECIES-rich set.

    psi_R_2 -> -psi_R_2 leaves the equal-weight condition untouched and only swaps the
    labels, so tying the sign to a physical observable is what keeps "basin A" the same
    branch across the sweep."""
    psi_R_2 = np.asarray(psi_R_2, dtype=float)
    in_A = psi_R_2 < 0.0
    w_A, w_B = theta[in_A].sum(), theta[~in_A].sum()
    if w_A <= 0.0 or w_B <= 0.0:
        raise NoCoexistingMacrostatesError(
            "the sign partition of the slow mode leaves all stationary "
            "weight on one side, so there are no two macrostates to "
            "balance; lambda_2 is not a two-basin switching mode here")
    mean_A = float(theta[in_A] @ cov[in_A]) / w_A
    mean_B = float(theta[~in_A] @ cov[~in_A]) / w_B
    return -psi_R_2 if mean_A < mean_B else psi_R_2


def solve_memkm_state(model_kwargs, tile, comm=None, n_eigs_scan=4, sigma_scale=1e-8,
                       factor=None, order_species="CO", coverage_cache=None):
    """One ME-MKM solve at a fixed parameter point: builds the generator, computes the
    stationary distribution and the n_eigs_scan slowest right eigenpairs via one
    shift-invert factorization, then orients and sign-partitions the slow mode.

    Pure -- no result caching across calls. CoexistencePipeline._state wraps this with
    its own per-axis-value cache for the Brent search; sweeps.steps.run_memkm_for_step
    calls it directly for a one-shot per-grid-step solve. `coverage_cache`, if given, is
    an externally-owned {species_name: array} dict reused across calls on the same
    tile -- coverage-per-microstate is purely structural (depends on tile + species
    ordering, not on the rates in model_kwargs), so it's safe to share across every
    solve on one tile.
    """
    builder = generate_model(tile=tile, **model_kwargs)
    W = backend.build_petsc_W(builder, comm)
    try:
        sigma = sigma_scale * backend.rate_scale(W)
        theta, eigvals, vecs = backend.right_eigenpairs(W, n_eigs_scan, sigma, factor)
    finally:
        W.destroy()
    if eigvals[1].real == 0.0:
        raise NoCoexistingMacrostatesError(
            "lambda_2 = 0: the generator is reducible, so there is no unique "
            "stationary distribution and no slow switching mode to partition on")
    if coverage_cache is not None and order_species in coverage_cache:
        cov = coverage_cache[order_species]
    else:
        cov = species_coverage_array(builder, order_species)
        if coverage_cache is not None:
            coverage_cache[order_species] = cov
    psi_R_2 = _orient(vecs[:, 1].real, theta, cov)
    in_A, in_B = _basin_split(psi_R_2)
    res = {"builder": builder, "theta": theta, "sigma": sigma,
           "eigvals": eigvals, "psi_R_2": psi_R_2, "in_A": in_A, "in_B": in_B,
           "vecs_R": vecs}
    res["coord"] = _reduced_coordinate(psi_R_2, theta)
    return res


def solve_left_modes(state, n_eigs, comm=None, factor=None):
    """Left eigenpairs (eigenvectors of W^T) at a `state` dict already produced by
    solve_memkm_state: the n_eigs slowest left eigenpairs plus the oriented psi_L_2,
    sign-fixed against state['psi_R_2']. Re-solves W from state['builder']."""
    W = backend.build_petsc_W(state["builder"], comm)
    try:
        eigvals, phi_slow = backend.left_eigenpairs(W, n_eigs, state["sigma"], factor)
    finally:
        W.destroy()
    psi_L_2 = phi_slow[:, 1].real
    if psi_L_2 @ state["psi_R_2"] < 0.0:
        psi_L_2 = -psi_L_2
    return {"eigvals": eigvals, "phi_slow": phi_slow, "psi_L_2": psi_L_2}


def coverage_grids(builder, theta, psi_L_2, psi_R_2):
    """Bin the stationary-weighted quantities into the coverage plane. Returns four
    (l+1, l+1) arrays indexed [N_species1, N_species2] (for CO oxidation: [N_CO, N_O]):

      cov_pop[a, b] = sum_{i in class} pi_i               (population)
      cov_phi[a, b] = mean_{i in class} psi_L_2,i         (slow coordinate)
      cov_r2[a, b]  = sum_{i in class} psi_R_2,i          (slow mode)
      cov_deg[a, b] = number of microstates in the class  (degeneracy)

    psi_R_2 is a density, not a coordinate, so it is *summed* over each class rather
    than pi-averaged: its class sums are the contribution of that coverage class to
    each macrostate's weight, and they sum to zero over the whole plane. Its sign is
    the partition itself.

    phi is NaN where the class is empty (or where psi_L_2 is None -- left modes were
    not solved); degeneracy lets a caller form the per-microstate mean weight
    cov_pop / cov_deg, which strips the combinatorial class-size factor from the
    population."""
    l = builder.l
    cov_pop = np.zeros((l + 1, l + 1))
    cov_phi = np.full((l + 1, l + 1), np.nan)
    cov_r2 = np.zeros((l + 1, l + 1))
    cov_deg = np.zeros((l + 1, l + 1))
    psi_R_2 = np.asarray(psi_R_2)
    for counts, idxs in coverage_classes(builder):
        a, b = int(counts[0]), int(counts[1])
        cov_pop[a, b] = theta[idxs].sum()
        cov_deg[a, b] = len(idxs)
        cov_r2[a, b] = psi_R_2[idxs].sum()
        if psi_L_2 is not None and len(idxs) > 0:
            cov_phi[a, b] = psi_L_2[idxs].mean()
    return cov_pop, cov_phi, cov_r2, cov_deg


class CoexistencePipeline:
    """Stateful driver over one tile and one bisection axis: caches per-axis-value
    results and the (axis-independent) coverage arrays so a sweep + Brent search reuses
    work. Correct on a size-1 communicator (serial) and across ranks.

    `axis_field` names the RunConfig/generate_model kwarg being varied (typically
    "k_o_ads", but any continuous physics parameter works); `base_kwargs` holds every
    other generate_model kwarg (physics rates, eps, temperature, khop_scale,
    k_o_des_scale) -- any entry under `axis_field` in `base_kwargs` is ignored, since the
    per-call value always comes from the scalar passed to _state/basin_log_ratio/etc.
    """

    def __init__(self, tile, axis_field="k_o_ads", base_kwargs=None, comm=None,
                 order_species="CO", n_eigs_scan=4, boundary_eps=0.05,
                 sigma_scale=1e-8, factor=None, basin_species=None):
        if n_eigs_scan < 3:
            raise ValueError("n_eigs_scan must be >= 3: lambda_1 (stationary), "
                             "lambda_2 (the partition) and lambda_3 (the "
                             f"spectral-gap denominator); got {n_eigs_scan}")
        if not 0.0 <= boundary_eps < 0.5:
            raise ValueError("boundary_eps is a fraction of the slow "
                             "coordinate's plateau separation and must lie in "
                             f"[0, 0.5) so the two plateau windows stay "
                             f"disjoint; got {boundary_eps}")
        self.tile = tile
        self.axis_field = axis_field
        self.base_kwargs = {k: v for k, v in (base_kwargs or {}).items()
                            if k != axis_field}
        self.comm = comm
        self.order_species = order_species
        self.basin_species = basin_species   # None -> auto-detect the other adsorbate
        self.n_eigs_scan = n_eigs_scan
        self.boundary_eps = boundary_eps
        self.sigma_scale = sigma_scale
        self.factor = factor
        self._cache = {}       # axis value -> theta + slow mode (no PETSc objects)
        self._left = {}        # axis value -> left eigenpairs (only solved at x*)
        self._cov_cache = {}   # species name -> per-microstate coverage array

    def _other_species(self, builder):
        """The adsorbate that characterizes macrostate B: the explicit
        `basin_species` if given, else the single non-vacancy species that is
        not ORDER_SPECIES. Used only for labelling."""
        if self.basin_species is not None:
            return self.basin_species
        others = [s for s in builder.species_names[1:] if s != self.order_species]
        if len(others) != 1:
            raise ValueError(
                "cannot auto-pick the basin-B species from "
                f"{builder.species_names}; pass basin_species explicitly")
        return others[0]

    def _species_coverage(self, builder, name):
        if name not in self._cov_cache:
            self._cov_cache[name] = species_coverage_array(builder, name)
        return self._cov_cache[name]

    # --- inner loop: stationary distribution + slow right mode ----------------
    def _state(self, x):
        """Theta, the slow eigenvalues and the oriented psi_R_2 at axis value x,
        cached. One shift-invert solve delivers both Theta (the lambda ~ 0 right
        eigenvector) and the nonstationary modes, so the scan costs a single
        factorization per x. W is destroyed before returning so a long Brent search
        never accumulates distributed matrices."""
        if x in self._cache:
            return self._cache[x]
        kwargs = {**self.base_kwargs, self.axis_field: x}
        try:
            res = solve_memkm_state(kwargs, self.tile, self.comm, self.n_eigs_scan,
                                    self.sigma_scale, self.factor, self.order_species,
                                    coverage_cache=self._cov_cache)
        except ValueError as exc:
            raise ValueError(f"{exc} (at {self.axis_field}={x:.6g})") from exc
        self._cache[x] = res
        return res

    def slow_mode(self, x):
        """The oriented slow right eigenvector psi_R_2 at x (a copy).

        Consecutive scan points should overlap strongly; a collapse of
        |<psi_R_2(x_k), psi_R_2(x_k-1))>| flags a mode crossing, i.e. the sweep
        stopped following the same physical process."""
        return self._state(x)["psi_R_2"].copy()

    def left_modes(self, x, n_eigs=3):
        """The n_eigs slowest left eigenpairs at x plus the oriented psi_L_2, cached;
        re-solves only if more modes are asked for than are held."""
        cached = self._left.get(x)
        if cached is not None and len(cached["eigvals"]) >= n_eigs:
            return cached
        res = solve_left_modes(self._state(x), n_eigs, self.comm, self.factor)
        self._left[x] = res
        return res

    # --- observables ----------------------------------------------------------
    def basin_weights(self, x):
        """(P_A, P_B): the stationary weight of each spectral macrostate. They
        sum to 1 because the sign partition is exhaustive."""
        s = self._state(x)
        theta = s["theta"]
        return float(theta[s["in_A"]].sum()), float(theta[s["in_B"]].sum())

    def basin_log_ratio(self, x):
        """ln pi(A)/pi(B): the coexistence objective, zero at x*."""
        P_A, P_B = self.basin_weights(x)
        return float(np.log(P_A / P_B))

    def diagnostics(self, x):
        """The checks that decide whether the two-state reading holds at x.

        `spectral_gap` = |Re lambda_3| / |Re lambda_2| must be well above 1 for
        a single slow process to control the partition; `im_re_ratio` must be
        negligible or the slow modes are a complex pair and the sign of one
        eigenvector is not a partition at all; `stationary_residual` is
        |lambda_1| / |Re lambda_2|, a check that the mode taken as stationary
        really is the null one.

        `boundary_mass` is the stationary weight sitting on *neither* plateau of
        the slow coordinate -- states further than `boundary_eps` * (plateau
        separation) from both conditional means -- i.e. the probability that is
        not firmly assigned and could switch sides under a small numerical or
        model perturbation. Note this measures distance from the plateaus rather
        than from the zero crossing, which the reference workflow suggests. The
        zero crossing is not usable here: probability conservation pins the
        Theta-weighted mean of the coordinate at zero, so whenever the two
        macrostates are lopsided the *heavier* one is pushed onto the dividing
        surface and a zero-centred window swallows it, reporting fragility that
        is really just imbalance. Plateau distance is scale-free in the weights
        and agrees with the zero-crossing measure when they are balanced -- as
        they are at x*, the only place the number is really being read."""
        s = self._state(x)
        theta, eigvals = s["theta"], s["eigvals"]
        lam1, lam2, lam3 = eigvals[0], eigvals[1], eigvals[2]
        coord, in_A, in_B = s["coord"], s["in_A"], s["in_B"]
        P_A, P_B = theta[in_A].sum(), theta[in_B].sum()
        m_A = float(theta[in_A] @ coord[in_A]) / P_A
        m_B = float(theta[in_B] @ coord[in_B]) / P_B
        cut = self.boundary_eps * abs(m_B - m_A)
        near = (np.abs(coord - m_A) > cut) & (np.abs(coord - m_B) > cut)
        return {
            "lambda2_re": float(lam2.real), "lambda2_im": float(lam2.imag),
            "lambda3_re": float(lam3.real),
            "spectral_gap": float(abs(lam3.real) / abs(lam2.real)),
            "im_re_ratio": float(abs(lam2.imag) / abs(lam2.real)),
            "stationary_residual": float(abs(lam1) / abs(lam2.real)),
            "boundary_mass": float(theta[near].sum()),
            "n_A": int(s["in_A"].sum()), "n_B": int(s["in_B"].sum()),
        }

    def coverages(self, x):
        """Mean fractional ME-MKM coverages (empty, CO, O) under Theta,
        directly comparable to the kMC steady coverages."""
        s = self._state(x)
        builder, theta = s["builder"], s["theta"]
        out = {}
        for name, code in _COVERAGE_NAMES:
            cov = self._species_coverage(builder, builder.species_names[code])
            out[name] = float(theta @ cov)   # cov is already a fraction n_s/l
        return out

    def basin_coverages(self, x):
        """Coverages conditioned on each spectral macrostate, keyed
        {species}_{A,B}.

        This is the physical-interpretation check: a partition worth calling
        two-state has to separate distinct surface regimes, not merely split the
        state space algebraically into two sets with the same observables."""
        s = self._state(x)
        builder, theta = s["builder"], s["theta"]
        out = {}
        for label, mask in (("A", s["in_A"]), ("B", s["in_B"])):
            w = theta[mask].sum()
            for name, code in _COVERAGE_NAMES:
                cov = self._species_coverage(builder, builder.species_names[code])
                out[f"{name}_{label}"] = float(theta[mask] @ cov[mask] / w)
        return out

    def coverage_marginal(self, x, species):
        """P(N_species): the stationary distribution marginalized onto one
        species' site count."""
        s = self._state(x)
        builder, theta = s["builder"], s["theta"]
        code = builder.species_names.index(species)
        P = np.zeros(builder.l + 1)
        for counts, idxs in coverage_classes(builder):
            P[counts[code - 1]] += theta[idxs].sum()
        return P

    # --- coexistence search ---------------------------------------------------
    def find_coexistence(self, x_values, log_ratios, xtol=1e-5):
        """Every x* where ln pi(A)/pi(B) changes sign across the sweep, each
        Brent-refined. Returns a sorted list of x* (possibly several, but only one
        should appear)."""
        b = np.asarray(x_values, dtype=float)
        r = np.asarray(log_ratios, dtype=float)
        good = np.isfinite(r)
        b, r = b[good], r[good]
        order = np.argsort(b)
        b, r = b[order], r[order]

        stars = []
        crossings = np.nonzero(np.diff(np.sign(r)))[0]
        for c in crossings:
            lo, hi = b[c], b[c + 1]
            if lo > 0.0 and hi > 0.0:   # refine in log-x like the reference
                log_star = brentq(lambda lb: self.basin_log_ratio(np.exp(lb)),
                                  np.log(lo), np.log(hi), xtol=xtol)
                stars.append(np.exp(log_star))
            else:
                stars.append(brentq(self.basin_log_ratio, lo, hi, xtol=xtol))
        return sorted(stars)

    def report(self, x_star, n_eigs=20):
        """Full spectral analysis at one x*.

        This is where the slow *left* eigenvector psi_L_2 enters: it is the
        natural reaction coordinate, while psi_R_2 is the density mode that
        defines the partition. For a reversible generator psi_R_2 = pi * psi_L_2
        pointwise, so the two agree in sign; the ME-MKM generator is driven,
        so `sign_agreement` (the stationary weight on which they do agree)
        measures how far that identity is from holding.

        Returns (row, arrays): `row` is a flat dict (one Coexistence-sheet row,
        including `axis_field` and `x_star` so it's self-describing regardless of
        which physics parameter was bisected); `arrays` holds the gathered
        eigenvectors for plotting. `vecs_R` (right eigenvectors, columns from the
        `n_eigs_scan`-sized right solve) and `phi_slow` (left eigenvectors, columns
        from this `n_eigs`-sized left solve) are independent solves of the same
        spectrum; their columns line up mode-for-mode only as long as
        `n_eigs_scan >= n_eigs` and no near-degenerate crossing reorders one solve
        relative to the other.
        """
        s = self._state(x_star)
        builder, theta, psi_R_2 = s["builder"], s["theta"], s["psi_R_2"]
        in_A, in_B, vecs_R = s["in_A"], s["in_B"], s["vecs_R"]

        m = self.left_modes(x_star, n_eigs)
        eigvals, phi_slow, psi_L_2 = m["eigvals"], m["phi_slow"], m["psi_L_2"]
        # W and W^T share a spectrum; the row quotes the scan's eigenvalues so
        # every number in it comes from one solve, and `eigvals` above is kept
        # only to label the left eigenvectors it was solved with.
        lam2 = s["eigvals"][1]
        sign_agreement = float(theta[(psi_L_2 >= 0.0) == in_B].sum())

        # psi_L_2, standardized, purely for the slow-coordinate density plot.
        phi_mean = theta @ psi_L_2
        phi_std = np.sqrt(theta @ (psi_L_2 - phi_mean) ** 2)
        phi_coord = (psi_L_2 - phi_mean) / phi_std

        P_A, P_B = self.basin_weights(x_star)
        row = dict(
            axis_field=self.axis_field, x_star=float(x_star),
            P_A=P_A, P_B=P_B,
            log_ratio=self.basin_log_ratio(x_star),
            **self.diagnostics(x_star),
            complex_slow_mode=bool(
                abs(lam2.imag) > IM_RE_TOL * abs(lam2.real)),
            sign_agreement=sign_agreement,
            **self.basin_coverages(x_star),
        )
        cov_pop, cov_phi, cov_r2, cov_deg = coverage_grids(
            builder, theta, psi_L_2, psi_R_2)
        arrays = {
            "axis_field": self.axis_field, "x_star": float(x_star),
            "species_A": self.order_species,            # A is rich in this
            "species_B": self._other_species(builder),  # B is rich in this
            "order_species": self.order_species,
            "eigvals": eigvals, "phi_slow": phi_slow, "vecs_R": vecs_R,
            "psi_L_2": psi_L_2, "psi_R_2": psi_R_2,
            "theta": theta,
            "in_A": in_A, "in_B": in_B,
            "phi_coord": phi_coord, "n_sites": builder.l,
            "cov_pop": cov_pop, "cov_phi": cov_phi,
            "cov_r2": cov_r2, "cov_deg": cov_deg,
            "marginal": self.coverage_marginal(x_star, self.order_species),
        }
        return row, arrays
