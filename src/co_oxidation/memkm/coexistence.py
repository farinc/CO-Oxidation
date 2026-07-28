"""Locate the Brennette acceptance ratio beta* where the two metastable macrostates
carry equal stationary weight, pi(A) = pi(B).

Basins are not hand-drawn coverage windows chosen up front. At each beta the
slowest nonstationary *right* eigenvector r_2 of the column generator W is
computed (through SLEPc) and every microstate is assigned by its sign,

    A(beta) = {i : r_2(i) <  0},
    B(beta) = {i : r_2(i) >= 0},

which is the approximate variational basins for a two-basin system

The finite-tile coexistence point is then defined operationally as the beta at
which the two spectral macrostates carry equal stationary weight, P_A = P_B = 1/2.

The overall sign of r_2 is arbitrary and only exchanges the labels A and B. That keeps
"basin A" on the same physical branch at every beta, which is what makes
log pi(A)/pi(B) monotone in beta so a sign change brackets beta*, refined by
Brent's method.
"""

import numpy as np
from me_mkm.microstates import coverage_classes, microstate_as_coverage
from scipy.optimize import brentq

from ..common import CO, EMPTY, O
from . import backend
from .model import generate_model

_COVERAGE_NAMES = (("empty", EMPTY), ("co", CO), ("o", O))
IM_RE_TOL = 1e-6


class CoexistencePipeline:
    """Stateful driver over one tile: caches per-beta results and the
    (beta-independent) coverage arrays so a sweep + Brent search reuses work.
    Correct on a size-1 communicator (serial) and across ranks."""

    def __init__(self, tile, comm=None, order_species="CO", n_eigs_scan=4,
                 boundary_eps=0.05, sigma_scale=1e-8,
                 factor=None, basin_species=None, delta_scale=1e-4, alpha=1.6,
                 gamma=1e-3, kr=1.0, khop_scale=1000.0, eps=8368.0,
                 temperature=500.0):
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
        self.comm = comm
        self.order_species = order_species
        self.basin_species = basin_species   # None -> auto-detect the other adsorbate
        self.n_eigs_scan = n_eigs_scan
        self.boundary_eps = boundary_eps
        self.sigma_scale = sigma_scale
        self.delta_scale = delta_scale   # delta = delta_scale * beta
        # Shared physics, forwarded to generate_model so the ME-MKM model uses
        # the same chemistry as the kMC sweep (see co_oxidation.memkm.model).
        self.physics = {"alpha": alpha, "gamma": gamma, "kr": kr,
                        "khop_scale": khop_scale, "eps": eps,
                        "temperature": temperature}
        self.factor = factor
        self._cache = {}       # beta -> theta + slow mode (no PETSc objects)
        self._left = {}        # beta -> left eigenpairs (only solved at beta*)
        self._cov_cache = {}   # species name -> per-microstate coverage array

    # --- per-microstate coverage (depends only on the tile, cached once) ------
    def _species_coverage(self, builder, name):
        if name not in self._cov_cache:
            code = builder.species_names.index(name)
            self._cov_cache[name] = np.array(
                [microstate_as_coverage(builder, i)[code]
                 for i in range(builder.n_states)]
            )
        return self._cov_cache[name]

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

    # --- inner loop: stationary distribution + slow right mode ----------------
    def _state(self, beta):
        """Theta, the slow eigenvalues and the oriented r_2 at beta, cached.

        One shift-invert solve delivers both Theta (the lambda ~ 0 right
        eigenvector) and the nonstationary modes, so the scan costs a single
        factorization per beta. W is destroyed before returning so a long Brent
        search never accumulates distributed matrices."""
        if beta in self._cache:
            return self._cache[beta]
        builder = generate_model(beta=beta, tile=self.tile,
                                 delta_scale=self.delta_scale, **self.physics)
        W = backend.build_petsc_W(builder, self.comm)
        try:
            sigma = self.sigma_scale * backend.rate_scale(W)
            theta, eigvals, vecs = backend.right_eigenpairs(
                W, self.n_eigs_scan, sigma, self.factor)
        finally:
            W.destroy()
        if eigvals[1].real == 0.0:
            raise ValueError(
                f"lambda_2 = 0 at beta={beta:.6g}: the generator is reducible, "
                "so there is no unique stationary distribution and no slow "
                "switching mode to partition on")
        cov = self._species_coverage(builder, self.order_species)
        r2 = self._orient(vecs[:, 1].real, theta, cov)
        in_A, in_B = self._basin_split(r2)
        res = {"builder": builder, "theta": theta, "sigma": sigma,
               "eigvals": eigvals, "r2": r2, "in_A": in_A, "in_B": in_B}
        res["coord"] = self._reduced_coordinate(r2, theta)
        self._cache[beta] = res
        return res

    @staticmethod
    def _reduced_coordinate(r2, theta):
        """r_2 / Theta: a cheap stand-in for the slow left eigenfunction, used
        to measure how firmly a microstate belongs to its macrostate."""
        return np.divide(r2, theta, out=np.zeros_like(r2), where=theta > 0.0)

    # --- the V2ST sign partition ----------------------------------------------
    @staticmethod
    def _basin_split(r2):
        """Boolean masks (in_A, in_B) assigning every microstate by the sign of
        the slow right mode: A = {r_2 < 0}, B = {r_2 >= 0}."""
        in_B = np.asarray(r2) >= 0.0
        return ~in_B, in_B

    @staticmethod
    def _orient(r2, theta, cov):
        """Sign-fix r_2 so that A = {r_2 < 0} is the ORDER_SPECIES-rich set.

        r_2 -> -r_2 leaves the equal-weight condition untouched and only swaps
        the labels, so tying the sign to a physical observable is what keeps
        "basin A" the same branch across the sweep."""
        r2 = np.asarray(r2, dtype=float)
        in_A = r2 < 0.0
        w_A, w_B = theta[in_A].sum(), theta[~in_A].sum()
        if w_A <= 0.0 or w_B <= 0.0:
            raise ValueError(
                "the sign partition of the slow mode leaves all stationary "
                "weight on one side, so there are no two macrostates to "
                "balance; lambda_2 is not a two-basin switching mode here")
        mean_A = float(theta[in_A] @ cov[in_A]) / w_A
        mean_B = float(theta[~in_A] @ cov[~in_A]) / w_B
        return -r2 if mean_A < mean_B else r2

    def slow_mode(self, beta):
        """The oriented slow right eigenvector r_2 at beta (a copy).

        Consecutive betas should overlap strongly; a collapse of
        |<r_2(b_k), r_2(b_k-1)>| flags a mode crossing, i.e. the sweep stopped
        following the same physical process (see the continuation check in
        sweeps._common.run_coexistence)."""
        return self._state(beta)["r2"].copy()

    def left_modes(self, beta, n_eigs=3):
        """The n_eigs slowest left eigenpairs at beta plus the oriented
        phi_2^L, cached; re-solves only if more modes are asked for than are
        held."""
        cached = self._left.get(beta)
        if cached is not None and len(cached["eigvals"]) >= n_eigs:
            return cached
        s = self._state(beta)
        W = backend.build_petsc_W(s["builder"], self.comm)
        try:
            eigvals, phi_slow = backend.left_eigenpairs(W, n_eigs, s["sigma"], self.factor)
        finally:
            W.destroy()
        phi2 = phi_slow[:, 1].real
        if phi2 @ s["r2"] < 0.0:
            phi2 = -phi2
        res = {"eigvals": eigvals, "phi_slow": phi_slow, "phi2": phi2}
        self._left[beta] = res
        return res

    # --- observables ----------------------------------------------------------
    def basin_weights(self, beta):
        """(P_A, P_B): the stationary weight of each spectral macrostate. They
        sum to 1 because the sign partition is exhaustive."""
        s = self._state(beta)
        theta = s["theta"]
        return float(theta[s["in_A"]].sum()), float(theta[s["in_B"]].sum())

    def basin_log_ratio(self, beta):
        """log10 pi(A)/pi(B): the coexistence objective, zero at beta*."""
        P_A, P_B = self.basin_weights(beta)
        return float(np.log(P_A / P_B))

    def diagnostics(self, beta):
        """The checks that decide whether the two-state reading holds at beta.

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
        they are at beta*, the only place the number is really being read."""
        s = self._state(beta)
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

    def coverages(self, beta):
        """Mean fractional ME-MKM coverages (empty, CO, O) under Theta,
        directly comparable to the kMC steady coverages."""
        s = self._state(beta)
        builder, theta = s["builder"], s["theta"]
        out = {}
        for name, code in _COVERAGE_NAMES:
            cov = self._species_coverage(builder, builder.species_names[code])
            out[name] = float(theta @ cov)   # cov is already a fraction n_s/l
        return out

    def basin_coverages(self, beta):
        """Coverages conditioned on each spectral macrostate, keyed
        {species}_{A,B}.

        This is the physical-interpretation check: a partition worth calling
        two-state has to separate distinct surface regimes, not merely split the
        state space algebraically into two sets with the same observables."""
        s = self._state(beta)
        builder, theta = s["builder"], s["theta"]
        out = {}
        for label, mask in (("A", s["in_A"]), ("B", s["in_B"])):
            w = theta[mask].sum()
            for name, code in _COVERAGE_NAMES:
                cov = self._species_coverage(builder, builder.species_names[code])
                out[f"{name}_{label}"] = float(theta[mask] @ cov[mask] / w)
        return out

    def coverage_marginal(self, beta, species):
        """P(N_species): the stationary distribution marginalized onto one
        species' site count."""
        s = self._state(beta)
        builder, theta = s["builder"], s["theta"]
        code = builder.species_names.index(species)
        P = np.zeros(builder.l + 1)
        for counts, idxs in coverage_classes(builder):
            P[counts[code - 1]] += theta[idxs].sum()
        return P

    # --- coexistence search ---------------------------------------------------
    def find_coexistence(self, betas, log_ratios, xtol=1e-5):
        """Every beta* where log10 pi(A)/pi(B) changes sign across the sweep,
        each Brent-refined. Returns a sorted list of beta* (possibly several,
        but only one should appear)."""
        b = np.asarray(betas, dtype=float)
        r = np.asarray(log_ratios, dtype=float)
        good = np.isfinite(r)
        b, r = b[good], r[good]
        order = np.argsort(b)
        b, r = b[order], r[order]

        stars = []
        crossings = np.nonzero(np.diff(np.sign(r)))[0]
        for c in crossings:
            lo, hi = b[c], b[c + 1]
            if lo > 0.0 and hi > 0.0:   # refine in log-beta like the reference
                log_star = brentq(lambda lb: self.basin_log_ratio(np.exp(lb)),
                                  np.log(lo), np.log(hi), xtol=xtol)
                stars.append(np.exp(log_star))
            else:
                stars.append(brentq(self.basin_log_ratio, lo, hi, xtol=xtol))
        return sorted(stars)

    def report(self, beta_star, n_eigs=20):
        """Full spectral analysis at one beta*.

        This is where the slow *left* eigenvector phi_2^L enters: it is the
        natural reaction coordinate, while r_2 is the density mode that defines
        the partition. For a reversible generator r_2 = pi * phi_2^L pointwise,
        so the two agree in sign; the ME-MKM generator is driven, so
        `sign_agreement` (the stationary weight on which they do agree) measures
        how far that identity is from holding.

        Returns (row, arrays): `row` is a flat dict (one {out}_coexistence.csv
        line); `arrays` holds the gathered eigenvectors for plotting.
        """
        s = self._state(beta_star)
        builder, theta, r2 = s["builder"], s["theta"], s["r2"]
        in_A, in_B = s["in_A"], s["in_B"]

        m = self.left_modes(beta_star, n_eigs)
        eigvals, phi_slow, phi2 = m["eigvals"], m["phi_slow"], m["phi2"]
        # W and W^T share a spectrum; the row quotes the scan's eigenvalues so
        # every number in it comes from one solve, and `eigvals` above is kept
        # only to label the left eigenvectors it was solved with.
        lam2 = s["eigvals"][1]
        sign_agreement = float(theta[(phi2 >= 0.0) == in_B].sum())

        # phi_2^L, standardized, purely for the slow-coordinate density plot
        # (plotting.py fig. 4).
        phi_mean = theta @ phi2
        phi_std = np.sqrt(theta @ (phi2 - phi_mean) ** 2)
        phi_coord = (phi2 - phi_mean) / phi_std

        P_A, P_B = self.basin_weights(beta_star)
        row = dict(
            beta_star=float(beta_star),
            P_A=P_A, P_B=P_B,
            log_ratio=self.basin_log_ratio(beta_star),
            **self.diagnostics(beta_star),
            complex_slow_mode=bool(
                abs(lam2.imag) > IM_RE_TOL * abs(lam2.real)),
            sign_agreement=sign_agreement,
            **self.basin_coverages(beta_star),
        )
        cov_pop, cov_phi, cov_r2, cov_deg = self._coverage_grids(
            builder, theta, phi2, r2)
        arrays = {
            "beta_star": float(beta_star),
            "species_A": self.order_species,            # A is rich in this
            "species_B": self._other_species(builder),  # B is rich in this
            "order_species": self.order_species,
            "eigvals": eigvals, "phi_slow": phi_slow, "phi2": phi2, "r2": r2,
            "theta": theta,
            "in_A": in_A, "in_B": in_B,
            "phi_coord": phi_coord, "n_sites": builder.l,
            "cov_pop": cov_pop, "cov_phi": cov_phi,
            "cov_r2": cov_r2, "cov_deg": cov_deg,
            "marginal": self.coverage_marginal(beta_star, self.order_species),
        }
        return row, arrays

    @staticmethod
    def _coverage_grids(builder, theta, phi2, r2):
        """Bin the stationary-weighted quantities into the coverage plane.
        Returns four (l+1, l+1) arrays indexed [N_species1, N_species2] (for CO
        oxidation: [N_CO, N_O]):

          cov_pop[a, b] = sum_{i in class} pi_i               (population)
          cov_phi[a, b] = <phi_2^L>_pi over the class         (slow coordinate)
          cov_r2[a, b]  = sum_{i in class} r_2,i / max|r_2|   (slow mode)
          cov_deg[a, b] = number of microstates in the class  (degeneracy)

        r_2 is a density, not a coordinate, so it is *summed* over each class
        rather than pi-averaged: its class sums are the contribution of that
        coverage class to each macrostate's weight, and they sum to zero over
        the whole plane. Its sign is the partition itself.

        phi is NaN where the class carries no stationary weight; degeneracy
        lets a caller form the per-microstate mean weight cov_pop / cov_deg,
        which strips the combinatorial class-size factor from the population."""
        l = builder.l
        cov_pop = np.zeros((l + 1, l + 1))
        cov_phi = np.full((l + 1, l + 1), np.nan)
        cov_r2 = np.zeros((l + 1, l + 1))
        cov_deg = np.zeros((l + 1, l + 1))
        r2_scaled = np.asarray(r2) / np.abs(r2).max()
        for counts, idxs in coverage_classes(builder):
            a, b = int(counts[0]), int(counts[1])
            w = theta[idxs].sum()
            cov_pop[a, b] = w
            cov_deg[a, b] = len(idxs)
            cov_r2[a, b] = r2_scaled[idxs].sum()
            if w > 0.0:
                cov_phi[a, b] = (theta[idxs] * phi2[idxs]).sum() / w
        return cov_pop, cov_phi, cov_r2, cov_deg
