"""Locate the coexistence point(s) beta* where the two metastable basins carry
equal stationary weight, pi(A) = pi(B), then compute the basin-to-basin
transition rates k(A->B), k(B->A) from committors.

The committor's Dirichlet sets (the *cores*) are fixed by coverage core A is the
states nearly saturated in ORDER_SPECIES, core B the states nearly saturated in 
the other adsorbate. They are beta-independent, so they are built once per tile.
`core_frac` is therefore a coverage tolerance (1 - core_frac of a full 
monolayer), not a threshold in eigenvector units.
"""

import numpy as np
from scipy.optimize import brentq

from me_mkm.microstates import coverage_classes, microstate_as_coverage

from ..common import CO, EMPTY, O
from . import backend
from .model import generate_model

_COVERAGE_NAMES = (("empty", EMPTY), ("co", CO), ("o", O))


class CoexistencePipeline:
    """Stateful driver over one tile: caches per-beta results and the
    (beta-independent) coverage arrays and basin cores so a sweep + Brent search
    reuses work. Correct on a size-1 communicator (serial) and across ranks."""

    def __init__(self, tile, comm=None, order_species="CO", core_frac=0.1,
                 sigma_scale=1e-8, factor=None, basin_species=None,
                 delta_scale=1e-4, alpha=1.6, gamma=1e-3, kr=1.0,
                 khop_scale=1000.0, eps=8368.0, temperature=500.0):
        if not 0.0 < core_frac < 0.5:
            raise ValueError("core_frac must be in (0, 0.5) so the two cores "
                             f"stay disjoint; got {core_frac}")
        self.tile = tile
        self.comm = comm
        self.order_species = order_species
        self.basin_species = basin_species   # None -> auto-detect the other adsorbate
        self.core_frac = core_frac
        self.sigma_scale = sigma_scale
        self.delta_scale = delta_scale   # delta = delta_scale * beta
        # Shared physics, forwarded to generate_model so the ME-MKM model uses
        # the same chemistry as the kMC sweep (see co_oxidation.memkm.model).
        self.physics = dict(alpha=alpha, gamma=gamma, kr=kr,
                            khop_scale=khop_scale, eps=eps,
                            temperature=temperature)
        self.factor = factor
        self._cache = {}       # beta -> theta + committor (no PETSc objects)
        self._spectral = {}    # (beta, k) -> eigenpairs
        self._cov_cache = {}   # species name -> per-microstate coverage array
        self._cores = None     # (core_A, core_B), beta-independent

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
        """The adsorbate defining core B: the explicit `basin_species` if given,
        else the single non-vacancy species that is not ORDER_SPECIES."""
        if self.basin_species is not None:
            return self.basin_species
        others = [s for s in builder.species_names[1:] if s != self.order_species]
        if len(others) != 1:
            raise ValueError(
                "cannot auto-pick the basin-B species from "
                f"{builder.species_names}; pass basin_species explicitly")
        return others[0]

    def basin_cores(self, builder):
        """Coverage-defined Dirichlet sets for the committor, cached.

        core_A: nearly saturated in ORDER_SPECIES; core_B: nearly saturated in
        the other adsorbate. Disjoint because the two coverages sum to <= 1 and
        core_frac < 0.5; non-empty because the fully covered states exist."""
        if self._cores is None:
            hi = 1.0 - self.core_frac
            cov_a = self._species_coverage(builder, self.order_species)
            cov_b = self._species_coverage(builder, self._other_species(builder))
            core_A, core_B = cov_a >= hi, cov_b >= hi
            if not core_A.any() or not core_B.any():
                raise ValueError(
                    f"empty basin core at core_frac={self.core_frac} "
                    f"(|A|={int(core_A.sum())}, |B|={int(core_B.sum())}); "
                    "widen core_frac")
            self._cores = (core_A, core_B)
        return self._cores

    # --- inner loop: stationary state + forward committor ---------------------
    def _state(self, beta):
        """Theta and the forward committor q+ at beta, cached.

        No eigensolve: q+ comes from a Dirichlet solve on the coverage cores.
        W is destroyed before returning so a long Brent search never
        accumulates distributed matrices."""
        if beta in self._cache:
            return self._cache[beta]
        builder = generate_model(beta=beta, tile=self.tile,
                                 delta_scale=self.delta_scale, **self.physics)
        core_A, core_B = self.basin_cores(builder)
        W = backend.build_petsc_W(builder, self.comm)
        sigma = self.sigma_scale * backend.rate_scale(W)
        theta = backend.stationary(W, sigma, self.factor)
        q_plus = backend.committor(W, core_A, core_B, self.factor)
        W.destroy()
        res = dict(builder=builder, theta=theta, q_plus=q_plus, sigma=sigma,
                   core_A=core_A, core_B=core_B)
        self._cache[beta] = res
        return res

    def slow_modes(self, beta, k):
        """The k slowest left eigenpairs at beta plus the oriented phi_2^L.

        Only needed for the diagnostics at beta*, never in the inner loop. The
        eigenvector's sign is arbitrary, so it is oriented to correlate
        positively with ORDER_SPECIES coverage under Theta."""
        key = (beta, k)
        if key in self._spectral:
            return self._spectral[key]
        s = self._state(beta)
        builder, theta = s["builder"], s["theta"]
        W = backend.build_petsc_W(builder, self.comm)
        eigvals, phi = backend.left_eigenpairs(W, k, s["sigma"], self.factor)
        W.destroy()
        phi2 = phi[:, 1].real
        cov = self._species_coverage(builder, self.order_species)
        covariance = theta @ (phi2 * cov) - (theta @ phi2) * (theta @ cov)
        if covariance < 0:
            phi2 = -phi2
        res = dict(eigvals=eigvals, phi_slow=phi, phi2=phi2, lam2=eigvals[1])
        self._spectral[key] = res
        return res

    # --- basins: the q+ = 1/2 isocommittor surface -----------------------------
    @staticmethod
    def _basin_split(q_plus):
        """Boolean masks (in_A, in_B) splitting ALL states at q+ = 1/2, the
        transition-path-theory dividing surface."""
        in_B = np.asarray(q_plus) > 0.5
        return ~in_B, in_B

    # --- observables ----------------------------------------------------------
    def basin_log_ratio(self, beta):
        s = self._state(beta)
        in_A, in_B = self._basin_split(s["q_plus"])
        pi_A, pi_B = s["theta"][in_A].sum(), s["theta"][in_B].sum()
        return float(np.log10(pi_A / pi_B))

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

    def tpt_rates(self, beta):
        """(k_AB, k_BA, F, q_plus, q_minus) between the basins at one beta.

            F    = sum_{i in A} pi_i (L q+)_i,   L = W^T (row convention),
            k_AB = F / <pi, q->,   k_BA = F / <pi, 1 - q->.

        q+ is reused from the cached inner-loop solve; only the backward
        committor and the reactive flux are computed here."""
        s = self._state(beta)
        builder, theta, q_plus = s["builder"], s["theta"], s["q_plus"]
        core_A, core_B = s["core_A"], s["core_B"]
        W = backend.build_petsc_W(builder, self.comm)
        q_minus = backend.committor_backward(W, core_A, core_B, theta,
                                             self.factor)
        F = backend.reactive_flux(W, theta, core_A, q_plus)
        W.destroy()
        m_A = float(theta @ q_minus)          # P(currently "coming from A")
        m_B = float(theta @ (1.0 - q_minus))  # P(currently "coming from B")
        return F / m_A, F / m_B, F, q_plus, q_minus

    # --- coexistence search ---------------------------------------------------
    def find_coexistence(self, betas, log_ratios, xtol=1e-5):
        """Every beta* where log10 pi(A)/pi(B) changes sign across the sweep,
        each Brent-refined. Returns a sorted list of beta* (possibly several)."""
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
                log_star = brentq(lambda lb: self.basin_log_ratio(10.0 ** lb),
                                  np.log10(lo), np.log10(hi), xtol=xtol)
                stars.append(10.0 ** log_star)
            else:
                stars.append(brentq(self.basin_log_ratio, lo, hi, xtol=xtol))
        return sorted(stars)

    def report(self, beta_star, n_eigs=20):
        """Full spectral + committor + TPT analysis at one beta*.

        This is the only place the eigensolve runs, so lambda_2, the spectral
        gap and the phi_2^L vs q+ collapse are available exactly as before.

        Returns (row, arrays): `row` is a flat dict (one {out}_coexistence.csv
        line); `arrays` holds the gathered eigenvectors/committor for plotting."""
        s = self._state(beta_star)
        builder, theta = s["builder"], s["theta"]
        m = self.slow_modes(beta_star, n_eigs)
        eigvals, phi_slow, phi2, lam2 = (m["eigvals"], m["phi_slow"],
                                         m["phi2"], m["lam2"])
        lam3 = eigvals[2]

        k_AB, k_BA, F, q_plus, q_minus = self.tpt_rates(beta_star)

        # pi-weighted affine fit phi_2^L ~ a + b q+ : R^2 -> 1 iff the slow mode
        # and the committor agree on the reaction coordinate (two-state).
        phi_mean = theta @ phi2
        phi_std = np.sqrt(theta @ (phi2 - phi_mean) ** 2)
        phi_coord = (phi2 - phi_mean) / phi_std
        X = np.column_stack([np.ones_like(q_plus), q_plus])
        wts = np.sqrt(theta)
        coef, *_ = np.linalg.lstsq(X * wts[:, None], phi_coord * wts, rcond=None)
        residual = phi_coord - X @ coef
        ss_res = theta @ residual ** 2
        ss_tot = theta @ (phi_coord - theta @ phi_coord) ** 2
        r_squared = float(1.0 - ss_res / ss_tot)

        row = dict(
            beta_star=float(beta_star),
            lambda2_re=float(lam2.real), lambda2_im=float(lam2.imag),
            lambda3_re=float(lam3.real),
            spectral_gap=float(lam3.real / lam2.real),
            im_re_ratio=float(abs(lam2.imag) / abs(lam2.real)),
            k_AB=float(k_AB), k_BA=float(k_BA), flux_F=float(F),
            residence_A=float(1.0 / k_AB), residence_B=float(1.0 / k_BA),
            rate_sum_ratio=float((k_AB + k_BA) / abs(lam2.real)),
            r_squared=r_squared,
        )
        in_A, in_B = self._basin_split(q_plus)
        cov_pop, cov_phi, cov_q, cov_deg = self._coverage_grids(
            builder, theta, phi2, q_plus)
        arrays = dict(beta_star=float(beta_star),
                      species_A=self.order_species,          # core A saturates this
                      species_B=self._other_species(builder),  # core B saturates this
                      order_species=self.order_species,
                      eigvals=eigvals, phi_slow=phi_slow, phi2=phi2,
                      theta=theta, q_plus=q_plus, in_A=in_A, in_B=in_B,
                      phi_coord=phi_coord, n_sites=builder.l,
                      cov_pop=cov_pop, cov_phi=cov_phi, cov_q=cov_q,
                      cov_deg=cov_deg,
                      marginal=self.coverage_marginal(beta_star, self.order_species))
        return row, arrays

    @staticmethod
    def _coverage_grids(builder, theta, phi2, q_plus):
        """Bin the stationary-weighted quantities into the coverage plane.
        Returns four (l+1, l+1) arrays indexed [N_species1, N_species2] (for CO
        oxidation: [N_CO, N_O]):

          cov_pop[a, b] = sum_{i in class} pi_i               (population)
          cov_phi[a, b] = <phi_2^L>_pi over the class         (slow mode)
          cov_q[a, b]   = <q+>_pi over the class              (committor)
          cov_deg[a, b] = number of microstates in the class  (degeneracy)

        q+ is the standard TPT forward committor throughout: q+ = 0 on core A
        (saturated in ORDER_SPECIES) and q+ = 1 on core B (saturated in the
        other adsorbate), i.e. the probability of reaching B before A.

        phi/q are NaN where the class carries no stationary weight; degeneracy
        lets a caller form the per-microstate mean weight cov_pop / cov_deg,
        which strips the combinatorial class-size factor from the population."""
        l = builder.l
        cov_pop = np.zeros((l + 1, l + 1))
        cov_phi = np.full((l + 1, l + 1), np.nan)
        cov_q = np.full((l + 1, l + 1), np.nan)
        cov_deg = np.zeros((l + 1, l + 1))
        q_plus = np.asarray(q_plus)
        for counts, idxs in coverage_classes(builder):
            a, b = int(counts[0]), int(counts[1])
            w = theta[idxs].sum()
            cov_pop[a, b] = w
            cov_deg[a, b] = len(idxs)
            if w > 0.0:
                cov_phi[a, b] = (theta[idxs] * phi2[idxs]).sum() / w
                cov_q[a, b] = (theta[idxs] * q_plus[idxs]).sum() / w
        return cov_pop, cov_phi, cov_q, cov_deg
