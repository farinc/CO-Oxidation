"""Locate the coexistence point(s) beta* where the two metastable basins carry
equal stationary weight, pi(A) = pi(B), then compute the basin-to-basin
transition rates k(A->B), k(B->A) from committors.

Basins are not hand-drawn coverage windows. At each beta the slow left
eigenvector phi_2^L of W is computed (through SLEPc); it is near-constant on
each metastable set and steps through the transition region, so splitting the
states at the midpoint between its two plateau values recovers the dynamical
basin membership directly (a spectral / PCCA-style split). Nothing about the
split knows this is CO oxidation, which keeps the pipeline portable.

The eigenvector's sign is arbitrary, so it is oriented to correlate positively
with ORDER_SPECIES coverage under Theta -- this pins "basin A" to the same
physical branch at every beta, making log10 pi(A)/pi(B) monotone in beta so a
sign change brackets beta*, refined by Brent's method.

This is the SLEPc/PETSc port of the former ``meta_stable.py``: every steady
state, eigenvector and committor solve goes through :mod:`.backend`, so the
whole pipeline runs distributed across MPI ranks and scales past what a serial
scipy solve reaches.
"""

import numpy as np
from scipy.optimize import brentq

from me_mkm.microstates import coverage_classes, microstate_as_coverage

from ..common import CO, EMPTY, O
from . import backend
from .model import generate_model

_COVERAGE_NAMES = (("empty", EMPTY), ("co", CO), ("o", O))


class CoexistencePipeline:
    """Stateful driver over one tile: caches per-beta spectral results and the
    (beta-independent) per-microstate coverage arrays so a sweep + Brent search
    reuses work. Correct on a size-1 communicator (serial) and across ranks."""

    def __init__(self, tile, comm=None, order_species="CO", core_frac=0.1,
                 sigma_scale=1e-8, n_eigs_scan=6, factor=None):
        self.tile = tile
        self.comm = comm
        self.order_species = order_species
        self.core_frac = core_frac
        self.sigma_scale = sigma_scale
        self.n_eigs_scan = n_eigs_scan
        self.factor = factor
        self._cache = {}       # beta -> lightweight numpy results (no PETSc objs)
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

    # --- lightweight per-beta solve -------------------------------------------
    def _slow(self, beta):
        """Steady state + oriented slow left eigenvector at beta, cached.

        Builds W, solves, then *destroys* the PETSc matrix and keeps only numpy
        arrays, so a long Brent search never accumulates distributed matrices."""
        if beta in self._cache:
            return self._cache[beta]
        builder = generate_model(beta=beta, tile=self.tile)
        W = backend.build_petsc_W(builder, self.comm)
        sigma = self.sigma_scale * backend.rate_scale(W)
        theta = backend.stationary(W, sigma, self.factor)
        eigvals, phi = backend.left_eigenpairs(W, self.n_eigs_scan, sigma,
                                               self.factor)
        W.destroy()

        lam2 = eigvals[1]
        phi2 = phi[:, 1].real
        cov = self._species_coverage(builder, self.order_species)
        covariance = theta @ (phi2 * cov) - (theta @ phi2) * (theta @ cov)
        if covariance < 0:
            phi2 = -phi2

        res = dict(builder=builder, theta=theta, eigvals=eigvals,
                   lam2=lam2, phi2=phi2, sigma=sigma)
        self._cache[beta] = res
        return res

    # --- spectral basin split (static, on a phi_2^L array) --------------------
    @staticmethod
    def _basin_split(phi2):
        """Boolean masks (in_A, in_B) splitting ALL states at the midpoint
        between the two plateau extremes of phi_2^L."""
        threshold = 0.5 * (phi2.min() + phi2.max())
        in_A = phi2 > threshold
        return in_A, ~in_A

    def _basin_cores(self, phi2):
        """Basin *cores*: only states on the flat plateaus of phi_2^L, leaving
        the transition region as committor interior."""
        span = phi2.max() - phi2.min()
        core_A = phi2 >= phi2.max() - self.core_frac * span
        core_B = phi2 <= phi2.min() + self.core_frac * span
        return core_A, core_B

    # --- observables ----------------------------------------------------------
    def basin_log_ratio(self, beta):
        s = self._slow(beta)
        in_A, in_B = self._basin_split(s["phi2"])
        pi_A, pi_B = s["theta"][in_A].sum(), s["theta"][in_B].sum()
        return float(np.log10(pi_A / pi_B))

    def coverages(self, beta):
        """Mean fractional ME-MKM coverages (empty, CO, O) under Theta,
        directly comparable to the kMC steady coverages."""
        s = self._slow(beta)
        builder, theta = s["builder"], s["theta"]
        out = {}
        for name, code in _COVERAGE_NAMES:
            cov = self._species_coverage(builder, builder.species_names[code])
            out[name] = float(theta @ cov)   # cov is already a fraction n_s/l
        return out

    def coverage_marginal(self, beta, species):
        """P(N_species): the stationary distribution marginalized onto one
        species' site count."""
        s = self._slow(beta)
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
        """
        s = self._slow(beta)
        builder, theta = s["builder"], s["theta"]
        core_A, core_B = self._basin_cores(s["phi2"])
        W = backend.build_petsc_W(builder, self.comm)
        q_plus = backend.committor(W, core_A, core_B, self.factor)
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

        Returns (row, arrays): `row` is a flat dict (one {out}_coexistence.csv
        line); `arrays` holds the gathered eigenvectors/committor for plotting."""
        s = self._slow(beta_star)
        builder, theta, phi2 = s["builder"], s["theta"], s["phi2"]
        lam2, sigma = s["lam2"], s["sigma"]

        W = backend.build_petsc_W(builder, self.comm)
        eigvals, phi_slow = backend.left_eigenpairs(W, n_eigs, sigma,
                                                    self.factor)
        W.destroy()
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
        in_A, in_B = self._basin_split(phi2)
        cov_pop, cov_phi, cov_q_A, cov_deg = self._coverage_grids(
            builder, theta, phi2, q_plus)
        arrays = dict(beta_star=float(beta_star), order_species=self.order_species,
                      eigvals=eigvals, phi_slow=phi_slow, phi2=phi2,
                      theta=theta, q_plus=q_plus, in_A=in_A, in_B=in_B,
                      phi_coord=phi_coord, n_sites=builder.l,
                      cov_pop=cov_pop, cov_phi=cov_phi, cov_q_A=cov_q_A,
                      cov_deg=cov_deg,
                      marginal=self.coverage_marginal(beta_star, self.order_species))
        return row, arrays

    @staticmethod
    def _coverage_grids(builder, theta, phi2, q_plus):
        """Bin the stationary-weighted spectral quantities into the coverage
        plane. Returns three (l+1, l+1) arrays indexed [N_species1, N_species2]
        (for CO oxidation: [N_CO, N_O]):

          cov_pop[a, b] = sum_{i in class} pi_i               (population)
          cov_phi[a, b] = <phi_2^L>_pi over the class         (slow mode)
          cov_q_A[a,b]  = <q_A>_pi over the class             (committor to A)
          cov_deg[a, b] = number of microstates in the class  (degeneracy)

        q_A = 1 - q+ is the probability of reaching basin A (the CO-rich core)
        *before* basin B (the O-rich core) -- the complement of the TPT-
        convention forward committor q+, which targets B. The TPT rate math
        below still uses q+ itself; only the reported/plotted map is oriented
        toward A, so "committor = 1" reads as "commits to the CO-rich side".

        phi/q are NaN where the class carries no stationary weight; degeneracy
        lets a caller form the per-microstate mean weight cov_pop / cov_deg,
        which strips the combinatorial class-size factor from the population."""
        l = builder.l
        cov_pop = np.zeros((l + 1, l + 1))
        cov_phi = np.full((l + 1, l + 1), np.nan)
        cov_q_A = np.full((l + 1, l + 1), np.nan)
        cov_deg = np.zeros((l + 1, l + 1))
        q_A = 1.0 - np.asarray(q_plus)          # orient toward the CO-rich core
        for counts, idxs in coverage_classes(builder):
            a, b = int(counts[0]), int(counts[1])
            w = theta[idxs].sum()
            cov_pop[a, b] = w
            cov_deg[a, b] = len(idxs)
            if w > 0.0:
                cov_phi[a, b] = (theta[idxs] * phi2[idxs]).sum() / w
                cov_q_A[a, b] = (theta[idxs] * q_A[idxs]).sum() / w
        return cov_pop, cov_phi, cov_q_A, cov_deg
