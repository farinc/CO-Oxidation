"""What an observable is handed: one context object per family, carrying everything
that parameter point already knows plus the means to ask the simulation for more.

The split follows the two simulations. A kMC observable gets trajectories and a runner;
an ME-MKM observable gets a solved spectrum and the generator. Both carry `options`, so
an observable can be tuned from the command line without a new CLI flag per knob.

Neither class imports PETSc or a sweep at module level: MemkmContext builds W only when
an observable actually asks for it, and the sweep-side glue (sweeps/observables.py) is
what constructs these from a RunConfig.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from ..kmc import KMCParams, run_first_passage, run_kmc


def _coerce(value, default):
    """Cast a command-line option string to the type of the caller's default, so
    ctx.option("restart_spread", "n_restarts", 4) hands back an int, not "4"."""
    if not isinstance(value, str) or default is None:
        return value
    if isinstance(default, bool):
        return value.strip().lower() in ("1", "true", "yes", "on")
    if isinstance(default, int):
        return int(value)
    if isinstance(default, float):
        return float(value)
    return value


@dataclass
class _BaseContext:
    options: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    label: str = ""          # e.g. "step 3", used only in messages

    def option(self, observable_name: str, key: str, default=None):
        """One tuning knob for `observable_name`, from --observable-option
        NAME.KEY=VALUE, cast to the type of `default`."""
        raw = self.options.get(observable_name, {}).get(key, default)
        return _coerce(raw, default)


@dataclass
class KMCContext(_BaseContext):
    """Context for family="kmc" observables.

    `results` holds the trajectories the sweep already ran at this parameter point
    (co_oxidation.kmc.KMCResult, or the sweep's KMCTrajectoryResult -- both expose
    times/cov_*/steady_*/t_final/steps/stuck), so a post-hoc observable never re-runs
    anything. `run` and `run_first_passage` are there for observables that genuinely
    need their own trajectories -- more samples than the sweep ran, a different initial
    condition, or a run stopped on a condition rather than at a fixed time.

    Seeds for those extra runs come from `seed(k)`, offset from the sweep's own seed
    stream by SEED_STRIDE so an observable's trajectories are reproducible without ever
    reusing a seed the sweep already spent on its own.
    """

    k_o_ads: float = 1.0
    params: KMCParams = field(default_factory=KMCParams)
    results: Sequence[Any] = ()
    seed_base: int = 0

    SEED_STRIDE = 1_000_003        # prime, so per-step offsets never alias

    def seed(self, k: int = 0) -> int:
        """The k-th deterministic seed reserved for observables at this point."""
        if self.params.seed < 0:
            return -1              # the run asked for an unseeded RNG; respect it
        return self.seed_base + self.SEED_STRIDE + k

    def run(self, init="empty", **overrides):
        """One extra kMC trajectory at this point's parameters. `overrides` are
        KMCParams fields (t_max, L, seed, ...)."""
        return run_kmc(self.k_o_ads, init=init, params=self.params, **overrides)

    def run_first_passage(self, predicate, init="empty", n_trajectories=1,
                          seed_offset=0, **overrides):
        """`n_trajectories` independent runs stopped the first time `predicate`
        (an njit (n_empty, n_co, n_o, N) -> bool, see kmc.coverage_predicate) holds.

        Returns a list of FirstPassageResult. Runs that hit params.t_max first come
        back with hit=False: right-censored samples, which the caller must handle
        rather than average in as if they were passage times."""
        out = []
        for i in range(n_trajectories):
            kw = dict(overrides)
            kw.setdefault("seed", self.seed(seed_offset + i))
            out.append(run_first_passage(self.k_o_ads, predicate, init=init,
                                         params=self.params, **kw))
        return out


@dataclass
class MemkmContext(_BaseContext):
    """Context for family="memkm" observables: one solved ME-MKM point.

    `state` is exactly what coexistence.solve_memkm_state returned -- builder, the
    stationary distribution theta, the slow eigenvalues/eigenvectors, the oriented
    psi_R_2 and the in_A/in_B spectral basin masks -- all already gathered to every
    rank. Anything that only reweights those (moments, marginals, basin conditionals)
    is pure numpy and costs nothing.

    `W` is the generator itself, rebuilt on demand as a distributed PETSc matrix for
    observables that need a linear solve or a matrix norm rather than a reweighting of
    theta. It is built at most once per context and released by `close()`, so a long
    sweep never accumulates distributed matrices. Use the context manager form
    when calling this yourself.
    """

    state: Mapping[str, Any] = field(default_factory=dict)
    comm: Any = None
    factor: str | None = None
    _W: Any = field(default=None, repr=False)
    _coverage_cache: dict = field(default_factory=dict, repr=False)

    # --- the cheap, already-computed pieces ---------------------------------
    @property
    def builder(self):
        return self.state["builder"]

    @property
    def theta(self):
        """Stationary distribution over microstates, normalized to sum 1."""
        return self.state["theta"]

    @property
    def eigvals(self):
        """The slowest eigenvalues of W, descending in real part (index 0 is ~0)."""
        return self.state["eigvals"]

    @property
    def psi_R_2(self):
        """The oriented slow right eigenvector whose sign defines the basins."""
        return self.state["psi_R_2"]

    @property
    def in_A(self):
        """Boolean mask of basin A (the ORDER_SPECIES-rich side)."""
        return self.state["in_A"]

    @property
    def in_B(self):
        return self.state["in_B"]

    def coverage(self, species: str):
        """Per-microstate fractional coverage n_species / l, cached per species."""
        if species not in self._coverage_cache:
            from ..memkm.coexistence import species_coverage_array
            self._coverage_cache[species] = species_coverage_array(self.builder, species)
        return self._coverage_cache[species]

    # --- the generator, on demand -------------------------------------------
    @property
    def W(self):
        """The generator as a distributed PETSc Mat, built once per context.

        Column convention: dp/dt = W p, so W[i, j] is the rate j -> i and every column
        sums to zero. The row generator of the textbook Markov-chain formulas is W^T."""
        if self._W is None:
            from ..memkm import backend
            self._W = backend.build_petsc_W(self.builder, self.comm)
        return self._W

    def close(self):
        """Release the generator if one was built. Idempotent."""
        if self._W is not None:
            self._W.destroy()
            self._W = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False
