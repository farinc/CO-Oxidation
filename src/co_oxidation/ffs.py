"""Forward flux sampling (FFS) and FFS in time (FFST) for rare transitions in the kMC.

This module is a skeleton. Settings, seeds, record types, result formatting and the
order-parameter check work, so the CLI, workbook and plots already connect to it. The
algorithm functions keep their signatures, say in their docstrings what belongs there,
and raise NotImplementedError.

The stages only call a propagator, so a toy Markov chain can stand in for the lattice
in the tests. `map_fn` maps a function over a list and keeps the order. The MPI driver
passes one that splits the list over ranks and gathers it back on every rank.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np

from . import kmc
from .common import CO, EMPTY, O
from .ffs_order import OrderParameter, co_clusters
from .ffs_order import get as get_order_parameter

SUCCESS = kmc.TRIAL_SUCCESS
FAILURE = kmc.TRIAL_FAILURE
CENSORED = kmc.TRIAL_CENSORED
STUCK = kmc.TRIAL_STUCK

MapFn = Callable[[Callable, Sequence], list]


def serial_map(fn, items):
    return [fn(item) for item in items]


class FFSError(RuntimeError):
    """The run can't produce a transition time, for example the flux run got stuck."""


# --- settings and seeds ---------------------------------------------------------------

@dataclass(frozen=True)
class FFSSettings:
    """Everything that defines one FFS run apart from the physics.

    Interfaces are in the order parameter's own units. With placement="static" give
    `lambdas` = (lambda_0, ..., lambda_n) ending at lam_B. With placement="dynamic"
    only lam_A, lam_0 and lam_B are used and the rest are placed on the fly."""

    lam_A: float
    lam_B: float
    lambdas: tuple[float, ...] | None = None
    lam_0: float | None = None
    placement: str = "static"
    n_crossings: int = 1000
    n_trials: int = 1000
    trial_t_max: float = 1e4
    flux_t_max: float = 1e7
    t_equil: float = 5.0
    max_steps: int = 2 ** 62
    dynamic_success: float = 0.1     # aim for this success fraction per interface
    dynamic_pilot_frac: float = 0.1  # pilot trials, as a fraction of n_trials
    max_interfaces: int = 200
    min_spacing: float = 0.0         # smallest gap between dynamic interfaces
    n_bootstrap: int = 200
    store_configs: bool = False

    def first_interface(self) -> float:
        if self.placement == "static":
            return float(self.lambdas[0])
        return float(self.lam_0 if self.lam_0 is not None else self.lam_A)

    def validate(self) -> None:
        if self.placement not in ("static", "dynamic"):
            raise ValueError(f"placement must be 'static' or 'dynamic', got {self.placement!r}")
        if self.placement == "static":
            if not self.lambdas or len(self.lambdas) < 2:
                raise ValueError("static placement needs at least lambda_0 and lambda_B")
            lam = np.asarray(self.lambdas, float)
            if np.any(np.diff(lam) <= 0):
                raise ValueError(f"interfaces must increase strictly, got {list(lam)}")
            if not math.isclose(lam[-1], self.lam_B):
                raise ValueError(f"the last interface {lam[-1]} must equal lam_B {self.lam_B}")
        if self.first_interface() < self.lam_A:
            raise ValueError("lambda_0 must not lie below lam_A")
        if self.lam_B <= self.first_interface():
            raise ValueError("lam_B must lie above lambda_0")
        if self.n_crossings < 1 or self.n_trials < 1:
            raise ValueError("n_crossings and n_trials must be at least 1")


class SeedStream:
    """Deterministic integer seeds for numba, keyed by where they are used.

    Every (step, stage, interface) gets its own SeedSequence child, so a rerun
    reproduces each trial exactly and ranks agree on every seed without talking."""

    FLUX, PICK, TRIAL, PILOT, BOOT = range(5)

    def __init__(self, base_seed: int, step_index: int = 0):
        self.base = int(base_seed)
        self.step = int(step_index)

    def _seq(self, stage: int, index: int) -> np.random.SeedSequence:
        return np.random.SeedSequence(entropy=self.base,
                                      spawn_key=(self.step, stage, index))

    def ints(self, stage: int, index: int, n: int) -> list[int]:
        return [int(s) for s in self._seq(stage, index).generate_state(n)]

    def rng(self, stage: int, index: int) -> np.random.Generator:
        return np.random.default_rng(self._seq(stage, index))


# --- propagator -------------------------------------------------------------------------

@dataclass
class TrialResult:
    outcome: int
    lam_end: float
    lam_max: float
    t: float
    steps: int
    config: np.ndarray


@dataclass
class FluxRun:
    """One stage-1 trajectory."""

    q: int
    t_hA: float
    t_total: float
    steps: int
    n_restarts: int
    stuck: bool
    snapshots: np.ndarray     # (q, size of a config)
    t_int: np.ndarray         # (q,), NaN where the entry time was unknown
    t_cross: np.ndarray       # (q,), h_A = 1 time at each crossing


class KMCPropagator:
    """Runs FFS trials and flux runs on the lattice kMC at one parameter point."""

    def __init__(self, k_o_ads: float, params: kmc.KMCParams,
                 order: OrderParameter | str = "theta_co"):
        self.order = get_order_parameter(order) if isinstance(order, str) else order
        self.k_o_ads = float(k_o_ads)
        self.params = params
        self.L = int(params.L)
        self.n_sites = self.L * self.L
        self.stride = self.order.resolved_stride(self.L)
        self.class_rate = kmc.class_rates_for(k_o_ads, params)

    def initial_config(self) -> np.ndarray:
        return kmc.make_lattice(self.L, "empty")

    def poisoned_config(self) -> np.ndarray:
        return kmc.make_lattice(self.L, "full")

    def lam(self, config: np.ndarray) -> float:
        return self.order.evaluate(config, self.L)

    def trial(self, config, lo, hi, t_max, max_steps, seed) -> TrialResult:
        r"""Run a single trial from a stored configuration and return where it ended up.
        """
        raise NotImplementedError("KMCPropagator.trial is part of the FFS skeleton")

    def flux(self, config, lam_A, lam_0, lam_B, n_crossings, t_equil, t_max,
             max_steps, seed) -> FluxRun:
        r"""Run one long stage-1 trajectory and collect its crossings of the first interface.

        kmc._run_ffs_flux writes into arrays you give it, so make those
        first: one lattice row per crossing, an array for the T_int samples that
        starts full of NaN, and one for the crossing times. Call the kernel on a copy
        of `config`. It tells you how many crossings q it found, so trim the arrays
        to q rows before building the FluxRun.
        """
        raise NotImplementedError("KMCPropagator.flux is part of the FFS skeleton")

    def describe(self, configs: np.ndarray) -> dict[str, np.ndarray]:
        """Per-configuration ensemble diagnostics for one interface's collection."""
        n = len(configs)
        out = {
            "theta_co": np.empty(n), "theta_o": np.empty(n), "theta_empty": np.empty(n),
            "largest_co_cluster": np.empty(n), "spans": np.empty(n, bool),
        }
        for k, lat in enumerate(configs):
            out["theta_co"][k] = np.count_nonzero(lat == CO) / self.n_sites
            out["theta_o"][k] = np.count_nonzero(lat == O) / self.n_sites
            out["theta_empty"][k] = np.count_nonzero(lat == EMPTY) / self.n_sites
            size, wraps = co_clusters(lat, self.L)
            out["largest_co_cluster"][k] = size / self.n_sites
            out["spans"][k] = bool(wraps)
        return out


# --- stage records ----------------------------------------------------------------------

@dataclass
class FluxStage:
    runs: list[FluxRun]

    @property
    def q(self) -> int:
        return sum(r.q for r in self.runs)

    @property
    def t_hA(self) -> float:
        return sum(r.t_hA for r in self.runs)

    @property
    def flux(self) -> float:
        """Allen's Phi_{A,0} / h_A = q / T_1."""
        return self.q / self.t_hA if self.t_hA > 0 else math.nan

    @property
    def snapshots(self) -> np.ndarray:
        return np.concatenate([r.snapshots for r in self.runs])

    @property
    def t_int_samples(self) -> np.ndarray:
        t = np.concatenate([r.t_int for r in self.runs])
        return t[np.isfinite(t)]

    @property
    def T_int(self) -> float:
        t = self.t_int_samples
        return float(t.mean()) if len(t) else math.nan

    def cycle_times(self) -> tuple[np.ndarray, float]:
        """Durations between crossings, plus the leftover time after the last one."""
        cycles, tail = [], 0.0
        for r in self.runs:
            if r.q:
                cycles.append(np.diff(r.t_cross, prepend=0.0))
                tail += r.t_hA - r.t_cross[-1]
            else:
                tail += r.t_hA
        return (np.concatenate(cycles) if cycles else np.zeros(0)), tail


@dataclass
class InterfaceStage:
    """Trials fired from interface lambda_i toward lambda_{i+1}."""

    index: int
    lam_lo: float
    lam_hi: float
    parent: np.ndarray        # index into the previous collection, per trial
    outcome: np.ndarray
    t: np.ndarray
    steps: np.ndarray
    lam_end: np.ndarray
    lam_max: np.ndarray
    success_configs: np.ndarray | None = None
    n_pilot: int = 0

    def count(self, outcome: int) -> int:
        return int(np.count_nonzero(self.outcome == outcome))

    @property
    def n_resolved(self) -> int:
        return int(np.count_nonzero(self.outcome != CENSORED))

    @property
    def p(self) -> float:
        """P(lambda_{i+1} | lambda_i). Stuck trials count as failures, censored are dropped."""
        m = self.n_resolved
        return self.count(SUCCESS) / m if m else math.nan

    def mean_time(self, outcome: int) -> float:
        t = self.t[self.outcome == outcome]
        return float(t.mean()) if len(t) else math.nan

    @property
    def overshoot(self) -> float:
        """Mean distance past lambda_{i+1} at which successes were noticed."""
        s = self.outcome == SUCCESS
        return float((self.lam_end[s] - self.lam_hi).mean()) if s.any() else math.nan


# --- stages -----------------------------------------------------------------------------

def run_flux_stage(prop, settings: FFSSettings, seeds: SeedStream,
                   map_fn: MapFn = serial_map, n_runs: int = 1) -> FluxStage:
    r"""Run stage 1 and pool the crossings from every flux trajectory.

    Split the requested crossings evenly over n_runs trajectories and
    give each its own seed from the FLUX stream. Run them through map_fn, so under MPI
    each rank does one trajectory. The pooled FluxStage then gives the flux through the
    first interface as

        $$\Phi = \frac{q}{T_1},$$

    where $q$ counts all crossings and $T_1$ is the total time spent with $h_A = 1$.
    A stuck run means this point has no reactive basin, so raise FFSError. Raise it
    too if nothing crossed $\lambda_0$.
    """
    raise NotImplementedError("run_flux_stage is part of the FFS skeleton")


def run_interface_stage(prop, index: int, configs: np.ndarray, lam_lo: float,
                        lam_hi: float, settings: FFSSettings, seeds: SeedStream,
                        map_fn: MapFn = serial_map, keep_configs: bool = True
                        ) -> InterfaceStage:
    r"""Fire a batch of trials from one interface and record how each one ended.

    Each trial starts from a configuration picked at random, with
    replacement, from `configs`. It runs until it reaches lam_hi or falls all the way
    back to basin A. Failure means returning to lam_A, not just dropping below lam_lo.
    Use the PICK stream for the picks and the TRIAL stream for the seeds, and run the
    trials through map_fn.

    Keep which configuration each trial started from. Those parent links are what
    most_likely_path follows later. The end lattices of the successful trials become
    the starting set for the next interface.
    """
    raise NotImplementedError("run_interface_stage is part of the FFS skeleton")


def place_next_interface(prop, index: int, configs: np.ndarray, lam_lo: float,
                         settings: FFSSettings, seeds: SeedStream,
                         map_fn: MapFn = serial_map) -> tuple[float, int]:
    r"""Choose where the next interface goes by sending out a few pilot trials.

    Fire a small batch of pilots toward $\lambda_B$, about
    dynamic_pilot_frac of n_trials and never fewer than 10, and note how far each one
    got. Call that $\lambda_{\max}$. To have a fraction $s$ of the real trials
    succeed, put the next interface at the matching quantile,

        $$\lambda_{i+1} = Q_{1-s}(\lambda_{\max}),$$

    where $s$ is dynamic_success. Keep it at least min_spacing above $\lambda_i$ and no
    higher than $\lambda_B$. Return the new value and how many pilots you used.
    """
    raise NotImplementedError("place_next_interface is part of the FFS skeleton")


# --- estimators -------------------------------------------------------------------------

def transition_times(flux: float, T_int: float, p_i: np.ndarray, t_succ: np.ndarray,
                     t_fail: np.ndarray) -> dict[str, float]:
    r"""Turn the per-interface statistics into the FFS and FFST transition times.

    For each interface $i$ you get the probability $p_i$ of reaching the
    next one, the mean time of a successful trial $\bar t_{i \to i+1}$ and the mean
    time of a failed one $\bar t_{i \to A}$. The chance of getting from $\lambda_0$ as
    far as $\lambda_i$ is $P_i = \prod_{j<i} p_j$, and the chance of going all the way
    is $p = P_n$.

    FFS only needs the flux and that total probability,

        $$k_{AB} = \Phi\, p, \qquad T_{\mathrm{FFS}} = \frac{1}{k_{AB}}.$$

    FFST adds up times instead. A successful climb takes
    $T_f = \sum_i \bar t_{i \to i+1}$. An excursion that dies between $\lambda_i$ and
    $\lambda_{i+1}$ first climbs to $\lambda_i$ and then falls back, so a failed
    excursion lasts on average

        $$T_{\mathrm{ext}} = \frac{1}{1-p} \sum_{i=0}^{n-1} P_i (1-p_i)
        \Big[\sum_{k<i} \bar t_{k \to k+1} + \bar t_{i \to A}\Big].$$

    About $1/p - 1$ attempts fail before one works, which gives

        $$T_{\mathrm{FFST}} = \Big(\frac{1}{p} - 1\Big)
        \big(T_{\mathrm{ext}} + T_{\mathrm{int}}\big) + T_f.$$

    Skip interfaces that had no failures, since their failure time is NaN. If $p = 0$
    both times are infinite, and if $p = 1$ then $T_{\mathrm{FFST}} = T_f$. Return all
    the pieces in a dict, because the workbook writes each one out.
    """
    raise NotImplementedError("transition_times is part of the FFS skeleton")


def bootstrap(flux_stage: FluxStage, stages: Sequence[InterfaceStage], n_boot: int,
              rng: np.random.Generator) -> dict[str, float]:
    r"""Estimate error bars on k_AB, T_FFS and T_FFST by bootstrapping.

    Each round, resample everything that went into the estimate and
    recompute it with transition_times. For the flux, resample the cycle times from
    FluxStage.cycle_times() but leave the tail time alone, so

        $$\Phi^* = \frac{n_c}{\sum_k c_k^* + t_{\mathrm{tail}}}.$$

    Resample the T_int samples too, and each interface's finished trials, keeping a
    trial's outcome and duration together. Report the spread as {key}_std,
    {key}_lo95 and {key}_hi95, which the summary row picks up. Resampling interfaces
    independently ignores that trials share parents. That's the usual FFS shortcut.
    """
    raise NotImplementedError("bootstrap is part of the FFS skeleton")


def binomial_rel_var(stages: Sequence[InterfaceStage]) -> float:
    r"""Give a quick error estimate for the total probability p.

    Treat each interface as an independent binomial experiment with
    $M_i$ finished trials. The relative variance of $p$ is then roughly

        $$\frac{\operatorname{Var}(p)}{p^2} \approx \sum_{i=0}^{n-1} \frac{1-p_i}{p_i M_i}.$$

    If any $p_i$ is zero the sum blows up, so return infinity.
    """
    raise NotImplementedError("binomial_rel_var is part of the FFS skeleton")


def most_likely_path(stages: Sequence[InterfaceStage], n_first: int) -> list[int] | None:
    r"""Find the single most likely route from the first interface to basin B.

    The configurations form a tree. Collection 0 is the flux snapshots,
    collection $i+1$ is the successes of interface $i$, and each success remembers
    where it came from. Give each configuration $c$ the weight $s_c / n_c$, its
    successes over its finished trials. The most likely path is the chain of parent
    links that maximizes

        $$\sum_{c \in \text{path}} \log \frac{s_c}{n_c}.$$

    Work backward from the last interface. A configuration's best score is its own log
    weight plus the best score among its successful children. Then start from the best
    configuration at $\lambda_0$ and follow the best child forward. Return one index
    per collection, or None if no trial reached $\lambda_B$.
    """
    raise NotImplementedError("most_likely_path is part of the FFS skeleton")


# --- driver ---------------------------------------------------------------------------

@dataclass
class FFSResult:
    settings: FFSSettings
    order_name: str
    order_units: str
    n_sites: int
    flux: FluxStage
    interfaces: list[InterfaceStage]
    levels: list[dict[str, np.ndarray]]       # describe() of each collection
    estimates: dict[str, float]
    errors: dict[str, float]
    completed: bool                            # some trial reached lam_B
    path: list[int] | None = None
    path_configs: np.ndarray | None = None
    warnings: list[str] = field(default_factory=list)

    @property
    def lambdas(self) -> list[float]:
        return [self.settings.first_interface()] + [s.lam_hi for s in self.interfaces]

    def committor(self) -> np.ndarray:
        """Forward committor estimate p_B(lambda_i) = prod_{j >= i} p_j at each interface."""
        p = np.array([s.p for s in self.interfaces] + [1.0])
        return np.cumprod(p[::-1])[::-1]

    def summary_row(self) -> dict:
        e = self.estimates
        row = {
            "order_parameter": self.order_name, "n_sites": self.n_sites,
            "completed": self.completed, "n_interfaces": len(self.interfaces),
            "flux": e["flux"], "T_int": e["T_int"], "T_ext": e["T_ext"],
            "T_f": e["T_f"], "P_AB": e["p"], "k_AB": e["k_AB"],
            "I_per_site": e["k_AB"] / self.n_sites,
            "T_FFS": e["T_FFS"], "T_FFST": e["T_FFST"],
            "T_f_over_T_FFST": (e["T_f"] / e["T_FFST"]
                                if np.isfinite(e["T_FFST"]) and e["T_FFST"] > 0
                                else math.nan),
            "binomial_rel_var_P": e["binomial_rel_var"],
            "flux_q": self.flux.q, "flux_t_hA": self.flux.t_hA,
            "flux_restarts": sum(r.n_restarts for r in self.flux.runs),
            "n_t_int_samples": len(self.flux.t_int_samples),
        }
        row.update(self.errors)
        for key in ("T_FFS", "T_FFST", "k_AB"):
            std = self.errors.get(f"{key}_std")
            row[f"{key}_rel_err"] = (std / e[key] if std is not None and
                                     np.isfinite(e[key]) and e[key] > 0 else math.nan)
        return row

    def interface_rows(self) -> list[dict]:
        committor = self.committor()
        rows = []
        for i, lam in enumerate(self.lambdas):
            level = self.levels[i] if i < len(self.levels) else {}
            row = {"interface": i, "lambda": lam, "committor": committor[i],
                   "n_configs": len(next(iter(level.values()))) if level else 0}
            for key, values in level.items():
                row[f"mean_{key}"] = float(np.mean(values)) if len(values) else math.nan
            if i < len(self.interfaces):
                s = self.interfaces[i]
                row.update({
                    "lambda_next": s.lam_hi, "n_trials": len(s.outcome),
                    "n_success": s.count(SUCCESS), "n_failure": s.count(FAILURE),
                    "n_stuck": s.count(STUCK), "n_censored": s.count(CENSORED),
                    "n_pilot": s.n_pilot, "P_next": s.p,
                    "t_success_mean": s.mean_time(SUCCESS),
                    "t_failure_mean": s.mean_time(FAILURE),
                    "overshoot_mean": s.overshoot,
                })
            if self.path is not None and i < len(self.path) and level:
                c = self.path[i]
                for key, values in level.items():
                    row[f"path_{key}"] = float(values[c])
            rows.append(row)
        return rows


def check_order_parameter(prop, settings: FFSSettings) -> None:
    """Refuse an order parameter that doesn't put A low and B high."""
    lam_a = prop.lam(prop.initial_config())
    lam_b = prop.lam(prop.poisoned_config())
    if lam_a > settings.lam_A:
        raise ValueError(f"order parameter {prop.order.name!r} is {lam_a:g} on the empty "
                         f"lattice, above lam_A = {settings.lam_A:g}. Basin A must have "
                         f"low values.")
    if lam_b < settings.lam_B:
        raise ValueError(f"order parameter {prop.order.name!r} is {lam_b:g} on the "
                         f"CO-covered lattice, below lam_B = {settings.lam_B:g}. It must "
                         f"increase from A to B.")


def run_ffs(prop, settings: FFSSettings, base_seed: int = 0, step_index: int = 0,
            map_fn: MapFn = serial_map, n_flux_runs: int = 1,
            progress: Callable[[str], None] | None = None) -> FFSResult:
    r"""Run a full FFS calculation at one parameter point and collect the results.

    This is the driver that ties the other pieces together. Run stage 1
    with run_flux_stage, then step through the interfaces one at a time. The next
    interface comes from settings.lambdas, or from place_next_interface when placement
    is dynamic. At each one, fire trials with run_interface_stage and keep
    prop.describe() of the new configurations for the workbook.

    If no trial gets past an interface, stop and warn that the time is only a lower
    bound. Also warn if more than 1% of trials hit the time limit or any got stuck.
    Then compute transition_times. Error bars and the most likely path only make sense
    if a trial reached lam_B, so skip them otherwise. Send status lines to `progress`
    and return an FFSResult.
    """
    raise NotImplementedError("run_ffs is part of the FFS skeleton")
