"""Tests for forward flux sampling: estimators, stages, kernels and order parameters.

The FFS algorithm is a skeleton, so tests that need it are marked needs_ffs. They show
as xfail until the stubs are implemented, then as xpass. Any failure other than
NotImplementedError still fails the run.
"""

import math
from types import SimpleNamespace

import numpy as np
import pytest
from numba import njit
from scipy import ndimage

import co_oxidation.kmc as kmc
from co_oxidation import ffs, ffs_order
from co_oxidation.ffs import (FAILURE, SUCCESS, FFSSettings, FluxRun, InterfaceStage,
                              KMCPropagator, TrialResult, check_order_parameter,
                              most_likely_path, run_ffs, transition_times)

needs_ffs = pytest.mark.xfail(raises=NotImplementedError, strict=False,
                              reason="the FFS algorithm in co_oxidation.ffs is a skeleton")


# --------------------------------------------------------------------------
# A birth-death chain with an exact mean first-passage time, driven through the
# same stage code as the lattice. Mirrors the test problem of Adams et al. App. B.
# --------------------------------------------------------------------------

class BirthDeathChain:
    """States 0..n_max, hopping up at rate up[x] and down at rate down[x]."""

    def __init__(self, up, down):
        self.up = np.asarray(up, float)
        self.down = np.asarray(down, float)
        self.n_max = len(self.up) - 1
        self.order = SimpleNamespace(name="x", units="state")
        self.n_sites = 1

    def initial_config(self):
        return np.array([0], np.int64)

    def poisoned_config(self):
        return np.array([self.n_max], np.int64)

    def lam(self, config):
        return float(config[0])

    def _step(self, x, rng):
        total = self.up[x] + self.down[x]
        dt = rng.exponential(1.0 / total)
        return (x + 1 if rng.random() * total < self.up[x] else x - 1), dt

    def trial(self, config, lo, hi, t_max, max_steps, seed):
        rng = np.random.default_rng(seed)
        x, t, steps, lam_max = int(config[0]), 0.0, 0, float(config[0])
        while True:
            if x >= hi:
                return TrialResult(SUCCESS, x, lam_max, t, steps, np.array([x]))
            if x <= lo:
                return TrialResult(FAILURE, x, lam_max, t, steps, np.array([x]))
            x, dt = self._step(x, rng)
            t += dt
            steps += 1
            lam_max = max(lam_max, x)

    def flux(self, config, lam_A, lam_0, lam_B, n_crossings, t_equil, t_max, max_steps,
             seed):
        rng = np.random.default_rng(seed)
        x = int(config[0])
        snaps, t_int, t_cross = [], [], []
        t = t_hA = 0.0
        h_A = armed = x <= lam_A
        t_enter = 0.0 if armed else -1.0
        while len(snaps) < n_crossings:
            x_new, dt = self._step(x, rng)
            if h_A:
                t_hA += dt
            x, t = x_new, t + dt
            if x <= lam_A:
                h_A = True
                if not armed:
                    armed, t_enter = True, t
            elif x >= lam_B:
                x, h_A, armed, t_enter = 0, True, True, t
            elif armed and x >= lam_0:
                snaps.append([x])
                t_int.append(t - t_enter if t_enter >= 0 else math.nan)
                t_cross.append(t_hA)
                armed = False
        return FluxRun(len(snaps), t_hA, t, 0, 0, False, np.array(snaps),
                       np.array(t_int), np.array(t_cross))

    def describe(self, configs):
        return {"x": configs[:, 0].astype(float)}

    def exact_mfpt(self, start, target):
        """Mean time from `start` to first reach `target`, reflecting at 0."""
        tau = np.zeros(target)            # tau[m] = mean time to go from m to m+1
        for m in range(target):
            tau[m] = (1.0 + self.down[m] * (tau[m - 1] if m else 0.0)) / self.up[m]
        return float(tau[start:target].sum())


def _chain():
    n = 12
    up = np.ones(n + 1)
    down = np.full(n + 1, 2.0)
    down[0] = 0.0
    return BirthDeathChain(up, down)


@needs_ffs
def test_ffs_and_ffst_match_the_exact_mean_first_passage_time():
    chain = _chain()
    settings = FFSSettings(lam_A=2, lam_B=12, lambdas=tuple(range(3, 13)),
                           n_crossings=3000, n_trials=3000, t_equil=0.0, n_bootstrap=0)
    result = run_ffs(chain, settings, base_seed=7)
    exact_from_lambda0 = chain.exact_mfpt(3, 12)     # FFST starts the clock at lambda_0
    assert result.completed
    assert result.estimates["T_FFST"] == pytest.approx(exact_from_lambda0, rel=0.15)
    assert result.estimates["T_FFS"] == pytest.approx(chain.exact_mfpt(0, 12), rel=0.15)


@needs_ffs
def test_transition_times_against_a_hand_calculation():
    est = transition_times(flux=2.0, T_int=0.5, p_i=np.array([0.5, 0.2]),
                           t_succ=np.array([1.0, 3.0]), t_fail=np.array([0.4, 2.0]))
    assert est["p"] == pytest.approx(0.1)
    assert est["T_FFS"] == pytest.approx(5.0)
    assert est["T_f"] == pytest.approx(4.0)
    # T_ext = (0.5 * 0.4 + 0.4 * (1 + 2)) / 0.9
    assert est["T_ext"] == pytest.approx(1.4 / 0.9)
    assert est["T_FFST"] == pytest.approx(9.0 * (1.4 / 0.9 + 0.5) + 4.0)


@needs_ffs
def test_transition_times_when_nothing_or_everything_succeeds():
    none = transition_times(1.0, 1.0, np.array([0.3, 0.0]), np.array([1.0, np.nan]),
                            np.array([1.0, 1.0]))
    assert math.isinf(none["T_FFS"]) and math.isinf(none["T_FFST"])
    every = transition_times(1.0, 1.0, np.array([1.0, 1.0]), np.array([1.0, 2.0]),
                             np.array([np.nan, np.nan]))
    assert every["T_FFST"] == pytest.approx(3.0)


def _stage(index, parent, outcome):
    n = len(outcome)
    return InterfaceStage(index=index, lam_lo=index, lam_hi=index + 1,
                          parent=np.array(parent), outcome=np.array(outcome, np.int8),
                          t=np.ones(n), steps=np.ones(n, np.int64), lam_end=np.zeros(n),
                          lam_max=np.zeros(n))


@needs_ffs
def test_most_likely_path_follows_the_best_product_of_success_fractions():
    # collection 0 has two configs. Config 1 succeeds 2/2, config 0 only 1/2.
    s0 = _stage(0, [0, 0, 1, 1], [SUCCESS, FAILURE, SUCCESS, SUCCESS])
    # collection 1 = successes of s0, in order: from parents 0, 1, 1
    s1 = _stage(1, [0, 1, 2, 2], [SUCCESS, FAILURE, SUCCESS, FAILURE])
    # config 0 (1/2) then child 0 (1/1) ties config 1 (2/2) then child 2 (1/2).
    # The first maximum wins, and child 0's success is the first in collection 2.
    assert most_likely_path([s0, s1], n_first=2) == [0, 0, 0]
    s1_only_child2 = _stage(1, [0, 1, 2, 2], [FAILURE, FAILURE, SUCCESS, SUCCESS])
    assert most_likely_path([s0, s1_only_child2], n_first=2)[:2] == [1, 2]


# --------------------------------------------------------------------------
# kMC kernels
# --------------------------------------------------------------------------

@needs_ffs
def test_ffs_trial_matches_first_passage_on_the_same_seed():
    L = 8
    N = L * L
    rng = np.random.default_rng(3)
    lat = np.zeros(N, np.int8)
    lat[rng.choice(N, 10, replace=False)] = kmc.CO
    lo_n, hi_n = 4, 20

    @njit
    def two_sided(n_empty, n_co, n_o, N):
        return n_co <= lo_n or n_co >= hi_n

    params = kmc.KMCParams(L=L)
    rates = kmc.class_rates_for(1.0, params)
    lat_a, lat_b = lat.copy(), lat.copy()
    outcome, lam_end, _, t, steps = kmc._run_ffs_trial(
        lat_a, L, rates, ffs_order.theta_co, 1, lo_n / N, hi_n / N, 1e6, 10 ** 9, 11)
    t_hit, hit, _, steps_b, _ = kmc._run_first_passage(lat_b, L, rates, two_sided,
                                                       1e6, 10 ** 9, 11)
    assert hit and outcome in (SUCCESS, FAILURE)
    assert steps == steps_b and t == pytest.approx(t_hit)
    assert np.array_equal(lat_a, lat_b)
    n_co = np.count_nonzero(lat_a == kmc.CO)
    assert (n_co >= hi_n) if outcome == SUCCESS else (n_co <= lo_n)
    assert lam_end == pytest.approx(n_co / N)


@needs_ffs
def test_flux_run_records_first_crossings_of_lambda0():
    prop = KMCPropagator(1.0, kmc.KMCParams(L=8), "theta_co")
    run = prop.flux(prop.initial_config(), 0.2, 0.25, 0.7, 8, 1.0, 1e5, 10 ** 10, 5)
    assert run.q == 8 and not run.stuck
    theta = (run.snapshots == kmc.CO).sum(axis=1) / 64
    assert np.allclose(theta, 0.25)                 # stride 1 stops exactly on lambda_0
    assert np.all(np.diff(run.t_cross) > 0)
    assert np.all(run.t_int[1:] > 0)
    assert run.t_hA >= run.t_cross[-1]


# --------------------------------------------------------------------------
# order parameters
# --------------------------------------------------------------------------

def _periodic_largest(lat2d):
    labels, n = ndimage.label(lat2d)
    parent = list(range(n + 1))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for a, b in list(zip(labels[0], labels[-1])) + list(zip(labels[:, 0], labels[:, -1])):
        if a and b:
            parent[find(a)] = find(b)
    sizes = {}
    for v in labels.ravel():
        if v:
            r = find(v)
            sizes[r] = sizes.get(r, 0) + 1
    return max(sizes.values(), default=0)


@pytest.mark.parametrize("seed", range(5))
def test_largest_cluster_matches_scipy_with_periodic_merging(seed):
    L = 10
    lat = np.random.default_rng(seed).choice([0, 1, 2], L * L, p=(0.3, 0.5, 0.2))
    lat = lat.astype(np.int8)
    size, _ = ffs_order.co_clusters(lat, L)
    assert size == _periodic_largest(lat.reshape(L, L) == kmc.CO)


def test_cluster_wrapping():
    L = 8
    row = np.zeros(L * L, np.int8)
    row[2 * L:3 * L] = kmc.CO
    assert ffs_order.co_clusters(row, L) == (L, True)
    block = np.zeros((L, L), np.int8)
    block[2:5, 2:5] = kmc.CO
    assert ffs_order.co_clusters(block.ravel(), L) == (9, False)


def test_registry_lists_builtins_and_loads_a_user_module(tmp_path):
    names = {op.name for op in ffs_order.available()}
    assert {"theta_co", "theta_co_minus_o", "largest_co_cluster"} <= names
    module = tmp_path / "my_order.py"
    module.write_text(
        "from numba import njit\n"
        "from co_oxidation.ffs_order import order_parameter\n"
        "@order_parameter('test_vacancy_deficit', replace=True)\n"
        "@njit\n"
        "def f(lat, L, n_empty, n_co, n_o):\n"
        "    '''One minus the vacancy coverage.'''\n"
        "    return 1.0 - n_empty / (L * L)\n")
    ffs_order.load_modules([str(module)])
    op = ffs_order.get("test_vacancy_deficit")
    lat = np.zeros(16, np.int8)
    lat[:4] = kmc.CO
    assert op.evaluate(lat, 4) == pytest.approx(0.25)
    assert op.doc == "One minus the vacancy coverage."
    assert ffs_order.get("largest_co_cluster").resolved_stride(8) == 64


def test_check_rejects_an_order_parameter_that_decreases_toward_B():
    @njit
    def minus_theta(lat, L, n_empty, n_co, n_o):
        return -n_co / (L * L)

    op = ffs_order.register("test_minus_theta", minus_theta, replace=True)
    prop = KMCPropagator(1.0, kmc.KMCParams(L=4), op)
    with pytest.raises(ValueError, match="increase from A to B"):
        check_order_parameter(prop, FFSSettings(lam_A=0.1, lam_B=0.5,
                                                lambdas=(0.1, 0.5)))


def test_settings_validation():
    with pytest.raises(ValueError):
        FFSSettings(lam_A=0.1, lam_B=0.5, lambdas=(0.1, 0.3, 0.2, 0.5)).validate()
    with pytest.raises(ValueError):
        FFSSettings(lam_A=0.1, lam_B=0.5, lambdas=(0.1, 0.4)).validate()
    FFSSettings(lam_A=0.1, lam_B=0.5, placement="dynamic").validate()


# --------------------------------------------------------------------------
# end to end
# --------------------------------------------------------------------------

def test_sweeps_ffs_writes_a_workbook(tmp_path):
    import pandas as pd

    from sweeps import ffs as ffs_driver

    out = tmp_path / "ffs_run"
    ffs_driver.main([
        "--ffs-lambda-A", "0.2", "--ffs-lambda-B", "0.5", "--ffs-n-interfaces", "3",
        "--ffs-n-trials", "20", "--ffs-n-crossings", "20", "--ffs-t-equil", "1",
        "--ffs-n-bootstrap", "10", "--kmc-L", "6", "--k-o-ads", "1.0", "--no-plot",
        "--out", str(out)])
    sheets = pd.read_excel(f"{out}.xlsx", sheet_name=None)
    assert {"Parameters", "FFS_Settings", "FFS_Summary"} <= set(sheets)
    summary = sheets["FFS_Summary"]
    assert len(summary) == 1
    if "error" in summary and summary["error"].notna().any():
        # the skeleton records each step as failed instead of crashing the sweep
        assert "NotImplementedError" in summary["error"].iloc[0]
    else:
        assert "S0_FFS" in sheets
        assert summary["T_FFST"].iloc[0] > 0


def test_split_argv_separates_ffs_options():
    from sweeps.ffs import split_argv
    ffs_args, rest = split_argv(["--ffs-lambda-A", "0.2", "--kmc-L", "8",
                                 "--ffs-store-configs", "--ffs-lambda-B=0.6",
                                 "--sweep", "k_o_ads=1,2"])
    assert ffs_args == ["--ffs-lambda-A", "0.2", "--ffs-store-configs",
                        "--ffs-lambda-B=0.6"]
    assert rest == ["--kmc-L", "8", "--sweep", "k_o_ads=1,2"]
