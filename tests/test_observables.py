"""Tests for the observable framework: the registry, the two contexts, and the kMC
stopping hook the KMCContext exposes."""

import numpy as np
import pytest

from co_oxidation import kmc
from co_oxidation.observables import registry
from co_oxidation.observables.context import KMCContext, MemkmContext


@pytest.fixture
def clean_registry():
    """Register into a scratch registry so a test's observables never leak into
    another's --list-observables."""
    saved = dict(registry._REGISTRY)
    registry._REGISTRY.clear()
    yield registry
    registry._REGISTRY.clear()
    registry._REGISTRY.update(saved)


# --- registry -------------------------------------------------------------------

def test_decorator_registers_and_returns_the_plain_function(clean_registry):
    @registry.observable("thing", family=registry.KMC)
    def thing(ctx):
        """A doc line."""
        return {"x": 1.0}

    obs = registry.get("thing")
    assert obs.family == registry.KMC
    assert obs.doc == "A doc line."
    # the decorator must not wrap: the function stays directly callable in a notebook
    assert thing(None) == {"x": 1.0}


def test_keys_are_namespaced_by_observable_name(clean_registry):
    registry.register("a", registry.KMC, lambda ctx: {"tau": 1.0})
    registry.register("b", registry.KMC, lambda ctx: {"tau": 2.0})
    values = registry.compute(["a", "b"], object(), family=registry.KMC)
    assert values == {"a.tau": 1.0, "b.tau": 2.0}


def test_a_bare_return_takes_the_observables_own_name(clean_registry):
    registry.register("scalar", registry.KMC, lambda ctx: 3.5)
    assert registry.compute(["scalar"], object()) == {"scalar": 3.5}


def test_compute_filters_by_family(clean_registry):
    registry.register("k", registry.KMC, lambda ctx: {"v": 1.0})
    registry.register("m", registry.MEMKM, lambda ctx: {"v": 2.0})
    # one --observable list is handed to both phases; each takes only its own
    assert registry.compute(["k", "m"], object(), family=registry.KMC) == {"k.v": 1.0}
    assert registry.compute(["k", "m"], object(), family=registry.MEMKM) == {"m.v": 2.0}


def test_a_failing_observable_does_not_take_the_run_down(clean_registry):
    def boom(ctx):
        raise ZeroDivisionError("nope")

    registry.register("boom", registry.KMC, boom)
    registry.register("fine", registry.KMC, lambda ctx: {"v": 1.0})
    errors = []
    values = registry.compute(["boom", "fine"], object(), on_error=errors.append)
    assert values["fine.v"] == 1.0                  # the good one still lands
    assert "ZeroDivisionError" in values["boom.error"]
    assert [e.name for e in errors] == ["boom"]


def test_strict_mode_reraises_with_the_observable_named(clean_registry):
    registry.register("boom", registry.KMC, lambda ctx: 1 / 0)
    with pytest.raises(registry.ObservableError, match="boom"):
        registry.compute(["boom"], object(), strict=True)


def test_duplicate_names_are_refused_unless_replacing(clean_registry):
    registry.register("dup", registry.KMC, lambda ctx: 1.0)
    with pytest.raises(ValueError, match="already registered"):
        registry.register("dup", registry.KMC, lambda ctx: 2.0)
    registry.register("dup", registry.KMC, lambda ctx: 2.0, replace=True)
    assert registry.compute(["dup"], object()) == {"dup": 2.0}


def test_unknown_family_is_refused(clean_registry):
    with pytest.raises(ValueError, match="family must be one of"):
        registry.register("x", "montecarlo", lambda ctx: 1.0)


def test_unknown_name_lists_what_is_known(clean_registry):
    registry.register("known", registry.KMC, lambda ctx: 1.0)
    with pytest.raises(KeyError, match="known"):
        registry.get("missing")


def test_load_modules_accepts_a_plain_file_path(clean_registry, tmp_path):
    plugin = tmp_path / "my_obs.py"
    plugin.write_text(
        "from co_oxidation.observables import observable\n"
        "@observable('from_file', family='kmc')\n"
        "def from_file(ctx):\n"
        "    return {'v': 42.0}\n")
    registry.load_modules([str(plugin)])
    assert registry.compute(["from_file"], object()) == {"from_file.v": 42.0}


def test_scalar_and_array_values_are_separated(clean_registry):
    values = {"a.x": 1.0, "a.n": 3, "a.flag": True, "a.arr": np.arange(4),
              "a.scalar_array": np.float64(2.0)}
    assert set(registry.scalar_items(values)) == {"a.x", "a.n", "a.flag",
                                                  "a.scalar_array"}
    assert set(registry.array_items(values)) == {"a.arr"}


# --- kMC context and the stopping hook -------------------------------------------

def _adsorption_only_params(L=4):
    """CO adsorption is the only event that can fire, so the first-passage time to
    'one CO on the lattice' is exactly Exponential(N * k_co_ads)."""
    return kmc.KMCParams(L=L, k_co_ads=1.0, k_co_des=0.0, k_rxn=0.0, k_o_des=0.0,
                         khop=0.0, t_max=1e6, seed=0)


def test_first_passage_time_matches_the_analytic_rate():
    params = _adsorption_only_params(L=4)
    N = params.L ** 2
    predicate = kmc.coverage_predicate("co", threshold=1.0 / N, above=True)
    n_runs = 600
    times = np.array([
        kmc.run_first_passage(0.0, predicate, init="empty", params=params,
                              seed=1000 + i).t_hit
        for i in range(n_runs)])
    assert np.all(np.isfinite(times))
    expected = 1.0 / (N * params.k_co_ads)      # only N empty sites can adsorb
    # the sample mean of n exponentials has standard error mean/sqrt(n)
    assert abs(times.mean() - expected) < 4.0 * expected / np.sqrt(n_runs)


def test_a_start_already_in_the_target_set_has_passage_time_zero():
    params = _adsorption_only_params(L=4)
    predicate = kmc.coverage_predicate("co", threshold=0.5, above=True)
    res = kmc.run_first_passage(0.0, predicate, init="full", params=params)
    assert res.hit and res.t_hit == 0.0 and res.steps == 0


def test_a_run_that_never_hits_is_reported_as_censored_not_as_a_time():
    params = _adsorption_only_params(L=4)
    # unreachable: nothing in this parameter set ever puts O on the lattice
    predicate = kmc.coverage_predicate("o", threshold=0.5, above=True)
    res = kmc.run_first_passage(0.0, predicate, init="empty", params=params,
                                t_max=0.5, seed=7)
    assert not res.hit
    assert np.isnan(res.t_hit)
    # censored at t_final, which is the first time at or past the limit: the loop
    # tests t < t_max before drawing a waiting time, so the last one overshoots.
    # Same convention as the main _run loop.
    assert res.t_final >= 0.5 and not res.stuck


def test_coverage_predicate_rejects_a_bad_species():
    with pytest.raises(ValueError, match="unknown species"):
        kmc.coverage_predicate("argon")


def test_context_seeds_are_deterministic_and_disjoint_from_the_sweeps_own():
    params = kmc.KMCParams(seed=0)
    step0 = KMCContext(k_o_ads=1.0, params=params, seed_base=0)
    step1 = KMCContext(k_o_ads=1.0, params=params,
                       seed_base=KMCContext.SEED_STRIDE)
    assert step0.seed(0) == step0.seed(0)                    # reproducible
    assert step0.seed(0) != step0.seed(1)                    # distinct per draw
    assert not ({step0.seed(i) for i in range(50)}
                & {step1.seed(i) for i in range(50)})        # distinct per step
    assert min(step0.seed(i) for i in range(50)) > 1000       # clear of sweep seeds


def test_an_unseeded_run_stays_unseeded():
    ctx = KMCContext(k_o_ads=1.0, params=kmc.KMCParams(seed=-1))
    assert ctx.seed(3) == -1


def test_context_options_are_cast_to_the_type_of_the_default():
    ctx = KMCContext(options={"obs": {"n": "8", "frac": "0.25", "on": "true",
                                      "name": "co"}})
    assert ctx.option("obs", "n", 1) == 8
    assert ctx.option("obs", "frac", 0.0) == 0.25
    assert ctx.option("obs", "on", False) is True
    assert ctx.option("obs", "name", "") == "co"
    assert ctx.option("obs", "absent", 5) == 5


def test_context_runs_extra_first_passage_trajectories():
    params = _adsorption_only_params(L=4)
    ctx = KMCContext(k_o_ads=0.0, params=params, seed_base=0)
    predicate = kmc.coverage_predicate("co", threshold=1.0 / params.L ** 2)
    runs = ctx.run_first_passage(predicate, n_trajectories=3)
    assert len(runs) == 3
    assert all(r.hit for r in runs)
    assert len({r.t_hit for r in runs}) == 3        # independent seeds, not a repeat


# --- ME-MKM context ---------------------------------------------------------------

def test_memkm_context_exposes_the_solved_state_without_touching_petsc():
    state = {"builder": object(), "theta": np.array([0.25, 0.75]),
             "eigvals": np.array([0.0, -1.0]), "psi_R_2": np.array([-1.0, 1.0]),
             "in_A": np.array([True, False]), "in_B": np.array([False, True])}
    ctx = MemkmContext(state=state)
    assert ctx.theta.tolist() == [0.25, 0.75]
    assert ctx.eigvals[1] == -1.0
    assert ctx.in_A.tolist() == [True, False]
    ctx.close()             # a context that never built W closes cleanly


@pytest.fixture(scope="module")
def small_builder():
    me_mkm = pytest.importorskip("me_mkm")
    from co_oxidation.memkm.model import generate_model
    tile = me_mkm.TileSettings.smallest_valid_square(4, False)
    return generate_model(k_o_ads=1.0, tile=tile)


def test_memkm_context_builds_the_generator_lazily_and_releases_it(small_builder):
    pytest.importorskip("petsc4py")
    n = small_builder.n_states
    state = {"builder": small_builder, "theta": np.full(n, 1.0 / n)}
    with MemkmContext(state=state) as ctx:
        assert ctx._W is None                       # nothing built until asked for
        W = ctx.W
        assert W.getSize() == (n, n)
        assert ctx.W is W                           # built at most once per context
    assert ctx._W is None                           # and released on the way out


def test_memkm_context_caches_per_microstate_coverages(small_builder):
    n = small_builder.n_states
    ctx = MemkmContext(state={"builder": small_builder, "theta": np.full(n, 1.0 / n)})
    cov = ctx.coverage("CO")
    assert cov.shape == (n,)
    assert cov.min() == 0.0 and cov.max() == 1.0    # a fraction n_CO / l
    assert ctx.coverage("CO") is cov
