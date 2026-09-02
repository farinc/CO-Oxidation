"""Per-grid-step drivers: one call per StepResult field (kMC trajectories, mean-field
time series, one-shot ME-MKM solve). Only run_memkm_for_step (and build_tile/
save_graph_html, which just build a tile) import co_oxidation.memkm.coexistence, so a
run with --no-memkm never triggers the PETSc/SLEPc lazy import in memkm/__init__.py."""

import numpy as np

from co_oxidation import meanfield
from co_oxidation.kmc import run_kmc

from .params import RunConfig
from .results import KMCTrajectoryResult, MeanFieldModelResult, MemkmStepResult

TAGS = ("empty", "full")
MEANFIELD_MODELS = ("mf", "ea")
_INIT_THETA = {"empty": (0.0, 0.0), "full": (1.0, 0.0)}


def kmc_seeds_for_step(base_seed: int, step_index: int, n_trajectories: int) -> dict[str, list[int]]:
    """Extends the sweep's seed(step, tag, trajectory) scheme so every trajectory of
    every (step, tag) pair gets a distinct, deterministic seed."""
    seeds = {}
    for j, tag in enumerate(TAGS):
        offset = step_index * 2 * n_trajectories + j * n_trajectories
        seeds[tag] = [base_seed + offset + t for t in range(n_trajectories)]
    return seeds


def run_kmc_for_step(config: RunConfig, seeds: dict[str, list[int]]) -> list[KMCTrajectoryResult]:
    """2 * n_trajectories calls to the unchanged co_oxidation.kmc.run_kmc, keeping the
    full times/coverage arrays (not just the steady-state averages)."""
    params = config.to_kmc_params()
    results = []
    for tag in TAGS:
        for i, seed in enumerate(seeds[tag]):
            res = run_kmc(config.k_o_ads, init=tag, params=params, seed=seed)
            results.append(KMCTrajectoryResult(
                tag=tag, index=i, seed=seed, times=res.times, cov_empty=res.cov_empty,
                cov_co=res.cov_co, cov_o=res.cov_o, steady_empty=res.steady_empty,
                steady_co=res.steady_co, steady_o=res.steady_o, t_final=res.t_final,
                steps=res.steps, stuck=res.stuck))
    return results


def run_meanfield_for_step(config: RunConfig,
                           kmc_trajectories: list[KMCTrajectoryResult] | None = None
                           ) -> list[MeanFieldModelResult]:
    """meanfield.integrate (time-resolved) + meanfield.rates for both rate laws
    ("mf"/MF-MKM and "ea"/Bragg-Williams), each from an empty and a CO-covered start
    to mirror the kMC init tags.

    When kmc_trajectories is given (this step's kMC results), the mean-field time
    series is sampled to exactly as many points as the longest kMC trajectory, so the
    kMC and mean-field sheets for this step come out the same length in the exported
    workbook. Without it (kMC off), config.meanfield_dt sets the sampling instead."""
    n_points = max((len(t.times) for t in kmc_trajectories), default=None) \
        if kmc_trajectories else None
    kw = config.to_meanfield_kwargs()
    results = []
    for model in MEANFIELD_MODELS:
        for tag, theta0 in _INIT_THETA.items():
            t, traj = meanfield.integrate(theta0, config.k_o_ads, config.meanfield_t_end,
                                          config.meanfield_dt, n_points=n_points,
                                          model=model, **kw)
            theta_co, theta_o = traj[:, 0], traj[:, 1]
            theta_empty = 1.0 - theta_co - theta_o
            r_ads_co, r_des_co, r_ads_o, r_oxi, r_des_o = meanfield.rates(
                theta_co, theta_o, config.k_o_ads, model=model, **kw)
            results.append(MeanFieldModelResult(
                model=model, tag=tag, t=t, theta_co=theta_co, theta_o=theta_o,
                theta_empty=theta_empty, r_ads_co=np.asarray(r_ads_co),
                r_des_co=np.asarray(r_des_co), r_ads_o=np.asarray(r_ads_o),
                r_oxi=np.asarray(r_oxi), r_des_o=np.asarray(r_des_o)))
    return results


def build_tile(memkm_sites: int):
    """Smallest valid square tile with `memkm_sites` sites for the ME-MKM run."""
    from me_mkm import TileSettings
    return TileSettings.smallest_valid_square(memkm_sites, True)


def save_graph_html(tile, config: RunConfig, out_prefix: str) -> str:
    """Write the coverage-class transition graph for `tile` as a self-contained
    {out_prefix}_graph.html viewer (structure only, from the tile/reactions -- no
    PETSc/SLEPc solve, no Theta weighting)."""
    from me_mkm import build_graph, save_html

    from co_oxidation.memkm.model import generate_model

    builder = generate_model(tile=tile, **config.to_memkm_kwargs())
    graph_data = build_graph(builder)
    path = save_html(graph_data, f"{out_prefix}_graph.html")
    print(f"Graph written to '{path}'.")
    return path


def run_memkm_for_step(config: RunConfig, tile, n_eigs_scan: int = 4,
                        sigma_scale: float = 1e-8, factor=None,
                        order_species: str = "CO", solve_left: bool = True,
                        n_eigs_left: int = 3, comm=None,
                        coverage_cache: dict | None = None
                        ) -> tuple[MemkmStepResult | None, dict | None]:
    """One-shot ME-MKM solve at this step's config, decoupled from any Brent search.

    `solve_left`, when False, skips the extra distributed left-eigenpair
    factorization (the dominant per-step cost of --memkm on a large grid) and leaves
    cov_phi as None.

    Returns (result, state). `state` is solve_memkm_state's raw output -- the
    stationary distribution, slow eigenpairs and basin masks -- handed back so
    memkm-family observables can run on this point without paying for a second
    eigensolve (sweeps.observables.run_memkm_observables); ignore it if you only want
    the summary. Both are None if this step turns out to have no two-basin structure to
    report on (a reducible generator or a monostable point, see
    NoCoexistingMacrostatesError) instead of failing the whole sweep over one step."""
    from co_oxidation.memkm.coexistence import (
        NoCoexistingMacrostatesError,
        coverage_grids,
        solve_left_modes,
        solve_memkm_state,
        species_coverage_array,
    )

    try:
        state = solve_memkm_state(config.to_memkm_kwargs(), tile, comm, n_eigs_scan,
                                  sigma_scale, factor, order_species, coverage_cache)
    except NoCoexistingMacrostatesError as exc:
        if comm is None or comm.Get_rank() == 0:
            print(f"  [memkm] skipped: {exc}", flush=True)
        return None, None
    builder, theta = state["builder"], state["theta"]

    psi_L_2 = None
    if solve_left:
        left = solve_left_modes(state, n_eigs_left, comm, factor)
        psi_L_2 = left["psi_L_2"]

    cov_pop, cov_phi, cov_r2, cov_deg = coverage_grids(builder, theta, psi_L_2,
                                                       state["psi_R_2"])

    def species_coverage(name):
        if coverage_cache is not None and name in coverage_cache:
            cov = coverage_cache[name]
        else:
            cov = species_coverage_array(builder, name)
            if coverage_cache is not None:
                coverage_cache[name] = cov
        return cov

    def mean_coverage(name):
        return float(theta @ species_coverage(name))

    in_A, in_B = state["in_A"], state["in_B"]
    p_a = float(theta[in_A].sum())
    basin = {}
    for label, mask in (("a", in_A), ("b", in_B)):
        w = theta[mask].sum()
        for name, key in (("*", "empty"), ("CO", "co"), ("O", "o")):
            cov = species_coverage(name)
            basin[f"{key}_{label}"] = float(theta[mask] @ cov[mask] / w) if w > 0 else float("nan")

    result = MemkmStepResult(
        eigvals=state["eigvals"], cov_pop=cov_pop, cov_r2=cov_r2, cov_phi=cov_phi,
        cov_deg=cov_deg, theta_empty=mean_coverage("*"), theta_co=mean_coverage("CO"),
        theta_o=mean_coverage("O"), n_sites=builder.l, p_a=p_a, **basin)
    return result, state
