"""
Serial sweep: runs the kMC, mean-field (MF-MKM/Bragg-Williams) and (opt-in) ME-MKM
phases over an N-D parameter grid on a single core, writing one {out}.xlsx workbook
(Parameters/Axes/Grid meta sheets, then per-step sheets, then a Coexistence sheet and
per-crossing ME-MKM sheets if --coexistence ran) plus the kept figures if --plot.

The ME-MKM phase is off by default here: even a small tile is too slow for a
laptop-run sweep. sweeps/mpi.py enables it by default since that's meant to run on a
cluster.

Usage:
    uv run python -m sweeps.linear
    uv run python -m sweeps.linear --kmc-L 24 --out case1
    uv run python -m sweeps.linear --sweep k_o_ads=0:10:0.4 --kmc-n-trajectories 3
    uv run python -m sweeps.linear --memkm --memkm-sites 8 --sweep k_o_ads=0:10:0.4 \\
        --coexistence --coexistence-axis k_o_ads   # + ME-MKM/SLEPc coexistence
"""

from sweeps.cli import parse_args
from sweeps.coexistence_driver import run_coexistence_for_grid, validate_bisection_axis
from sweeps.excel import write_workbook
from sweeps.axes import SweepGrid
from sweeps.observables import run_kmc_observables, run_memkm_observables
from sweeps.results import StepResult, SweepResult
from sweeps.steps import (build_tile, kmc_seeds_for_step, run_kmc_for_step,
                          run_meanfield_for_step, run_memkm_for_step, save_graph_html)


def main():
    args = parse_args(None, memkm_default=False)
    grid = SweepGrid.build(args.base, args.axes)

    if args.memkm and args.coexistence:
        validate_bisection_axis(grid, args.coexistence_axis, True, args.coexistence_fixed)

    tile_cache: dict[int, object] = {}

    def tile_for(sites):
        if sites not in tile_cache:
            tile_cache[sites] = build_tile(sites)
        return tile_cache[sites]

    if args.memkm and args.save_graph:
        save_graph_html(tile_for(args.base.memkm_sites), args.base, args.out)

    # Per-microstate coverage arrays depend only on the tile, not the physics -- scope
    # the cache per tile (memkm_sites) so steps at different site counts never share it.
    coverage_caches: dict[int, dict] = {}

    def coverage_cache_for(sites):
        return coverage_caches.setdefault(sites, {})

    steps = []
    for grid_step in grid.steps:
        cfg = grid_step.config
        kmc = (run_kmc_for_step(cfg, kmc_seeds_for_step(cfg.kmc_seed, grid_step.index,
                                                        cfg.kmc_n_trajectories))
               if args.kmc else None)
        meanfield = run_meanfield_for_step(cfg, kmc) if args.meanfield else None
        memkm = memkm_state = None
        if args.memkm:
            memkm, memkm_state = run_memkm_for_step(
                cfg, tile_for(cfg.memkm_sites), n_eigs_scan=args.memkm_n_eigs_scan,
                factor=args.memkm_factor_solver, order_species=args.memkm_order_species,
                solve_left=not args.memkm_skip_left_modes, comm=None,
                coverage_cache=coverage_cache_for(cfg.memkm_sites))
        # Observables run last: both families see everything this step already
        # produced, and neither triggers a solve of its own.
        observables = {}
        if args.observables:
            observables.update(run_kmc_observables(
                args.observables, cfg, kmc, grid_step.index, args.observable_options))
            observables.update(run_memkm_observables(
                args.observables, memkm_state, comm=None,
                factor=args.memkm_factor_solver, options=args.observable_options,
                step_index=grid_step.index))
        steps.append(StepResult(step=grid_step, kmc=kmc, meanfield=meanfield,
                                memkm=memkm, observables=observables))
        print(f"step {grid_step.index}: {grid_step.axis_values or '(no swept axes)'}",
              flush=True)

    coexistence = None
    if args.memkm and args.coexistence:
        coexistence = run_coexistence_for_grid(
            grid, args.coexistence_axis, comm=None, n_eigs_scan=args.memkm_n_eigs_scan,
            boundary_eps=args.memkm_boundary_eps, n_eigs=args.memkm_n_eigs,
            xtol=args.memkm_brent_xtol, factor=args.memkm_factor_solver,
            order_species=args.memkm_order_species, fixed_value=args.coexistence_fixed)

    result = SweepResult(grid=grid, steps=steps, coexistence=coexistence)

    out_path = f"{args.out}.xlsx"
    write_workbook(result, out_path)
    print(f"Data written to '{out_path}'.")

    if args.plot:
        from sweeps.plotting import plot_all
        for path in plot_all(result, args.out, args.plot_axis, args.plot_memkm_steps):
            print(f"  [plot] wrote {path}")


if __name__ == "__main__":
    main()
