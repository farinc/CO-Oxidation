"""
MPI sweep. Phase A (kMC): grid steps are split round-robin across MPI ranks (each rank
runs all 2*n_trajectories trajectories of its assigned steps) and gathered on rank 0 --
embarrassingly parallel. Phase B (ME-MKM, optional): every rank cooperates on each
step's distributed generator, one step at a time, so a --coexistence Brent search stays
in lockstep across ranks. Phase C (mean field): computed serially on rank 0 (cheap ODE
integration, not worth parallelizing). Rank 0 writes {out}.xlsx and the figures. Each
phase can be turned off with --no-kmc / --memkm/--no-memkm / --no-meanfield (ME-MKM is
on by default here, unlike sweeps/linear.py). Requires the native-petsc + native-slepc
or source-petsc + source-slepc extras (mpi4py, petsc4py, slepc4py).

Usage:
    mpirun -np 4 uv run python -m sweeps.mpi --memkm-sites 8 --out case1
    mpirun -np 24 uv run python -m sweeps.mpi --memkm-sites 12 --out big -eps_monitor
"""

from mpi4py import MPI

from sweeps.axes import SweepGrid
from sweeps.cli import parse_args
from sweeps.coexistence_driver import run_coexistence_for_grid, validate_bisection_axis
from sweeps.excel import write_workbook
from sweeps.observables import run_kmc_observables, run_memkm_observables
from sweeps.results import StepResult, SweepResult
from sweeps.steps import (build_tile, kmc_seeds_for_step, run_kmc_for_step,
                          run_meanfield_for_step, run_memkm_for_step, save_graph_html)


def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    args = parse_args(None, memkm_default=True)
    grid = SweepGrid.build(args.base, args.axes)

    if args.memkm and args.coexistence:
        validate_bisection_axis(grid, args.coexistence_axis, True, args.coexistence_fixed)

    tile_cache: dict[int, object] = {}

    def tile_for(sites):
        if sites not in tile_cache:
            tile_cache[sites] = build_tile(sites)
        return tile_cache[sites]

    if args.memkm and args.save_graph and rank == 0:
        save_graph_html(tile_for(args.base.memkm_sites), args.base, args.out)

    # Phase A: kMC, step-sharded round-robin across ranks. kmc-family observables run
    # here too, on the rank that owns the step, and ride the same gather home.
    local_kmc = {}
    local_obs = {}
    if args.kmc:
        for grid_step in grid.steps:
            if grid_step.index % size != rank:
                continue
            cfg = grid_step.config
            local_kmc[grid_step.index] = run_kmc_for_step(
                cfg, kmc_seeds_for_step(cfg.kmc_seed, grid_step.index, cfg.kmc_n_trajectories))
            if args.observables:
                local_obs[grid_step.index] = run_kmc_observables(
                    args.observables, cfg, local_kmc[grid_step.index],
                    grid_step.index, args.observable_options)
            print(f"[rank {rank}] kMC step {grid_step.index} done", flush=True)
    gathered_kmc = comm.gather(local_kmc, root=0)
    gathered_obs = comm.gather(local_obs, root=0)
    kmc_by_step = {}
    observables_by_step: dict[int, dict] = {}
    if rank == 0:
        for part in gathered_kmc:
            kmc_by_step.update(part)
        for part in gathered_obs:
            for idx, values in part.items():
                observables_by_step.setdefault(idx, {}).update(values)

    # Phase B: ME-MKM, collective per step (all ranks in lockstep -- every rank must
    # take part in every step's distributed PETSc/SLEPc solve).
    memkm_by_step = {}
    if args.memkm:
        if rank == 0:
            print("ME-MKM / SLEPc phase", flush=True)
        # Per-microstate coverage arrays depend only on the tile, not the physics --
        # scope the cache per tile (memkm_sites) so steps at different site counts
        # never share it.
        coverage_caches: dict[int, dict] = {}
        for grid_step in grid.steps:
            cfg = grid_step.config
            cache = coverage_caches.setdefault(cfg.memkm_sites, {})
            result, state = run_memkm_for_step(
                cfg, tile_for(cfg.memkm_sites), n_eigs_scan=args.memkm_n_eigs_scan,
                factor=args.memkm_factor_solver, order_species=args.memkm_order_species,
                solve_left=not args.memkm_skip_left_modes, comm=comm, coverage_cache=cache)
            memkm_by_step[grid_step.index] = result
            if args.observables:
                # Collective: every rank runs these on the same replicated state, so an
                # observable that touches the distributed generator stays in lockstep.
                # Only rank 0 keeps the values (and reports failures).
                values = run_memkm_observables(
                    args.observables, state, comm=comm, factor=args.memkm_factor_solver,
                    options=args.observable_options, step_index=grid_step.index,
                    quiet=rank != 0)
                if rank == 0:
                    observables_by_step.setdefault(grid_step.index, {}).update(values)
            if rank == 0 and result is not None:
                print(f"  [memkm] step {grid_step.index} done", flush=True)

    # Phase C: mean-field branches, serial on rank 0. kmc_by_step is only populated on
    # rank 0 (gathered above), which is exactly where this loop runs, so each step's
    # mean-field series can be sampled to match its kMC trajectory length directly.
    meanfield_by_step = {}
    if rank == 0 and args.meanfield:
        for grid_step in grid.steps:
            meanfield_by_step[grid_step.index] = run_meanfield_for_step(
                grid_step.config, kmc_by_step.get(grid_step.index))

    # Coexistence, collective across comm (see run_coexistence_for_grid docstring).
    coexistence = None
    if args.memkm and args.coexistence:
        coexistence = run_coexistence_for_grid(
            grid, args.coexistence_axis, comm=comm, n_eigs_scan=args.memkm_n_eigs_scan,
            boundary_eps=args.memkm_boundary_eps, n_eigs=args.memkm_n_eigs,
            xtol=args.memkm_brent_xtol, factor=args.memkm_factor_solver,
            order_species=args.memkm_order_species, fixed_value=args.coexistence_fixed)

    if rank != 0:
        return

    steps = [StepResult(step=s, kmc=kmc_by_step.get(s.index),
                        meanfield=meanfield_by_step.get(s.index),
                        memkm=memkm_by_step.get(s.index),
                        observables=observables_by_step.get(s.index, {}))
             for s in grid.steps]
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
