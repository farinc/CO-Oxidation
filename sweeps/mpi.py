"""
MPI beta sweep. 

Phase A: 
The (beta, init) kMC cases are split round-robin
across MPI ranks and gathered on rank 0 (embarrassingly parallel). 

Phase B:
The ME-MKM / SLEPc coexistence analysis runs together with all ranks cooperating
on each beta's distributed generator. It goes one beta at a time so the Brent search
for beta* stays in lockstep across ranks. 

Phase C: 
The mean-field branches are computed serially on rank 0. Rank 0 writes {out}_kmc_sweep.csv,
{out}_coexistence.csv and {out}_meanfield.csv. Each phase can be turned off
with --no-kmc / --no-memkm / --no-meanfield. Requires the `native-petsc-slepc`
or `source-petsc-slepc` extra (taht is mpi4py, petsc4py, and slepc4py).

Usage:
    mpirun -np 4 uv run python -m sweeps.mpi --memkm-sites 8 --out case1
    mpirun -np 24 uv run python -m sweeps.mpi --memkm-sites 12 --out big -eps_monitor
"""

from mpi4py import MPI

from sweeps._common import (assemble, build_argparser, build_betas,
                            build_meanfield_betas, build_tasks, build_tile,
                            delta_scale_of, maybe_plot_coexistence,
                            maybe_plot_sweep, meanfield_physics_from_args,
                            params_from_args, run_coexistence, run_meanfield,
                            run_task, save_coexistence_csv, save_meanfield_csv,
                            save_sweep_csv)


def run_sweep(betas, params, seed, comm=None, delta_scale=0.0):
    """Run every kMC (beta, init) task round-robin across MPI ranks.

    Returns the {out}_kmc_sweep.csv dict on rank 0, None on every other rank
    (mirrors comm.gather's root-only result). Works with COMM_WORLD's
    default single-rank world too, e.g. when called outside of mpirun.
    """
    comm = comm or MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    tasks = build_tasks(betas, seed)
    local_tasks = [t for t in tasks if t[0] % size == rank]
    local_results = [run_task(task, params, delta_scale=delta_scale,
                              verbose_prefix=f"[rank {rank}] ")
                     for task in local_tasks]

    gathered = comm.gather(local_results, root=0)

    if rank != 0:
        return None
    results = [r for part in gathered for r in part]
    return assemble(betas, results)


def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    ap = build_argparser(__doc__.splitlines()[1])
    args, _ = ap.parse_known_args()          # let any PETSc/SLEPc options pass

    params = params_from_args(args)
    betas = build_betas(args.beta_min, args.beta_max, args.beta_step)
    dscale = delta_scale_of(args)

    # Phase A: kMC, beta-parallel.
    if args.no_kmc:
        sweep = assemble(betas, []) if rank == 0 else None
    else:
        sweep = run_sweep(betas, params, args.kmc_seed, comm=comm,
                          delta_scale=dscale)

    # Phase B: ME-MKM / SLEPc, flat collective per beta (all ranks in lockstep).
    if not args.no_memkm:
        if rank == 0:
            print("ME-MKM / SLEPc coexistence phase")
        tile = build_tile(args)
        cols, rows, arrays = run_coexistence(betas, tile, args, comm=comm)
        if rank == 0:
            sweep.update(cols)
            coex_path = args.memkm_coexistence_out or f"{args.out}_coexistence.csv"
            if save_coexistence_csv(rows, coex_path):
                print(f"Coexistence data written to '{coex_path}'.")
            if args.plot:
                maybe_plot_coexistence(cols, rows, arrays, betas, args.out)

    if rank == 0:
        save_sweep_csv(betas, sweep, args.kmc_L, f"{args.out}_kmc_sweep.csv",
                       delta_scale=dscale)
        print(f"Data written to '{args.out}_kmc_sweep.csv'.")

        # Phase C: mean-field branches (serial, rank 0 only).
        branches = None
        if not args.no_meanfield:
            betas_fine = build_meanfield_betas(betas, args.meanfield_beta_step)
            branches = run_meanfield(betas_fine, delta_scale=dscale,
                                     **meanfield_physics_from_args(args))
            save_meanfield_csv(branches, f"{args.out}_meanfield.csv")
            print(f"Mean-field branches written to '{args.out}_meanfield.csv'.")

        if args.plot:
            maybe_plot_sweep(sweep, betas, args, dscale, branches)


if __name__ == "__main__":
    main()
