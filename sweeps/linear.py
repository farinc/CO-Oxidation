"""
Serial beta sweep: runs the kMC cases, the mean-field branches, and (opt-in)
the ME-MKM / SLEPc coexistence analysis on a single core, writing
{out}_kmc_sweep.csv (kMC coverages, plus the ME-MKM columns if --memkm is on),
{out}_coexistence.csv (only with --memkm: transition rates at each beta*) and
{out}_meanfield.csv (the MF-MK / Ea-MK branches). Each phase can be toggled
independently with --no-kmc / --memkm / --no-meanfield.

The ME-MKM phase is off by default here: even a small tile is too slow for a
laptop-run sweep. sweeps/mpi.py enables it by default since that's meant to run on a cluster.

Usage:
    uv run python -m sweeps.linear
    uv run python -m sweeps.linear --kmc-L 24 --out case1
    uv run python -m sweeps.linear --memkm --memkm-sites 8     # + ME-MKM/SLEPc
    uv run python -m sweeps.linear --no-kmc --memkm            # ME-MKM only
"""

from sweeps._common import (
    assemble,
    build_argparser,
    build_betas,
    build_meanfield_betas,
    build_tasks,
    build_tile,
    delta_scale_of,
    maybe_plot_coexistence,
    maybe_plot_sweep,
    meanfield_physics_from_args,
    params_from_args,
    run_coexistence,
    run_meanfield,
    run_task,
    save_coexistence_csv,
    save_meanfield_csv,
    save_sweep_csv,
)


def run_sweep(betas, params, seed, delta_scale=0.0):
    """Run every (beta, init) task serially. Returns the {out}_kmc_sweep.csv dict."""
    tasks = build_tasks(betas, seed)
    results = [run_task(task, params, delta_scale=delta_scale)
              for task in tasks]
    return assemble(betas, results)


def main():
    ap = build_argparser(__doc__.splitlines()[1], memkm_default=False)
    args, _ = ap.parse_known_args()          # let any PETSc/SLEPc options pass

    params = params_from_args(args)
    betas = build_betas(args.beta_min, args.beta_max, args.beta_step)
    dscale = delta_scale_of(args)

    # Phase A: kMC.
    if args.no_kmc:
        sweep = assemble(betas, [])          # all-NaN kMC columns
    else:
        sweep = run_sweep(betas, params, args.kmc_seed, delta_scale=dscale)

    # Phase B: ME-MKM / SLEPc coexistence (off by default, see --memkm).
    if args.memkm:
        tile = build_tile(args)
        print("ME-MKM / SLEPc coexistence phase")
        cols, rows, arrays = run_coexistence(betas, tile, args, comm=None)
        sweep.update(cols)
        coex_path = args.memkm_coexistence_out or f"{args.out}_coexistence.csv"
        if save_coexistence_csv(rows, coex_path):
            print(f"Coexistence data written to '{coex_path}'.")
        if args.plot:
            maybe_plot_coexistence(cols, rows, arrays, betas, args.out)

    save_sweep_csv(betas, sweep, args.kmc_L, f"{args.out}_kmc_sweep.csv",
                   delta_scale=dscale)
    print(f"Data written to '{args.out}_kmc_sweep.csv'.")

    # Phase C: mean-field branches on a filled-in grid through the sweep betas.
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
