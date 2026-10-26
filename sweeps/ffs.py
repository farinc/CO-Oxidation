"""
Forward flux sampling sweep for the reactive -> CO-rich (poisoning) transition.

At every grid step this runs direct FFS and reports both the FFS transition time
(Allen et al. 2006) and the FFST one (Adams et al. 2010). Physics, kMC and --sweep
options are the same as in sweeps.linear and sweeps.mpi. The FFS options all start
with --ffs-. Phase toggles such as --memkm or --observable are accepted but ignored.

Runs serially, or under mpirun, where every interface's trials are split across ranks.

Usage:
    uv run python -m sweeps.ffs --ffs-lambda-A 0.2 --ffs-lambda-B 0.7 --kmc-L 32 \\
        --sweep k_o_ads=2.0,2.2,2.4 --out ffs_case1
    mpirun -np 8 uv run python -m sweeps.ffs --ffs-lambda-A 0.2 --ffs-lambda-B 0.7 \\
        --ffs-placement dynamic --sweep kmc_L=24,32,48 --out ffs_size
    uv run python -m sweeps.ffs --ffs-list-order-parameters

The FFS algorithm in co_oxidation.ffs is a skeleton. Until it is written, each step
shows up as failed in the FFS_Summary sheet.
"""

import sys
from dataclasses import dataclass, field

import numpy as np
import typer
from typer.main import get_command

from co_oxidation import ffs_order
from co_oxidation.ffs import FFSError, FFSSettings, KMCPropagator, check_order_parameter, run_ffs

from sweeps.axes import SweepGrid
from sweeps.cli import parse_args

FLAG_OPTIONS = {"--ffs-store-configs", "--ffs-no-store-configs",
                "--ffs-list-order-parameters", "--list-order-parameters"}


@dataclass
class FFSArgs:
    lam_A: float | None = None
    lam_B: float | None = None
    lam_0: float | None = None
    lambdas: list[float] | None = None
    n_interfaces: int = 8
    placement: str = "static"
    n_trials: int = 1000
    n_crossings: int = 1000
    trial_t_max: float = 1e4
    flux_t_max: float = 1e7
    t_equil: float = 5.0
    dynamic_success: float = 0.1
    n_bootstrap: int = 200
    store_configs: bool = False
    order_parameter: str = "theta_co"
    order_modules: list[str] = field(default_factory=list)
    list_order_parameters: bool = False


def build_ffs_app(results: list[FFSArgs]) -> typer.Typer:
    app = typer.Typer(add_completion=False, pretty_exceptions_enable=False)

    @app.command(help="FFS options. Every other option is shared with sweeps.linear.")
    def _cmd(
        lam_A: float = typer.Option(
            None, "--ffs-lambda-A",
            help="Upper edge of basin A (reactive) in order parameter units."),
        lam_B: float = typer.Option(
            None, "--ffs-lambda-B",
            help="Lower edge of basin B (CO-rich). Reaching it ends a transition."),
        lam_0: float = typer.Option(
            None, "--ffs-lambda-0",
            help="First interface. Defaults to lambda_A."),
        lambdas: str = typer.Option(
            None, "--ffs-lambdas",
            help="Explicit interfaces lambda_0,...,lambda_B as a comma list. "
                 "Overrides --ffs-lambda-0 and --ffs-n-interfaces."),
        n_interfaces: int = typer.Option(
            8, "--ffs-n-interfaces",
            help="Number of evenly spaced interface steps from lambda_0 to "
                 "lambda_B, for static placement."),
        placement: str = typer.Option(
            "static", "--ffs-placement",
            help="'static' uses the interfaces given. 'dynamic' places each next "
                 "interface from pilot trials (Adams et al.)."),
        n_trials: int = typer.Option(1000, "--ffs-n-trials",
                                     help="Trials fired from each interface."),
        n_crossings: int = typer.Option(1000, "--ffs-n-crossings",
                                        help="Crossings of lambda_0 to collect in stage 1."),
        trial_t_max: float = typer.Option(
            1e4, "--ffs-trial-t-max",
            help="Longest a single trial may run, in kMC seconds. Longer trials are "
                 "dropped and reported."),
        flux_t_max: float = typer.Option(
            1e7, "--ffs-flux-t-max",
            help="Longest the stage-1 run may go, in kMC seconds."),
        t_equil: float = typer.Option(
            5.0, "--ffs-t-equil",
            help="kMC seconds to relax the stage-1 run from the empty lattice "
                 "before counting."),
        dynamic_success: float = typer.Option(
            0.1, "--ffs-dynamic-success",
            help="Target success fraction per interface for dynamic placement."),
        n_bootstrap: int = typer.Option(200, "--ffs-n-bootstrap",
                                        help="Bootstrap resamples for error bars."),
        store_configs: bool = typer.Option(
            False, "--ffs-store-configs/--ffs-no-store-configs",
            help="Keep every interface configuration and save the most likely "
                 "path's lattices to {out}_S{n}_ffs_path.npz."),
        order_parameter: str = typer.Option(
            "theta_co", "--ffs-order-parameter",
            help="Name of the order parameter. See --ffs-list-order-parameters."),
        order_module: list[str] = typer.Option(
            [], "--ffs-order-module",
            help="Module or .py file that registers extra order parameters. "
                 "Repeatable."),
        list_order_parameters: bool = typer.Option(
            False, "--ffs-list-order-parameters", "--list-order-parameters",
            help="Print the registered order parameters and exit."),
    ) -> None:
        parsed_lambdas = ([float(v) for v in lambdas.split(",") if v.strip()]
                          if lambdas else None)
        results.append(FFSArgs(
            lam_A=lam_A, lam_B=lam_B, lam_0=lam_0, lambdas=parsed_lambdas,
            n_interfaces=n_interfaces, placement=placement, n_trials=n_trials,
            n_crossings=n_crossings, trial_t_max=trial_t_max, flux_t_max=flux_t_max,
            t_equil=t_equil, dynamic_success=dynamic_success, n_bootstrap=n_bootstrap,
            store_configs=store_configs, order_parameter=order_parameter,
            order_modules=list(order_module),
            list_order_parameters=list_order_parameters))

    return app


def split_argv(argv: list[str]) -> tuple[list[str], list[str]]:
    """Separate the --ffs-* options from the shared ones."""
    ffs_args, rest = [], []
    i = 0
    while i < len(argv):
        token = argv[i]
        name = token.split("=", 1)[0]
        if name.startswith("--ffs-") or name in FLAG_OPTIONS:
            ffs_args.append(token)
            if "=" not in token and name not in FLAG_OPTIONS and i + 1 < len(argv):
                ffs_args.append(argv[i + 1])
                i += 1
        else:
            rest.append(token)
        i += 1
    return ffs_args, rest


def parse_ffs_args(argv: list[str]) -> FFSArgs:
    results: list[FFSArgs] = []
    command = get_command(build_ffs_app(results))
    try:
        command.main(args=argv, prog_name="python -m sweeps.ffs", standalone_mode=True)
    except SystemExit as e:
        if results and e.code in (0, None):
            return results[0]
        raise
    raise AssertionError("click always raises SystemExit from command.main()")


def settings_for(args: FFSArgs, n_sites: int, kind: str) -> FFSSettings:
    """FFSSettings for one grid step. Dynamic spacing is at least one site for count
    order parameters, so interfaces never sit closer than the lattice can resolve."""
    lam_0 = args.lam_0 if args.lam_0 is not None else args.lam_A
    if args.placement == "static":
        lambdas = (tuple(args.lambdas) if args.lambdas else
                   tuple(np.linspace(lam_0, args.lam_B, args.n_interfaces + 1)))
    else:
        lambdas = None
    return FFSSettings(
        lam_A=args.lam_A, lam_B=args.lam_B, lambdas=lambdas, lam_0=lam_0,
        placement=args.placement, n_crossings=args.n_crossings, n_trials=args.n_trials,
        trial_t_max=args.trial_t_max, flux_t_max=args.flux_t_max, t_equil=args.t_equil,
        dynamic_success=args.dynamic_success, n_bootstrap=args.n_bootstrap,
        store_configs=args.store_configs,
        min_spacing=1.0 / n_sites if kind == ffs_order.COUNTS else 0.0)


def make_mpi_map(comm):
    """map_fn that splits items round-robin over ranks and allgathers the results.

    Every rank calls it with the same items and gets the full result list back, so the
    orchestration in co_oxidation.ffs runs in lockstep on all ranks."""
    rank, size = comm.Get_rank(), comm.Get_size()

    def mpi_map(fn, items):
        local = [(k, fn(item)) for k, item in enumerate(items) if k % size == rank]
        out = [None] * len(items)
        for part in comm.allgather(local):
            for k, value in part:
                out[k] = value
        return out

    return mpi_map


def _get_comm():
    try:
        from mpi4py import MPI
    except ImportError:
        return None
    return MPI.COMM_WORLD if MPI.COMM_WORLD.Get_size() > 1 else None


def main(argv: list[str] | None = None):
    argv = list(sys.argv[1:] if argv is None else argv)
    ffs_argv, rest = split_argv(argv)
    if "--help" in rest or "-h" in rest:
        try:
            parse_ffs_args(["--help"])
        except SystemExit:
            pass
        print("\nShared physics, kMC and sweep options:\n")
        parse_args(["--help"], memkm_default=False)
    fargs = parse_ffs_args(ffs_argv)
    ffs_order.load_modules(fargs.order_modules)
    if fargs.list_order_parameters:
        print(ffs_order.describe_available())
        return
    if fargs.lam_A is None or fargs.lam_B is None:
        raise SystemExit("sweeps.ffs needs --ffs-lambda-A and --ffs-lambda-B")
    order = ffs_order.get(fargs.order_parameter)
    args = parse_args(rest, memkm_default=False)
    grid = SweepGrid.build(args.base, args.axes)

    comm = _get_comm()
    rank = comm.Get_rank() if comm is not None else 0
    size = comm.Get_size() if comm is not None else 1
    map_fn = make_mpi_map(comm) if comm is not None else None
    say = print if rank == 0 else (lambda msg: None)

    results = []
    for grid_step in grid.steps:
        cfg = grid_step.config
        params = cfg.to_kmc_params()
        prop = KMCPropagator(cfg.k_o_ads, params, order)
        settings = settings_for(fargs, prop.n_sites, order.kind)
        say(f"[ffs] step {grid_step.index} {grid_step.axis_values or ''} "
            f"(k_o_ads={cfg.k_o_ads}, L={cfg.kmc_L})")
        try:
            check_order_parameter(prop, settings)
            kwargs = {"map_fn": map_fn} if map_fn is not None else {}
            result = run_ffs(prop, settings, base_seed=cfg.kmc_seed,
                             step_index=grid_step.index, n_flux_runs=size,
                             progress=lambda m: say(f"  {m}"), **kwargs)
        except (FFSError, ValueError, NotImplementedError) as e:
            # The FFS skeleton raises NotImplementedError. Record the step as failed
            # and carry on with the rest of the sweep.
            say(f"  [ffs] step {grid_step.index} failed: {e}")
            result = e
        results.append((grid_step, result))

    if rank != 0:
        return

    from sweeps.excel import write_ffs_workbook
    settings_rows = {"order_parameter": order.name, "order_kind": order.kind,
                     "order_units": order.units, "mpi_ranks": size,
                     **{k: v for k, v in vars(fargs).items()
                        if k not in ("list_order_parameters",)}}
    out_path = f"{args.out}.xlsx"
    write_ffs_workbook(grid, results, settings_rows, out_path)
    print(f"Data written to '{out_path}'.")

    for grid_step, result in results:
        if not isinstance(result, Exception) and result.path_configs is not None:
            path_file = f"{args.out}_S{grid_step.index}_ffs_path.npz"
            np.savez_compressed(path_file, lattices=result.path_configs,
                                lambdas=np.array(result.lambdas), L=grid_step.config.kmc_L)
            print(f"  [ffs] most likely path written to '{path_file}'")

    if args.plot:
        from sweeps.plotting import plot_ffs
        for path in plot_ffs(grid, results, args.out, args.plot_axis):
            print(f"  [plot] wrote {path}")


if __name__ == "__main__":
    main()
