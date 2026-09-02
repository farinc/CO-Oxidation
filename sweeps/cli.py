"""Typer-based CLI shared by sweeps/linear.py and sweeps/mpi.py: parses argv into a
ParsedArgs (a RunConfig base + swept axes + phase/coexistence/plotting toggles).

context_settings={"ignore_unknown_options": True, "allow_extra_args": True} on the
single command replicates argparse's old parse_known_args() passthrough: arbitrary
single-dash PETSc/SLEPc flags (e.g. -eps_monitor) are left untouched in sys.argv rather
than raising, for petsc4py's own argv-scanning init (triggered later, lazily, inside
sweeps.steps.run_memkm_for_step) to consume."""

from dataclasses import dataclass

import typer
from typer.main import get_command

from .axes import AxisSpec
from .observables import describe_available, parse_observable_options, prepare
from .params import RunConfig


@dataclass
class ParsedArgs:
    base: RunConfig
    axes: list[AxisSpec]
    out: str
    plot: bool
    save_graph: bool
    kmc: bool
    meanfield: bool
    memkm: bool
    coexistence: bool
    coexistence_axis: str
    coexistence_fixed: float | None
    memkm_order_species: str
    memkm_n_eigs_scan: int
    memkm_boundary_eps: float
    memkm_n_eigs: int
    memkm_brent_xtol: float
    memkm_factor_solver: str | None
    memkm_skip_left_modes: bool
    plot_axis: str | None
    plot_memkm_steps: str | None
    observables: list[str]
    observable_options: dict[str, dict[str, str]]
    observable_modules: list[str]


def build_app(memkm_default: bool, results: list[ParsedArgs]) -> typer.Typer:
    app = typer.Typer(add_completion=False, pretty_exceptions_enable=False)

    @app.command(context_settings={"ignore_unknown_options": True, "allow_extra_args": True})
    def _cmd(
        ctx: typer.Context,
        out: str = typer.Option(
            "co_oxidation",
            help="The prefix used for every file this run writes, including the "
                 "workbook and any figures."),
        sweep: list[str] = typer.Option(
            [], "--sweep",
            help="Add an axis to sweep over. Write it as NAME=START:STOP:STEP for an "
                 "even range, or as NAME=V1,V2,V3 for an explicit list of values. "
                 "Repeat this flag to sweep more than one axis at once. NAME can be "
                 "any physics parameter, for example k_o_ads, eps, temperature, or "
                 "memkm_sites."),
        plot: bool = typer.Option(
            True, help="Whether to render figures once the run finishes."),
        save_graph: bool = typer.Option(
            False,
            help="Write the ME-MKM coverage class transition graph out as a self "
                 "contained {out}_graph.html file you can open in a browser. This "
                 "needs the memkm phase turned on."),

        plot_axis: str = typer.Option(
            None,
            help="Which axis to put on the x axis of the bifurcation and ratio "
                 "curve plots. If you leave this unset it defaults to the "
                 "coexistence axis when coexistence ran, and otherwise to "
                 "whichever single axis you swept.",
            rich_help_panel="Plotting"),
        plot_memkm_steps: str = typer.Option(
            None,
            help='Which grid steps to plot ME-MKM snapshots for. Give "all" to '
                 "plot every step, or a comma separated list of step indices. If "
                 "you leave this unset, only the steps at coexistence crossings "
                 "are plotted.",
            rich_help_panel="Plotting"),

        observable: list[str] = typer.Option(
            [], "--observable",
            help="Also compute this observable at every grid step. Repeat the flag, "
                 "or pass a comma separated list, to ask for several. Run with "
                 "--list-observables to see what is available. Values land on an "
                 "Observables sheet in the workbook.",
            rich_help_panel="Observables"),
        observable_module: list[str] = typer.Option(
            [], "--observable-module",
            help="Import this module before the run so the observables it defines "
                 "can be named by --observable. Give an importable dotted path, or "
                 "the path to a .py file, so an observable can live in a single "
                 "script next to the run that needs it. Repeatable.",
            rich_help_panel="Observables"),
        observable_option: list[str] = typer.Option(
            [], "--observable-option",
            help="Pass a tuning value to one observable, written as "
                 "NAME.KEY=VALUE. Repeatable. Each observable documents the keys "
                 "it reads and the type it reads them as.",
            rich_help_panel="Observables"),
        list_observables: bool = typer.Option(
            False, "--list-observables",
            help="Print every registered observable, with the module(s) named by "
                 "--observable-module already loaded, and exit without running "
                 "anything.",
            rich_help_panel="Observables"),

        k_co_ads: float = typer.Option(
            1.6, help="The CO adsorption rate, in inverse seconds.",
            rich_help_panel="Physics"),
        k_co_des: float = typer.Option(
            1e-3, help="The CO desorption prefactor, in inverse seconds.",
            rich_help_panel="Physics"),
        k_o_ads: float = typer.Option(
            1.0, help="The O2 impingement rate, in inverse seconds. This is usually "
                      "the parameter you sweep.",
            rich_help_panel="Physics"),
        k_rxn: float = typer.Option(
            1.0, help="The CO plus O reaction prefactor, in inverse seconds.",
            rich_help_panel="Physics"),
        k_o_des_scale: float = typer.Option(
            1e-4, help="How reversible O2 adsorption is. The O2 desorption rate is "
                       "this value times k_o_ads, so setting it to 0.0 means O2 "
                       "adsorption never reverses.",
            rich_help_panel="Physics"),
        eps: float = typer.Option(
            8368.0, help="The nearest neighbour repulsion between two adsorbed CO "
                        "molecules, in joules per mole.",
            rich_help_panel="Physics"),
        temperature: float = typer.Option(
            500.0, "--temperature", "--temp", help="The temperature, in kelvin.",
            rich_help_panel="Physics"),
        khop_scale: float = typer.Option(
            1000.0, "--khop-scale", "--kmc-khop-scale",
            help="How much faster diffusion hops are than the fastest adsorption "
                 "process. The hop rate is this value times whichever of k_o_ads "
                 "and k_co_ads is larger.",
            rich_help_panel="Physics"),

        kmc: bool = typer.Option(
            True, "--kmc/--no-kmc",
            help="Whether to run the kMC phase at all.",
            rich_help_panel="Runners"),
        memkm: bool = typer.Option(
            memkm_default, "--memkm/--no-memkm",
            help="Whether to run the ME-MKM phase, which solves the full joint "
                 "distribution over coverages at every step with SLEPc. It's the "
                 "most expensive phase, and it's turned "
                 + ("on" if memkm_default else "off")
                 + " by default from this entry point.",
            rich_help_panel="Runners"),
        meanfield: bool = typer.Option(
            True, "--meanfield/--no-meanfield",
            help="Whether to run the mean field phase, which covers both the "
                 "MF-MKM and Bragg-Williams rate laws.",
            rich_help_panel="Runners"),

        meanfield_t_end: float = typer.Option(
            None, "--meanfield-t-end",
            help="How long the mean field integration runs, in simulated "
                 "seconds. If you leave this unset it matches the kMC time "
                 "limit above, so both phases cover the same span of simulated "
                 "time.",
            rich_help_panel="MF-MKM"),
        meanfield_dt: float = typer.Option(
            0.05, "--meanfield-dt",
            help="How often to sample the mean field trajectory, in simulated "
                 "seconds. This is only a fallback for when the kMC phase is off. "
                 "Whenever a step has kMC data, the mean field trajectory is "
                 "instead sampled to have exactly as many points as that step's "
                 "kMC trajectories, so the two line up in the workbook.",
            rich_help_panel="MF-MKM"),

        kmc_L: int = typer.Option(
            16, "--kmc-L", "--L",
            help="The lattice is a square this many sites on a side, so the total "
                 "number of sites is L squared.",
            rich_help_panel="kMC"),
        kmc_t_max: float = typer.Option(
            30.0, "--kmc-tmax", "--tmax",
            help="How long a kMC trajectory is allowed to run, in simulated "
                 "seconds. The mean field phase matches this by default too, see "
                 "meanfield-t-end below.",
            rich_help_panel="kMC"),
        kmc_max_steps: int = typer.Option(
            1_000_000_000, "--kmc-max-steps", "--max-steps",
            help="The most events a kMC trajectory is allowed to run before "
                 "stopping, whichever of this and the time limit is hit first.",
            rich_help_panel="kMC"),
        kmc_sample_interval: int = typer.Option(
            10_000, "--kmc-sample-interval", "--sample-interval",
            help="How often to record coverages while a trajectory runs, measured "
                 "in events.",
            rich_help_panel="kMC"),
        kmc_seed: int = typer.Option(
            0, "--kmc-seed", "--seed",
            help="The base random seed for the kMC trajectories. Every trajectory "
                 "gets its own seed derived from this one, so a run is fully "
                 "reproducible.",
            rich_help_panel="kMC"),
        kmc_n_trajectories: int = typer.Option(
            1, "--kmc-n-trajectories", "--n-trajectories",
            help="How many trajectories to run from each starting configuration. "
                 "Two starting configurations are always tried, an empty lattice "
                 "and a fully CO covered one, so twice this many trajectories run "
                 "at every grid step.",
            rich_help_panel="kMC"),

        memkm_sites: int = typer.Option(
            8, "--memkm-sites", "--sites",
            help="How many sites the ME-MKM tile should cover. The smallest "
                 "square tile that reaches at least this many sites is used.",
            rich_help_panel="ME-MKM"),
        memkm_order_species: str = typer.Option(
            "CO", "--memkm-order-species", "--order-species",
            help="Which species' coverage is used to orient the slow eigenvector, "
                 "so the coverage classes are ordered consistently from step to "
                 "step.",
            rich_help_panel="ME-MKM"),
        memkm_n_eigs_scan: int = typer.Option(
            4, "--memkm-n-eigs-scan", "--n-eigs-scan",
            help="How many right eigenpairs to solve at every step. This has to "
                 "be at least 3.",
            rich_help_panel="ME-MKM"),
        memkm_boundary_eps: float = typer.Option(
            0.1, "--memkm-boundary-eps", "--boundary-eps",
            help="How wide the boundary mass window is, given as a fraction of "
                 "the separation between the two plateaus.",
            rich_help_panel="ME-MKM"),
        memkm_n_eigs: int = typer.Option(
            4, "--memkm-n-eigs", "--n-eigs-report",
            help="How many left eigenpairs to solve at each coexistence "
                 "crossing.",
            rich_help_panel="ME-MKM"),
        memkm_brent_xtol: float = typer.Option(
            1e-5, "--memkm-brent-xtol", "--brent-xtol",
            help="The tolerance for Brent's method along the bisection axis, "
                 "which is searched in log space.",
            rich_help_panel="ME-MKM"),
        memkm_factor_solver: str = typer.Option(
            None, "--memkm-factor-solver", "--factor-solver",
            help="Which PETSc LU solver to use instead of the default. Choose "
                 "one of mumps, superlu_dist, pastix, or petsc.",
            rich_help_panel="ME-MKM"),
        memkm_skip_left_modes: bool = typer.Option(
            False, "--memkm-skip-left-modes",
            help="Skip solving the left eigenvector at every step. That solve is "
                 "usually the most expensive part of a large grid run, so this "
                 "speeds things up a lot, but psi_L_2 is then exported as NaN.",
            rich_help_panel="ME-MKM"),

        coexistence: bool = typer.Option(
            False, "--coexistence/--no-coexistence",
            help="Whether to search for the point where the two ME-MKM spectral "
                 "macrostates carry equal weight, using Brent's method along the "
                 "coexistence axis below. This needs the memkm phase turned on.",
            rich_help_panel="ME-MKM Coexistence"),
        coexistence_axis: str = typer.Option(
            "k_o_ads",
            help="Which swept axis the coexistence search bisects along. It has "
                 "to be a continuous physics parameter, not memkm_sites.",
            rich_help_panel="ME-MKM Coexistence"),
        coexistence_fixed: float = typer.Option(
            None,
            help="Skip the Brent search entirely and treat this value as the "
                 "coexistence point directly.",
            rich_help_panel="ME-MKM Coexistence"),
    ) -> None:
        # ignore_unknown_options/allow_extra_args (above) exist only to let single-dash
        # PETSc/SLEPc flags pass through untouched (see module docstring): a stray
        # double-dash typo would otherwise be silently swallowed as an "extra arg" and
        # the sweep would run to completion with defaults instead of erroring out.
        unrecognized = [a for a in ctx.args if a.startswith("--")]
        if unrecognized:
            raise typer.BadParameter(
                f"unrecognized option(s): {' '.join(unrecognized)}", ctx=ctx)

        # --observable accepts both "--observable a --observable b" and
        # "--observable a,b"; flatten the two forms into one list of names.
        observable_names = [n.strip() for spec in observable
                            for n in spec.split(",") if n.strip()]
        if list_observables:
            typer.echo(describe_available(observable_module))
            raise typer.Exit(0)
        try:
            # Resolve now, before any physics runs: an unknown --observable should
            # fail in the first second rather than after the ME-MKM phase.
            prepare(observable_names, observable_module)
            observable_opts = parse_observable_options(observable_option)
        except (KeyError, ValueError, ImportError, FileNotFoundError) as e:
            raise typer.BadParameter(str(e).strip('"'), ctx=ctx,
                                     param_hint="'--observable'") from e

        resolved_meanfield_t_end = (
            meanfield_t_end if meanfield_t_end is not None else kmc_t_max)
        base = RunConfig(
            k_co_ads=k_co_ads, k_co_des=k_co_des, k_o_ads=k_o_ads, k_rxn=k_rxn,
            k_o_des_scale=k_o_des_scale, eps=eps, temperature=temperature,
            khop_scale=khop_scale, kmc_L=kmc_L, kmc_t_max=kmc_t_max,
            kmc_max_steps=kmc_max_steps, kmc_sample_interval=kmc_sample_interval,
            kmc_seed=kmc_seed, kmc_n_trajectories=kmc_n_trajectories,
            meanfield_t_end=resolved_meanfield_t_end, meanfield_dt=meanfield_dt,
            memkm_sites=memkm_sites)
        try:
            axes = [AxisSpec.parse(s) for s in sweep]
        except ValueError as e:
            raise typer.BadParameter(str(e), ctx=ctx, param_hint="'--sweep'") from e
        results.append(ParsedArgs(
            base=base, axes=axes, out=out, plot=plot, save_graph=save_graph,
            kmc=kmc, meanfield=meanfield, memkm=memkm,
            coexistence=coexistence, coexistence_axis=coexistence_axis,
            coexistence_fixed=coexistence_fixed,
            memkm_order_species=memkm_order_species,
            memkm_n_eigs_scan=memkm_n_eigs_scan, memkm_boundary_eps=memkm_boundary_eps,
            memkm_n_eigs=memkm_n_eigs, memkm_brent_xtol=memkm_brent_xtol,
            memkm_factor_solver=memkm_factor_solver,
            memkm_skip_left_modes=memkm_skip_left_modes, plot_axis=plot_axis,
            plot_memkm_steps=plot_memkm_steps, observables=observable_names,
            observable_options=observable_opts,
            observable_modules=list(observable_module)))

    return app


def parse_args(argv: list[str] | None, memkm_default: bool) -> ParsedArgs:
    """Parse argv into a ParsedArgs, or exit the process (--help, --version, or a usage
    error) exactly as a standalone typer/click CLI would -- printing the formatted
    help/error to the right stream and using the matching exit code.

    We can't just call `command.main(standalone_mode=False)` and use its return value:
    in non-standalone mode click *returns* the exit code (an int) instead of exiting on
    --help or other early-exit paths, which would silently hand callers an int where
    they expect a ParsedArgs. So instead we run in standalone mode (which always raises
    SystemExit, with --help/errors already printed) and smuggle the parsed result out
    through `results`, populated only when the command body actually ran to completion.
    """
    results: list[ParsedArgs] = []
    app = build_app(memkm_default, results)
    command = get_command(app)
    try:
        command.main(args=argv, standalone_mode=True)
    except SystemExit as e:
        if results and e.code in (0, None):
            return results[0]
        raise
    raise AssertionError("typer/click always raises SystemExit from command.main()")
