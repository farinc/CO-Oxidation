"""Sweep-side glue for co_oxidation.observables: turns the CLI's --observable /
--observable-option / --observable-module flags into contexts, runs the selected
observables at each grid step, and hands back a flat {key: value} dict for StepResult.

The point of the split is that co_oxidation.observables knows nothing about sweeps
(RunConfig, grids, workbooks) and sweeps needs no per-observable code. Adding an
observable touches neither side.

Both families are driven from the same --observable list: run_kmc_observables and
run_memkm_observables each filter it down to their own family, so a user never has to
remember which of their functions acts on which simulation.
"""

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from co_oxidation.observables import (
    KMC,
    MEMKM,
    KMCContext,
    MemkmContext,
    available,
    compute,
    get,
    load_modules,
    resolve,
)

from .params import RunConfig


def parse_observable_options(specs: Iterable[str]) -> dict[str, dict[str, str]]:
    """--observable-option NAME.KEY=VALUE, repeatable, into {name: {key: value}}.

    Values stay strings here; ctx.option() casts each to the type of the default the
    observable asks with, so an observable declares its own option types."""
    options: dict[str, dict[str, str]] = {}
    for spec in specs:
        if "=" not in spec or "." not in spec.split("=", 1)[0]:
            raise ValueError(
                f"--observable-option {spec!r} must be NAME.KEY=VALUE, e.g. "
                "kmc_branching.n_trajectories=32")
        lhs, value = spec.split("=", 1)
        name, key = lhs.split(".", 1)
        options.setdefault(name, {})[key] = value
    return options


def prepare(names: Sequence[str], modules: Sequence[str]) -> list[str]:
    """Import any --observable-module plugins, then check every requested name resolves.

    Called once before the sweep starts: an unknown --observable should fail in the
    first second, not after the ME-MKM phase has been running for an hour."""
    if modules:
        load_modules(modules)
    for name in names:
        get(name)                       # raises KeyError with the list of known names
    return list(names)


def describe_available(modules: Sequence[str] = ()) -> str:
    """The --list-observables table, plugins included."""
    if modules:
        load_modules(modules)
    lines = []
    for family in (KMC, MEMKM):
        obs = available(family)
        lines.append(f"{family} observables (act on the "
                     + ("kMC trajectories" if family == KMC
                        else "solved ME-MKM generator") + "):")
        if not obs:
            lines.append("    (none registered)")
        for o in obs:
            lines.append(f"    {o.name:<22} {o.doc}")
        lines.append("")
    return "\n".join(lines)


def has_family(names: Sequence[str], family: str) -> bool:
    """Whether any requested observable belongs to `family` -- lets a driver skip
    building a context (and, for ME-MKM, keeping a state around) for nothing."""
    return bool(resolve(names, family))


def run_kmc_observables(names: Sequence[str], config: RunConfig, trajectories,
                        step_index: int, options: Mapping[str, Mapping[str, str]],
                        quiet: bool = False) -> dict[str, Any]:
    """Every kmc-family observable in `names`, at one grid step.

    `trajectories` are this step's already-run results (KMCTrajectoryResult works: the
    context only needs the attributes it shares with KMCResult). An observable that
    wants its own runs calls ctx.run_first_passage(...), whose seeds come from a stream
    offset well past the sweep's own by KMCContext.SEED_STRIDE."""
    if not has_family(names, KMC):
        return {}
    ctx = KMCContext(
        options=options, label=f"step {step_index}", k_o_ads=config.k_o_ads,
        params=config.to_kmc_params(), results=list(trajectories or ()),
        seed_base=config.kmc_seed + step_index * KMCContext.SEED_STRIDE)
    return compute(names, ctx, family=KMC,
                   on_error=(lambda e: None) if quiet else None)


def run_memkm_observables(names: Sequence[str], state, comm=None, factor=None,
                          options: Mapping[str, Mapping[str, str]] = None,
                          step_index: int | None = None,
                          quiet: bool = False) -> dict[str, Any]:
    """Every memkm-family observable in `names`, at one solved ME-MKM point.

    `state` is coexistence.solve_memkm_state's output. The context is closed on the way
    out, so any generator an observable built for a linear solve is released before the
    next step rather than piling up across the grid.

    Collective: every rank runs the same observables on the same replicated state, so
    an observable that touches the distributed generator stays in lockstep. Only rank 0
    reports failures."""
    if state is None or not has_family(names, MEMKM):
        return {}
    label = "" if step_index is None else f"step {step_index}"
    with MemkmContext(options=options or {}, label=label, state=state, comm=comm,
                      factor=factor) as ctx:
        return compute(names, ctx, family=MEMKM,
                       on_error=(lambda e: None) if quiet else None)
