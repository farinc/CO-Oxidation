"""Coexistence: a separate, later pass over a finished sweep grid that Brent-searches
one user-designated continuous axis for the point(s) where the two ME-MKM spectral
macrostates carry equal stationary weight, independently for every combination of the
grid's *other* swept axes."""

from collections import OrderedDict

import numpy as np

from .axes import SweepGrid
from .results import CoexistenceCrossing
from .steps import build_tile

DISCRETE_MEMKM_AXES = frozenset({"memkm_sites"})


def validate_bisection_axis(grid: SweepGrid, bisection_axis: str, coexistence_enabled: bool,
                            fixed_value: float | None = None) -> None:
    """Raise ValueError if coexistence_enabled and the request can't produce a valid
    Brent search:
      - bisection_axis is a discrete/structural ME-MKM parameter (site count changes
        the microstate space itself, so the pipeline can't be evaluated "between" two
        site counts and Brent's method requires continuity);
      - bisection_axis isn't a swept axis and no fixed_value was given (there is then
        no scan to bracket a sign change from).
    A fixed (non-swept) memkm_sites value is still fully supported for building the
    tile; the first check only fires when it is *also* the designated bisection axis.
    """
    if not coexistence_enabled:
        return
    if bisection_axis in DISCRETE_MEMKM_AXES:
        raise ValueError(
            f"--coexistence-axis {bisection_axis} is invalid: {bisection_axis} is a "
            "discrete ME-MKM structural parameter that changes the microstate space "
            "itself (via TileSettings.smallest_valid_square), so the coexistence "
            "pipeline cannot be evaluated 'between' two values of it and Brent's "
            "method requires continuity. Keep it fixed (a scalar, e.g. --memkm-sites "
            "8) or sweep it only as an *outer* axis, and pick a continuous physics "
            "axis (k_o_ads, eps, temperature, k_co_ads, k_co_des, k_rxn, "
            "k_o_des_scale, khop_scale) as --coexistence-axis.")
    swept_names = {a.name for a in grid.axes}
    if bisection_axis not in swept_names and fixed_value is None:
        raise ValueError(
            f"--coexistence-axis {bisection_axis} is not a swept axis (no "
            f"--sweep {bisection_axis}=... was given), so there is no scan to "
            "bracket a coexistence crossing from. Either sweep it or pass "
            "--coexistence-fixed VALUE to evaluate a single report() there directly.")


def _slices(grid: SweepGrid, bisection_axis: str):
    """Group grid.steps by every swept axis EXCEPT bisection_axis. Each group ("slice")
    shares one combination of the other swept axes' values; a grid with only
    bisection_axis swept (or nothing swept) yields exactly one slice."""
    other_names = [a.name for a in grid.axes if a.name != bisection_axis]
    groups: "OrderedDict[tuple, list]" = OrderedDict()
    for step in grid.steps:
        key = tuple(step.axis_values[name] for name in other_names)
        groups.setdefault(key, []).append(step)
    return [(dict(zip(other_names, key)), steps) for key, steps in groups.items()]


def run_coexistence_for_grid(grid: SweepGrid, bisection_axis: str, comm=None,
                              n_eigs_scan: int = 4, boundary_eps: float = 0.1,
                              n_eigs: int = 4, xtol: float = 1e-5, factor=None,
                              order_species: str = "CO",
                              fixed_value: float | None = None) -> list[CoexistenceCrossing]:
    """One independent CoexistencePipeline + Brent search per slice (see _slices);
    each slice's tile is built from that slice's own resolved memkm_sites (so
    memkm_sites may legally be an *outer* swept axis, just never the bisection axis --
    validate_bisection_axis enforces that). Scans the slice's own bisection-axis grid
    values (no separate scan-resolution flag needed) unless `fixed_value` is given, in
    which case that value stands in for the crossing directly and the scan/Brent step
    is skipped (the CLI's --coexistence-fixed shortcut). Collective across `comm`: every
    rank runs the identical Brent search in lockstep since each objective evaluation is
    a globally-consistent collective solve.
    """
    from co_oxidation.memkm import CoexistencePipeline

    validate_bisection_axis(grid, bisection_axis, True, fixed_value)
    rank = comm.Get_rank() if comm is not None else 0

    tile_cache: dict[int, object] = {}

    def tile_for(sites):
        if sites not in tile_cache:
            tile_cache[sites] = build_tile(sites)
        return tile_cache[sites]

    crossings: list[CoexistenceCrossing] = []
    for outer_axis_values, steps in _slices(grid, bisection_axis):
        base_config = steps[0].config
        tile = tile_for(base_config.memkm_sites)
        pipe = CoexistencePipeline(
            tile, axis_field=bisection_axis, base_kwargs=base_config.to_memkm_kwargs(),
            comm=comm, order_species=order_species, n_eigs_scan=n_eigs_scan,
            boundary_eps=boundary_eps, factor=factor)

        if fixed_value is not None:
            stars = [float(fixed_value)]
            scan_values = np.array([float(fixed_value)])
            scan_log_ratios = np.array([np.nan])
            if rank == 0:
                print(f"  [coexistence] fixed {bisection_axis}={stars[0]:.6g} "
                      "(no Brent search)", flush=True)
        else:
            scan_values = np.array([s.axis_values[bisection_axis] for s in steps], float)
            scan_log_ratios = np.full(scan_values.shape, np.nan)
            for i, x in enumerate(scan_values):
                try:
                    scan_log_ratios[i] = pipe.basin_log_ratio(x)
                except Exception as exc:   # deterministic across ranks -> lockstep safe
                    if rank == 0:
                        print(f"  [coexistence] {bisection_axis}={x:.6g}: solve "
                              f"skipped ({exc})", flush=True)
            stars = pipe.find_coexistence(scan_values, scan_log_ratios, xtol=xtol)
            if rank == 0:
                where = f" ({outer_axis_values})" if outer_axis_values else ""
                print(f"  [coexistence] {bisection_axis}* crossing(s){where}: "
                      f"{', '.join(f'{s:.6g}' for s in stars) or 'none'}", flush=True)

        for x_star in stars:
            row, arrays = pipe.report(x_star, n_eigs=n_eigs)
            crossings.append(CoexistenceCrossing(
                outer_axis_values=outer_axis_values, bisection_axis=bisection_axis,
                value_star=x_star, row=row, arrays=arrays, scan_values=scan_values,
                scan_log_ratios=scan_log_ratios))
    return crossings
