"""Plotting: the four kept plot families, all reading from the same StepResult/
SweepResult/CoexistenceCrossing objects the Excel writer (sweeps.excel) uses.

  - coverage maps    : plot_coverage_map      (population, psi_R_2, psi_L_2)
  - eigenspectrum     : plot_eigenspectrum     (sorted/partitioned only)
  - ratio curve       : plot_ratio_curve       (only when coexistence ran)
  - bifurcation       : plot_bifurcation       (kMC + mean-field + ME-MKM vs one axis)

plot_all orchestrates all four with the N-D grid faceting rule (see its docstring) and
is what both the in-process --plot path (sweeps/linear.py, sweeps/mpi.py, on the live
SweepResult) and the standalone replot CLI below (on a SweepResult read back from a
saved .xlsx) call.

Usage (standalone replot):
    uv run python -m sweeps.plotting co_oxidation.xlsx
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle

from .excel import read_workbook
from .results import CoexistenceCrossing, MemkmStepResult, StepResult, SweepResult

sns.set_theme(style="white", context="talk")

MODEL_LABELS = {"mf": "MF-MKM", "ea": "Bragg-Williams"}
MODEL_COLORS = {"mf": None, "ea": "green"}


# --- coverage maps ---------------------------------------------------------------

def _coverage_pcolor(ax, grid, l, cmap, norm=None, vmin=None, vmax=None):
    """Draw one coverage-class grid on `ax` as a brick-stack pyramid: row N_O (of
    length l+1-N_O) is offset right by N_O/2 as a whole, centering it under the
    full-width N_O=0 row, so the accessible region reads as a symmetric triangle of
    unit *squares*. Each valid (n_CO, n_O) class is one square, drawn individually."""
    n_co, n_o = np.where(np.isfinite(grid))
    vals = grid[n_co, n_o]
    squares = [Rectangle((co + o / 2 - 0.5, o - 0.5), 1, 1) for co, o in zip(n_co, n_o)]
    im = PatchCollection(squares, cmap=cmap, edgecolor="0.35", linewidth=1.2)
    im.set_array(vals)
    if norm is not None:
        im.set_norm(norm)
    else:
        im.set_clim(vmin, vmax)
    ax.add_collection(im)
    ax.set_aspect(np.sqrt(3) / 2)
    ax.set_xlabel(r"$N_\mathrm{CO}$")
    ax.set_ylabel(r"$N_\mathrm{O}$")
    ax.set_xticks([])
    ax.set_yticks(np.arange(0, l + 1))
    ax.set_xlim(-0.5, l + 0.5)
    ax.set_ylim(-0.5, l + 0.5)
    return im


def _inset_colorbar(fig, ax, im, label=None):
    return fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=label)


def _diverging_norm(grid, center=0.0):
    if np.isfinite(grid).any() and np.nanmin(grid) < 0 < np.nanmax(grid):
        from matplotlib.colors import TwoSlopeNorm
        return TwoSlopeNorm(vcenter=center, vmin=np.nanmin(grid), vmax=np.nanmax(grid))
    return None


def plot_coverage_map(memkm: MemkmStepResult, out_path_prefix: str) -> list[str]:
    """Coverage-class maps over (N_CO, N_O): population, the slow right mode sum
    (psi_R_2, whose sign is the macrostate partition), and the slow left mode class
    mean (psi_L_2, the reaction coordinate -- skipped if not solved)."""
    l = memkm.n_sites
    a = np.arange(l + 1)
    outside = (a[:, None] + a[None, :]) > l
    pop_m = np.where((memkm.cov_pop > 0) & ~outside, memkm.cov_pop, np.nan)
    empty = outside | (memkm.cov_deg <= 0)
    r2_m = np.where(empty, np.nan, memkm.cov_r2)
    figsize, dpi = (12, 5.5), 150
    written = []

    fig, ax = plt.subplots(1, 1, figsize=figsize, constrained_layout=True)
    im0 = _coverage_pcolor(ax, pop_m, l, "Blues")
    ax.set_title("Stationary Distribution by Coverage-class", pad=10)
    _inset_colorbar(fig, ax, im0, label=r"$P(N_\mathrm{CO}, N_\mathrm{O})$")
    path = f"{out_path_prefix}_coverage-population.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    written.append(path)

    fig, ax = plt.subplots(1, 1, figsize=figsize, constrained_layout=True)
    im1 = _coverage_pcolor(ax, r2_m, l, "coolwarm", norm=_diverging_norm(r2_m))
    ax.set_title(r"Slowest Relaxation Mode $\Psi_2^R(n_\mathrm{CO},n_\mathrm{O})$", pad=10)
    _inset_colorbar(fig, ax, im1, label=r"$\Psi_2^R(n_\mathrm{CO},n_\mathrm{O})$")
    path = f"{out_path_prefix}_coverage-psi_R_2.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    written.append(path)

    if memkm.cov_phi is not None:
        phi_m = np.where(outside, np.nan, memkm.cov_phi)
        fig, ax = plt.subplots(1, 1, figsize=figsize, constrained_layout=True)
        im2 = _coverage_pcolor(ax, phi_m, l, "coolwarm", norm=_diverging_norm(phi_m))
        ax.set_title(r"Reaction Coordinate $\Psi_2^L(n_\mathrm{CO},n_\mathrm{O})$", pad=10)
        _inset_colorbar(fig, ax, im2, label=r"$\Psi_2^L(n_\mathrm{CO},n_\mathrm{O})$")
        path = f"{out_path_prefix}_coverage-psi_L_2.png"
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        written.append(path)

    return written


# --- eigenspectrum -----------------------------------------------------------------

_EIGENSPECTRUM_KEYS = ("phi_slow", "vecs_R", "psi_L_2", "psi_R_2", "theta", "in_A", "in_B")


def plot_eigenspectrum(arrays: dict, out_path: str) -> str | None:
    """Sorted/partitioned eigenspectrum: the eigenvalue bar chart on top, the slowest
    left eigenvectors in the left column and right eigenvectors in the right column,
    each basin drawn as its own segment split at the A/B boundary.

    Needs the raw per-microstate arrays from CoexistencePipeline.report() -- present
    when plotting chains directly off a live SweepResult, but not reconstructable from
    a saved .xlsx (sweeps.excel only carries the adsorbate-count joint distributions,
    not the per-microstate vectors), in which case this prints a note and returns None.
    """
    if any(k not in arrays for k in _EIGENSPECTRUM_KEYS):
        print(f"  [plot] skipping eigenspectrum for {out_path}: per-microstate "
              "eigenvectors not available (likely loaded from a saved workbook)")
        return None

    eigvals = np.sort(arrays["eigvals"])[::-1].copy()
    eigvals.real[np.abs(eigvals.real) < 1e-12] = 0.0
    eigvals.imag[np.abs(eigvals.imag) < 1e-12] = 0.0
    phi_slow, vecs_R = arrays["phi_slow"], arrays["vecs_R"]
    psi_L_2, psi_R_2 = arrays["psi_L_2"], arrays["psi_R_2"]
    in_A, in_B = arrays["in_A"], arrays["in_B"]
    palette = sns.color_palette("deep")
    K = len(eigvals)
    Ks = np.arange(1, K + 1)

    idx_A, idx_B = np.where(in_A)[0], np.where(in_B)[0]
    idx_A = idx_A[np.argsort(psi_L_2[idx_A])]
    idx_B = idx_B[np.argsort(psi_L_2[idx_B])]
    state_order = np.concatenate([idx_A, idx_B])
    x = np.concatenate([
        np.linspace(0.0, 1.0, len(idx_A), endpoint=False),
        1.0 + np.linspace(0.0, 1.0, len(idx_B), endpoint=False),
    ])
    n_panel = K
    n_panel_R = min(n_panel, vecs_R.shape[1])
    row_h, width, top_units = 2, 11.0, 2
    nA = len(idx_A)

    fig = plt.figure(figsize=(width, row_h * (n_panel + top_units)),
                     constrained_layout=True)
    gs = fig.add_gridspec(n_panel + 1, 2, hspace=0.05, wspace=0.05,
                          height_ratios=[top_units] + [1] * n_panel)
    ax_re = fig.add_subplot(gs[0, :])
    bars = ax_re.bar(Ks, eigvals.real, color=[palette[m] for m in range(K)])
    bar_labels = [f"{v:.2f}" if v != 0.0 else "" for v in eigvals.real]
    ax_re.bar_label(bars, labels=bar_labels, label_type="center")
    ax_re.set_title("Real Component of Eigenvalues")
    max_yticks = 3
    nonzero = np.abs(eigvals.real[eigvals.real != 0.0])
    linthresh = nonzero.min() if nonzero.size else 1.0
    ax_re.set_yscale("symlog", linthresh=linthresh)
    ax_re.set_ylim(top=linthresh)
    ax_re.axhline(0, color="k", lw=0.8)
    ax_re.set_xticks(Ks)
    ax_re.set_xticklabels([rf"$\lambda_{{{i}}}$" for i in Ks])
    ticks = ax_re.yaxis.get_majorticklocs()
    ticks = ticks[np.abs(ticks) > linthresh]
    if len(ticks) > max_yticks:
        keep = np.unique(np.linspace(0, len(ticks) - 1, max_yticks).round().astype(int))
        ticks = ticks[keep]
    ax_re.set_yticks(ticks)

    def _draw_column(col, vecs, n_col, symbol, sign_ref, sharex):
        axes = [fig.add_subplot(gs[1, col], sharex=sharex)]
        axes += [fig.add_subplot(gs[i + 1, col], sharex=axes[0], sharey=axes[0])
                for i in range(1, n_col)]
        for m, ax in enumerate(axes):
            psi = vecs[:, m].real
            if np.dot(psi, sign_ref) < 0:
                psi = -psi
            for x_seg, psi_seg in ((x[:nA], psi[state_order][:nA]),
                                   (x[nA:], psi[state_order][nA:])):
                if len(x_seg) == 0:
                    continue
                ax.plot(x_seg, psi_seg, lw=0.9, color=palette[m],
                        marker="o" if len(x_seg) < 20 else None, ms=3)
            ax.axhline(0.0, color="0.8", lw=0.8)
            ax.axvline(1.0, color="black", ls="--", lw=1.0)
            ax.set_title(rf"$\Psi_{m + 1}^{symbol}$")
            ax.set_xlim(0.0, 2.0)
            ax.set_xticks([0.0, 1.0, 2.0])
            ax.set_xticklabels([])
            ax.tick_params(axis="x", which="minor", length=0, labelbottom=False)
        axes[-1].set_xticks([0.5, 1.5], minor=True)
        axes[-1].set_xticklabels([r"$\Psi_2^R < 0$", r"$\Psi_2^R \geq 0$"], minor=True)
        axes[-1].tick_params(axis="x", which="minor", labelbottom=True)
        return axes

    eig_axes_L = _draw_column(0, phi_slow, n_panel, "L", psi_L_2, sharex=None)
    _draw_column(1, vecs_R, n_panel_R, "R", psi_R_2, sharex=eig_axes_L[0])

    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


# --- ratio curve ---------------------------------------------------------------------

def plot_ratio_curve(crossing: CoexistenceCrossing, out_path: str) -> str:
    """Basin log-ratio and P_A/P_B vs. the bisection axis, marking the coexistence
    point. P_A is recovered from the log-ratio scan via the logistic identity
    P_A = 1/(1+exp(-log_ratio)), since log_ratio = ln(P_A/P_B) and P_A+P_B=1."""
    x = np.asarray(crossing.scan_values, float)
    lr = np.asarray(crossing.scan_log_ratios, float)
    good = np.isfinite(lr)
    order = np.argsort(x[good])
    x_good, lr_good = x[good][order], lr[good][order]
    palette = sns.color_palette("deep")

    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(7, 7))
    axes[0].plot(x_good, lr_good, "-o", color=palette[0])
    axes[0].axhline(0.0, color="0.6", lw=1)
    axes[0].annotate(rf"{crossing.bisection_axis}* = {crossing.value_star:.4g}",
                     (crossing.value_star, 0.0), textcoords="offset points",
                     xytext=(8, 8))
    axes[0].set_ylabel(r"$\ln\,p_{ss}(A)/p_{ss}(B)$")
    axes[0].set_title(f"Spectral macrostate weights vs. {crossing.bisection_axis}")

    P_A = 1.0 / (1.0 + np.exp(-lr_good))
    axes[1].plot(x_good, P_A, "-o", color=palette[0], label=r"$P_A$")
    axes[1].plot(x_good, 1.0 - P_A, "-o", color=palette[1], label=r"$P_B$")
    axes[1].axhline(0.5, color="0.6", lw=1)
    axes[1].set_ylabel("spectral macrostate weight")
    axes[1].set_ylim(-0.02, 1.02)
    axes[1].legend()
    for ax in axes:
        ax.axvline(crossing.value_star, color="0.6", lw=1, ls="--")
    axes[-1].set_xlabel(crossing.bisection_axis)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


# --- bifurcation ---------------------------------------------------------------------

def plot_bifurcation(steps: list[StepResult], plot_axis: str, out_path: str,
                     title: str | None = None) -> str:
    """kMC hysteresis scatter + MF-MKM/Bragg-Williams branch lines (steady-state
    proxy: the last time-sample of each model's empty/full trajectory) + ME-MKM
    spectral-basin coverages, vs. plot_axis, over one group of steps (every other
    swept axis fixed within the group)."""
    ordered = sorted(steps, key=lambda s: s.step.config.__dict__[plot_axis])
    xs = np.array([s.step.config.__dict__[plot_axis] for s in ordered], float)

    fig, axes = plt.subplots(2, 1, figsize=(8, 9), sharex=True, layout="constrained")
    for ax, key, ylabel in ((axes[0], "co", r"$\theta_{CO}$"),
                            (axes[1], "o", r"$\theta_O$")):
        for model, label in MODEL_LABELS.items():
            full_x, full_y, empty_x, empty_y = [], [], [], []
            for x, s in zip(xs, ordered):
                if s.meanfield is None:
                    continue
                for r in s.meanfield:
                    if r.model != model:
                        continue
                    val = (r.theta_co if key == "co" else r.theta_o)[-1]
                    (full_x if r.tag == "full" else empty_x).append(x)
                    (full_y if r.tag == "full" else empty_y).append(val)
            color = MODEL_COLORS[model]
            if full_x:
                line, = ax.plot(full_x, full_y, "-", color=color,
                               label=f"{label} (CO start)")
                color = line.get_color()
            if empty_x:
                ax.plot(empty_x, empty_y, "--", color=color, label=f"{label} (empty start)")

        for tag, marker, marker_label in (("full", "o", "CO-covered start"),
                                          ("empty", "s", "empty start")):
            kx, ky = [], []
            for x, s in zip(xs, ordered):
                if s.kmc is None:
                    continue
                for t in s.kmc:
                    if t.tag != tag:
                        continue
                    kx.append(x)
                    ky.append(t.steady_co if key == "co" else t.steady_o)
            if kx:
                ax.scatter(kx, ky, marker=marker, zorder=5, label=f"kMC ({marker_label})")

        for basin, style in (("a", "-"), ("b", "--")):
            mx, my = [], []
            for x, s in zip(xs, ordered):
                if s.memkm is None:
                    continue
                mx.append(x)
                my.append(getattr(s.memkm, f"{key}_{basin}"))
            if mx:
                ax.plot(mx, my, style, color="0.2", lw=1.4, zorder=4,
                       label=f"ME-MKM basin {basin.upper()}")

        ax.set_ylabel(ylabel)
        ax.set_ylim(-0.02, 1.02)
    axes[-1].set_xlabel(plot_axis)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=3, loc="outside upper center")
    if title:
        fig.suptitle(title)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


# --- orchestration + N-D faceting ------------------------------------------------------

def group_steps(steps: list[StepResult], plot_axis: str) -> list[tuple[dict, list[StepResult]]]:
    """Group `steps` by the value-tuple of every swept axis EXCEPT plot_axis; one
    group == one facet/figure. A grid with plot_axis as the only swept axis yields
    exactly one group."""
    other_names = sorted({n for s in steps for n in s.step.axis_values if n != plot_axis})
    groups: dict[tuple, list[StepResult]] = {}
    order = []
    for s in steps:
        key = tuple(s.step.axis_values[n] for n in other_names)
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append(s)
    return [(dict(zip(other_names, key)), groups[key]) for key in order]


def _resolve_plot_axis(result: SweepResult, plot_axis: str | None) -> str | None:
    if plot_axis:
        return plot_axis
    if result.coexistence:
        return result.coexistence[0].bisection_axis
    if len(result.grid.axes) == 1:
        return result.grid.axes[0].name
    return None


def plot_all(result: SweepResult, out_prefix: str, plot_axis: str | None = None,
            plot_memkm_steps: str | None = None) -> list[str]:
    """Concrete N-D faceting rule:
      - plot_axis resolves (see _resolve_plot_axis) to the coexistence bisection axis
        if coexistence ran, else the explicit argument, else the sole swept axis; with
        more than one swept axis and no coexistence, an explicit plot_axis is required.
      - bifurcation: one figure per group from group_steps(); "{out}_bifurcation.png"
        if there's exactly one group, else "_g{i}.png" (the fixed axis=value context
        for that group is printed and put in the figure title; still traceable via
        the workbook's Grid sheet by step index).
      - ratio-curve: one figure per CoexistenceCrossing (each already is one
        outer-axis slice), only produced when coexistence ran.
      - eigenspectrum + coverage maps: per-step/per-crossing snapshots, not grouped by
        axis. If coexistence ran: one snapshot per crossing (on by default). Otherwise:
        skipped by default (still exported to Excel regardless) unless
        plot_memkm_steps is "all" or a comma-separated list of step indices.
    """
    written = []
    axis = _resolve_plot_axis(result, plot_axis)
    if axis is None:
        print("  [plot] no plot axis resolved (more than one swept axis and no "
              "coexistence ran); pass --plot-axis to enable bifurcation/ratio-curve "
              "plots. Skipping.")
    else:
        groups = group_steps(result.steps, axis)
        for i, (fixed, group) in enumerate(groups):
            suffix = "" if len(groups) == 1 else f"_g{i}"
            path = f"{out_prefix}_bifurcation{suffix}.png"
            title = ", ".join(f"{k}={v:g}" for k, v in fixed.items()) or None
            if title:
                print(f"  [plot] group {i}: {title}")
            written.append(plot_bifurcation(group, axis, path, title=title))

    if result.coexistence:
        for i, crossing in enumerate(result.coexistence):
            tag = "" if len(result.coexistence) == 1 else f"_{i}"
            written.append(plot_ratio_curve(crossing, f"{out_prefix}_ratio-curve{tag}.png"))
            snap_prefix = f"{out_prefix}_coexistence{tag}"
            written.extend(plot_coverage_map(MemkmStepResult(
                eigvals=crossing.arrays["eigvals"], cov_pop=crossing.arrays["cov_pop"],
                cov_r2=crossing.arrays["cov_r2"], cov_phi=crossing.arrays["cov_phi"],
                cov_deg=crossing.arrays["cov_deg"], theta_empty=0.0, theta_co=0.0,
                theta_o=0.0, n_sites=crossing.arrays["n_sites"], p_a=crossing.row["P_A"],
                empty_a=0.0, co_a=0.0, o_a=0.0, empty_b=0.0, co_b=0.0, o_b=0.0),
                snap_prefix))
            eig_path = plot_eigenspectrum(crossing.arrays, f"{snap_prefix}_eigenspectrum.png")
            if eig_path:
                written.append(eig_path)
    elif plot_memkm_steps:
        wanted = (None if plot_memkm_steps.strip().lower() == "all"
                 else {int(v) for v in plot_memkm_steps.split(",")})
        for s in result.steps:
            if s.memkm is None or (wanted is not None and s.step.index not in wanted):
                continue
            snap_prefix = f"{out_prefix}_S{s.step.index}"
            written.extend(plot_coverage_map(s.memkm, snap_prefix))

    return [p for p in written if p]


def main():
    ap = argparse.ArgumentParser(
        description="Replot the four kept figures from a saved sweep workbook.")
    ap.add_argument("xlsx", help="path to a sweep .xlsx workbook")
    ap.add_argument("--plot-axis", default=None)
    ap.add_argument("--plot-memkm-steps", default=None,
                    help='"all" or comma-separated step indices')
    args = ap.parse_args()

    result = read_workbook(args.xlsx)
    out_prefix = str(Path(args.xlsx).with_suffix(""))
    for path in plot_all(result, out_prefix, args.plot_axis, args.plot_memkm_steps):
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
