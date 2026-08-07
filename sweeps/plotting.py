"""
Plot: render the bifurcation and rate figures for a beta sweep.

Reads a {out}_kmc_sweep.csv file (as written by sweeps/linear.py or
sweeps/mpi.py) and the matching {out}_meanfield.csv branch file (the mean-field
phase; if it is absent the branches are recomputed on the fly), then writes
{out}_bifurcation.png (Fig. 3) and {out}_rates.png (Fig. 4) next to them. Fig. 3
/ Fig. 4 style plots of Tian & Rangarajan (2021). delta = beta * 1e-4 by
default, matching the sweeps' --delta-scale-beta flag.

Usage:
    uv run python -m sweeps.plotting co_oxidation_kmc_sweep.csv
"""

import argparse
from pathlib import Path

from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from co_oxidation import meanfield

sns.set_theme(style="white", context="talk")

# The mean-field branches are computed by the sweep's mean-field phase
# (sweeps._common.run_meanfield); this module only draws them.
from sweeps._common import (
    DELTA_SCALE,
    MEANFIELD_MODELS,
    build_meanfield_betas,
    run_meanfield,
)

MODEL_LABELS = {"mf": "MF-MK", "ea": "Ea-MK"}
MODEL_COLORS = {"mf": None, "ea": "green"}

def rates_dataframe(beta, theta_o, delta_scale=DELTA_SCALE, mf_physics=None):
    """Fig. 4 rate curves vs theta_CO at fixed beta and theta_O.

    `mf_physics` is the shared mean-field chemistry (alpha, gamma, kr, eps,
    temperature); None uses meanfield.rates' defaults."""
    mf_kw = dict(mf_physics or {})
    if "temperature" in mf_kw:
        mf_kw["T"] = mf_kw.pop("temperature")
    theta_co = np.linspace(1e-4, 1.0 - theta_o - 1e-4, 200)
    frames = []
    for model in MEANFIELD_MODELS:
        _, r_des_co, r_ads_o, r_oxi, _ = meanfield.rates(
            theta_co, theta_o, beta, model=model, delta=beta * delta_scale,
            **mf_kw)
        frames.append(pd.DataFrame({
            "model": model, "beta": beta, "theta_o": theta_o,
            "theta_co": theta_co, "r_oxi": r_oxi, "r_ads_o": r_ads_o,
            "r_des_co": np.broadcast_to(r_des_co, theta_co.shape),
        }))
    return pd.concat(frames, ignore_index=True)


def plot_bifurcation(branches, sweep, path):
    """Fig. 3 style plot: theta_CO and theta_O vs beta.

    `branches` is the mean-field dataframe (as written to {out}_meanfield.csv);
    pass None to draw only the kMC scatter points, e.g. when the mean-field
    phase was skipped with --no-meanfield. When the sweep carries the ME-MKM
    columns, the coverages conditioned on the two spectral macrostates are drawn
    as well: they are the master-equation counterpart of the kMC hysteresis
    branches, computed from a single ergodic steady state rather than from two
    initial conditions.
    """
    L = int(sweep["L"].iloc[0])
    fig, axes = plt.subplots(2, 1, figsize=(8, 9), sharex=True,
                             layout="constrained")
    for ax, col, ylabel in ((axes[0], "theta_co", r"$\theta_{CO}$"),
                            (axes[1], "theta_o", r"$\theta_O$")):
        if branches is not None:
            for model in MEANFIELD_MODELS:
                mdf = branches[branches["model"] == model]
                label = MODEL_LABELS[model]
                hi = mdf[mdf["branch"] == "stable_hi"].sort_values("beta")
                lo = mdf[mdf["branch"] == "stable_lo"].sort_values("beta")
                un = mdf[mdf["branch"] == "unstable"].sort_values("beta")
                line, = ax.plot(hi["beta"], hi[col], "-",
                                color=MODEL_COLORS[model], label=f"{label} stable")
                ax.plot(lo["beta"], lo[col], "-", color=line.get_color())
                ax.plot(un["beta"], un[col], "--", color=line.get_color(),
                        label=f"{label} unstable")
        key = "co" if col == "theta_co" else "o"
        ax.scatter(sweep["beta"], sweep[f"{key}_full"], marker="o", zorder=5,
                  label="kMC (CO-covered start)")
        ax.scatter(sweep["beta"], sweep[f"{key}_empty"], marker="s", zorder=5,
                  label="kMC (empty start)")
        for basin, style in (("A", "-"), ("B", "--")):
            memkm_col = f"memkm_{key}_{basin}"
            if memkm_col not in sweep.columns:
                continue
            branch = sweep.dropna(subset=[memkm_col]).sort_values("beta")
            if branch.empty:
                continue
            ax.plot(branch["beta"], branch[memkm_col], style, color="0.2",
                    lw=1.4, zorder=4, label=f"ME-MKM basin {basin}")
        ax.set_ylabel(ylabel)
        ax.set_ylim(-0.02, 1.02)
    axes[-1].set_xlabel(r"$\beta$ (O$_2$ impingement rate, s$^{-1}$)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=9, ncol=3, loc="outside upper center")
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_rates(rates, path):
    """Fig. 4 style plot: log10(rate) vs theta_CO at fixed theta_O and beta."""
    beta = rates["beta"].iloc[0]
    theta_o = rates["theta_o"].iloc[0]
    mf = rates[rates["model"] == "mf"].sort_values("theta_co")
    ea = rates[rates["model"] == "ea"].sort_values("theta_co")
    fig, ax = plt.subplots(figsize=(7, 5))
    for col, label in (("r_oxi", "CO oxidation"), ("r_ads_o", "O2 adsorption"),
                       ("r_des_co", "CO desorption")):
        ax.plot(mf["theta_co"], np.log10(mf[col]), "-", label=f"{label} (MF-MK)")
        ax.plot(ea["theta_co"], np.log10(ea[col]), "--", color="green",
               label=f"{label} (Ea-MK)")
    ax.set_xlabel(r"$\theta_{CO}$")
    ax.set_ylabel("log10(rate)")
    ax.legend(fontsize=8, ncol=3, loc="lower center",
             bbox_to_anchor=(0.5, 1.02))
    fig.suptitle(rf"Rate comparison at $\beta$={beta}, $\theta_O$={theta_o}",
                y=1.1)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_trajectory(traj, path):
    """Coverage-vs-time trajectories from empty and CO-covered starts."""
    beta = traj["beta"].iloc[0]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    for ax, init, title in ((axes[0], "empty", "empty start"),
                            (axes[1], "full", "CO-covered start")):
        tdf = traj[traj["init"] == init].sort_values("t")
        ax.plot(tdf["t"], tdf["theta_CO"], label=r"$\theta_{CO}$")
        ax.plot(tdf["t"], tdf["theta_O"], label=r"$\theta_O$")
        ax.plot(tdf["t"], tdf["theta_empty"], label=r"$\theta_*$")
        ax.set_xlabel("t (s)")
        ax.set_title(title)
    axes[0].set_ylabel("coverage")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=8, ncol=3, loc="lower center",
              bbox_to_anchor=(0.5, 1.0))
    fig.suptitle(rf"Coverage trajectories at $\beta$={beta}", y=1.08)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)

def plot_coexistence(arrays, betas, cols, out_prefix, tag=""):
    """The ME-MKM spectral diagnostics at one coexistence point beta*, from the
    in-memory report arrays (see CoexistencePipeline.report). Writes several
    {out_prefix}_coexistence{tag}_*.png figures. `betas`/`cols` are the full
    sweep (the MEMKM_COLS dict), drawn as the basin-weight ratio curve and the
    partition-validity continuity panel."""
    beta_star = arrays["beta_star"]
    eigvals = np.sort(arrays["eigvals"])[::-1] # most positive first, for the slowest modes
    phi_slow = arrays["phi_slow"]
    phi2 = arrays["phi2"]
    theta = arrays["theta"]
    phi_coord = arrays["phi_coord"]
    in_A, in_B = arrays["in_A"], arrays["in_B"]
    species = arrays["order_species"]
    log_ratios = cols["log_ratio"]
    palette = sns.color_palette("deep")
    K = len(eigvals)
    Ks = np.arange(1,K+1)

    # 1. Eigenvalue spectrum (real / imaginary). Re(lambda) <= 0 for every mode
    # but lambda_1 (the generator has no unstable direction), and most modes
    # have Im(lambda) = 0 (only the handful of complex-conjugate pairs don't) --
    # a plain 'log' scale drops every non-positive bar, which is why the old
    # plot came out empty. symlog keeps the sign (so the negative real parts
    # still read as negative) while going log-scale away from a linear band
    # around zero sized to the smallest nonzero |eigenvalue|, so the many
    # exactly-zero imaginary parts still draw as zero-height bars instead of
    # disappearing.
    fig, axes = plt.subplots(1, 2, figsize=(max(9.0, 0.3 * K), 5.5))
    fig.suptitle(rf"Eigenvalues of $W$ at $\beta^*$ = {beta_star:.4g}")
    axes[0].bar(Ks, eigvals.real)
    axes[0].set_title("Real Component", fontsize=14)
    axes[1].bar(Ks, eigvals.imag)
    axes[1].set_title("Imaginary Component", fontsize=14)
    max_yticks = 7
    for ax, vals, label_all in zip(axes, [eigvals.real, eigvals.imag],
                                   [True, False]):
        nonzero_mask = vals != 0.0
        nonzero = np.abs(vals[nonzero_mask])
        linthresh = nonzero.min() if nonzero.size else 1.0
        ax.set_yscale('symlog', linthresh=linthresh)
        ax.axhline(0, color='k', lw=0.8)
        # Every mode is informative for the real panel; the imaginary panel's
        # exact zeros (every purely real mode) carry no information beyond
        # "zero" and sit on top of each other at the axis, so only the modes
        # that actually stick out get a tick label there.
        shown = Ks if label_all else Ks[nonzero_mask]
        ax.set_xticks(shown)
        ax.set_xticklabels([rf"$\lambda_{{{i}}}$" for i in shown])
        # Thin the symlog locator's one-tick-per-decade default down to a
        # readable handful, keeping it symmetric about zero.
        ticks = ax.yaxis.get_majorticklocs()
        if len(ticks) > max_yticks:
            keep = np.unique(np.linspace(0, len(ticks) - 1, max_yticks)
                             .round().astype(int))
            ticks = ticks[keep]
        ax.set_yticks(ticks)
    fig.tight_layout()
    fig.savefig(f"{out_prefix}_coexistence{tag}_eigenvalues.png", dpi=150,
                bbox_inches="tight")
    plt.close(fig)

    idx_A, idx_B = np.where(in_A)[0], np.where(in_B)[0]
    idx_A = idx_A[np.argsort(phi2[idx_A])]
    idx_B = idx_B[np.argsort(phi2[idx_B])]
    state_order = np.concatenate([idx_A, idx_B])
    x = np.concatenate([
        np.linspace(0.0, 1.0, len(idx_A), endpoint=False),
        1.0 + np.linspace(0.0, 1.0, len(idx_B), endpoint=False),
    ])
    n_panel = min(4, K)
    fig, axes = plt.subplots(n_panel, 1, sharex=True, figsize=(7, 9))
    fig.suptitle(rf"Slowest left eigenvectors of $W$ at $\beta^*$ = {beta_star:.4g}")
    for m, ax in enumerate(np.atleast_1d(axes)):
        lam = eigvals[m]
        psi = phi_slow[:, m].real
        if np.dot(psi, phi2) < 0:
            psi = -psi
        psi = psi / np.max(np.abs(psi))
        # Draw the two basins as separate segments so a basin holding only a
        # handful of microstates (a line plot renders a single point as
        # nothing) still shows up, via markers.
        nA = len(idx_A)
        for x_seg, psi_seg in ((x[:nA], psi[state_order][:nA]),
                               (x[nA:], psi[state_order][nA:])):
            if len(x_seg) == 0:
                continue
            ax.plot(x_seg, psi_seg, lw=0.9, color=palette[m],
                    marker="o" if len(x_seg) < 20 else None, ms=4)
        ax.axhline(0.0, color="0.8", lw=0.8)
        ax.set_ylabel(rf"$\phi_{m + 1}^L$")
        label = rf"$\lambda_{m + 1}$ = {lam.real:.3e}"
        if abs(lam.imag) > 1e-8 * max(abs(lam.real), 1e-300):
            label += rf" (Im = {lam.imag:.1e}!)"
        ax.text(0.02, 0.85, label, transform=ax.transAxes, fontsize=9)
        ax.set_xlim(0.0, 2.0)
        ax.set_xticks([0.0, 1.0, 2.0])
        ax.set_xticklabels([])
        ax.set_xticks([0.5, 1.5], minor=True)
        ax.set_xticklabels(["A", "B"], minor=True)
        ax.tick_params(axis="x", which="minor", length=0)
    fig.savefig(f"{out_prefix}_coexistence{tag}_eigenvectors.png", dpi=150,
                bbox_inches="tight")
    plt.close(fig)

    # 3. Coverage marginal of the ordering species at beta*.
    marginal = arrays["marginal"]
    fig, ax = plt.subplots()
    ax.plot(np.arange(len(marginal)), marginal, "-o", color=palette[0])
    ax.set_yscale("log")
    ax.set_xlabel(rf"$N_\mathrm{{{species}}}$")
    ax.set_ylabel(rf"$P(N_\mathrm{{{species}}})$")
    fig.suptitle(rf"{species}-count marginal at $\beta^*$ = {beta_star:.4g}")
    fig.savefig(f"{out_prefix}_coexistence{tag}_marginal.png", dpi=150,
                bbox_inches="tight")
    plt.close(fig)

    # 4. Stationary density on the slow coordinate.
    sort_idx = np.argsort(phi_coord)
    pmf, edges = np.histogram(phi_coord[sort_idx], bins=50, weights=theta[sort_idx])
    fig, ax = plt.subplots()
    ax.stairs(pmf, edges, fill=True, color=palette[0])
    ax.set_yscale("log")
    ax.set_xlabel(r"slow coordinate $\phi_2^L$")
    ax.set_ylabel(r"$\rho(\phi_2^L)$")
    fig.suptitle(rf"Stationary density on the slow mode at $\beta^*$ = {beta_star:.4g}")
    fig.savefig(f"{out_prefix}_coexistence{tag}_slow-coordinate.png", dpi=150,
                bbox_inches="tight")
    plt.close(fig)

    # 5. Basin-weight curves over the sweep, marking beta*: the objective
    #    ln pi(A)/pi(B) and the weights themselves, whose crossing at 1/2 is
    #    the coexistence definition.
    betas = np.asarray(betas, float)
    log_ratios = np.asarray(log_ratios, float)
    good = np.isfinite(log_ratios)
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(7, 7))
    axes[0].plot(betas[good], log_ratios[good], "-o", color=palette[0])
    axes[0].axhline(0.0, color="0.6", lw=1)
    axes[0].annotate(rf"$\beta^*$ = {beta_star:.4g}", (beta_star, 0.0),
                     textcoords="offset points", xytext=(8, 8))
    axes[0].set_ylabel(r"$\ln\,p_{ss}(A)/p_{ss}(B)$")
    P_A = np.asarray(cols["memkm_P_A"], float)
    ok = np.isfinite(P_A)
    axes[1].plot(betas[ok], P_A[ok], "-o", color=palette[0], label=r"$P_A$")
    axes[1].plot(betas[ok], 1.0 - P_A[ok], "-o", color=palette[1], label=r"$P_B$")
    axes[1].axhline(0.5, color="0.6", lw=1)
    axes[1].set_ylabel("spectral macrostate weight")
    axes[1].set_ylim(-0.02, 1.02)
    axes[1].legend(fontsize=8)
    for ax in axes:
        ax.axvline(beta_star, color="0.6", lw=1, ls="--")
    axes[-1].set_xlabel(r"$\beta$")
    fig.suptitle("Spectral macrostate weights vs. adsorption rate")
    fig.tight_layout()
    fig.savefig(f"{out_prefix}_coexistence{tag}_ratio-curve.png", dpi=150,
                bbox_inches="tight")
    plt.close(fig)

    # 6. Continuity / validity of the partition across the sweep.
    plot_partition_diagnostics(betas, cols, beta_star, out_prefix, tag=tag)

    # 7. Coverage-class map over (N_CO, N_O).
    if "cov_pop" in arrays:
        plot_coverage_map(arrays, out_prefix, tag=tag)


def plot_partition_diagnostics(betas, cols, beta_star, out_prefix, tag=""):
    """Whether the two-state reading actually holds across the sweep, from the
    per-beta MEMKM_COLS: the two slowest eigenvalues and their gap, the
    stationary mass sitting on the sign boundary, the overlap of consecutive
    slow modes, and the conditional coverages of the two macrostates."""
    betas = np.asarray(betas, float)
    palette = sns.color_palette("deep")

    def finite(key):
        y = np.asarray(cols[key], float)
        m = np.isfinite(y)
        return betas[m], y[m]

    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), sharex=True)
    ax = axes[0, 0]
    for key, label, color in (("lambda2_re", r"$|\mathrm{Re}\,\lambda_2|$", 0),
                              ("lambda3_re", r"$|\mathrm{Re}\,\lambda_3|$", 1)):
        x, y = finite(key)
        ax.plot(x, np.abs(y), "-o", ms=3, color=palette[color], label=label)
    ax.set_yscale("log")
    ax.set_ylabel("relaxation rate, s$^{-1}$")
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    x, y = finite("spectral_gap")
    ax.plot(x, y, "-o", ms=3, color=palette[0])
    ax.axhline(1.0, color="0.6", lw=1)
    ax.set_yscale("log")
    ax.set_ylabel(r"gap $|\mathrm{Re}\,\lambda_3 / \mathrm{Re}\,\lambda_2|$")
    ax.set_title("large = one slow process controls the split", fontsize=9)

    # Exact zeros are common and meaningful here (no ambiguous weight at all, a
    # perfectly continued mode), so they are drawn on the floor of the log axis
    # rather than dropped, and the floor is set a decade under the smallest
    # positive value so the informative range keeps the height of the panel.
    def _floored(y, default=1e-12):
        positive = y[y > 0.0]
        floor = positive.min() / 10.0 if positive.size else default
        return np.maximum(y, floor), floor

    ax = axes[1, 0]
    x, y = finite("boundary_mass")
    y, floor = _floored(y)
    ax.plot(x, y, "-o", ms=3, color=palette[0])
    ax.set_yscale("log")
    ax.set_ylim(bottom=floor / 2)
    ax.set_ylabel(r"$P_\mathrm{boundary}$")
    ax.set_xlabel(r"$\beta$")
    im_re = np.asarray(cols["im_re_ratio"], float)
    worst = np.nanmax(im_re) if np.isfinite(im_re).any() else np.nan
    ax.set_title(rf"max $|\mathrm{{Im}}\,\lambda_2/\mathrm{{Re}}\,\lambda_2|$ = "
                 rf"{worst:.1e} (real mode required)", fontsize=9)

    # Plotted as 1 - overlap: the overlaps themselves sit at 0.9999... where a
    # linear axis resolves nothing. A sign flip shows up as a value near 2.
    ax = axes[1, 1]
    x, y = finite("mode_overlap")
    y, floor = _floored(1.0 - y)
    ax.plot(x, y, "-o", ms=3, color=palette[0])
    ax.set_yscale("log")
    ax.set_ylim(bottom=floor / 2)
    ax.set_ylabel(r"$1-\langle \phi_2^R(\beta_k), \phi_2^R(\beta_{k-1})\rangle$")
    ax.set_xlabel(r"$\beta$")
    ax.set_title("spike (or a value near 2, a sign flip) = mode crossing",
                 fontsize=9)

    for ax in axes.ravel():
        ax.axvline(beta_star, color="0.6", lw=1, ls="--")
    fig.suptitle("Validity of the spectral two-state partition")
    fig.tight_layout()
    fig.savefig(f"{out_prefix}_coexistence{tag}_partition-diagnostics.png",
                dpi=150, bbox_inches="tight")
    plt.close(fig)


def _coverage_pcolor(ax, grid, l, cmap, norm=None, vmin=None, vmax=None):
    """Draw one coverage-class grid on `ax` as a brick-stack pyramid: row N_O
    (of length l+1-N_O) is offset right by N_O/2 as a whole, centering it
    under the full-width N_O=0 row, so the accessible region reads as a
    symmetric triangle of unit *squares* -- not a sheared parallelogram mesh,
    where the offset would vary between a cell's own top and bottom edges and
    leave every cell a slightly slanted trapezoid. Each valid (n_CO, n_O)
    class is one square, drawn individually since adjacent rows don't share
    edges (this is a staggered brick wall, not a connected grid)."""
    n_co, n_o = np.where(np.isfinite(grid))
    vals = grid[n_co, n_o]
    squares = [Rectangle((co + o / 2 - 0.5, o - 0.5), 1, 1)
              for co, o in zip(n_co, n_o)]
    im = PatchCollection(squares, cmap=cmap, edgecolor="0.35", linewidth=1.2)
    im.set_array(vals)
    if norm is not None:
        im.set_norm(norm)
    else:
        im.set_clim(vmin, vmax)
    ax.add_collection(im)
    # Base (N_O=0 row) and total height both span l+1 data units, so an
    # "equal" aspect draws a triangle taller than it is wide (height/base=1
    # vs. sqrt(3)/2 for equilateral). Squashing the y-unit by sqrt(3)/2 fixes
    # the overall proportions while keeping each class an axis-aligned
    # rectangle rather than shearing the mesh.
    ax.set_aspect(np.sqrt(3) / 2)
    ax.set_xlabel(r"$N_\mathrm{CO}$")
    ax.set_ylabel(r"$N_\mathrm{O}$")
    ax.set_xticks([])                            # x no longer maps 1:1 to N_CO once shifted
    ax.set_yticks(np.arange(0, l + 1))
    ax.set_xlim(-0.5, l + 0.5)
    ax.set_ylim(-0.5, l + 0.5)
    return im


def _inset_colorbar(fig, ax, im, label=None):
    """A colorbar sized to match its axes. Goes through fig.colorbar(ax=...)
    rather than the axes_grid1 divider trick, since these figures use
    constrained_layout=True and the layout engine only reserves space for
    colorbars it knows about -- an axes_grid1 divider axes is invisible to it
    and ends up drawn on top of whatever is already there."""
    return fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=label)


def plot_coverage_map(arrays, out_prefix, tag=""):
    """Coverage-class maps over the (N_CO, N_O) plane at beta*, in two figures:

      {..}_coverage-population.png : the stationary marginal log10 sum_i pi_i,
        which states are populated (the two basins + transition valley),
      {..}_coverage-reaction.png  : the slow right mode sum phi_^_2 (whose *sign*
        is the macrostate partition), the unweighted class-mean slow left mode
        <phi_2^L>.

    Cells are centered on integer (N_CO, N_O); the inaccessible corner
    (N_CO + N_O > l) and empty classes are masked."""
    from matplotlib.colors import TwoSlopeNorm

    beta_star = arrays["beta_star"]
    l = arrays["n_sites"]
    pop = arrays["cov_pop"]
    phi = arrays["cov_phi"]
    r2map = arrays["cov_r2"]   # class sums of r_2; sign = partition

    deg = arrays["cov_deg"]
    a = np.arange(l + 1)
    outside = (a[:, None] + a[None, :]) > l
    logpop = np.where((pop > 0) & ~outside, np.log(np.where(pop > 0, pop, 1)),np.nan)
    empty = outside | (deg <= 0)
    phi_m = np.where(outside, np.nan, phi)
    r2_m = np.where(empty, np.nan, r2map)

    # Figure 1: stationary population, with the degeneracy-corrected view beside it.
    fig, ax = plt.subplots(1, 1, figsize=(12, 5.5), constrained_layout=True)
    im0 = _coverage_pcolor(ax, logpop, l, "viridis")
    ax.set_title(rf"Stationary Distribution by Coverage-class at $\beta^*$ = {beta_star:.4g}",
                pad=10)
    _inset_colorbar(fig, ax, im0, label=r"$\ln(w_{n_\mathrm{CO}, n_\mathrm{O}})$")
    fig.savefig(f"{out_prefix}_coexistence{tag}_coverage-population.png",
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Figure 2: the partition (r_2) and the reaction coordinate (phi_2^L),
    # which should both place the dividing surface in the same place.
    def _diverging(grid, center=0.0):
        if np.isfinite(grid).any() and np.nanmin(grid) < 0 < np.nanmax(grid):
            return TwoSlopeNorm(vcenter=center, vmin=np.nanmin(grid),
                                vmax=np.nanmax(grid))
        return None

    fig, axes = plt.subplots(1, 2, figsize=(11, 5.5), constrained_layout=True)
    im0 = _coverage_pcolor(axes[0], r2_m, l, "coolwarm", norm=_diverging(r2_m))
    axes[0].set_title(r"Slowest Relaxation Mode $\phi_2^R(n_\mathrm{CO},n_\mathrm{O})$",
                      pad=36)
    _inset_colorbar(fig, axes[0], im0)
    im1 = _coverage_pcolor(axes[1], phi_m, l, "coolwarm", norm=_diverging(phi_m))
    axes[1].set_title(r"Reaction Coordinate $\phi_2^L(n_\mathrm{CO},n_\mathrm{O})$",
                      pad=36)
    _inset_colorbar(fig, axes[1], im1)
    fig.suptitle(rf"Partition and reaction coordinate in coverage space at "
                 rf"$\beta^*$ = {beta_star:.4g}")
    fig.savefig(f"{out_prefix}_coexistence{tag}_coverage-reaction.png",
                dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_sweep(sweep, out_prefix, branches=None, delta_scale=DELTA_SCALE,
               rates_beta=4.0, rates_theta_o=0.01, mf_physics=None):
    """Bifurcation + rate figures from a sweep dataframe, at the delta the
    sweep actually used.

    `branches` is the precomputed mean-field dataframe (the mean-field phase's
    {out}_meanfield.csv). When it is None the mean-field lines and the rate
    figure are skipped and only the kMC scatter bifurcation is drawn.
    `mf_physics` is the shared mean-field chemistry used to recompute the rate
    curves. Writes {out_prefix}_bifurcation.png and, when branches are
    available, _rates.png.
    """
    plot_bifurcation(branches, sweep, f"{out_prefix}_bifurcation.png")
    written = [f"{out_prefix}_bifurcation.png"]
    if branches is not None:
        plot_rates(rates_dataframe(rates_beta, rates_theta_o, delta_scale,
                                   mf_physics=mf_physics),
                   f"{out_prefix}_rates.png")
        written.append(f"{out_prefix}_rates.png")
    return written


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("csv", help="path to a {out}_kmc_sweep.csv file")
    ap.add_argument("--meanfield-beta-step", "--beta-fine-step",
                    dest="meanfield_beta_step", type=float, default=0.05,
                    help="fill-in beta grid step used only when no "
                         "{out}_meanfield.csv is found and the branches must "
                         "be recomputed")
    ap.add_argument("--rates-beta", type=float, default=4.0,
                    help="fixed beta for the Fig. 4 rate plot")
    ap.add_argument("--rates-theta-o", type=float, default=0.01,
                    help="fixed theta_O for the Fig. 4 rate plot")
    args = ap.parse_args()

    p = Path(args.csv)
    dir = p.parents[0]

    # Drop only rows whose kMC coverages are missing; ME-MKM columns (added by
    # the coexistence phase) may legitimately be NaN at some betas. A sweep run
    # with --no-kmc has no complete kMC rows -- keep the frame so the
    # mean-field branches still plot.
    kmc_cols = ["e_empty", "co_empty", "o_empty", "e_full", "co_full", "o_full"]
    raw = pd.read_csv(args.csv)
    sweep = raw.dropna(subset=kmc_cols)
    if sweep.empty:
        sweep = raw
    stem = p.stem.replace("_kmc_sweep", "")

    # use the delta the sweep recorded, so the branches match its kMC points
    delta_scale = (float(sweep["delta_scale"].iloc[0])
                   if "delta_scale" in sweep.columns else DELTA_SCALE)

    # Prefer the branches written by the sweep's mean-field phase; only
    # recompute (on the fly, at the sweep's delta) if that file is absent.
    mf_path = dir / f"{stem}_meanfield.csv"
    if mf_path.exists():
        branches = pd.read_csv(mf_path)
        print(f"read mean-field branches from {mf_path.name}")
    else:
        betas_fine = build_meanfield_betas(sweep["beta"].to_numpy(),
                                           args.meanfield_beta_step)
        branches = run_meanfield(betas_fine, delta_scale=delta_scale)

    for path in plot_sweep(sweep, f"{dir}/{stem}", branches=branches,
                           delta_scale=delta_scale,
                           rates_beta=args.rates_beta,
                           rates_theta_o=args.rates_theta_o):
        print(f"wrote {Path(path).name}  (delta = {delta_scale:g} * beta)")

if __name__ == "__main__":
    main()