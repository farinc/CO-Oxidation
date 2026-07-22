"""
Plot: render the bifurcation and rate figures for a beta sweep.

Reads a {out}_kmc_sweep.csv file (as written by sweeps/linear.py or
sweeps/mpi.py), computes the matching mean-field branches (Fig. 3) and
rate curves (Fig. 4) on the fly via meanfield.branches/meanfield.rates,
and writes {out}_bifurcation.png and {out}_rates.png next to it. Fig. 3 /
Fig. 4 style plots of Tian & Rangarajan (2021). delta = beta * 1e-4
throughout, matching the sweeps' --delta-scale-beta flag.

Usage:
    uv run python -m co_oxidation.plotting co_oxidation_kmc_sweep.csv
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from co_oxidation import meanfield

sns.set_theme(style="whitegrid")

DELTA_SCALE = 1e-4
MODELS = ("mf", "ea")
MODEL_LABELS = {"mf": "MF-MK", "ea": "Ea-MK"}
MODEL_COLORS = {"mf": None, "ea": "green"}

def branch_dataframe(betas_fine):
    """Branches of both models in the shape plot.plot_bifurcation expects."""
    frames = []
    for model in MODELS:
        rows = {"stable_hi": [], "stable_lo": [], "unstable": []}
        for b in betas_fine:
            hi, lo, un = meanfield.branches(
                [b], model=model, delta=b * DELTA_SCALE)
            rows["stable_hi"].append(hi[0])
            rows["stable_lo"].append(lo[0])
            rows["unstable"].append(un[0])
        for branch, arr in rows.items():
            arr = np.asarray(arr)
            frames.append(pd.DataFrame({
                "model": model, "branch": branch, "beta": betas_fine,
                "theta_empty": arr[:, 0], "theta_co": arr[:, 1],
                "theta_o": arr[:, 2],
            }))
    return pd.concat(frames, ignore_index=True)


def rates_dataframe(beta, theta_o):
    """Fig. 4 rate curves vs theta_CO at fixed beta and theta_O."""
    theta_co = np.linspace(1e-4, 1.0 - theta_o - 1e-4, 200)
    frames = []
    for model in MODELS:
        _, r_des_co, r_ads_o, r_oxi, _ = meanfield.rates(
            theta_co, theta_o, beta, model=model, delta=beta * DELTA_SCALE)
        frames.append(pd.DataFrame({
            "model": model, "beta": beta, "theta_o": theta_o,
            "theta_co": theta_co, "r_oxi": r_oxi, "r_ads_o": r_ads_o,
            "r_des_co": np.broadcast_to(r_des_co, theta_co.shape),
        }))
    return pd.concat(frames, ignore_index=True)


def plot_bifurcation(branches, sweep, path):
    """Fig. 3 style plot: theta_CO and theta_O vs beta."""
    L = int(sweep["L"].iloc[0])
    fig, axes = plt.subplots(2, 1, figsize=(7, 8), sharex=True)
    for ax, col, ylabel in ((axes[0], "theta_co", r"$\theta_{CO}$"),
                            (axes[1], "theta_o", r"$\theta_O$")):
        for model in ("mf", "ea"):
            mdf = branches[branches["model"] == model]
            label = MODEL_LABELS[model]
            hi = mdf[mdf["branch"] == "stable_hi"].sort_values("beta")
            lo = mdf[mdf["branch"] == "stable_lo"].sort_values("beta")
            un = mdf[mdf["branch"] == "unstable"].sort_values("beta")
            line, = ax.plot(hi["beta"], hi[col], "-", color=MODEL_COLORS[model],
                            label=f"{label} stable")
            ax.plot(lo["beta"], lo[col], "-", color=line.get_color())
            ax.plot(un["beta"], un[col], "--", color=line.get_color(),
                    label=f"{label} unstable")
        key = "co" if col == "theta_co" else "o"
        ax.scatter(sweep["beta"], sweep[f"{key}_full"], marker="o", zorder=5,
                  label="kMC (CO-covered start)")
        ax.scatter(sweep["beta"], sweep[f"{key}_empty"], marker="s", zorder=5,
                  label="kMC (empty start)")
        ax.set_ylabel(ylabel)
        ax.set_ylim(-0.02, 1.02)
    axes[0].legend(fontsize=8, ncol=3, loc="lower center",
                  bbox_to_anchor=(0.5, 1.02))
    axes[-1].set_xlabel(r"$\beta$ (O$_2$ impingement rate, s$^{-1}$)")
    fig.suptitle(f"CO oxidation bifurcation diagram (L={L})", y=1.08)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
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

def plot_coexistence(arrays, betas, log_ratios, out_prefix, tag=""):
    """The ME-MKM spectral diagnostics at one coexistence point beta*, from the
    in-memory report arrays (see CoexistencePipeline.report). Writes several
    {out_prefix}_coexistence{tag}_*.png figures. `betas`/`log_ratios` are the
    full sweep, drawn as the basin-weight ratio curve."""
    beta_star = arrays["beta_star"]
    eigvals = arrays["eigvals"]
    phi_slow = arrays["phi_slow"]
    phi2 = arrays["phi2"]
    theta = arrays["theta"]
    q_plus = arrays["q_plus"]
    phi_coord = arrays["phi_coord"]
    in_A, in_B = arrays["in_A"], arrays["in_B"]
    species = arrays["order_species"]
    palette = sns.color_palette("deep")
    K = len(eigvals)

    # 1. Eigenvalue spectrum (real / imaginary).
    fig, axes = plt.subplots(1, 2, sharex=True)
    fig.suptitle(rf"Eigenvalues of $W$ at $\beta^*$ = {beta_star:.4g}")
    axes[0].bar(np.arange(K), eigvals.real)
    axes[0].set_title("Real Component")
    axes[1].bar(np.arange(K), eigvals.imag)
    axes[1].set_title("Imaginary Component")
    fig.savefig(f"{out_prefix}_coexistence{tag}_eigenvalues.png", dpi=150,
                bbox_inches="tight")
    plt.close(fig)

    # 2. Prinz-style panel: the four slowest left eigenvectors over the two
    #    basins, each basin stretched to equal axis width.
    idx_A, idx_B = np.where(in_A)[0], np.where(in_B)[0]
    idx_A = idx_A[np.argsort(-phi2[idx_A])]
    idx_B = idx_B[np.argsort(-phi2[idx_B])]
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
        ax.plot(x, psi[state_order], lw=0.9, color=palette[m])
        ax.axhline(0.0, color="0.8", lw=0.8)
        ax.set_ylabel(rf"$\psi_{m + 1}^L$")
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
    pmf, edges = np.histogram(phi_coord, bins=50, weights=theta)
    fig, ax = plt.subplots()
    ax.stairs(pmf, edges, fill=True, color=palette[0])
    ax.set_yscale("log")
    ax.set_xlabel(r"slow coordinate $\phi_2^L$")
    ax.set_ylabel(r"$\rho(\phi_2^L)$")
    fig.suptitle(rf"Stationary density on the slow mode at $\beta^*$ = {beta_star:.4g}")
    fig.savefig(f"{out_prefix}_coexistence{tag}_slow-coordinate.png", dpi=150,
                bbox_inches="tight")
    plt.close(fig)

    # 5. Slow mode vs. forward committor (affine collapse = two-state kinetics).
    fig, ax = plt.subplots()
    ax.scatter(q_plus, phi_coord, s=6, alpha=0.4, color=palette[0],
               edgecolors="none")
    ax.set_xlabel(r"forward committor $q^+$")
    ax.set_ylabel(r"slow coordinate $\phi_2^L$")
    fig.suptitle(rf"Slow mode vs. committor at $\beta^*$ = {beta_star:.4g}")
    fig.savefig(f"{out_prefix}_coexistence{tag}_committor.png", dpi=150,
                bbox_inches="tight")
    plt.close(fig)

    # 6. Basin-weight ratio curve over the sweep, marking beta*.
    betas = np.asarray(betas, float)
    log_ratios = np.asarray(log_ratios, float)
    good = np.isfinite(log_ratios)
    fig, ax = plt.subplots()
    ax.plot(betas[good], log_ratios[good], "-o", color=palette[0])
    ax.axhline(0.0, color="0.6", lw=1)
    ax.axvline(beta_star, color="0.6", lw=1, ls="--")
    ax.annotate(rf"$\beta^*$ = {beta_star:.4g}", (beta_star, 0.0),
                textcoords="offset points", xytext=(8, 8))
    ax.set_xlabel(r"$\beta$")
    ax.set_ylabel(r"$\log_{10}\,\pi(A)/\pi(B)$")
    fig.suptitle("Basin-weight ratio vs. adsorption rate")
    fig.savefig(f"{out_prefix}_coexistence{tag}_ratio-curve.png", dpi=150,
                bbox_inches="tight")
    plt.close(fig)

    # 7. Coverage-class map over (N_CO, N_O).
    if "cov_pop" in arrays:
        plot_coverage_map(arrays, out_prefix, tag=tag)


def _coverage_pcolor(ax, grid, l, cmap, norm=None, vmin=None, vmax=None):
    """Draw one coverage-class grid on `ax` as a pcolormesh. Cells are centered
    on integer (N_CO, N_O) with edges at the half-integers, so (0, 0) is the
    empty-tile cell; every cell is outlined (grid line at every integer)."""
    edges = np.arange(l + 2) - 0.5              # cell boundaries: -0.5 .. l+0.5
    data = np.ma.masked_invalid(grid).T          # -> [N_O (row), N_CO (col)]
    kw = {"norm": norm} if norm is not None else {"vmin": vmin, "vmax": vmax}
    im = ax.pcolormesh(edges, edges, data, cmap=cmap,
                       edgecolors="0.35", linewidth=1.2, **kw)
    ax.grid(False)                               # only the cell edges, no theme grid
    ax.set_aspect("equal")
    ax.set_xlabel(r"$N_\mathrm{CO}$")
    ax.set_ylabel(r"$N_\mathrm{O}$")
    ax.set_xticks(np.arange(0, l + 1))           # integer labels every 1
    ax.set_yticks(np.arange(0, l + 1))
    ax.set_xlim(-0.5, l + 0.5)
    ax.set_ylim(-0.5, l + 0.5)
    return im


def plot_coverage_map(arrays, out_prefix, tag=""):
    """Coverage-class maps over the (N_CO, N_O) plane at beta*, in two figures:

      {..}_coverage-population.png : the stationary marginal log10 sum_i pi_i,
        which states are populated (the two basins + transition valley),
      {..}_coverage-reaction.png  : the pi-weighted slow mode <phi_2^L> and the
        pi-weighted forward committor <q+> side by side -- they tell the same
        two-state story (phi_2^L is ~ affine in q+), so they are grouped.

    Cells are centered on integer (N_CO, N_O); the inaccessible corner
    (N_CO + N_O > l) and empty classes are masked."""
    from matplotlib.colors import TwoSlopeNorm

    beta_star = arrays["beta_star"]
    l = arrays["n_sites"]
    pop = arrays["cov_pop"]
    phi = arrays["cov_phi"]
    qmap = arrays["cov_q_A"]   # q_A = 1 - q+ : P(reach the CO-rich core first)

    deg = arrays["cov_deg"]
    a = np.arange(l + 1)
    outside = (a[:, None] + a[None, :]) > l
    logpop = np.where((pop > 0) & ~outside, np.log10(np.where(pop > 0, pop, 1)),
                      np.nan)
    # Per-microstate mean weight = population / degeneracy: strips the
    # combinatorial class-size factor to show the intrinsic per-config preference.
    perstate = np.where((pop > 0) & (deg > 0), pop / np.where(deg > 0, deg, 1),
                        np.nan)
    logperstate = np.where((perstate > 0) & ~outside,
                           np.log10(np.where(perstate > 0, perstate, 1)), np.nan)
    phi_m = np.where(outside, np.nan, phi)
    q_m = np.where(outside, np.nan, qmap)

    # Figure 1: stationary population, with the degeneracy-corrected view beside it.
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), constrained_layout=True)
    im0 = _coverage_pcolor(axes[0], logpop, l, "viridis")
    axes[0].set_title(r"Distribution of Microstates $\sum_i \pi_i$")
    fig.colorbar(im0, ax=axes[0], label=r"$\log_{10}\sum_i \pi_i$", shrink=0.85)
    im1 = _coverage_pcolor(axes[1], logperstate, l, "viridis")
    axes[1].set_title(r"Normalized Distribution of Microstates $(\sum_i \pi_i) / g$")
    fig.colorbar(im1, ax=axes[1], label=r"$\log_{10}(\sum_i \pi_i / g)$",
                 shrink=0.85)
    fig.suptitle(rf"Coverage-class population at $\beta^*$ = {beta_star:.4g}")
    fig.savefig(f"{out_prefix}_coexistence{tag}_coverage-population.png",
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Figure 2: slow mode + committor together (the reaction coordinate).
    finite = np.isfinite(phi_m)
    if finite.any() and np.nanmin(phi_m) < 0 < np.nanmax(phi_m):
        phi_norm = TwoSlopeNorm(vcenter=0.0, vmin=np.nanmin(phi_m),
                                vmax=np.nanmax(phi_m))
    else:
        phi_norm = None

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), constrained_layout=True)
    im0 = _coverage_pcolor(axes[0], phi_m, l, "coolwarm", norm=phi_norm)
    axes[0].set_title(r"slow mode $\langle\phi_2^L\rangle_\pi$")
    fig.colorbar(im0, ax=axes[0], shrink=0.85)
    im1 = _coverage_pcolor(axes[1], q_m, l, "Blues", vmin=0.0, vmax=1.0)
    axes[1].set_title(r"committor to CO-rich basin $\langle q_A\rangle_\pi$")
    fig.colorbar(im1, ax=axes[1], label=r"$\langle q_A\rangle_\pi = 1-\langle q^+\rangle_\pi$",
                 shrink=0.85)
    fig.suptitle(rf"Reaction coordinate in coverage space at "
                 rf"$\beta^*$ = {beta_star:.4g}")
    fig.savefig(f"{out_prefix}_coexistence{tag}_coverage-reaction.png",
                dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("csv", help="path to a {out}_kmc_sweep.csv file")
    ap.add_argument("--beta-fine-step", type=float, default=0.05,
                    help="beta grid step for the mean-field branches")
    ap.add_argument("--rates-beta", type=float, default=4.0,
                    help="fixed beta for the Fig. 4 rate plot")
    ap.add_argument("--rates-theta-o", type=float, default=0.01,
                    help="fixed theta_O for the Fig. 4 rate plot")
    args = ap.parse_args()

    p = Path(args.csv)
    dir = p.parents[0]

    # Drop only rows whose kMC coverages are missing; ME-MKM columns (added by
    # the coexistence phase) may legitimately be NaN at some betas.
    kmc_cols = ["e_empty", "co_empty", "o_empty", "e_full", "co_full", "o_full"]
    sweep = pd.read_csv(args.csv).dropna(subset=kmc_cols)
    stem = p.stem.replace("_kmc_sweep", "")

    betas_fine = np.arange(sweep["beta"].min(),
                           sweep["beta"].max() + 0.5 * args.beta_fine_step,
                           args.beta_fine_step)
    branches = branch_dataframe(betas_fine)

    plot_bifurcation(branches, sweep, f"{dir}/{stem}_bifurcation.png")
    print(f"wrote {stem}_bifurcation.png")

    rates = rates_dataframe(args.rates_beta, args.rates_theta_o)
    plot_rates(rates, f"{dir}/{stem}_rates.png")
    print(f"wrote {stem}_rates.png")

if __name__ == "__main__":
    main()