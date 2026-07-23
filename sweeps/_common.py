"""Shared task list, single-task runner and CSV writer for the beta sweeps.
A sweep runs up to three independent phases, each toggleable from the CLI:
  - kMC          : the (beta, init) trajectories        (--no-kmc to skip)
  - ME-MKM       : the SLEPc/PETSc coexistence analysis  (--no-memkm to skip)
  - mean field   : the MF-MK / Ea-MK steady-state branches (--no-meanfield)
"""

import argparse

import numpy as np
import pandas as pd

from co_oxidation import KMCParams, run_kmc

TAGS = ("empty", "full")

# O2 desorption rate delta = DELTA_SCALE * beta when --delta-scale-beta is on
# (the default). The mean-field phase uses the same value for its branches --
# keep the two in step or the kMC points and the MF/Ea curves are computed at
# different delta and stop being comparable.
DELTA_SCALE = 1e-4

# ME-MKM sweep columns added alongside the kMC coverages in {out}_kmc_sweep.csv.
MEMKM_COLS = ("memkm_empty", "memkm_co", "memkm_o", "log_ratio")

# The two mean-field rate laws (MF-MK and Ea-MK in Tian & Rangarajan 2021).
MEANFIELD_MODELS = ("mf", "ea")


def build_betas(beta_min, beta_max, beta_step):
    return np.arange(beta_min, beta_max + 0.5 * beta_step, beta_step)


def delta_scale_of(args):
    """The single O2-desorption scale used by the kMC sweep, the ME-MKM model
    and the mean-field branches: delta = delta_scale * beta."""
    return args.delta_scale if args.delta_scale_beta else 0.0


def physics_from_args(args):
    """Full shared chemistry as generate_model / KMCParams keywords: the same
    values feed the kMC model and the ME-MKM generator."""
    return dict(alpha=args.alpha, gamma=args.gamma, kr=args.kr,
                khop_scale=args.khop_scale, eps=args.eps,
                temperature=args.temperature)


def meanfield_physics_from_args(args):
    """Mean-field subset of the shared chemistry (no diffusion, so khop_scale
    is dropped) for run_meanfield and the rate figure."""
    return dict(alpha=args.alpha, gamma=args.gamma, kr=args.kr,
                eps=args.eps, temperature=args.temperature)


def build_tasks(betas, seed):
    """List of (idx, beta, init_tag, seed), two entries per beta."""
    tasks = []
    for i, b in enumerate(betas):
        for j, tag in enumerate(TAGS):
            tasks.append((i, float(b), tag, seed + 2 * i + j))
    return tasks


def run_task(task, params, delta_scale=DELTA_SCALE, verbose_prefix=""):
    """Run one (idx, beta, init, seed) task, returning (idx, tag, steady coverages).

    `delta_scale` ties the O2 desorption rate to the impingement rate,
    delta = delta_scale * beta; 0.0 gives irreversible O2 adsorption. The same
    value is handed to the ME-MKM model so both describe the same chemistry.
    """
    idx, beta, tag, seed = task
    overrides = {"delta": beta * delta_scale}
    res = run_kmc(beta, init=tag, params=params, seed=seed, **overrides)
    note = " [absorbing state]" if res.stuck else ""
    print(f"{verbose_prefix}beta={beta:5.2f} init={tag:5s}  "
          f"theta_CO={res.steady_co:.3f} theta_O={res.steady_o:.3f}  "
          f"({res.steps:.3e} events){note}", flush=True)
    return idx, tag, res.steady_empty, res.steady_co, res.steady_o


def assemble(betas, results):
    """Combine run_task() outputs into the {out}_kmc_sweep.csv dict shape.

    An empty `results` (e.g. --no-kmc) yields all-NaN kMC columns, so the CSV
    and any ME-MKM columns attached later keep their shape.
    """
    n = len(betas)
    out = {key: np.full(n, np.nan) for key in
           ("e_empty", "co_empty", "o_empty", "e_full", "co_full", "o_full")}
    for idx, tag, e, co, o in results:
        out[f"e_{tag}"][idx] = e
        out[f"co_{tag}"][idx] = co
        out[f"o_{tag}"][idx] = o
    return out


def save_sweep_csv(betas, sweep, L, path, delta_scale=DELTA_SCALE):
    df = pd.DataFrame({"beta": betas, "L": L, "delta_scale": delta_scale,
                       **sweep})
    df.to_csv(path, index=False)


# --- mean-field phase ---------------------------------------------------------

def build_meanfield_betas(betas, fine_step):
    """Fine beta grid for smooth mean-field curves that still contains every
    sweep beta.

    Returns the sorted union of the sweep betas and a uniform grid of spacing
    `fine_step` spanning the same range, so the MF/Ea branches line up exactly
    with the kMC / ME-MKM sample points while filling in the gaps between them.
    """
    betas = np.asarray(betas, float)
    if betas.size == 0:
        return betas
    lo, hi = float(betas.min()), float(betas.max())
    fine = np.arange(lo, hi + 0.5 * fine_step, fine_step)
    return np.unique(np.concatenate([betas, fine]))


def run_meanfield(betas_fine, delta_scale=DELTA_SCALE, alpha=1.6, gamma=1e-3,
                  kr=1.0, eps=8368.0, temperature=500.0):
    """MF-MK and Ea-MK steady-state branches over `betas_fine`.

    The chemistry (alpha, gamma, kr, eps, temperature) is the shared physics,
    so the branches match the kMC / ME-MKM phases; the mean-field model has no
    diffusion, so khop_scale does not enter. Returns the long-form dataframe
    (columns model, branch, beta, theta_empty, theta_co, theta_o) that
    plot_bifurcation expects. `betas_fine` should already include every sweep
    beta (see build_meanfield_betas).
    """
    from co_oxidation import meanfield
    mf_kw = dict(alpha=alpha, gamma=gamma, kr=kr, eps=eps, T=temperature)
    frames = []
    for model in MEANFIELD_MODELS:
        rows = {"stable_hi": [], "stable_lo": [], "unstable": []}
        for b in betas_fine:
            hi, lo, un = meanfield.branches(
                [b], model=model, delta=b * delta_scale, **mf_kw)
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


def save_meanfield_csv(branches, path):
    branches.to_csv(path, index=False)


# --- ME-MKM coexistence phase -------------------------------------------------

def build_tile(args):
    """Smallest valid square tile with `args.memkm_sites` sites for the ME-MKM run."""
    from me_mkm import TileSettings
    return TileSettings.smallest_valid_square(args.memkm_sites, True)


def run_coexistence(betas, tile, args, comm=None):
    """ME-MKM phase: per-beta coverages + basin log-ratio, then Brent-refined
    coexistence point(s) with the full spectral/committor/TPT report at each.

    Collective across `comm` (an mpi4py communicator or None): every rank runs
    the identical Brent search in lockstep because each objective evaluation is
    a globally-consistent collective solve. Returns
    (memkm_columns, coexistence_rows, report_arrays):
      - memkm_columns: dict of length-len(betas) arrays (MEMKM_COLS),
      - coexistence_rows: one flat dict per beta* for {out}_coexistence.csv,
      - report_arrays: the matching per-beta* arrays for plotting.
    """
    from co_oxidation.memkm import CoexistencePipeline

    rank = comm.Get_rank() if comm is not None else 0
    pipe = CoexistencePipeline(
        tile, comm=comm, order_species=args.memkm_order_species,
        core_frac=args.memkm_core_frac, factor=args.memkm_factor_solver,
        delta_scale=delta_scale_of(args), **physics_from_args(args))

    n = len(betas)
    cols = {key: np.full(n, np.nan) for key in MEMKM_COLS}
    for i, b in enumerate(betas):
        b = float(b)
        try:
            lr = pipe.basin_log_ratio(b)
            cov = pipe.coverages(b)
        except Exception as exc:  # deterministic across ranks -> lockstep safe
            if rank == 0:
                print(f"  [memkm] beta={b:.4g}: solve skipped ({exc})", flush=True)
            continue
        cols["log_ratio"][i] = lr
        cols["memkm_empty"][i] = cov["empty"]
        cols["memkm_co"][i] = cov["co"]
        cols["memkm_o"][i] = cov["o"]
        if rank == 0:
            print(f"  [memkm] beta={b:5.3f}  theta_CO={cov['co']:.3f} "
                  f"theta_O={cov['o']:.3f}  log10(A/B)={lr:+.3f}", flush=True)

    stars = pipe.find_coexistence(betas, cols["log_ratio"], xtol=args.memkm_brent_xtol)
    if rank == 0:
        print(f"  [memkm] coexistence point(s): "
              f"{', '.join(f'{s:.6g}' for s in stars) or 'none'}", flush=True)

    rows, arrays = [], []
    for bstar in stars:
        row, arr = pipe.report(bstar, n_eigs=args.memkm_n_eigs)
        rows.append(row)
        arrays.append(arr)
        if rank == 0:
            print(f"  [memkm] beta*={bstar:.6g}  k(A->B)={row['k_AB']:.4e}  "
                  f"k(B->A)={row['k_BA']:.4e}  F={row['flux_F']:.4e}", flush=True)
    return cols, rows, arrays


def save_coexistence_csv(rows, path):
    """Write the per-beta* coexistence report rows; empty file with a header
    note is skipped (no crossings found)."""
    if not rows:
        return False
    pd.DataFrame(rows).to_csv(path, index=False)
    return True


def build_argparser(description):
    ap = argparse.ArgumentParser(description=description)

    # Shared sweep controls (physics + book-keeping common to every phase).
    ap.add_argument("--out", default="co_oxidation", help="output file prefix")
    ap.add_argument("--beta-min", type=float, default=0.0)
    ap.add_argument("--beta-max", type=float, default=10.0)
    ap.add_argument("--beta-step", type=float, default=0.4)
    ap.add_argument("--delta-scale", type=float, default=DELTA_SCALE,
                    help="O2 desorption scale: delta = delta_scale * beta. "
                         "Applied to the kMC sweep, the ME-MKM model and the "
                         "mean-field branches alike")
    ap.add_argument("--delta-scale-beta", action=argparse.BooleanOptionalAction,
                    default=True,
                    help="O2 desorption rate delta = delta_scale * beta per "
                         "task (default), shared by all three phases. Pass "
                         "--no-delta-scale-beta for the fixed delta=0 "
                         "irreversible-O2 case")
    ap.add_argument("--plot", action=argparse.BooleanOptionalAction, default=True,
                    help="render figures after the run (default): the "
                         "bifurcation/rate plots, plus the ME-MKM spectral "
                         "diagnostics at each beta*. Rank 0 only; --no-plot skips")

    # Shared physics: one set of chemistry threaded into the kMC model, the
    # ME-MKM generator and the mean-field branches, so all three phases agree.
    phys = ap.add_argument_group("shared physics (kMC + ME-MKM + mean field)")
    phys.add_argument("--alpha", type=float, default=1.6,
                      help="CO adsorption rate, s^-1")
    phys.add_argument("--gamma", type=float, default=1e-3,
                      help="CO desorption prefactor, s^-1")
    phys.add_argument("--kr", type=float, default=1.0,
                      help="CO + O reaction prefactor, s^-1")
    phys.add_argument("--khop-scale", "--kmc-khop-scale", dest="khop_scale",
                      type=float, default=1000.0,
                      help="fast-diffusion factor: khop = khop_scale*max(beta, alpha)")
    phys.add_argument("--eps", type=float, default=8368.0,
                      help="CO-CO nearest-neighbour repulsion, J/mol")
    phys.add_argument("--temperature", "--temp", dest="temperature",
                      type=float, default=500.0, help="temperature, K")

    # Phase toggles: turn any phase off entirely.
    phases = ap.add_argument_group("phase toggles")
    phases.add_argument("--no-kmc", action="store_true",
                        help="skip the kMC sweep (its coverage columns stay NaN)")
    phases.add_argument("--no-memkm", "--no-coexistence", dest="no_memkm",
                        action="store_true",
                        help="skip the ME-MKM / SLEPc coexistence phase "
                             "(--no-coexistence is a deprecated alias)")
    phases.add_argument("--no-meanfield", action="store_true",
                        help="skip the MF-MK / Ea-MK mean-field branch phase "
                             "(no {out}_meanfield.csv, no bifurcation/rate lines)")

    # kMC phase parameters.
    kmc = ap.add_argument_group("kMC parameters")
    kmc.add_argument("--kmc-L", "--L", dest="kmc_L", type=int, default=16,
                     help="lattice edge length")
    kmc.add_argument("--kmc-tmax", "--tmax", dest="kmc_tmax", type=float,
                     default=30.0, help="kMC time limit, s")
    kmc.add_argument("--kmc-max-steps", "--max-steps", dest="kmc_max_steps",
                     type=int, default=1_000_000_000,
                     help="event limit (whichever of tmax/max-steps hits first)")
    kmc.add_argument("--kmc-seed", "--seed", dest="kmc_seed", type=int, default=0,
                     help="base RNG seed for the kMC tasks")
    kmc.add_argument("--kmc-sample-interval", "--sample-interval",
                     dest="kmc_sample_interval", type=int, default=10_000,
                     help="record coverages every this many events")

    # ME-MKM phase parameters.
    memkm = ap.add_argument_group("ME-MKM parameters")
    memkm.add_argument("--memkm-sites", "--sites", dest="memkm_sites", type=int,
                       default=8,
                       help="ME-MKM tile site count (smallest valid square tile)")
    memkm.add_argument("--memkm-order-species", "--order-species",
                       dest="memkm_order_species", default="CO",
                       help="species whose coverage orients the slow eigenvector "
                            "(fixes which branch is 'basin A')")
    memkm.add_argument("--memkm-core-frac", "--core-frac", dest="memkm_core_frac",
                       type=float, default=0.1,
                       help="coverage tolerance defining the committor's basin "
                            "cores: a core is the states within core_frac of a "
                            "full monolayer of its species (must be < 0.5). Keep "
                            "it small: widening the cores pulls transition-region "
                            "states into the basins and inflates the rates -- "
                            "check (k_AB+k_BA)/|lambda_2| stays near 1")
    memkm.add_argument("--memkm-n-eigs", "--n-eigs-report", dest="memkm_n_eigs",
                       type=int, default=20,
                       help="left eigenpairs solved at each coexistence point "
                            "(the sweep/Brent inner loop needs no eigensolve)")
    memkm.add_argument("--memkm-brent-xtol", "--brent-xtol",
                       dest="memkm_brent_xtol", type=float, default=1e-5,
                       help="Brent tolerance on (log-)beta for beta*")
    memkm.add_argument("--memkm-factor-solver", "--factor-solver",
                       dest="memkm_factor_solver", default=None,
                       help="override the PETSc LU solver (mumps/superlu_dist/"
                            "pastix/petsc); default auto-selects the best available")
    memkm.add_argument("--memkm-coexistence-out", "--coexistence-out",
                       dest="memkm_coexistence_out", default=None,
                       help="coexistence CSV path (default {out}_coexistence.csv)")

    # Mean-field phase parameters.
    mf = ap.add_argument_group("mean-field parameters")
    mf.add_argument("--meanfield-beta-step", "--beta-fine-step",
                    dest="meanfield_beta_step", type=float, default=0.05,
                    help="fill-in beta spacing for the smooth MF/Ea branches "
                         "(the sweep betas are always included on top of this grid)")

    return ap


def maybe_plot_sweep(sweep, betas, args, delta_scale, branches):
    """Render the bifurcation + rate figures from the finished sweep, at the
    delta the sweep actually used. `branches` is the mean-field dataframe (or
    None when --no-meanfield: the kMC points are still drawn, the MF/Ea lines
    and the rate figure are skipped). Called on rank 0 only; a missing
    matplotlib is a no-op with a note."""
    try:
        import pandas as pd
        from sweeps.plotting import plot_sweep
    except ImportError:
        print("  [plot] matplotlib not installed; skipping sweep figures")
        return
    df = pd.DataFrame({"beta": betas, "L": args.kmc_L, **sweep})
    for path in plot_sweep(df, args.out, branches=branches,
                           delta_scale=delta_scale,
                           mf_physics=meanfield_physics_from_args(args)):
        print(f"  [plot] wrote {path}")


def maybe_plot_coexistence(cols, rows, arrays, betas, out_prefix):
    """Render the per-beta* spectral diagnostics if matplotlib is available.
    Called on rank 0 only; a missing `plot` extra is a no-op with a note."""
    if not rows:
        return
    try:
        from sweeps.plotting import plot_coexistence
    except ImportError:
        print("  [plot] matplotlib not installed (pip install '.[plot]'); "
              "skipping figures")
        return
    for i, arr in enumerate(arrays):
        tag = "" if len(arrays) == 1 else f"-{i}"
        plot_coexistence(arr, betas, cols["log_ratio"], out_prefix, tag=tag)
    print(f"  [plot] wrote spectral diagnostics for {len(arrays)} "
          f"coexistence point(s)")


def params_from_args(args):
    return KMCParams(L=args.kmc_L, t_max=args.kmc_tmax,
                     max_steps=args.kmc_max_steps,
                     khop_scale=args.khop_scale, seed=args.kmc_seed,
                     sample_interval=args.kmc_sample_interval,
                     alpha=args.alpha, gamma=args.gamma, kr=args.kr,
                     eps=args.eps, T=args.temperature)
