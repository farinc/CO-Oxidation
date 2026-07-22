"""Shared task list, single-task runner and CSV writer for the beta sweeps.

Both sweeps/linear.py (serial) and sweeps/mpi.py (MPI-parallel) build the
same (idx, beta, init, seed) task list and combine per-task results into the
same {out}_kmc_sweep.csv shape, so the two drivers stay interchangeable.
"""

import argparse

import numpy as np
import pandas as pd

from co_oxidation import KMCParams, run_kmc

TAGS = ("empty", "full")

# ME-MKM sweep columns added alongside the kMC coverages in {out}_kmc_sweep.csv.
MEMKM_COLS = ("memkm_empty", "memkm_co", "memkm_o", "log_ratio")


def build_betas(beta_min, beta_max, beta_step):
    return np.arange(beta_min, beta_max + 0.5 * beta_step, beta_step)


def build_tasks(betas, seed):
    """List of (idx, beta, init_tag, seed), two entries per beta."""
    tasks = []
    for i, b in enumerate(betas):
        for j, tag in enumerate(TAGS):
            tasks.append((i, float(b), tag, seed + 2 * i + j))
    return tasks


def run_task(task, params, delta_scale_beta=False, verbose_prefix=""):
    """Run one (idx, beta, init, seed) task, returning (idx, tag, steady coverages).

    delta_scale_beta ties the O2 desorption rate to the impingement rate,
    delta = beta * 1e-4, instead of the fixed params.delta (0 by default).
    """
    idx, beta, tag, seed = task
    overrides = {"delta": beta * 1e-4} if delta_scale_beta else {}
    res = run_kmc(beta, init=tag, params=params, seed=seed, **overrides)
    note = " [absorbing state]" if res.stuck else ""
    print(f"{verbose_prefix}beta={beta:5.2f} init={tag:5s}  "
          f"theta_CO={res.steady_co:.3f} theta_O={res.steady_o:.3f}  "
          f"({res.steps:.3e} events){note}", flush=True)
    return idx, tag, res.steady_empty, res.steady_co, res.steady_o


def assemble(betas, results):
    """Combine run_task() outputs into the {out}_kmc_sweep.csv dict shape."""
    n = len(betas)
    out = {key: np.full(n, np.nan) for key in
           ("e_empty", "co_empty", "o_empty", "e_full", "co_full", "o_full")}
    for idx, tag, e, co, o in results:
        out[f"e_{tag}"][idx] = e
        out[f"co_{tag}"][idx] = co
        out[f"o_{tag}"][idx] = o
    return out


def save_sweep_csv(betas, sweep, L, path):
    df = pd.DataFrame({"beta": betas, "L": L, **sweep})
    df.to_csv(path, index=False)


# --- ME-MKM coexistence phase -------------------------------------------------

def build_tile(args):
    """Smallest valid square tile with `args.sites` sites for the ME-MKM run."""
    from me_mkm import TileSettings
    return TileSettings.smallest_valid_square(args.sites, True)


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
        tile, comm=comm, order_species=args.order_species,
        core_frac=args.core_frac, n_eigs_scan=args.n_eigs_scan,
        factor=args.factor_solver)

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

    stars = pipe.find_coexistence(betas, cols["log_ratio"], xtol=args.brent_xtol)
    if rank == 0:
        print(f"  [memkm] coexistence point(s): "
              f"{', '.join(f'{s:.6g}' for s in stars) or 'none'}", flush=True)

    rows, arrays = [], []
    for bstar in stars:
        row, arr = pipe.report(bstar, n_eigs=args.n_eigs_report)
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
    ap.add_argument("--L", type=int, default=16, help="lattice edge length")
    ap.add_argument("--tmax", type=float, default=30.0, help="kMC time limit, s")
    ap.add_argument("--max-steps", type=int, default=1_000_000_000,
                    help="event limit (whichever hits first)")
    ap.add_argument("--beta-min", type=float, default=0.0)
    ap.add_argument("--beta-max", type=float, default=10.0)
    ap.add_argument("--beta-step", type=float, default=0.4)
    ap.add_argument("--khop-scale", type=float, default=1000.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sample-interval", type=int, default=10_000)
    ap.add_argument("--delta-scale-beta", action="store_true",
                    help="O2 desorption rate delta = beta * 1e-4 per task, "
                         "instead of the fixed delta=0 (irreversible O2 ads)")
    ap.add_argument("--out", default="co_oxidation", help="output file prefix")

    # ME-MKM coexistence phase
    ap.add_argument("--no-coexistence", action="store_true",
                    help="run the kMC sweep only; skip the ME-MKM / SLEPc phase")
    ap.add_argument("--sites", type=int, default=8,
                    help="ME-MKM tile site count (smallest valid square tile)")
    ap.add_argument("--order-species", default="CO",
                    help="species whose coverage orients the slow eigenvector "
                         "(fixes which branch is 'basin A')")
    ap.add_argument("--core-frac", type=float, default=0.1,
                    help="plateau fraction of the phi_2^L span defining basin cores")
    ap.add_argument("--n-eigs-scan", type=int, default=6,
                    help="left eigenpairs solved per beta during the sweep/Brent")
    ap.add_argument("--n-eigs-report", type=int, default=20,
                    help="left eigenpairs solved at each coexistence point")
    ap.add_argument("--brent-xtol", type=float, default=1e-5,
                    help="Brent tolerance on (log-)beta for beta*")
    ap.add_argument("--factor-solver", default=None,
                    help="override the PETSc LU solver (mumps/superlu_dist/"
                         "pastix/petsc); default auto-selects the best available")
    ap.add_argument("--coexistence-out", default=None,
                    help="coexistence CSV path (default {out}_coexistence.csv)")
    ap.add_argument("--plot", action="store_true",
                    help="also render the ME-MKM spectral diagnostic figures "
                         "at each beta* (needs the 'plot' extra; rank 0 only)")
    return ap


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
    return KMCParams(L=args.L, t_max=args.tmax, max_steps=args.max_steps,
                     khop_scale=args.khop_scale, seed=args.seed,
                     sample_interval=args.sample_interval)
