"""
Single-run CLI: one (L, k_o_ads, init, seed, ...) -> one kMC trajectory.

Usage:
    uv run kmc-run --L 16 --k-o-ads 5.0 --init full
    uv run kmc-run --L 24 --k-o-ads 4.0 --init empty --seed 1 --tmax 15
"""

import argparse
import time

from .kmc import KMCParams, run_kmc


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--L", type=int, default=16, help="lattice edge length")
    ap.add_argument("--k-o-ads", type=float, required=True,
                    help="O2 impingement rate")
    ap.add_argument("--init", choices=("empty", "full"), default="empty",
                    help="initial lattice state")
    ap.add_argument("--seed", type=int, default=-1)
    ap.add_argument("--tmax", type=float, default=30.0, help="kMC time limit, s")
    ap.add_argument("--max-steps", type=int, default=1_000_000_000,
                    help="event limit (whichever hits first)")
    ap.add_argument("--khop-scale", type=float, default=1000.0,
                    help="khop = khop_scale * max(k_o_ads, k_co_ads)")
    ap.add_argument("--sample-interval", type=int, default=10_000)
    ap.add_argument("--k-co-ads", type=float, default=1.6, help="CO adsorption rate")
    ap.add_argument("--k-co-des", type=float, default=1e-3, help="CO desorption prefactor")
    ap.add_argument("--k-rxn", type=float, default=1.0, help="CO+O reaction prefactor")
    ap.add_argument("--k-o-des", type=float, default=0.0, help="O2 desorption rate")
    ap.add_argument("--eps", type=float, default=8368.0, help="CO-CO NN repulsion, J/mol")
    args = ap.parse_args()

    params = KMCParams(L=args.L, k_co_ads=args.k_co_ads, k_co_des=args.k_co_des, k_rxn=args.k_rxn,
                       k_o_des=args.k_o_des, eps=args.eps, t_max=args.tmax,
                       max_steps=args.max_steps, khop_scale=args.khop_scale,
                       sample_interval=args.sample_interval, seed=args.seed)

    t0 = time.perf_counter()
    res = run_kmc(args.k_o_ads, init=args.init, params=params)
    wall = time.perf_counter() - t0

    note = " [absorbing state]" if res.stuck else ""
    print(f"k_o_ads={args.k_o_ads:5.2f} init={args.init:5s}  theta_CO={res.steady_co:.3f} "
          f"theta_O={res.steady_o:.3f}  ({res.steps:.3e} events, "
          f"{wall:.1f} s, {res.steps / max(wall, 1e-9):.2e} events/s)"
          f"{note}")


if __name__ == "__main__":
    main()
