"""A worked example of writing your own observables, in a single file.

Nothing here is installed or registered in advance: point a sweep at this file and the
two functions below become available by name.

    uv run python -m sweeps.linear \
        --observable-module examples/observables_example.py \
        --observable generator_scale,restart_spread \
        --memkm --memkm-sites 8 --sweep k_o_ads=0.5:4.0:0.5

Between them they show the two things an observable can do beyond reading numbers that
are already lying around:

  generator_scale  touches the ME-MKM generator itself (ctx.W), the entry point for
                   anything that needs a linear solve or a matrix norm rather than a
                   reweighting of the stationary distribution.
  restart_spread   drives the kMC simulation (ctx.run), running trajectories the sweep
                   did not ask for, on seeds reserved so they never collide with it.
"""

import numpy as np

from co_oxidation.observables import observable


@observable("generator_scale", family="memkm")
def generator_scale(ctx):
    """Stiffness of the ME-MKM generator at this point: fastest exit rate and sparsity.

    ctx.W is the generator as a distributed PETSc matrix, built on first use and
    released when the step is done. The ratio of the fastest exit rate to |lambda_2| is
    the timescale separation the shift-invert solve has to resolve, so this is the
    number to look at when SLEPc starts struggling on a big tile.
    """
    W = ctx.W
    diagonal = W.getDiagonal()
    try:
        diagonal.abs()
        fastest = float(diagonal.max()[1])
    finally:
        diagonal.destroy()
    lambda2 = abs(float(np.asarray(ctx.eigvals)[1].real))
    return {
        "fastest_exit_rate": fastest,
        "n_states": int(W.getSize()[0]),
        "nnz": int(W.getInfo()["nz_used"]),
        "stiffness": fastest / lambda2 if lambda2 > 0 else np.inf,
    }


@observable("restart_spread", family="kmc")
def restart_spread(ctx):
    """Does the steady coverage depend on where the lattice started?

    The sweep already runs one trajectory from empty and one from CO-covered, but with
    a single seed each you cannot tell a genuine branch split from run-to-run noise.
    This runs `n_restarts` extra trajectories from each start (ctx.run, on the seed
    stream reserved for observables) and reports the gap between the two ensembles
    against their own scatter: a gap of several standard deviations is bistability, a
    gap inside the scatter is one branch being sampled twice.

    Tune with --observable-option restart_spread.n_restarts=8
    """
    n_restarts = ctx.option("restart_spread", "n_restarts", 4)
    means, spreads = {}, {}
    for k, init in enumerate(("empty", "full")):
        theta = [ctx.run(init=init, seed=ctx.seed(10 * k + i)).steady_co
                 for i in range(n_restarts)]
        means[init] = float(np.mean(theta))
        spreads[init] = float(np.std(theta))
    pooled = np.hypot(spreads["empty"], spreads["full"])
    gap = abs(means["empty"] - means["full"])
    return {
        "n_restarts": n_restarts,
        "steady_co_from_empty": means["empty"],
        "steady_co_from_full": means["full"],
        "gap": gap,
        "gap_over_spread": gap / pooled if pooled > 0 else np.inf,
    }
