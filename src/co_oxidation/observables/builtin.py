"""Two reference observables, one per family, kept deliberately small.

They exist to pin the plumbing down with something real -- and to be the thing you copy
when writing your own. Between them they show both shapes an observable takes: reading
what the sweep already computed (kmc_branching) and reweighting a solved distribution
(coverage_moments). Neither needs an extra solve, so both are free to leave on.

An observable that must *drive* the simulation instead of reading it goes through
ctx.run() / ctx.run_first_passage() -- see KMCContext.
"""

import numpy as np

from .registry import KMC, MEMKM, observable


@observable("kmc_branching", family=KMC)
def kmc_branching(ctx):
    """How this point's kMC trajectories split between the CO-rich and O-rich branches.

    Purely post-hoc: it reads the trajectories the sweep already ran (ctx.results) and
    costs nothing. Near a bistable point the two initial conditions land on different
    branches, so frac_co_rich sits strictly between 0 and 1 and steady_co_std is large;
    on a monostable point every trajectory agrees.
    """
    results = list(ctx.results)
    if not results:
        return {"n_trajectories": 0}
    steady_co = np.array([r.steady_co for r in results], float)
    steady_o = np.array([r.steady_o for r in results], float)
    return {
        "n_trajectories": len(results),
        "frac_co_rich": float(np.mean(steady_co > steady_o)),
        "steady_co_mean": float(steady_co.mean()),
        "steady_co_std": float(steady_co.std()),
        "steady_o_mean": float(steady_o.mean()),
        "steady_o_std": float(steady_o.std()),
        "frac_stuck": float(np.mean([bool(r.stuck) for r in results])),
    }


@observable("coverage_moments", family=MEMKM)
def coverage_moments(ctx):
    """Fluctuations of the adsorbate counts under the ME-MKM stationary distribution.

    Pure numpy over theta and the per-microstate coverages -- no second solve, so this
    is what a cheap ME-MKM observable looks like. The variances are the coverage
    susceptibilities and peak where the distribution goes bimodal, which makes them an
    independent read on the same transition the spectral basins are tracking.
    """
    theta = np.asarray(ctx.theta, float)
    l = ctx.builder.l
    adsorbates = ctx.builder.species_names[1:]     # index 0 is the vacancy
    counts = {name: np.asarray(ctx.coverage(name), float) * l for name in adsorbates}

    out = {}
    means = {}
    for name, n in counts.items():
        mean = float(theta @ n)
        means[name] = mean
        out[f"n_{name}_mean"] = mean
        out[f"n_{name}_var"] = float(theta @ (n - mean) ** 2)
    for i, a in enumerate(adsorbates):
        for b in adsorbates[i + 1:]:
            out[f"cov_{a}_{b}"] = float(
                theta @ ((counts[a] - means[a]) * (counts[b] - means[b])))
    return out
