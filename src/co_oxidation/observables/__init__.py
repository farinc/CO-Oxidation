"""Pluggable functions that act on the simulations.

Write one, decorate it, and it runs at every grid step of a sweep and lands in the
workbook -- no edits to the kMC loop, the ME-MKM pipeline, the sweep drivers, the
workbook writer or the CLI:

    # my_observables.py
    import numpy as np
    from co_oxidation.observables import observable

    @observable("co_susceptibility", family="memkm")
    def co_susceptibility(ctx):
        '''Variance of the CO count under the stationary distribution.'''
        n = ctx.coverage("CO") * ctx.builder.l
        mean = ctx.theta @ n
        return {"chi": float(ctx.theta @ (n - mean) ** 2)}

    $ uv run python -m sweeps.linear --memkm \
        --observable-module my_observables.py --observable co_susceptibility

`family` picks which simulation the function acts on and therefore what `ctx` is:

    "kmc"    KMCContext   -- the trajectories already run at this point, plus
                            ctx.run(...) and ctx.run_first_passage(...) to run more
    "memkm"  MemkmContext -- the solved stationary distribution, slow eigenpairs and
                            spectral basin masks, plus ctx.W for a linear solve

Return a mapping of numbers; scalars become columns of the workbook's Observables
sheet, arrays become a per-step S{i}_OBS sheet. Keys are namespaced by the
observable's name, so nothing collides. `co_oxidation.observables.builtin` ships one
worked example per family to copy from.
"""

from . import builtin  # noqa: F401  -- importing registers the shipped examples
from .context import KMCContext, MemkmContext
from .registry import (
    FAMILIES,
    KMC,
    MEMKM,
    Observable,
    ObservableError,
    array_items,
    available,
    compute,
    get,
    is_scalar,
    load_modules,
    observable,
    register,
    resolve,
    scalar_items,
    unregister,
)

__all__ = [
    "FAMILIES", "KMC", "MEMKM", "Observable", "ObservableError", "KMCContext",
    "MemkmContext", "observable", "register", "unregister", "get", "available",
    "resolve", "compute", "load_modules", "is_scalar", "scalar_items", "array_items",
]
