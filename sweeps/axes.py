"""Sweep-axis specification and the N-D grid it generates.

Any RunConfig field can be designated a swept axis via AxisSpec.parse (the CLI's
repeatable --sweep NAME=SPEC flag, sweeps/cli.py). SweepGrid.build takes the Cartesian
product of every designated axis; each grid point is a GridStep with a fully-resolved
RunConfig."""

from dataclasses import dataclass, fields, replace
from itertools import product

import numpy as np

from .params import RunConfig

_FIELD_NAMES = frozenset(f.name for f in fields(RunConfig))
_INT_AXES = frozenset({
    "kmc_L", "kmc_max_steps", "kmc_sample_interval", "kmc_seed",
    "kmc_n_trajectories", "memkm_sites",
})


@dataclass(frozen=True)
class AxisSpec:
    name: str
    values: tuple[float, ...]

    @classmethod
    def parse(cls, spec: str) -> "AxisSpec":
        """'NAME=START:STOP:STEP' (inclusive range, like np.arange with a half-step
        epsilon) or 'NAME=V1,V2,V3' (explicit list). NAME must be a RunConfig field."""
        if "=" not in spec:
            raise ValueError(f"--sweep {spec!r} must be NAME=... (range or comma list)")
        name, rhs = spec.split("=", 1)
        if name not in _FIELD_NAMES:
            raise ValueError(
                f"--sweep {spec!r}: {name!r} is not a RunConfig parameter; "
                f"choose from {sorted(_FIELD_NAMES)}")
        cast = int if name in _INT_AXES else float
        if ":" in rhs:
            parts = rhs.split(":")
            if len(parts) != 3:
                raise ValueError(f"--sweep {spec!r}: range must be START:STOP:STEP")
            start, stop, step = (float(p) for p in parts)
            if step == 0:
                raise ValueError(f"--sweep {spec!r}: STEP must be nonzero")
            values = tuple(cast(v) for v in np.arange(start, stop + 0.5 * step, step))
        else:
            values = tuple(cast(v) for v in rhs.split(","))
        if not values:
            raise ValueError(f"--sweep {spec!r} produced no values")
        return cls(name=name, values=values)


@dataclass(frozen=True)
class GridStep:
    index: int
    axis_values: dict[str, float]   # swept axes only, at this step
    config: RunConfig               # base + this step's overrides, fully resolved


@dataclass(frozen=True)
class SweepGrid:
    base: RunConfig
    axes: list[AxisSpec]
    steps: list[GridStep]

    @classmethod
    def build(cls, base: RunConfig, axes: list[AxisSpec]) -> "SweepGrid":
        if not axes:
            return cls(base=base, axes=[], steps=[GridStep(0, {}, base)])
        names = [a.name for a in axes]
        steps = []
        for i, combo in enumerate(product(*(a.values for a in axes))):
            axis_values = dict(zip(names, combo))
            steps.append(GridStep(index=i, axis_values=axis_values,
                                   config=replace(base, **axis_values)))
        return cls(base=base, axes=list(axes), steps=steps)
