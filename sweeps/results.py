"""In-memory result model for one sweep run, shared by sweeps.excel (the workbook
writer/reader) and sweeps.plotting -- neither re-derives data the other already holds."""

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .axes import GridStep, SweepGrid


@dataclass
class KMCTrajectoryResult:
    tag: str                # "empty" | "full"
    index: int               # 0..n_trajectories-1 within this tag
    seed: int
    times: np.ndarray
    cov_empty: np.ndarray
    cov_co: np.ndarray
    cov_o: np.ndarray
    steady_empty: float
    steady_co: float
    steady_o: float
    t_final: float
    steps: int
    stuck: bool


@dataclass
class MeanFieldModelResult:
    model: str               # "mf" | "ea"
    tag: str                  # "empty" | "full"
    t: np.ndarray
    theta_co: np.ndarray
    theta_o: np.ndarray
    theta_empty: np.ndarray
    r_ads_co: np.ndarray
    r_des_co: np.ndarray
    r_ads_o: np.ndarray
    r_oxi: np.ndarray
    r_des_o: np.ndarray


@dataclass
class MemkmStepResult:
    eigvals: np.ndarray       # complex, length n_eigs_scan
    cov_pop: np.ndarray       # (l+1, l+1)
    cov_r2: np.ndarray        # (l+1, l+1)
    cov_phi: np.ndarray | None  # (l+1, l+1), None unless left modes were solved
    cov_deg: np.ndarray       # (l+1, l+1)
    theta_empty: float
    theta_co: float
    theta_o: float
    n_sites: int
    # spectral two-basin split (from the sign of psi_R_2, always computed as part of
    # the underlying solve): P_A + P_B = 1, used by the bifurcation plot's ME-MKM lines.
    p_a: float
    empty_a: float
    co_a: float
    o_a: float
    empty_b: float
    co_b: float
    o_b: float


@dataclass
class StepResult:
    step: GridStep
    kmc: list[KMCTrajectoryResult] | None
    meanfield: list[MeanFieldModelResult] | None
    memkm: MemkmStepResult | None
    # --observable values at this step, both families merged into one flat dict keyed
    # "{observable}.{key}" (see co_oxidation.observables). Empty when none were asked
    # for. Deliberately untyped in the values: an observable may return scalars (which
    # become Observables-sheet columns) or arrays (which get their own S{i}_OBS sheet),
    # and the result model does not need to know which in advance.
    observables: dict[str, Any] = field(default_factory=dict)


@dataclass
class CoexistenceCrossing:
    outer_axis_values: dict[str, float]   # non-bisection swept axes, this slice
    bisection_axis: str
    value_star: float
    row: dict                              # CoexistencePipeline.report()'s row dict
    arrays: dict                           # CoexistencePipeline.report()'s arrays dict
    scan_values: np.ndarray                # this slice's bisection-axis grid points
    scan_log_ratios: np.ndarray            # matching basin_log_ratio values


@dataclass
class SweepResult:
    grid: SweepGrid
    steps: list[StepResult]
    coexistence: list[CoexistenceCrossing] | None
