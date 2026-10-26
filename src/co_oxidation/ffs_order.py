"""Order parameters for forward flux sampling, selectable by name.

An order parameter is an njit function of the lattice and its running counts that
returns a float. It must be low in basin A (reactive) and high in basin B (CO-rich):

    from numba import njit
    from co_oxidation.ffs_order import order_parameter

    @order_parameter("theta_co_weighted", kind="counts")
    @njit
    def theta_co_weighted(lat, L, n_empty, n_co, n_o):
        '''CO coverage minus half the O coverage.'''
        return (n_co - 0.5 * n_o) / (L * L)

`kind="counts"` order parameters only read the counts, so the kMC kernel evaluates
them after every event. `kind="lattice"` ones read `lat` and are usually O(N), so by
default they are evaluated once every N events. That stride means a trial can step
past an interface before it is noticed.

Name one on the command line with --ffs-order-parameter, and load your own module
with --ffs-order-module.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass

import numpy as np
from numba import njit

from .common import CO, EMPTY, O

COUNTS = "counts"
LATTICE = "lattice"
KINDS = (COUNTS, LATTICE)


@dataclass(frozen=True)
class OrderParameter:
    """One registered order parameter plus what the kernels and workbook need."""

    name: str
    fn: Callable              # njit (lat, L, n_empty, n_co, n_o) -> float
    kind: str = COUNTS
    stride: int | None = None  # events between evaluations, None for the kind default
    units: str = "fraction of sites"
    doc: str = ""

    def resolved_stride(self, L: int) -> int:
        """Events between evaluations on an L x L lattice."""
        if self.stride is not None:
            return max(1, int(self.stride))
        return 1 if self.kind == COUNTS else L * L

    def evaluate(self, lat: np.ndarray, L: int) -> float:
        """Value of the order parameter for one lattice, from Python."""
        n_co = int(np.count_nonzero(lat == CO))
        n_o = int(np.count_nonzero(lat == O))
        n_empty = int(np.count_nonzero(lat == EMPTY))
        return float(self.fn(lat, L, n_empty, n_co, n_o))


_REGISTRY: dict[str, OrderParameter] = {}


def register(name: str, fn: Callable, kind: str = COUNTS, stride: int | None = None,
             units: str = "fraction of sites", replace: bool = False) -> OrderParameter:
    """Add an order parameter to the registry and return its record."""
    if kind not in KINDS:
        raise ValueError(f"order parameter {name!r}: kind must be one of {KINDS}")
    if name in _REGISTRY and not replace:
        raise ValueError(f"order parameter {name!r} is already registered")
    doc = (getattr(fn, "__doc__", None) or getattr(getattr(fn, "py_func", None),
                                                    "__doc__", None) or "").strip()
    record = OrderParameter(name=name, fn=fn, kind=kind, stride=stride, units=units,
                            doc=doc.splitlines()[0] if doc else "")
    _REGISTRY[name] = record
    return record


def order_parameter(name: str | None = None, *, kind: str = COUNTS,
                    stride: int | None = None, units: str = "fraction of sites",
                    replace: bool = False):
    """Decorator form of register(). Put it above @njit."""
    def decorate(fn):
        fn_name = name or getattr(fn, "__name__", None) or fn.py_func.__name__
        register(fn_name, fn, kind=kind, stride=stride, units=units, replace=replace)
        return fn
    return decorate


def get(name: str) -> OrderParameter:
    if name not in _REGISTRY:
        raise KeyError(f"unknown order parameter {name!r}, choose from {sorted(_REGISTRY)}")
    return _REGISTRY[name]


def available() -> list[OrderParameter]:
    return [_REGISTRY[k] for k in sorted(_REGISTRY)]


def load_modules(specs: Iterable[str]) -> list[str]:
    """Import user modules so their @order_parameter decorators run.

    Same rules as observable modules: a dotted import path or a path to a .py file."""
    from .observables.registry import load_modules as _load
    return _load(specs)


def describe_available() -> str:
    lines = ["Registered order parameters:"]
    for op in available():
        lines.append(f"  {op.name:22s} [{op.kind}] {op.doc}")
    return "\n".join(lines)


# --- clusters ---------------------------------------------------------------------

@njit(cache=True)
def co_clusters(lat, L):
    """Largest CO cluster on the periodic L x L lattice and whether it wraps.

    Clusters are 4-neighbour connected. A cluster wraps when the search reaches one
    of its sites by two routes whose unwrapped positions differ, which means it
    closes on itself around the torus. Returns (largest_size, largest_wraps)."""
    N = L * L
    seen = np.zeros(N, np.bool_)
    ux = np.zeros(N, np.int64)      # unwrapped coordinates from the cluster seed
    uy = np.zeros(N, np.int64)
    stack = np.empty(N, np.int64)
    best = 0
    best_wraps = False
    dxs = (1, -1, 0, 0)
    dys = (0, 0, 1, -1)
    for s in range(N):
        if lat[s] != CO or seen[s]:
            continue
        seen[s] = True
        ux[s] = 0
        uy[s] = 0
        top = 0
        stack[top] = s
        top += 1
        size = 0
        wraps = False
        while top > 0:
            top -= 1
            i = stack[top]
            size += 1
            row = i // L
            col = i % L
            for k in range(4):
                nx = ux[i] + dxs[k]
                ny = uy[i] + dys[k]
                j = ((row + dys[k]) % L) * L + (col + dxs[k]) % L
                if lat[j] != CO:
                    continue
                if seen[j]:
                    if ux[j] != nx or uy[j] != ny:
                        wraps = True
                    continue
                seen[j] = True
                ux[j] = nx
                uy[j] = ny
                stack[top] = j
                top += 1
        if size > best:
            best = size
            best_wraps = wraps
    return best, best_wraps


# --- built-in order parameters ------------------------------------------------------

@order_parameter("theta_co", kind=COUNTS)
@njit(cache=True)
def theta_co(lat, L, n_empty, n_co, n_o):
    """CO coverage n_CO / N, the choice of Adams, Ziff and Sander (2010)."""
    return n_co / (L * L)


@order_parameter("theta_co_minus_o", kind=COUNTS)
@njit(cache=True)
def theta_co_minus_o(lat, L, n_empty, n_co, n_o):
    """(n_CO - n_O) / N, which also pushes O-rich states away from B."""
    return (n_co - n_o) / (L * L)


@order_parameter("largest_co_cluster", kind=LATTICE)
@njit(cache=True)
def largest_co_cluster(lat, L, n_empty, n_co, n_o):
    """Size of the largest periodic CO cluster over N, a nucleation coordinate."""
    size, _ = co_clusters(lat, L)
    return size / (L * L)
