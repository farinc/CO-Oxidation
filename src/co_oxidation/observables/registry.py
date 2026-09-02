"""The observable registry: the one place a new function that acts on a simulation has
to announce itself.

An *observable* is a plain function of a single context object that returns numbers:

    from co_oxidation.observables import observable

    @observable("my_thing", family="kmc")
    def my_thing(ctx):
        '''One line of help, shown by --list-observables.'''
        return {"mean_co": float(np.mean([t.steady_co for t in ctx.results]))}

Two families exist because there are two simulations to act on, and they hand you
different things (see .context):

  family="kmc"    ctx is a KMCContext: the trajectories already run at this parameter
                  point, plus ctx.run(...)/ctx.run_first_passage(...) to launch more.
  family="memkm"  ctx is a MemkmContext: the solved ME-MKM state -- stationary
                  distribution, slow eigenpairs, spectral basin masks -- plus ctx.W,
                  the generator itself, for anything that needs a linear solve.

The return value is a mapping {key: scalar-or-array}; a bare scalar/array is also
accepted and takes the observable's own name. Keys are namespaced as "{name}.{key}" in
the sweep output, so two observables can both return "tau" without colliding.

Nothing here imports PETSc, numba or a sweep: importing the registry is cheap, so the
CLI can list observables without paying for any backend.
"""

from __future__ import annotations

import importlib
import importlib.util
import sys
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

KMC = "kmc"
MEMKM = "memkm"
FAMILIES = (KMC, MEMKM)


class ObservableError(RuntimeError):
    """An observable failed. Carries the observable's name so a sweep can report which
    one broke without losing the original traceback (chained as __cause__)."""

    def __init__(self, name: str, cause: BaseException):
        super().__init__(f"observable {name!r} failed: {cause}")
        self.name = name


@dataclass(frozen=True)
class Observable:
    """One registered function plus the metadata the CLI and workbook need."""

    name: str
    family: str
    fn: Callable[[Any], Any]
    doc: str

    def compute(self, ctx) -> dict[str, Any]:
        """Call the function and normalize whatever it returned into a flat
        {"{name}.{key}": value} dict (or {name: value} for a bare return)."""
        out = self.fn(ctx)
        if isinstance(out, Mapping):
            return {f"{self.name}.{k}": v for k, v in out.items()}
        return {self.name: out}


_REGISTRY: dict[str, Observable] = {}


def register(name: str, family: str, fn: Callable[[Any], Any],
             doc: str | None = None, replace: bool = False) -> Observable:
    """Register `fn` under `name`. Prefer the @observable decorator; this is the
    functional form, for registering something you did not write."""
    if family not in FAMILIES:
        raise ValueError(f"observable {name!r}: family must be one of {FAMILIES}, "
                         f"got {family!r}")
    if name in _REGISTRY and not replace:
        raise ValueError(
            f"an observable named {name!r} is already registered (family "
            f"{_REGISTRY[name].family!r}). Pick another name, or pass replace=True "
            "to deliberately shadow it.")
    doc = doc if doc is not None else (fn.__doc__ or "").strip().split("\n")[0]
    obs = Observable(name=name, family=family, fn=fn, doc=doc)
    _REGISTRY[name] = obs
    return obs


def observable(name: str | None = None, *, family: str, replace: bool = False):
    """Decorator registering the wrapped function as an observable of `family`.

    The function is returned unchanged, so it stays directly callable and testable."""
    def decorate(fn):
        register(name or fn.__name__, family, fn, replace=replace)
        return fn
    return decorate


def unregister(name: str) -> None:
    _REGISTRY.pop(name, None)


def get(name: str) -> Observable:
    try:
        return _REGISTRY[name]
    except KeyError:
        known = ", ".join(sorted(_REGISTRY)) or "(none registered)"
        raise KeyError(f"no observable named {name!r}; known: {known}") from None


def available(family: str | None = None) -> list[Observable]:
    """Every registered observable, optionally filtered to one family, name-sorted."""
    obs = [o for o in _REGISTRY.values() if family is None or o.family == family]
    return sorted(obs, key=lambda o: o.name)


def resolve(names: Iterable[str], family: str | None = None) -> list[Observable]:
    """Look up `names`, keeping the caller's order. With `family` given, names from
    other families are dropped -- that is how one --observable list is fed to both the
    kMC and the ME-MKM phase of a sweep without the caller sorting them by hand."""
    out = []
    for n in names:
        obs = get(n)
        if family is None or obs.family == family:
            out.append(obs)
    return out


def compute(names: Iterable[str], ctx, family: str | None = None,
            strict: bool = False, on_error=None) -> dict[str, Any]:
    """Run every named observable of `family` against `ctx` and merge the results.

    A failing observable does not take the sweep down with it unless `strict`: by
    default its exception is reported through `on_error(ObservableError)` (or printed)
    and the run continues, with "{name}.error" recorded in place of its values. Losing
    one derived number is not a reason to throw away hours of kMC and SLEPc work."""
    values: dict[str, Any] = {}
    for obs in resolve(names, family):
        try:
            values.update(obs.compute(ctx))
        except Exception as exc:
            if strict:
                raise ObservableError(obs.name, exc) from exc
            err = ObservableError(obs.name, exc)
            if on_error is not None:
                on_error(err)
            else:
                print(f"  [observable] {err}", flush=True)
            values[f"{obs.name}.error"] = f"{type(exc).__name__}: {exc}"
    return values


def load_modules(specs: Iterable[str]) -> list[str]:
    """Import each spec so its @observable decorators run, and return the module names.

    A spec is either an importable dotted path (my_package.my_observables) or a path to
    a .py file, which is loaded under its file stem. The file form is the point: an
    observable can live in a single scratch script next to the run that needs it,
    with nothing installed and nothing added to this package."""
    loaded = []
    for spec in specs:
        if spec.endswith(".py") or "/" in spec or "\\" in spec:
            path = Path(spec).expanduser().resolve()
            if not path.is_file():
                raise FileNotFoundError(f"--observable-module {spec!r}: no such file")
            mod_name = f"co_oxidation_observable_plugin_{path.stem}"
            module_spec = importlib.util.spec_from_file_location(mod_name, path)
            module = importlib.util.module_from_spec(module_spec)
            sys.modules[mod_name] = module          # so dataclasses/pickle can find it
            module_spec.loader.exec_module(module)
            loaded.append(str(path))
        else:
            importlib.import_module(spec)
            loaded.append(spec)
    return loaded


# --- value helpers, shared by the workbook writer and the plots -----------------------

def is_scalar(value) -> bool:
    """Whether a value belongs in a one-row-per-step table rather than its own column
    block. Anything numpy-shaped with more than one element is an array."""
    if isinstance(value, (str, bool, int, float)):
        return True
    shape = getattr(value, "shape", None)
    if shape is None:
        return True
    return len(shape) == 0


def scalar_items(values: Mapping[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in values.items() if is_scalar(v)}


def array_items(values: Mapping[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in values.items() if not is_scalar(v)}
