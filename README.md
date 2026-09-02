# CO oxidation
Features a number of kMC, mean field, and ME-MKM kinetic models for CO oxidation following the model of [Tian & Rangarajan, *J. Phys. Chem. C* 2021, 125, 20275](https://doi.org/10.1021/acs.jpcc.1c04495)

## Model
For the purposes of using the Master Equation Microkinetic model and for using a slightly more realistic system, we add desorption of the adsorbed molecular oxygen compared to the original Tian and Rangarajan CO oxidation model. In addition, since the use of the Greek lettering is difficult to parse, the naming here uses the more common $k$ lettering with descriptive subscripts. 

<div align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/co_oxidation_latex/mechanism_white.svg">
    <source media="(prefers-color-scheme: light)" srcset="docs/co_oxidation_latex/mechanism.svg">
    <img alt="CO Oxidation Reaction Equations" src="docs/co_oxidation_latex/mechanism.svg" width="464">
  </picture>
</div>

Note the differences in arrows. The first two equation are competitive rates whereas the last two diffusion equations follow detailed balance. Hence whole system does not obey detailed balence. The model only differs from Tian & Rangarajan by the addition of a desorption step for $\mathrm{O}^\ast$. The only real reason for this is providing a model that features ergodicity, although it is physically more realistic. In genereal oxygen binds strongly to most catalyst surfaces so $k_{O,des} \ll k_{O,ads}$ and can be set to zero to restore the model of Tian & Rangarajan. Only $\mathrm{CO}^\ast$ has repuslive lateral interactions in this model. For diffusion, `khop = khop_scale * max(k_o_ads, k_co_ads)` mimics the fast-diffusion limit (the default `khop_scale` is 1000, i.e. three orders of magnitude, per the paper).

## kMC Algorithm

The kinetic Monte Carlo method is a rejection-free n-fold (BKL) following the algorithms reviewed by [Chatterjee & Vlachos, *J. Comput.-Aided Mater. Des.* 2007, 14, 253](https://doi.org/10.1007/s10820-006-9042-9) (n-fold method sec. 6.3, linear search sec. 6.1.1, local updates sec. 6.4) on a periodic square lattice. The n-fold works well here because the only interaction is nearest-neighbour, allowing every event rate to belong to one of 20 discrete classes (neighbour counts 0-4). Each kMC step selects a class by linear search over 20 cumulative weights. After an event, only the events within graph distance 2 of the changed sites are re-classified (local update), so the cost per event is O(1) and is independent of lattice size. Class membership uses swap-with-last lists for O(1) add/remove.

## Development

I would highly recommed to use `git` when making changes to the project. Install [Git Bash](https://git-scm.com/install/windows) then also [Github Desktop](https://desktop.github.com/download/). Git acts as like a code journal, it comes in handy when dealing with complicated projects where you should be concise of every detail you have made to the project and to easily reverse changes if they dont work. I would also [setup SSH keys](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent) for your desktop and on the cluster so you can pull/push to the code repository without manaully syncing individual files back and forth.

The manual approach is using the command line interface (CLI) such as CMD, GitBash, etc. This is definitly needed on the cluster.
```sh
git clone https://github.com/farinc/kMC_CO_oxidation
git checkout MFPT
```
Github Desktop can do most of the `git` actions, so on your laptop its a bit easier to get started.

If this is something you would rather not do then skip and download the project as a zip file and carry on.

To actually have the intended python enviroment, download and install `uv` [by Astral](https://docs.astral.sh/uv/getting-started/installation/). If you plan to run this on the cluster then install it there too (its a linux machine) for your user. If you have a conda enviroment active I would deactivate it before setup.

To make use of the Tensor Train Format solver using MALS, use the `tt` group using the `schkit_tt` package.
```sh
uv sync --extra tt 
```
For running the kMC runs in parallel using MPI, then install the `mpi` group. Note that this requires that a MPI runtime is aviable on the system.
```sh
module load openmpi/4.1.2
uv sync --extra mpi 
```
There are two mutually exclusive options for installing the PETSc/SLEPc dependencies needed for ME-MKM support. Note that both install the `mpi` group automatically. When using the native option, make sure the `PETSC_DIR` and `SLEPC_DIR` enviroment variables are set before hand alongside mpi runtime.

**You must run the two `uv sync` calls below separately, one after the other -- do not pass both extras in a single `uv sync` call.** `petsc4py` and `slepc4py` both build with `--no-build-isolation`, and `slepc4py`'s build needs `petsc4py` to already be installed; uv builds no-build-isolation packages within a single `uv sync` concurrently, so requesting both extras at once races the two builds and `slepc4py` fails to find `petsc4py`'s headers. Splitting into two sequential syncs guarantees `petsc4py` is fully installed before `slepc4py`'s build starts.

```sh
# Option 1: link against a PETSc/SLEPc that already exists on the system
# (e.g. a cluster module, or system packages). Set PETSC_DIR/SLEPC_DIR first.
module load gcc/10.3.0 petsc/3.25.3-real slepc/3.25.1-real openmpi/4.1.2 cmake/3.28.4
UV_LOCK_TIMEOUT=600 uv sync --extra native-petsc -v && uv sync --extra native-slepc -v
```
```sh
# Option 2: build PETSc/SLEPc from source via the PyPI `petsc`/`slepc`
# packages; no external install needed, but the first sync compiles
# PETSc/SLEPc, which is slow.
UV_LOCK_TIMEOUT=1200 uv sync --extra source-petsc && uv sync --extra source-slepc
```
Once that done your ready to code! IDE's and the code editor VS Code is aware of python enivroments and will activate them for you to run files and code hints related to the dependencies. Otherwise, use `source .venv/bin/activate` on your linux machine/Git Bash. There probably a way to do this in other terminals...

## Usage
```sh
uv run python -m sweeps.linear                 # single point: L=16, k_o_ads=1.0, t_max 30 s
uv run python -m sweeps.linear --sweep k_o_ads=0:10:0.4 --kmc-L 24 --out case1

uv run pytest                                  # test suite for development purposes
```

Both `sweeps/linear.py` and `sweeps/mpi.py` run up to **three independent
phases** and write everything to one Excel workbook, `{out}.xlsx`:

- **kMC** (`--no-kmc` to skip): coverage trajectories from the empty and
  CO-covered starts. `--kmc-n-trajectories N` runs N independent trajectories
  from each start (so 2N total per point) instead of just one of each.
- **ME-MKM** (`--memkm`/`--no-memkm`; **off by default** in
  `sweeps/linear.py` since even a small tile is too slow for a laptop, **on
  by default** in `sweeps/mpi.py`): the stationary distribution and slow
  eigenvectors at every point, expressed as joint distributions over
  adsorbate counts. Needs the `native-petsc`+`native-slepc` or
  `source-petsc`+`source-slepc` extras.
- **Mean field** (`--no-meanfield` to skip): time-resolved coverages and
  rates for both rate laws, MF-MKM (plain mean field) and Bragg-Williams
  (coverage-dependent activation barriers), each from an empty and a
  CO-covered start.

Instead of a single k_o_ads value, any physics or simulation parameter can be
swept over a range or an explicit list with `--sweep NAME=START:STOP:STEP` or
`--sweep NAME=V1,V2,V3`. You can pass `--sweep` more than once to sweep
several parameters at once, which sweeps the full grid of every combination.
For example `--sweep k_o_ads=0:10:0.4 --sweep temperature=450,500,550` runs
the whole pipeline at every (k_o_ads, temperature) pair. Every parameter name
matches a CLI flag, so `--sweep eps=6000,8368,10000` sweeps the CO-CO
repulsion instead of `--eps 8368`.

The workbook always starts with a `Parameters` sheet (every value used in the
run), an `Axes` sheet (what was swept) and a `Grid` sheet (which parameter
values each numbered step actually ran at). Every step then gets its own
`S{n}_kMC`, `S{n}_MFMKM`, `S{n}_BW` and `S{n}_MEMKM` sheets.

With `--memkm` on, `--coexistence` runs a separate pass after the sweep that
finds the point along one swept axis where the two ME-MKM basins carry equal
stationary weight (Brent's method on the sign change of `ln pi(A)/pi(B)`).
Pick the axis to search with `--coexistence-axis` (defaults to `k_o_ads`); it
has to be a real physics rate, not something like `memkm_sites`, since Brent's
method needs a continuous parameter to bisect on and the tile's site count
changes the whole state space rather than shifting it smoothly. If you sweep
more than one axis, coexistence runs once per combination of the other axes.
Results land in a `Coexistence` sheet plus a `C{n}_MEMKM` sheet per crossing
found.

### Observables: your own functions on the simulation

Anything you want computed at every grid point that the three phases don't
already produce goes in as an **observable**: a function that is handed the
simulation and returns numbers. Write it, decorate it, name it on the command
line.

```python
# my_observables.py
import numpy as np
from co_oxidation.observables import observable

@observable("co_susceptibility", family="memkm")
def co_susceptibility(ctx):
    """Variance of the CO count under the stationary distribution."""
    n = ctx.coverage("CO") * ctx.builder.l
    mean = ctx.theta @ n
    return {"chi": float(ctx.theta @ (n - mean) ** 2)}

@observable("vacancies", family="kmc")
def vacancies(ctx):
    """Mean steady-state vacancy coverage over this point's trajectories."""
    return {"theta_empty": float(np.mean([r.steady_empty for r in ctx.results]))}
```

```sh
uv run python -m sweeps.linear --memkm --sweep k_o_ads=0:10:0.4 \
    --observable-module my_observables.py \
    --observable co_susceptibility,vacancies
```

`--observable-module` takes a file path as well as a dotted import path, so an
observable can live in a scratch file next to the run that needs it.
`--list-observables` shows everything registered, your modules included.

`family` picks which simulation the function acts on, and so what `ctx` is:

- **`kmc`** -- `ctx.results` holds the trajectories this point already ran.
  `ctx.run(init=..., **overrides)` runs another one, on seeds reserved so they
  never collide with the sweep's own, and `ctx.run_first_passage(predicate)`
  runs one stopped the first time a condition on the site counts holds
  (`co_oxidation.kmc.coverage_predicate` builds the common predicates).
- **`memkm`** -- `ctx.theta`, `ctx.eigvals`, `ctx.psi_R_2`, `ctx.in_A`/`in_B`
  and `ctx.coverage(species)` are the point the sweep already solved, so
  reweighting the stationary distribution adds no solve. `ctx.W` is the
  generator itself as a PETSc matrix, built only if asked for, for anything
  needing a linear solve.

`--observable-option NAME.KEY=VALUE` feeds `ctx.option("name", "key", default)`,
cast to the type of the default, so an observable adds its own knobs without
adding CLI flags.

Scalars you return become columns of an `Observables` sheet, one row per grid
step; arrays get a per-step `S{n}_OBS` sheet. Keys are namespaced by the
observable's name, so two observables can both return `chi` without colliding.
One that raises is reported and skipped rather than taking the run down with
it, with its message in the sheet as `{name}.error`.

`examples/observables_example.py` is a worked pair to copy from.

### Running on an HPC cluster

`sweeps/mpi.py` splits the sweep's grid points round-robin across MPI ranks
for the kMC phase (gathered on rank 0), then runs the ME-MKM / SLEPc phase
*collectively* -- all ranks cooperate on each point's distributed generator,
one point at a time.
```sh
qsub submit_sweep_job.sh --memkm-sites 12 --out big --sweep k_o_ads=0:10:0.4
```
Everything after the script name is forwarded to `sweeps/mpi.py` (including any
`-eps_*`/`-st_*` PETSc/SLEPc runtime options). The cluster build should include
MUMPS (`--download-mumps`).

## Using as a Dependency
Since this is a `uv` hybrid project and library you can use this as a dependency in other projects:
```sh
uv init
uv add "co_oxidation @ git+https://github.com/farinc/CO-Oxidation.git"
``` 
A few examples:

```python
# kMC: run one trajectory at a given O2 impingement rate k_o_ads
from co_oxidation.kmc import KMCParams, run_kmc

result = run_kmc(k_o_ads=5.0, init="empty", params=KMCParams(L=16))
print(result.steady_co, result.steady_o)
```

```python
# Mean field: steady-state branches over a k_o_ads range
from co_oxidation.meanfield import steady_states, branches

state = steady_states(k_o_ads=5.0)                       # single k_o_ads
curves = branches(k_o_ads_values=[0, 2, 4, 6, 8, 10])     # full bifurcation sweep
```

```python
# Observables outside a sweep: build a context and call one.
from co_oxidation.kmc import KMCParams, run_kmc
from co_oxidation.observables import KMCContext, compute

params = KMCParams(L=16, seed=0)
runs = [run_kmc(5.0, init=tag, params=params) for tag in ("empty", "full")]
ctx = KMCContext(k_o_ads=5.0, params=params, results=runs)
print(compute(["kmc_branching"], ctx))
```

```python
# ME-MKM needs the some `petsc` and `slepc` aviable.
from me_mkm import TileSettings
from co_oxidation.memkm import generate_model, CoexistencePipeline

tile = TileSettings.smallest_valid_square(8, True)  # 8-site ME-MKM tile
pipeline = CoexistencePipeline(tile)                 # bisects on k_o_ads by default
log_ratio = pipeline.basin_log_ratio(5.0)            # ln pi(A)/pi(B) at k_o_ads=5.0
checks = pipeline.diagnostics(5.0)                   # is it really two-state?
row, arrays = pipeline.report(k_o_ads_star)          # full analysis at k_o_ads*
```

To bisect on a different parameter, pass `axis_field` and give the rest of
the physics through `base_kwargs`:
```python
pipeline = CoexistencePipeline(tile, axis_field="temperature",
                               base_kwargs={"k_o_ads": 5.0})
log_ratio = pipeline.basin_log_ratio(500.0)          # ln pi(A)/pi(B) at temperature=500 K
```