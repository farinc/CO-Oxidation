"""ME-MKM model + the SLEPc/PETSc coexistence pipeline.

    from co_oxidation.memkm import generate_model, CoexistencePipeline

`generate_model(beta, tile)` builds the master-equation generator for the CO
oxidation mechanism and needs only `me_mkm`. `CoexistencePipeline` locates the
coexistence point(s) beta* and the basin transition rates from its spectrum;
its heavy numerics live in `.backend` on PETSc/SLEPc, which are imported
*lazily* so that importing the model (or the kMC-only code) never requires a
PETSc/SLEPc install.
"""

from .model import generate_model

__all__ = ["generate_model", "CoexistencePipeline"]


def __getattr__(name):
    # Defer the SLEPc/PETSc import until CoexistencePipeline is actually used.
    if name == "CoexistencePipeline":
        from .coexistence import CoexistencePipeline
        return CoexistencePipeline
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
