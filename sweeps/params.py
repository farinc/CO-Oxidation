"""RunConfig: the single canonical parameter schema shared by the kMC, mean-field and
ME-MKM phases of a sweep. Every field name here is also a legal --sweep axis name (see
sweeps/axes.py). Reconciles the naming mismatches between the three physics backends
(T vs temperature, k_o_des vs k_o_des_scale, khop vs khop_scale)."""

from dataclasses import dataclass

from co_oxidation.kmc import KMCParams


@dataclass(frozen=True)
class RunConfig:
    # shared physics
    k_co_ads: float = 1.6
    k_co_des: float = 1e-3
    k_o_ads: float = 1.0
    k_rxn: float = 1.0
    k_o_des_scale: float = 1e-4          # k_o_des = k_o_des_scale * k_o_ads
    eps: float = 8368.0
    temperature: float = 500.0
    khop_scale: float = 1000.0

    # kMC structural / simulation
    kmc_L: int = 16
    kmc_t_max: float = 30.0
    kmc_max_steps: int = 1_000_000_000
    kmc_sample_interval: int = 10_000
    kmc_seed: int = 0
    kmc_n_trajectories: int = 1          # 2 * kmc_n_trajectories runs per grid step

    # mean-field time resolution
    meanfield_t_end: float = 30.0
    meanfield_dt: float = 0.05

    # ME-MKM structural (never a legal coexistence bisection axis, see coexistence_driver.py)
    memkm_sites: int = 8

    def resolved_k_o_des(self) -> float:
        return self.k_o_des_scale * self.k_o_ads

    def to_kmc_params(self) -> KMCParams:
        return KMCParams(
            L=self.kmc_L, k_co_ads=self.k_co_ads, k_co_des=self.k_co_des,
            k_rxn=self.k_rxn, k_o_des=self.resolved_k_o_des(), eps=self.eps,
            T=self.temperature, khop_scale=self.khop_scale, t_max=self.kmc_t_max,
            max_steps=self.kmc_max_steps, sample_interval=self.kmc_sample_interval,
            seed=self.kmc_seed)

    def to_meanfield_kwargs(self) -> dict:
        return {
            "k_co_ads": self.k_co_ads, "k_co_des": self.k_co_des, "k_rxn": self.k_rxn,
            "eps": self.eps, "T": self.temperature, "k_o_des": self.resolved_k_o_des(),
        }

    def to_memkm_kwargs(self) -> dict:
        """Keys match generate_model's / CoexistencePipeline's kwarg names 1:1 with
        RunConfig's field names (k_o_ads included), so a coexistence bisection axis
        can be popped out of this dict by its RunConfig field name directly."""
        return {
            "k_o_ads": self.k_o_ads, "k_o_des_scale": self.k_o_des_scale,
            "k_co_ads": self.k_co_ads, "k_co_des": self.k_co_des, "k_rxn": self.k_rxn,
            "khop_scale": self.khop_scale, "eps": self.eps,
            "temperature": self.temperature,
        }
