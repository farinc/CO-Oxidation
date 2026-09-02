"""Tests for the n-fold local-update kMC and the mean-field model."""

import numpy as np
import pytest

import co_oxidation.kmc as kmc
from co_oxidation import meanfield

R_GAS = kmc.R_GAS


# --------------------------------------------------------------------------
# Brute-force rate oracle: full N x J_MAX rate matrix in plain Python,
# following the direct-kMC reference implementation (with the hop
# self-count fix).
# --------------------------------------------------------------------------

def oracle_rate_matrix(lat, L, k_co_ads, k_co_des, k_o_ads, k_rxn, khop, k_o_des=0.0,
                       eps=8368.0, T=500.0):
    w = eps / (R_GAS * T)
    N = L * L

    def right(i):
        return (i // L) * L + (i + 1) % L

    def down(i):
        return (i + L) % N

    def nco(i):
        nbrs = (right(i), (i // L) * L + (i - 1) % L, down(i), (i - L) % N)
        return sum(1 for q in nbrs if lat[q] == kmc.CO)

    rates = np.zeros((N, kmc.J_MAX))
    for i in range(N):
        if lat[i] == kmc.EMPTY:
            rates[i, 0] = k_co_ads
        elif lat[i] == kmc.CO:
            rates[i, 1] = k_co_des * np.exp(nco(i) * w)
        for joff, p in ((0, right(i)), (1, down(i))):
            si, sp = lat[i], lat[p]
            if si == kmc.EMPTY and sp == kmc.EMPTY:
                rates[i, 2 + joff] = k_o_ads
            if si == kmc.CO and sp == kmc.O:
                rates[i, 4 + joff] = k_rxn * np.exp(nco(i) * w)
            if si == kmc.O and sp == kmc.CO:
                rates[i, 4 + joff] = k_rxn * np.exp(nco(p) * w)
            if si == kmc.CO and sp == kmc.EMPTY:
                dn = (nco(p) - 1) - nco(i)
                rates[i, 6 + joff] = khop * np.exp(-dn * w / 2.0)
            if si == kmc.EMPTY and sp == kmc.CO:
                dn = (nco(i) - 1) - nco(p)
                rates[i, 6 + joff] = khop * np.exp(-dn * w / 2.0)
            if (si == kmc.O and sp == kmc.EMPTY) or (si == kmc.EMPTY and sp == kmc.O):
                rates[i, 8 + joff] = khop
            if si == kmc.O and sp == kmc.O:
                rates[i, 10 + joff] = k_o_des
    return rates


def class_rate_matrix(lat, L, class_rate):
    """Per-event rates implied by the n-fold classification."""
    N = L * L
    ec, ep, cm, cc = kmc._init_tables(lat, L)
    out = np.zeros((N, kmc.J_MAX))
    for i in range(N):
        for j in range(kmc.J_MAX):
            c = ec[i * kmc.J_MAX + j]
            if c >= 0:
                out[i, j] = class_rate[c]
    return out


PARAMS = dict(k_co_ads=1.6, k_co_des=1e-3, k_o_ads=5.0, k_rxn=1.0, khop=800.0, k_o_des=0.3)


def random_lattice(L, seed, p=(0.4, 0.35, 0.25)):
    rng = np.random.default_rng(seed)
    return rng.choice([0, 1, 2], size=L * L, p=p).astype(np.int8)


def test_classification_matches_rate_oracle():
    L = 12
    class_rate = kmc.make_class_rates(**PARAMS)
    for seed in (1, 2, 3):
        lat = random_lattice(L, seed)
        got = class_rate_matrix(lat, L, class_rate)
        want = oracle_rate_matrix(lat, L, **PARAMS)
        np.testing.assert_allclose(got, want, rtol=1e-10, atol=0.0)


def test_local_updates_match_full_rebuild():
    """After many locally-updated events, rebuilding the class tables from
    scratch must give the identical classification -- proves the distance-2
    update neighbourhood never misses an affected event."""
    L = 12
    class_rate = kmc.make_class_rates(**PARAMS)
    for seed, init_p in ((7, (0.5, 0.3, 0.2)), (8, (0.9, 0.05, 0.05))):
        lat = random_lattice(L, seed, init_p)
        ec, ep, cm, cc = kmc._init_tables(lat, L)
        done = kmc._advance(lat, L, class_rate, ec, ep, cm, cc, 30_000, seed)
        assert done == 30_000
        ec2, ep2, cm2, cc2 = kmc._init_tables(lat, L)
        np.testing.assert_array_equal(ec, ec2)
        np.testing.assert_array_equal(cc, cc2)
        for k in range(kmc.K_CLASSES):
            got = np.sort(cm[k, :cc[k]])
            want = np.sort(cm2[k, :cc2[k]])
            np.testing.assert_array_equal(got, want)
        # positional index consistency: cm[c][ep[e]] == e for every active e
        for e in range(L * L * kmc.J_MAX):
            c = ec[e]
            if c >= 0:
                assert cm[c, ep[e]] == e


def test_class_rates_collapse_without_interactions():
    r = kmc.make_class_rates(k_co_ads=1.6, k_co_des=1e-3, k_o_ads=5.0, k_rxn=1.0,
                             khop=800.0, eps=0.0)
    np.testing.assert_allclose(r[kmc.CLASS_CO_HOP0:kmc.CLASS_CO_HOP0 + 7], 800.0)
    np.testing.assert_allclose(r[kmc.CLASS_CO_DES0:kmc.CLASS_CO_DES0 + 5], 1e-3)
    np.testing.assert_allclose(r[kmc.CLASS_RXN0:kmc.CLASS_RXN0 + 5], 1.0)


def test_langmuir_equilibrium():
    """k_o_ads=0, eps=0: pure CO adsorption/desorption must equilibrate at the
    Langmuir coverage k_co_ads/(k_co_ads+k_co_des)."""
    k_co_ads, k_co_des = 1.0, 1.0
    res = kmc.run_kmc(k_o_ads=0.0, init="empty", L=10, k_co_ads=k_co_ads, k_co_des=k_co_des,
                      eps=0.0, khop=10.0, t_max=400.0, t_equil=100.0,
                      sample_interval=1_000, seed=1234)
    assert not res.stuck
    assert res.steady_o == pytest.approx(0.0, abs=1e-12)
    assert res.steady_co == pytest.approx(k_co_ads / (k_co_ads + k_co_des), abs=0.02)


def test_o2_desorption_reduces_o_coverage():
    """With k_co_ads=k_rxn=0, only O2 ads/des is active. Irreversible O2 adsorption
    (k_o_des=0) jams near the dimer RSA limit (isolated vacancies can't accept
    an O2 molecule on their own); making desorption reversible (k_o_des>0)
    must relax that to a measurably lower steady-state O coverage."""
    common = dict(L=12, k_co_ads=0.0, k_rxn=0.0, khop=50.0, t_max=20.0,
                  t_equil=5.0, sample_interval=5_000, seed=1)
    irrev = kmc.run_kmc(k_o_ads=2.0, init="empty", k_o_des=0.0, **common)
    rev = kmc.run_kmc(k_o_ads=2.0, init="empty", k_o_des=5.0, **common)
    assert irrev.steady_o > 0.9
    assert 0.0 < rev.steady_o < irrev.steady_o - 0.05


def test_bistability_branches():
    """At k_o_ads = 4 (inside the bistable window) the steady state depends on
    the initial condition: full-CO lattice stays CO-rich (branch I), empty
    lattice ends O-rich / CO-poor (branch II)."""
    common = dict(L=12, t_max=15.0, khop_scale=200.0, sample_interval=50_000)
    hi = kmc.run_kmc(k_o_ads=4.0, init="full", seed=42, **common)
    lo = kmc.run_kmc(k_o_ads=4.0, init="empty", seed=43, **common)
    assert hi.steady_co > 0.45
    assert lo.steady_co < 0.25
    assert lo.steady_o > 0.4


def test_meanfield_bistable_window():
    ss = meanfield.steady_states(5.0)
    stable = [s for s in ss if s.stable]
    unstable = [s for s in ss if not s.stable]
    assert len(stable) == 2
    # unstable states: the fully O-poisoned corner (theta_O = 1, always a
    # fixed point -- the MF image of the kMC absorbing state) plus the
    # interior separatrix of the bistable window
    interior = [s for s in unstable if 1e-6 < s.theta_co < 1.0 - 1e-6]
    assert len(interior) == 1
    assert any(s.theta_o == pytest.approx(1.0, abs=1e-9) for s in unstable)
    lo, hi = stable[0], stable[-1]
    assert hi.theta_co > 0.9
    assert lo.theta_co < 0.2
    assert lo.theta_co < interior[0].theta_co < hi.theta_co
    # every returned state must actually satisfy dtheta/dt = 0
    for s in ss:
        f1, f2 = meanfield._rhs(s.theta_co, s.theta_o, 5.0, 1.6, 1e-3, 1.0, 4)
        assert abs(f1) < 1e-6 and abs(f2) < 1e-6
    # coverages sum to 1
    for s in ss:
        assert s.theta_empty + s.theta_co + s.theta_o == pytest.approx(1.0, abs=1e-6)


def test_meanfield_langmuir_limit():
    """k_o_ads = 0: single stable state at the Langmuir coverage, no O."""
    ss = meanfield.steady_states(0.0, k_co_ads=1.0, k_co_des=1.0)
    stable = [s for s in ss if s.stable]
    assert len(stable) == 1
    assert stable[0].theta_co == pytest.approx(0.5, abs=1e-9)
    assert stable[0].theta_o == pytest.approx(0.0, abs=1e-9)


def test_ea_mk_rate_corrections():
    """Ea-MK (Table 2 of the paper): desorption gains exp(Z*eps*x/RT),
    oxidation exp((Z-1)*eps*x/RT), the adsorption steps are untouched."""
    x, y = 0.3, 0.1
    mf = meanfield.rates(x, y, 5.0)
    ea = meanfield.rates(x, y, 5.0, model="ea")
    eps_rt = 8368.0 / (meanfield.R_GAS * 500.0)
    assert ea[0] == pytest.approx(mf[0])
    assert ea[2] == pytest.approx(mf[2])
    assert ea[1] == pytest.approx(mf[1] * np.exp(4 * eps_rt * x))
    assert ea[3] == pytest.approx(mf[3] * np.exp(3 * eps_rt * x))


def test_jacobian_matches_finite_differences():
    h = 1e-7
    for model in ("mf", "ea"):
        for x, y in ((0.3, 0.2), (0.7, 0.1), (0.05, 0.6)):
            J = meanfield._jacobian(x, y, 5.0, model=model)
            f0 = meanfield._rhs(x, y, 5.0, model=model)
            fx = meanfield._rhs(x + h, y, 5.0, model=model)
            fy = meanfield._rhs(x, y + h, 5.0, model=model)
            num = np.array([[(fx[0] - f0[0]) / h, (fy[0] - f0[0]) / h],
                            [(fx[1] - f0[1]) / h, (fy[1] - f0[1]) / h]])
            np.testing.assert_allclose(J, num, rtol=1e-5, atol=1e-4)


def test_ea_mk_steady_states():
    """Paper Fig. 3: at intermediate k_o_ads the Ea-MK model finds only the
    low-CO branch II, unlike plain MF-MK."""
    ss = meanfield.steady_states(5.0, model="ea")
    for s in ss:
        f1, f2 = meanfield._rhs(s.theta_co, s.theta_o, 5.0, model="ea")
        assert abs(f1) < 1e-7 and abs(f2) < 1e-7
    stable = [s for s in ss if s.stable]
    assert len(stable) == 1
    assert stable[0].theta_co < 0.2


def test_meanfield_integration_reaches_steady_state():
    ts, traj = meanfield.integrate((0.0, 0.0), k_o_ads=5.0, t_end=50.0, dt=1e-3)
    ss = meanfield.steady_states(5.0)
    lo = min((s for s in ss if s.stable), key=lambda s: s.theta_co)
    assert traj[-1, 0] == pytest.approx(lo.theta_co, abs=1e-4)
    assert traj[-1, 1] == pytest.approx(lo.theta_o, abs=1e-4)
