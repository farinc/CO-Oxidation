"""
Mean-field microkinetic models for the CO oxidation system of
Tian & Rangarajan (J. Phys. Chem. C 2021, 125, 20275), square lattice Z = 4.

    dx/dt = r_ads_co - r_des_co - r_oxi         (x = theta_CO)
    dy/dt = r_ads_o - r_oxi - r_des_o           (y = theta_O, w = 1 - x - y)

O2 desorption (O* + O* -> O2(g) + 2*, rate k_o_des) mirrors O2 adsorption's
mean-field form, r_des_o = Z*k_o_des*y^2, the mass-action rate of a site
finding an O neighbour among its Z bonds. Like O2 adsorption it carries no
CO-CO repulsion correction in either model, matching kmc.py's CLASS_O2_DES
(k_o_des is a flat rate, no BEP term). k_o_des = 0 recovers the original
irreversible-O2-adsorption model.

Two rate laws are implemented (the model argument):

    "mf"  plain mean field, MF-MK in the paper. Rates follow mass action
          with no interaction correction.
    "ea"  mean field with coverage-dependent activation barriers, Ea-MK in
          the paper (Table 2). Desorption is boosted by exp(Z*eps*x/RT) and
          oxidation by exp((Z-1)*eps*x/RT), because a CO has on average
          Z*x CO neighbours, one fewer in the oxidation case since one
          neighbour is the reacting O. O2 adsorption keeps its mean-field
          form, its barrier does not involve CO.

Steady states of the plain model are found exactly. Subtracting the two
equations gives x = (k_co_ads*w - Z*k_o_ads*w^2)/k_co_des, and substituting into
k_rxn*x*y = k_o_ads*w^2 with y = 1 - w - x reduces the system to a single quartic
in w. np.roots then delivers every root at once, stable and unstable
branches alike. The Ea model is transcendental, so its steady states come
from Newton iterations started on a grid of coverages instead. In both
cases stability follows from the eigenvalues of the analytic 2x2 Jacobian.
This replaces the earlier sympy nonlinsolve approach, which was far slower
and could miss branches.
"""

from typing import NamedTuple

import numpy as np
from scipy.integrate import solve_ivp

R_GAS = 8.314462618  # J / mol / K


class SteadyState(NamedTuple):
    theta_empty: float
    theta_co: float
    theta_o: float
    stable: bool


def rates(x, y, k_o_ads, k_co_ads=1.6, k_co_des=1e-3, k_rxn=1.0, Z=4, model="mf",
          eps=8368.0, T=500.0, k_o_des=0.0):
    """Rates of the five steps at coverages x = theta_CO, y = theta_O.

    Accepts scalars or numpy arrays. Returns the tuple
    (r_ads_co, r_des_co, r_ads_o, r_oxi, r_des_o).
    """
    w = 1.0 - x - y
    r_ads_co = k_co_ads * w
    r_des_co = k_co_des * x
    r_ads_o = Z * k_o_ads * w * w
    r_oxi = Z * k_rxn * x * y
    r_des_o = Z * k_o_des * y * y
    if model == "ea":
        eps_rt = eps / (R_GAS * T)
        r_des_co = r_des_co * np.exp(Z * eps_rt * x)
        r_oxi = r_oxi * np.exp((Z - 1) * eps_rt * x)
    elif model != "mf":
        raise ValueError(f"unknown model {model!r}")
    return r_ads_co, r_des_co, r_ads_o, r_oxi, r_des_o


def _rhs(x, y, k_o_ads, k_co_ads=1.6, k_co_des=1e-3, k_rxn=1.0, Z=4, model="mf",
         eps=8368.0, T=500.0, k_o_des=0.0):
    r_ads_co, r_des_co, r_ads_o, r_oxi, r_des_o = rates(
        x, y, k_o_ads, k_co_ads, k_co_des, k_rxn, Z, model, eps, T, k_o_des)
    return r_ads_co - r_des_co - r_oxi, r_ads_o - r_oxi - r_des_o


def _jacobian(x, y, k_o_ads, k_co_ads=1.6, k_co_des=1e-3, k_rxn=1.0, Z=4, model="mf",
              eps=8368.0, T=500.0, k_o_des=0.0):
    """Analytic Jacobian of the right-hand side with respect to (x, y)."""
    w = 1.0 - x - y
    if model == "ea":
        a = Z * eps / (R_GAS * T)        # desorption exponent coefficient
        b = (Z - 1) * eps / (R_GAS * T)  # oxidation exponent coefficient
    else:
        a = 0.0
        b = 0.0
    e_des = np.exp(a * x)
    e_oxi = np.exp(b * x)
    # d/dx of x*exp(c*x) is exp(c*x)*(1 + c*x), which is where the (1 + ...)
    # factors below come from
    return np.array([
        [-k_co_ads - k_co_des * e_des * (1.0 + a * x)
         - Z * k_rxn * y * e_oxi * (1.0 + b * x),
         -k_co_ads - Z * k_rxn * x * e_oxi],
        [-2.0 * Z * k_o_ads * w - Z * k_rxn * y * e_oxi * (1.0 + b * x),
         -2.0 * Z * k_o_ads * w - Z * k_rxn * x * e_oxi - 2.0 * Z * k_o_des * y],
    ])


def _append_state(states, x, y, k_o_ads, k_co_ads, k_co_des, k_rxn, Z, model, eps, T,
                  k_o_des):
    """Validate, deduplicate and stability-classify one candidate root."""
    w = 1.0 - x - y
    tol = 1e-6
    if not (-tol <= x <= 1 + tol and -tol <= y <= 1 + tol
            and -tol <= w <= 1 + tol):
        return
    w = min(max(w, 0.0), 1.0)
    x = min(max(x, 0.0), 1.0)
    y = min(max(y, 0.0), 1.0)
    if any(abs(s.theta_co - x) < 1e-6 and abs(s.theta_o - y) < 1e-6
           for s in states):
        return
    eig = np.linalg.eigvals(_jacobian(x, y, k_o_ads, k_co_ads, k_co_des, k_rxn, Z,
                                      model, eps, T, k_o_des))
    states.append(SteadyState(w, x, y, bool(np.max(eig.real) < 0.0)))


def _steady_states_poly(k_o_ads, k_co_ads, k_co_des, k_rxn, Z):
    """Exact steady states of the plain mean-field model via one quartic.

    Only valid for k_o_des = 0 (irreversible O2 adsorption): the quartic
    reduction below relies on the two-equation system reducing to a single
    variable w, which breaks once r_des_o = Z*k_o_des*y^2 adds an independent
    y-dependence to dy/dt.
    """
    # with P(w) = k_co_ads*w - Z*k_o_ads*w^2 and Q(w) = k_co_des*(1-w) - P(w) the
    # conditions reduce to k_rxn*P(w)*Q(w) - k_co_des^2*k_o_ads*w^2 = 0
    P = np.array([-Z * k_o_ads, k_co_ads, 0.0])              # descending coeffs
    Q = np.array([Z * k_o_ads, -(k_co_ads + k_co_des), k_co_des])
    poly = k_rxn * np.polymul(P, Q)
    poly[-3] -= k_co_des * k_co_des * k_o_ads
    coeffs = np.trim_zeros(poly, "f")
    states: list[SteadyState] = []
    if len(coeffs) < 2:
        return states
    for r in np.roots(coeffs):
        if abs(r.imag) > 1e-9:
            continue
        w = float(r.real)
        if not -1e-9 <= w <= 1.0 + 1e-9:
            continue
        x = (k_co_ads * w - Z * k_o_ads * w * w) / k_co_des
        y = 1.0 - w - x
        _append_state(states, x, y, k_o_ads, k_co_ads, k_co_des, k_rxn, Z, "mf",
                      8368.0, 500.0, 0.0)
    return states


def _steady_states_newton(k_o_ads, k_co_ads, k_co_des, k_rxn, Z, model, eps, T, k_o_des):
    """Steady states by damped-free Newton from a grid of starting coverages.

    The Ea model has exponential terms, and any model with k_o_des != 0 has an
    extra y^2 desorption term, so there is no polynomial reduction. A coarse
    grid of starts is enough because the system only ever has a handful of
    roots, and every converged root is verified and deduplicated.
    """
    guesses = [(xg, yg)
               for xg in np.linspace(0.02, 0.98, 9)
               for yg in np.linspace(0.02, 0.98, 9)
               if xg + yg <= 1.0]
    # corners where branch I, an O-poisoned state and a dilute state live
    guesses += [(0.99, 0.005), (0.005, 0.99), (0.005, 0.005)]
    states: list[SteadyState] = []
    for x0, y0 in guesses:
        x, y = x0, y0
        converged = False
        for _ in range(100):
            if not (-1.0 < x < 2.0 and -1.0 < y < 2.0):
                break  # wandered far outside the simplex, give up this start
            f1, f2 = _rhs(x, y, k_o_ads, k_co_ads, k_co_des, k_rxn, Z, model, eps, T, k_o_des)
            J = _jacobian(x, y, k_o_ads, k_co_ads, k_co_des, k_rxn, Z, model, eps, T, k_o_des)
            det = J[0, 0] * J[1, 1] - J[0, 1] * J[1, 0]
            if abs(det) < 1e-300:
                break
            step_x = (J[1, 1] * f1 - J[0, 1] * f2) / det
            step_y = (J[0, 0] * f2 - J[1, 0] * f1) / det
            x -= step_x
            y -= step_y
            if abs(step_x) < 1e-13 and abs(step_y) < 1e-13:
                converged = True
                break
        if not converged:
            continue
        f1, f2 = _rhs(x, y, k_o_ads, k_co_ads, k_co_des, k_rxn, Z, model, eps, T, k_o_des)
        if abs(f1) > 1e-9 or abs(f2) > 1e-9:
            continue
        _append_state(states, x, y, k_o_ads, k_co_ads, k_co_des, k_rxn, Z, model, eps, T,
                     k_o_des)
    return states


def steady_states(k_o_ads, k_co_ads=1.6, k_co_des=1e-3, k_rxn=1.0, Z=4, model="mf",
                  eps=8368.0, T=500.0, k_o_des=0.0):
    """All physical steady states at a given k_o_ads, sorted by theta_CO.

    The exact quartic solver only applies to the plain mean-field model with
    irreversible O2 adsorption (k_o_des = 0); any other combination falls back
    to the Newton solver.
    """
    if model == "mf" and k_o_des == 0.0:
        states = _steady_states_poly(k_o_ads, k_co_ads, k_co_des, k_rxn, Z)
    else:
        states = _steady_states_newton(k_o_ads, k_co_ads, k_co_des, k_rxn, Z, model,
                                       eps, T, k_o_des)
    states.sort(key=lambda s: s.theta_co)
    return states


def integrate(theta0, k_o_ads, t_end, dt, n_points=None, k_co_ads=1.6, k_co_des=1e-3,
              k_rxn=1.0, Z=4, model="mf", eps=8368.0, T=500.0, k_o_des=0.0):
    """Time integration from theta0 = (theta_CO, theta_O) via scipy's RK45.

    Returns (t, traj) with traj[:, 0] = theta_CO and traj[:, 1] = theta_O, sampled on
    a fixed grid for a predictable return shape: n_points evenly spaced points over
    [0, t_end] if n_points is given, otherwise a grid of spacing dt. n_points lets a
    caller match another series' sample count exactly (sweeps/steps.py uses it to tie
    the mean-field row count to a step's kMC trajectory length for the workbook).
    """
    def f(t, theta):
        return _rhs(theta[0], theta[1], k_o_ads, k_co_ads, k_co_des, k_rxn, Z, model, eps, T, k_o_des)

    if n_points is not None:
        ts = np.linspace(0.0, t_end, max(2, n_points))
    else:
        n = max(1, int(np.ceil(t_end / dt)))
        ts = np.linspace(0.0, n * dt, n + 1)
    sol = solve_ivp(f, (0.0, ts[-1]), theta0, t_eval=ts, method="RK45",
                    rtol=1e-8, atol=1e-10)
    return sol.t, sol.y.T


def branches(k_o_ads_values, k_co_ads=1.6, k_co_des=1e-3, k_rxn=1.0, Z=4, model="mf",
             eps=8368.0, T=500.0, k_o_des=0.0):
    """Organize steady states over a k_o_ads sweep into plottable branches.

    Returns (branch_hi, branch_lo, unstable), each an array of shape
    (len(k_o_ads_values), 3) holding (theta_empty, theta_co, theta_o), with NaN
    where the branch does not exist. branch_hi and branch_lo are the
    stable high and low theta_CO branches, the paper's branch I and II.
    """
    hi = np.full((len(k_o_ads_values), 3), np.nan)
    lo = np.full((len(k_o_ads_values), 3), np.nan)
    un = np.full((len(k_o_ads_values), 3), np.nan)
    for i, b in enumerate(k_o_ads_values):
        ss = steady_states(b, k_co_ads, k_co_des, k_rxn, Z, model, eps, T, k_o_des)
        stab = [s for s in ss if s.stable]
        unst = [s for s in ss if not s.stable]
        if stab:
            s_hi = max(stab, key=lambda s: s.theta_co)
            s_lo = min(stab, key=lambda s: s.theta_co)
            if len(stab) >= 2:
                hi[i] = s_hi[:3]
                lo[i] = s_lo[:3]
            elif s_hi.theta_co > 0.5:
                hi[i] = s_hi[:3]
            else:
                lo[i] = s_lo[:3]
        if unst:
            # the interior unstable state, the separatrix of the bistable
            # window. The O-poisoned corner theta_O = 1 is skipped.
            cand = [s for s in unst if 1e-6 < s.theta_co < 1.0 - 1e-6]
            if cand:
                un[i] = cand[0][:3]
    return hi, lo, un
