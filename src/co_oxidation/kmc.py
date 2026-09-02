"""
Rejection-free n-fold (BKL) kinetic Monte Carlo for CO oxidation on a periodic
square lattice, after Tian & Rangarajan (J. Phys. Chem. C 2021, 125, 20275).

Model (site states: 0 = empty, 1 = CO*, 2 = O*):

    CO(g) + *      -> CO*            rate  k_co_ads
    CO*            -> CO(g) + *      rate  k_co_des * exp(+n*eps/RT)
    O2(g) + 2*     -> O* + O*        rate  k_o_ads          (adjacent empty pair)
    O* + O*        -> O2(g) + 2*     rate  k_o_des         
    CO* + O*       -> CO2(g) + 2*    rate  k_rxn * exp(+n*eps/RT)
    CO* + *        -> * + CO*        rate  khop * exp(-dn*eps/2RT)   (NN hop)
    O*  + *        -> * + O*         rate  khop

Here n is the number of CO nearest neighbours of the CO* involved, and eps > 0
is the CO-CO nearest-neighbour repulsion. Barriers follow BEP relations with
omega = 1 for desorption and reaction and omega = 1/2 for hops. For a hop,
dn = n_final - n_initial, and the moving CO is excluded from the destination
neighbour count. That exclusion fixes a self-counting bias in the earlier
direct-kMC script and restores detailed balance for diffusion.

The algorithm is the n-fold method described by Chatterjee & Vlachos
(J. Comput.-Aided Mater. Des. 2007, 14, 253, section 6.3). Every event 
belongs to one of 20 rate classes (see the class layout below). A step 
picks a class by linear search over the 20 cumulative class weights 
(section 6.1.1), then picks a random member of that class. Every attempt 
fires, so there are no null events. After an event only the events 
within graph distance 2 of the changed sites are looked at again, 
which makes the cost per event independent of lattice size.

Event encoding: event id = site*J_MAX + j, with per-site slots

    j=0 CO ads, j=1 CO des, j=2,3 O2 ads right/down, j=4,5 rxn right/down,
    j=6,7 CO hop across the right/down bond, j=8,9 O hop right/down,
    j=10,11 O2 des right/down.

Pair events live on the right or down bond of a site, so each bond appears
exactly once. A hop slot is active whenever one end of the bond holds the
mobile species and the other end is empty. The rate depends on which end
the adsorbate sits on, and _classify works that out.
"""

from dataclasses import dataclass, replace

import numpy as np
from numba import njit

from .common import CO, EMPTY, O

R_GAS = 8.314462618  # J / mol / K

J_MAX = 12  # event slots per site

# n-fold class layout
# An n-fold class groups every event on the lattice that currently has the
# same rate (Bortz, Kalos and Lebowitz). That grouping is possible here
# because the only lateral interaction is the nearest-neighbour CO-CO pair
# term. A rate can then depend on the configuration only through an integer
# CO neighbour count, so the full rate spectrum collapses to the 20 discrete
# values below. The function _classify maps an event to its class, and a kMC 
# step only searches 20 cumulative class weights instead of all N*10 per-site 
# rates.
#
# The ranges: n runs 0..4 for desorption. For a hop, both ends of the
# bond are excluded from the counts, because the destination is empty and
# the mover must not count itself there. Each end then sees at most 3 CO,
# so dn runs -3..+3. Reaction never actually reaches n = 4 either, since
# one neighbour of the CO is its O partner. Slots 0..4 are allocated anyway
# to keep the indexing uniform.
K_CLASSES = 21
CLASS_CO_ADS = 0        # rate k_co_ads, the same on every empty site
CLASS_CO_DES0 = 1       # 1..5:   k_co_des * exp(n*eps/RT), repulsion boosts desorption
CLASS_O2_ADS = 6        # rate k_o_ads, per adjacent empty pair
CLASS_RXN0 = 7          # 7..11:  k_rxn * exp(n*eps/RT), repulsion boosts reaction
CLASS_CO_HOP0 = 12      # 12..18: khop * exp(-dn*eps/2RT), hopping into crowding is slow
CLASS_O_HOP = 19        # rate khop, O* feels no lateral interaction
CLASS_O2_DES = 20       # rate k_o_des, flat like O2 ads and O hop (no O-O interaction term)


def make_class_rates(k_co_ads, k_co_des, k_o_ads, k_rxn, khop, k_o_des=0.0, eps=8368.0, T=500.0):
    """Rate of each of the 21 n-fold classes.

    This is the only place where rates are computed. The simulation itself
    only ever looks them up by class index.
    """
    eps_rt = eps / (R_GAS * T)  # dimensionless repulsion energy per CO-CO pair
    rates = np.zeros(K_CLASSES)
    rates[CLASS_CO_ADS] = k_co_ads
    for n in range(5):
        rates[CLASS_CO_DES0 + n] = k_co_des * np.exp(n * eps_rt)
        rates[CLASS_RXN0 + n] = k_rxn * np.exp(n * eps_rt)
    rates[CLASS_O2_ADS] = k_o_ads
    for dn in range(-3, 4):
        # BEP with omega = 1/2, only half the energy change enters the barrier
        rates[CLASS_CO_HOP0 + dn + 3] = khop * np.exp(-dn * eps_rt / 2.0)
    rates[CLASS_O_HOP] = khop
    rates[CLASS_O2_DES] = k_o_des
    return rates


# local geometry
# Sites are numbered row by row. The index arithmetic implements the
# periodic wrapping: (i+1)%L wraps within a row, (i+L)%(L*L) wraps rows.

@njit(cache=True)
def _right(i, L):
    return (i // L) * L + (i + 1) % L


@njit(cache=True)
def _down(i, L):
    return (i + L) % (L * L)


@njit(cache=True)
def _count_co_neighbors(lat, i, L):
    """Number of CO among the 4 nearest neighbours of site i."""
    N = L * L
    count = 0
    if lat[(i // L) * L + (i + 1) % L] == CO:
        count += 1
    if lat[(i // L) * L + (i - 1) % L] == CO:
        count += 1
    if lat[(i + L) % N] == CO:
        count += 1
    if lat[(i - L) % N] == CO:
        count += 1
    return count


# event classification

@njit(cache=True)
def _classify(lat, i, j, L):
    """Class of event slot (i, j) in the current configuration, -1 if inactive.

    This function is the single source of truth for the rate physics. Setup,
    local updates and the tests all go through it, so the bookkeeping can
    never disagree with the model.
    """
    site_state = lat[i]
    if j == 0:  # CO adsorption needs an empty site
        if site_state == EMPTY:
            return CLASS_CO_ADS
        return -1
    if j == 1:  # CO desorption, the rate depends on how crowded the CO is
        if site_state == CO:
            return CLASS_CO_DES0 + _count_co_neighbors(lat, i, L)
        return -1
    # every remaining slot is a pair event on the bond from i to its partner
    partner = _right(i, L) if j % 2 == 0 else _down(i, L)
    partner_state = lat[partner]
    if j <= 3:  # O2 adsorption needs both bond sites empty
        if site_state == EMPTY and partner_state == EMPTY:
            return CLASS_O2_ADS
        return -1
    if j <= 5:  # CO + O reaction, either ordering of the pair counts once
        if site_state == CO and partner_state == O:
            return CLASS_RXN0 + _count_co_neighbors(lat, i, L)
        if site_state == O and partner_state == CO:
            return CLASS_RXN0 + _count_co_neighbors(lat, partner, L)
        return -1
    if j <= 7:  # CO hop, one end CO and the other end empty
        # dn is the change in CO-CO contacts if the hop happens. The raw
        # neighbour count at the empty destination includes the mover itself
        # because the two sites are adjacent, hence the -1.
        if site_state == CO and partner_state == EMPTY:
            dn = (_count_co_neighbors(lat, partner, L) - 1) - _count_co_neighbors(lat, i, L)
        elif site_state == EMPTY and partner_state == CO:
            dn = (_count_co_neighbors(lat, i, L) - 1) - _count_co_neighbors(lat, partner, L)
        else:
            return -1
        return CLASS_CO_HOP0 + dn + 3
    if j <= 9:  # O hop, one end O and the other end empty, always the same rate
        if (site_state == O and partner_state == EMPTY) or \
           (site_state == EMPTY and partner_state == O):
            return CLASS_O_HOP
        return -1
    # O2 desorption, both bond sites must hold O
    if site_state == O and partner_state == O:
        return CLASS_O2_DES
    return -1 # no event


# class-list bookkeeping
# Each class keeps a plain array of its member event ids (class_members and
# class_count). event_class holds the current class of every event and
# event_pos remembers where the event sits inside its class list, so removal
# is O(1): overwrite the slot with the last member and shrink the list.

@njit(cache=True)
def _update_site(lat, i, L, event_class, event_pos, class_members, class_count):
    """Re-classify all 10 event slots of site i after nearby occupancy changed."""
    base = i * J_MAX
    for j in range(J_MAX):
        event = base + j
        new_class = _classify(lat, i, j, L)
        old_class = event_class[event]
        if new_class == old_class:
            continue  # most slots keep their class and nothing moves
        if old_class >= 0:
            # remove from the old class by swapping the last member into
            # this event's slot
            pos = event_pos[event]
            last = class_count[old_class] - 1
            moved = class_members[old_class, last]
            class_members[old_class, pos] = moved
            event_pos[moved] = pos
            class_count[old_class] = last
        if new_class >= 0:
            # append to the new class
            count = class_count[new_class]
            class_members[new_class, count] = event
            event_pos[event] = count
            class_count[new_class] = count + 1
        event_class[event] = new_class


@njit(cache=True)
def _init_tables(lat, L):
    """Build the class member lists from scratch by sweeping every site.

    Used once at the start of a run. The tests also use it to check that a
    long chain of local updates lands on exactly this state.
    """
    N = L * L
    event_class = np.full(N * J_MAX, -1, np.int64)  # class of each event, -1 inactive
    event_pos = np.zeros(N * J_MAX, np.int64)       # position inside its class list
    class_members = np.zeros((K_CLASSES, 2 * N), np.int64)  # at most 2N per class
    class_count = np.zeros(K_CLASSES, np.int64)
    for i in range(N):
        for j in range(J_MAX):
            c = _classify(lat, i, j, L)
            if c >= 0:
                event = i * J_MAX + j
                event_class[event] = c
                class_members[c, class_count[c]] = event
                event_pos[event] = class_count[c]
                class_count[c] += 1
    return event_class, event_pos, class_members, class_count


@njit(cache=True)
def _total_rate(class_count, class_rate):
    """Total rate of the whole lattice, a 20-term sum over the classes."""
    total = 0.0
    for k in range(K_CLASSES):
        total += class_count[k] * class_rate[k]
    return total


# selection and execution

@njit(cache=True)
def _select(class_rate, class_members, class_count, total_rate):
    """Pick the next event: linear search over classes, then a random member.

    A class is chosen with probability (members * rate) / total, which is
    the n-fold selection rule. All members of a class share one rate, so
    any member of the chosen class is then equally likely.
    """
    target = np.random.rand() * total_rate
    cum_weight = 0.0
    chosen = -1
    for k in range(K_CLASSES):
        cum_weight += class_count[k] * class_rate[k]
        if target < cum_weight:
            chosen = k
            break
    if chosen < 0:
        # rounding can leave the target a hair above the final accumulated
        # sum, in that case take the last class that carries any weight
        for k in range(K_CLASSES - 1, -1, -1):
            if class_count[k] > 0 and class_rate[k] > 0.0:
                chosen = k
                break
    if chosen < 0:
        return -1
    member = int(np.random.rand() * class_count[chosen])
    if member >= class_count[chosen]:  # guard the rand() ~ 1.0 edge
        member = class_count[chosen] - 1
    return class_members[chosen, member]


@njit(cache=True)
def _apply(lat, event, L, event_class, event_pos, class_members, class_count,
           affected):
    """Execute the event and repair the class tables locally.

    Only events within graph distance 2 of a changed site can change class,
    because an event on a bond depends on nothing beyond the two bond sites
    and their nearest neighbours. Returns the change in the CO and O counts.
    """
    i = event // J_MAX
    j = event % J_MAX
    d_co = 0
    d_o = 0
    changed2 = -1  # second changed site, set if the event touches a bond
    if j == 0:
        lat[i] = CO
        d_co = 1
    elif j == 1:
        lat[i] = EMPTY
        d_co = -1
    else:
        partner = _right(i, L) if j % 2 == 0 else _down(i, L)
        changed2 = partner
        if j <= 3:      # O2 adsorption fills both sites
            lat[i] = O
            lat[partner] = O
            d_o = 2
        elif j <= 5:    # reaction empties both sites
            lat[i] = EMPTY
            lat[partner] = EMPTY
            d_co = -1
            d_o = -1
        elif j <= 9:    # a hop is just a swap of the two bond sites
            tmp = lat[i]
            lat[i] = lat[partner]
            lat[partner] = tmp
        else:           # O2 desorption empties both sites
            lat[i] = EMPTY
            lat[partner] = EMPTY
            d_o = -2
    # collect every site within Manhattan distance 2 of a changed site.
    # Each changed site contributes a diamond of 13 sites, and duplicates
    # are dropped because the two diamonds overlap.
    n_affected = 0
    for k in range(2):
        center = i if k == 0 else changed2
        if center < 0:
            continue
        row0 = center // L
        col0 = center % L
        for dy in range(-2, 3):
            abs_dy = dy if dy >= 0 else -dy
            row = ((row0 + dy) % L) * L
            for dx in range(-(2 - abs_dy), 3 - abs_dy):
                site = row + (col0 + dx) % L
                dup = False
                for m in range(n_affected):
                    if affected[m] == site:
                        dup = True
                        break
                if not dup:
                    affected[n_affected] = site
                    n_affected += 1
    for m in range(n_affected):
        _update_site(lat, affected[m], L, event_class, event_pos,
                     class_members, class_count)
    return d_co, d_o


@njit(cache=True)
def _advance(lat, L, class_rate, event_class, event_pos, class_members,
             class_count, n_steps, seed):
    """Run n_steps events on externally held tables.

    The tests use this to check the local bookkeeping against a rebuild.
    Returns the number of steps actually executed.
    """
    if seed >= 0:
        np.random.seed(seed)
    affected = np.empty(26, np.int64)
    for step in range(n_steps):
        total = _total_rate(class_count, class_rate)
        if total <= 0.0:
            return step
        event = _select(class_rate, class_members, class_count, total)
        if event < 0:
            return step
        _apply(lat, event, L, event_class, event_pos, class_members,
               class_count, affected)
    return n_steps


@njit(cache=True)
def _run(lat, L, class_rate, t_max, max_steps, sample_interval, t_equil, seed):
    """Main kMC loop. Stops at t_max or max_steps, whichever comes first."""
    if seed >= 0:
        np.random.seed(seed)
    N = L * L
    event_class, event_pos, class_members, class_count = _init_tables(lat, L)
    # particle counts are kept incrementally so sampling never scans the lattice
    n_co = 0
    n_o = 0
    for i in range(N):
        if lat[i] == CO:
            n_co += 1
        elif lat[i] == O:
            n_o += 1
    buf_len = max_steps // sample_interval + 2
    times = np.empty(buf_len)
    cov_empty = np.empty(buf_len)
    cov_co = np.empty(buf_len)
    cov_o = np.empty(buf_len)
    n_samples = 0
    affected = np.empty(26, np.int64)  # scratch for _apply, allocated once
    t = 0.0
    steps = 0
    # steady-state accumulators. Each visited configuration is weighted by
    # the time the system actually spends in it. Averaging per event instead
    # would over-count short-lived states.
    time_weight = 0.0
    sum_empty = 0.0
    sum_co = 0.0
    sum_o = 0.0
    stuck = False
    while t < t_max and steps < max_steps:
        total = _total_rate(class_count, class_rate)
        if total <= 0.0:
            # nothing can happen anymore. With this chemistry that is the
            # O-poisoned lattice: no CO left and no empty sites.
            stuck = True
            break
        event = _select(class_rate, class_members, class_count, total)
        if event < 0:
            stuck = True
            break
        # exponential waiting time of the current state.
        dt = -np.log(1.0 - np.random.rand()) / total
        if t >= t_equil:
            # the state before the event is the one that lived for dt
            time_weight += dt
            sum_empty += (N - n_co - n_o) * dt
            sum_co += n_co * dt
            sum_o += n_o * dt
        d_co, d_o = _apply(lat, event, L, event_class, event_pos,
                           class_members, class_count, affected)
        n_co += d_co
        n_o += d_o
        t += dt
        steps += 1
        if steps % sample_interval == 0 and n_samples < buf_len:
            times[n_samples] = t
            cov_empty[n_samples] = (N - n_co - n_o) / N
            cov_co[n_samples] = n_co / N
            cov_o[n_samples] = n_o / N
            n_samples += 1
    if time_weight > 0.0:
        avg_empty = sum_empty / (time_weight * N)
        avg_co = sum_co / (time_weight * N)
        avg_o = sum_o / (time_weight * N)
    else:
        # the run ended before t_equil, report the final configuration
        avg_empty = (N - n_co - n_o) / N
        avg_co = n_co / N
        avg_o = n_o / N
    return (times[:n_samples].copy(), cov_empty[:n_samples].copy(),
            cov_co[:n_samples].copy(), cov_o[:n_samples].copy(),
            t, steps, avg_empty, avg_co, avg_o, stuck)


# user-facing API

@dataclass
class KMCParams:
    L: int = 16                     # lattice edge, N = L*L sites
    k_co_ads: float = 1.6              # CO adsorption rate, s^-1
    k_co_des: float = 1e-3             # CO desorption prefactor, s^-1
    k_rxn: float = 1.0                 # CO+O reaction prefactor, s^-1
    k_o_des: float = 0.0              # O2 desorption rate, s^-1; 0 keeps O2 ads irreversible
    eps: float = 8368.0             # CO-CO NN repulsion, J/mol
    T: float = 500.0                # temperature, K
    khop: float | None = None       # hop rate, defaults to khop_scale*max(k_o_ads, k_co_ads)
    khop_scale: float = 1000.0      # fast-diffusion factor, the paper asks for 3+ orders
    t_max: float = 30.0             # kMC time limit, s
    max_steps: int = 1_000_000_000  # event limit, whichever of the two hits first
    sample_interval: int = 10_000   # record coverages every this many events
    t_equil: float | None = None    # start of steady-state averaging, defaults to t_max/2
    seed: int = -1                  # RNG seed, -1 means do not seed


@dataclass
class KMCResult:
    k_o_ads: float
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
    lattice: np.ndarray


def make_lattice(L, init="empty", theta_co=0.0, theta_o=0.0, rng=None):
    """'empty', 'full' (all CO), or 'random' with given coverages."""
    N = L * L
    if init == "empty":
        return np.zeros(N, np.int8)
    if init == "full":
        return np.full(N, CO, np.int8)
    if init == "random":
        rng = rng or np.random.default_rng()
        n_co = int(round(theta_co * N))
        n_o = int(round(theta_o * N))
        lat = np.zeros(N, np.int8)
        lat[:n_co] = CO
        lat[n_co:n_co + n_o] = O
        rng.shuffle(lat)
        return lat
    raise ValueError(f"unknown init {init!r}")


def run_kmc(k_o_ads, init="empty", params=None, lat0=None, **overrides):
    """Run one kMC trajectory at O2 impingement rate k_o_ads.

    Stops at t >= t_max or after max_steps events, whichever comes first.
    Steady-state coverages are time-weighted averages over t >= t_equil.
    """
    params = replace(params or KMCParams(), **overrides)
    lat = lat0.copy() if lat0 is not None else make_lattice(params.L, init)
    khop = (params.khop if params.khop is not None
            else params.khop_scale * max(k_o_ads, params.k_co_ads))
    class_rate = make_class_rates(params.k_co_ads, params.k_co_des, k_o_ads, params.k_rxn,
                                  khop, params.k_o_des, params.eps, params.T)
    t_equil = params.t_equil if params.t_equil is not None else 0.5 * params.t_max
    (times, cov_empty, cov_co, cov_o, t_final, steps,
     avg_empty, avg_co, avg_o, stuck) = _run(lat, params.L, class_rate,
                                             params.t_max, params.max_steps,
                                             params.sample_interval, t_equil,
                                             params.seed)
    return KMCResult(k_o_ads=k_o_ads, times=times, cov_empty=cov_empty, cov_co=cov_co,
                     cov_o=cov_o, steady_empty=avg_empty, steady_co=avg_co,
                     steady_o=avg_o, t_final=t_final, steps=steps,
                     stuck=bool(stuck), lattice=lat)


# first-passage runs
# The same n-fold machinery as _run, stopped the first time a caller-supplied
# predicate on the particle counts turns true. That is what a mean-first-passage
# time needs: the *hitting* time of a set of configurations, not a time average
# over a fixed window. The predicate is an njit function passed straight into the
# compiled loop, so a user-written stopping condition costs a call per event and
# nothing else. Passing a function argument defeats numba's on-disk cache, hence
# cache=False here (the loop body itself is the cached functions above).

@njit(cache=False)
def _run_first_passage(lat, L, class_rate, predicate, t_max, max_steps, seed):
    """Run until predicate(n_empty, n_co, n_o, N) is true, or until the time/step
    limit. Returns (t_hit, hit, t_final, steps, stuck); t_hit is NaN when the run
    ended without ever satisfying the predicate."""
    if seed >= 0:
        np.random.seed(seed)
    N = L * L
    event_class, event_pos, class_members, class_count = _init_tables(lat, L)
    n_co = 0
    n_o = 0
    for i in range(N):
        if lat[i] == CO:
            n_co += 1
        elif lat[i] == O:
            n_o += 1
    affected = np.empty(26, np.int64)
    t = 0.0
    steps = 0
    # a starting configuration already in the target set has passage time zero;
    # testing before the loop keeps that case from being reported as a miss
    if predicate(N - n_co - n_o, n_co, n_o, N):
        return 0.0, True, 0.0, 0, False
    while t < t_max and steps < max_steps:
        total = _total_rate(class_count, class_rate)
        if total <= 0.0:
            return np.nan, False, t, steps, True
        event = _select(class_rate, class_members, class_count, total)
        if event < 0:
            return np.nan, False, t, steps, True
        dt = -np.log(1.0 - np.random.rand()) / total
        d_co, d_o = _apply(lat, event, L, event_class, event_pos, class_members,
                           class_count, affected)
        n_co += d_co
        n_o += d_o
        t += dt
        steps += 1
        if predicate(N - n_co - n_o, n_co, n_o, N):
            return t, True, t, steps, False
    return np.nan, False, t, steps, False


@dataclass
class FirstPassageResult:
    k_o_ads: float
    t_hit: float            # first time the predicate held, NaN if it never did
    hit: bool               # False means the sample is right-censored at t_final
    t_final: float
    steps: int
    stuck: bool             # the lattice ran out of possible events (O-poisoned)
    lattice: np.ndarray


def coverage_predicate(species="co", threshold=0.5, above=True):
    """An njit stopping predicate on one species' fractional coverage.

    `above=True` stops the first time theta_species >= threshold, `above=False` the
    first time it drops to <= threshold. This is the predicate factory to reach for
    first; write your own njit (n_empty, n_co, n_o, N) -> bool for anything else."""
    codes = {"empty": 0, "co": 1, "o": 2}
    if species not in codes:
        raise ValueError(f"unknown species {species!r}, choose from {sorted(codes)}")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(f"threshold is a coverage fraction in [0, 1], got {threshold}")
    code = codes[species]     # closed over as a compile-time constant by numba
    hi = bool(above)

    @njit(cache=False)
    def predicate(n_empty, n_co, n_o, N):
        if code == 0:
            n = n_empty
        elif code == 1:
            n = n_co
        else:
            n = n_o
        if hi:
            return n >= threshold * N
        return n <= threshold * N

    return predicate


def run_first_passage(k_o_ads, predicate, init="empty", params=None, lat0=None,
                      **overrides):
    """Run one kMC trajectory until `predicate` first holds, and report when.

    `predicate` is an njit function (n_empty, n_co, n_o, N) -> bool; build a common one
    with coverage_predicate(). Parameters are the same KMCParams as run_kmc, and
    params.t_max / params.max_steps still bound the run: a trajectory that reaches
    either without hitting comes back with hit=False and t_hit=NaN, i.e. censored at
    t_final rather than silently reported as a passage time."""
    params = replace(params or KMCParams(), **overrides)
    lat = lat0.copy() if lat0 is not None else make_lattice(params.L, init)
    khop = (params.khop if params.khop is not None
            else params.khop_scale * max(k_o_ads, params.k_co_ads))
    class_rate = make_class_rates(params.k_co_ads, params.k_co_des, k_o_ads, params.k_rxn,
                                  khop, params.k_o_des, params.eps, params.T)
    t_hit, hit, t_final, steps, stuck = _run_first_passage(
        lat, params.L, class_rate, predicate, params.t_max, params.max_steps,
        params.seed)
    return FirstPassageResult(k_o_ads=k_o_ads, t_hit=float(t_hit), hit=bool(hit),
                              t_final=float(t_final), steps=int(steps),
                              stuck=bool(stuck), lattice=lat)
