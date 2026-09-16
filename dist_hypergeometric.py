"""
Hypergeometric sampler FP-error analysis.
Two regimes, matching numpy's dispatch (random_hypergeometric, see _use_hrua):
  _HRUA_SWITCH <= n <= N - _HRUA_SWITCH : HRUA (ratio-of-uniforms rejection,
                                          distributions/hypergeometric_hrua.c),
                                          analysed like dist_poisson.py's PTRS
                                          (eps_floor / eps_accept split)
  otherwise                             : HYP  (inversion-style loop,
                                          distributions/hypergeometric_hyp.c)

Every runner takes N, K, n either as points or as (lo, hi) integer intervals
(interval mode, --N-range / --K-range / --n-range): see dist_common's
"Interval (box) mode" section.  A box's HRUA analysis works from enclosures
of the integers the templates embed (popsize, min(K, N-K), m = min(n, N-n),
the mode index d9) -- taken over the box's valid, non-degenerate HRUA points
only (hrua_consts) -- and decouples the once-per-point constants (d6, d8,
d10) from the floor/accept queries, which otherwise don't finish (see
_run_hrua_fptaylor).
"""
import math
import time
from pathlib import Path

from dist_common import (
    ROOT, FP_TO_FPTAYLOR_RND,
    run_command, extract_abs_errors_by_problem,
    save_loglog_plot,
    loggam_defs, run_fptaylor_query,
    ulp_rnd_op,
    iv, interval_ivar,
    rou_proposal_deviation, acceptance_tv,
    vprint, elapsed_since, format_seconds,
    floor_x_abs_tol_vars, accept_x_abs_tol_vars,
    dist_switch,
    BoxTooWide, analyse_param_box, max_fields, parse_range, bisect_box,
    safe_box_name, box_label, csv_num, fmt_num, with_param_tols,
    int_or_float_str,
)

NAME = "hypergeometric"
CSV_FIELDS = ["N", "K", "n", "N_lo", "N_hi", "K_lo", "K_hi", "n_lo", "n_hi",
              "fp", "regime", "delta", "eps_floor", "eps_accept", "tv",
              "n_boxes", "time_s"]

# sample-count threshold: HYP below, HRUA above (see _use_hrua) --
# overridable via fptaylor_settings.toml's [hypergeometric].switch
# (dist_common.dist_switch).
_HRUA_SWITCH = dist_switch(NAME, 10)

_D1 = 1.7155277699214135   # 2*sqrt(2/e)
_D2 = 0.8989161620588988   # 3 - 2*sqrt(3/e)

# Interval mode enumerates a box's integer points to maximise the modal
# probability (_hrua_modal_pmf_max) when there are at most this many.
_PMF_ENUM_MAX = 200_000


# ---------------------------------------------------------------------------
# HRUA FPTaylor templates  (_use_hrua)
# ---------------------------------------------------------------------------

def _is_box(N, K, n):
    return any(isinstance(x, tuple) for x in (N, K, n))


def hrua_consts(N, K, n):
    """
    The d4..d11 setup constants of random_hypergeometric_hrua
    (distributions/hypergeometric_hrua.c lines 81-93), computed in exact
    Python arithmetic.

    For a box, the enclosures (lo, hi) of popsize, mingoodbad, m, n, d6, d7,
    d8, d9 and d11 over the box's valid (K, n <= N), non-degenerate
    (mingoodbad >= 1) HRUA (_HRUA_SWITCH <= n <= N - _HRUA_SWITCH, hence m >=
    _HRUA_SWITCH) points -- the only ones the HRUA analysis has to cover.
    Every factor below is positive and monotone, so each bound is a product
    of the factors' own bounds.
    """
    if _is_box(N, K, n):
        return _hrua_box_consts(N, K, n)
    good, bad  = K, N - K
    mingoodbad = min(good, bad)
    maxgoodbad = max(good, bad)
    popsize    = good + bad
    m          = min(n, popsize - n)

    d4  = mingoodbad / popsize
    d5  = 1.0 - d4
    d6  = m * d4 + 0.5
    d7  = math.sqrt((popsize - m) * n * d4 * d5 / (popsize - 1) + 0.5)
    d8  = _D1 * d7 + _D2
    d9  = int(math.floor((m + 1) * (mingoodbad + 1) / (popsize + 2)))
    d10 = (math.lgamma(d9 + 1) + math.lgamma(mingoodbad - d9 + 1)
           + math.lgamma(m - d9 + 1) + math.lgamma(maxgoodbad - m + d9 + 1))
    d11 = min(min(m, mingoodbad) + 1.0, math.floor(d6 + 16 * d7))

    return dict(mingoodbad=mingoodbad, maxgoodbad=maxgoodbad, popsize=popsize,
                m=m, d6=d6, d7=d7, d8=d8, d10=d10, d11=d11)


def _hrua_box_consts(N, K, n):
    (N0, N1), (K0, K1), (n0, n1) = iv(N), iv(K), iv(n)
    s = _HRUA_SWITCH
    P   = (N0, N1)
    nn  = (max(n0, s), min(n1, N1 - s))
    mgb = (max(1, min(K0, N0 - K1)), min(K1, N1 - K0, N1 // 2))
    m   = (max(s, min(nn[0], N0 - nn[1])), min(nn[1], N1 - nn[0], N1 // 2))
    if nn[0] > nn[1] or mgb[0] > mgb[1] or m[0] > m[1]:
        raise RuntimeError(f"{box_label({'N': N, 'K': K, 'n': n})}: "
                           "no non-degenerate HRUA point in the box")
    d4 = (mgb[0] / P[1], min(0.5, mgb[1] / P[0]))
    d45 = (d4[0] * (1.0 - d4[0]), d4[1] * (1.0 - d4[1]))    # d4*d5: increasing on [0, 1/2]
    Pm = (max(P[0] - m[1], P[0] / 2), P[1] - m[0])          # popsize - m >= popsize/2
    var = (Pm[0] * nn[0] * d45[0] / (P[1] - 1), Pm[1] * nn[1] * d45[1] / (P[0] - 1))
    d6 = (m[0] * d4[0] + 0.5, m[1] * d4[1] + 0.5)
    d7 = (math.sqrt(var[0] + 0.5), math.sqrt(var[1] + 0.5))
    d8 = (_D1 * d7[0] + _D2, _D1 * d7[1] + _D2)
    d9 = (math.floor((m[0] + 1) * (mgb[0] + 1) / (P[1] + 2)),
          math.floor((m[1] + 1) * (mgb[1] + 1) / (P[0] + 2)))
    d11 = (min(min(m[0], mgb[0]) + 1.0, math.floor(d6[0] + 16 * d7[0])),
           min(min(m[1], mgb[1]) + 1.0, math.floor(d6[1] + 16 * d7[1])))
    return dict(popsize=P, mingoodbad=mgb, m=m, n=nn, d6=d6, d7=d7, d8=d8,
                d9=d9, d11=d11)


# Variable names a box's HRUA templates use for its derived integers.
_BOX_VARS = {"popsize": "Pv", "mingoodbad": "mgbv", "m": "mv", "n": "nv", "d9": "d9v"}


def hrua_setup_defs(rnd, N, K, n, exact=False, prefix=""):
    """
    random_hypergeometric_hrua's setup block [hypergeometric_hrua.c lines
    85-92] as FPTaylor Definitions.  d4..d8 are derived from the integer
    parameters, not free inputs, so they are written as expressions: that
    charges the rounding of the setup arithmetic, which embedding them as
    literals silently drops.

    mingoodbad, maxgoodbad, popsize and m are integer-valued and exact, so
    for a point they stay literals; for a box they are the _BOX_VARS
    Variables.  exact=True drops the rounding markers, for the copies that
    feed Z (see hrua_z_defs).
    """
    r = "=" if exact else f"{rnd}="
    if _is_box(N, K, n):
        P, M, mgb, nn = "Pv", "mv", "mgbv", "nv"
        P_minus_M, P_minus_1 = "(Pv - mv)", "(Pv - 1.0)"
    else:
        c = hrua_consts(N, K, n)
        P, M, mgb = (f"{float(c[k]):.1f}" for k in ("popsize", "m", "mingoodbad"))
        nn = f"{float(n):.1f}"
        P_minus_M = f"{float(c['popsize'] - c['m']):.1f}"
        P_minus_1 = f"{float(c['popsize'] - 1):.1f}"
    return [
        f"  {prefix}d4_ {r} {mgb} / {P},",
        f"  {prefix}d5_ {r} 1.0 - {prefix}d4_,",
        f"  {prefix}d6_ {r} {M} * {prefix}d4_ + 0.5,",
        f"  {prefix}d7_ {r} sqrt({P_minus_M} * {nn}"
        f" * {prefix}d4_ * {prefix}d5_ / {P_minus_1} + 0.5),",
        f"  {prefix}d8_ {r} {_D1:.20e} * {prefix}d7_ + {_D2:.20e},",
    ]


def _box_var_lines(c, names):
    """float64 Variables for a box's derived integers: they are exact
    doubles in the C code (see dist_common.param_ivar on why not `real`)."""
    return [interval_ivar(_BOX_VARS[k], *c[k], kind="float64") for k in names]


def hrua_accept_z_range(N, K, n):
    """
    Z window the accept query covers.  Z = floor(W) lives in [0, d11 - 1];
    narrow it to keep every inlined lgamma argument > 0 -- a hard
    domain-validity floor, same role as dist_poisson._K_BOUNDARY_MARGIN.
    Z = W - f only guarantees Z > W_lo - 1, so the template's W window
    starts one above z_lo.  For a box, the union over its points (z_lo is 0
    for every valid point: maxgoodbad >= popsize/2 >= m).
    """
    c = hrua_consts(N, K, n)
    if _is_box(N, K, n):
        z_lo = 0.0
        z_hi = float(min(int(c["d11"][1]) - 1, c["mingoodbad"][1], c["m"][1]))
    else:
        mgb, Mgb, m = c["mingoodbad"], c["maxgoodbad"], c["m"]
        z_lo = float(max(0, -(Mgb - m)))
        z_hi = float(min(int(c["d11"]) - 1, mgb, m))
    if z_lo >= z_hi:
        raise RuntimeError(
            f"{_NKn_label(N, K, n)}: no Z range with all loggam arguments > 0 "
            f"(z_lo={z_lo}, z_hi={z_hi}); HRUA analysis not applicable")
    return z_lo, z_hi


def _check_hrua_setup_box(c):
    """Raise BoxTooWide unless the box setup query's sqrt/division stay
    well-defined with its integers ranging independently (popsize - m > 0
    for d7's sqrt, mingoodbad < popsize for d5 = 1 - d4 > 0)."""
    P0 = c["popsize"][0]
    bad = [what for what, v in (("popsize - m", P0 - c["m"][1]),
                                ("popsize - mingoodbad", P0 - c["mingoodbad"][1]))
           if v <= 0]
    if bad:
        raise BoxTooWide(f"box too wide for {', '.join(bad)} to stay positive")


def hrua_box_arg_ranges(c, z_lo, z_hi):
    """
    Ranges of the eight lgamma arguments of the acceptance test over a box:
    A1..A4 = Z+1, mingoodbad-Z+1, m-Z+1, maxgoodbad-m+Z+1 and B1..B4 = the
    same with d9 for Z (d10's).  The C code forms every one of them in exact
    int64 arithmetic [hypergeometric_hrua.c lines 91-92, 106-107], and each
    is >= 1 at every point (Z <= min(mingoodbad, m); d9 <= (m+1)/2 and
    <= (mingoodbad+1)/2), so each range is floored at 1 -- which is what
    lets the box accept query take them as independent Variables.
    """
    (P0, P1), (g0, g1), (m0, m1), (d0, d1) = (c["popsize"], c["mingoodbad"],
                                              c["m"], c["d9"])
    Mm = (max(0, P0 - g1 - m1), P1 - g0 - m0)           # maxgoodbad - m >= 0
    return {
        "A1": (z_lo + 1, z_hi + 1),
        "A2": (max(1, g0 - z_hi + 1), g1 - z_lo + 1),
        "A3": (max(1, m0 - z_hi + 1), m1 - z_lo + 1),
        "A4": (Mm[0] + z_lo + 1, Mm[1] + z_hi + 1),
        "B1": (d0 + 1, d1 + 1),
        "B2": (max(1, g0 - d1 + 1), g1 - d0 + 1),
        "B3": (max(1, m0 - d1 + 1), m1 - d0 + 1),
        "B4": (Mm[0] + d0 + 1, Mm[1] + d1 + 1),
    }


def _log_choose(n, k):
    if k < 0 or k > n:
        return -math.inf
    return math.lgamma(n + 1.0) - math.lgamma(k + 1.0) - math.lgamma(n - k + 1.0)


def hrua_modal_pmf(N, K, n):
    c = hrua_consts(N, K, n)
    z = int(math.floor((c["m"] + 1) * (c["mingoodbad"] + 1)
                       / (c["popsize"] + 2)))
    log_p = (_log_choose(c["mingoodbad"], z)
             + _log_choose(c["maxgoodbad"], c["m"] - z)
             - _log_choose(c["popsize"], c["m"]))
    return math.exp(log_p)


def _hrua_modal_pmf_max(N, K, n):
    """
    (max of hrua_modal_pmf over the box's HRUA points, how): exact
    enumeration when the box has at most _PMF_ENUM_MAX integer points
    (widened by 1e-6 relative for lgamma's rounding), else the trivial
    bound 1.  The modal probability isn't monotone in (N, K, n), so no
    corner evaluation is sound here.
    """
    (N0, N1), (K0, K1), (n0, n1) = iv(N), iv(K), iv(n)
    count = (N1 - N0 + 1) * (K1 - K0 + 1) * (n1 - n0 + 1)
    if count > _PMF_ENUM_MAX:
        return 1.0, f"trivial bound (box has {count} points)"
    best = 0.0
    for NN in range(N0, N1 + 1):
        for KK in range(K0, min(K1, NN) + 1):
            if min(KK, NN - KK) == 0:
                continue
            for nn in range(max(n0, _HRUA_SWITCH), min(n1, NN - _HRUA_SWITCH) + 1):
                best = max(best, hrua_modal_pmf(NN, KK, nn))
    return min(1.0, best * (1.0 + 1e-6)), f"enumerated {count} points"


def hrua_xtail(N, K, n, ulps=8.0):
    """
    The X cutoff that minimizes the total eps_floor error budget.

    Restricting X to [xtail, 1] trades two terms against each other.  A draw
    survives the fast rejection  (W < 0) || (W >= d11)  iff

        -d6*X/d8  <=  Y - 0.5  <  (d11 - d6)*X/d8            [line 99-105]

    so P(survive | X) = min(1, d11*X/d8) -- linear in X, never zero.  Cutting
    the domain at xtail therefore discards surviving draws with probability

        eps_tail(x) = int_0^x d11*t/d8 dt = d11 * x^2 / (2*d8)     (quadratic)

    while the analysis error grows as the worst case |W| = d6 + d8/(2x):

        eps_floor(x) ~ rho * d8 / (2x),  rho ~ `ulps` * 2^-53      (hyperbolic)

    rho is the measured relative error of the FPTaylor bound, flat at ~6e-16
    (about 5 ulps) over 24 decades of xtail; `ulps` defaults to 8 for margin.
    Minimizing the sum gives x* = (rho * d8^2 / (2*d11))^(1/3).

    The hyperbolic term is an artifact of the analysis box: FPTaylor treats X
    and Y as independent and so evaluates at |Y-0.5| = 0.5, where the real
    algorithm would already have rejected the draw.  A joint constraint on
    (X, Y) would remove it, at which point xtail could go much lower.

    For a box, x* is taken at its smallest (d8's lo, d11's hi), so the
    analysed X range contains every point's own.

    Returns (xtail, eps_floor_estimate, eps_tail).
    """
    c = hrua_consts(N, K, n)
    d6, d8, d11 = iv(c["d6"])[1], iv(c["d8"])[0], iv(c["d11"])[1]
    rho = ulps * 2.0 ** -53
    xtail = (rho * d8 * d8 / (2.0 * d11)) ** (1.0 / 3.0)
    return xtail, rho * (d6 + d8 / (2.0 * xtail)), d11 * xtail ** 2 / (2.0 * d8)


def hrua_z_defs():
    """
    Z = floor(W) as exact (unrounded) Definitions.

    W = d6 + d8*(Y - 0.5)/X is a *derived* quantity, so Z must not be a free
    variable.  But the reachable region is a bowtie in (X, Y) -- the sampler
    rejects W < 0 and W >= d11 before ever forming Z, and no box in (X, Y)
    excludes those -- which leaves FPTaylor's conservative range for Z
    spanning zero, tripping its division-by-zero check on 1/(Z+1).

    For fixed X the map Y -> W is affine and invertible, so (X, W) describes
    exactly the same draws as (X, Y) and there the reachable region *is* a
    box.  The template therefore takes W as the second variable and recovers
    Y from it.  As in the BTRS/PTRS templates, these definitions carry no
    rounding: Z is one integer, computed once, and both samplers feed the
    same integer into the acceptance test.
    """
    return ["  Z_  = W - f,", "  Z1_ = Z_ + 1.0,"]


def make_hrua_floor_template(N, K, n, fp, x_lo, x_hi, d6=None, d8=None):
    """
    FPTaylor expression for eps_floor: absolute error of the candidate

        W = d6 + d8 * (Y - 0.5) / X          [hypergeometric_hrua.c line 99]

    whose floor is the proposed Z.  X, Y are independent uniforms; X is
    restricted to [x_lo, x_hi] because W blows up as X -> 0 (those draws
    are rejected by the W >= d11 test).

    For a box, d6 and d8 are float64 Variables over the given enclosures of
    the doubles the C code computed once (make_hrua_setup_template bounds
    that computation's own error, which the caller adds): folding the whole
    setup chain in here instead, over the box's four independently-ranging
    integers, measured 582s for a bound 30x looser.
    """
    rnd = FP_TO_FPTAYLOR_RND[fp]
    if _is_box(N, K, n):
        return (
            "Variables\n"
            f"  real X in [{x_lo:.20e}, {x_hi:.20e}],\n"
            f"  real Y in [0.0, 1.0],\n"
            + interval_ivar("d6_", *d6, kind="float64") + ",\n"
            + interval_ivar("d8_", *d8, kind="float64") + ";\n\n"
            + "Definitions\n"
            + f"  hrua_floor {rnd}= d6_ + d8_ * (Y - 0.5) / X;\n\n"
            + "Expressions\n"
              "  eps_floor = hrua_floor;\n"
        )

    return (
        "Variables\n"
        f"  real X in [{x_lo:.20e}, {x_hi:.20e}],\n"
        f"  real Y in [0.0, 1.0];\n\n"
        + "Definitions\n"
        + "\n".join(hrua_setup_defs(rnd, N, K, n)) + "\n"
        + f"  hrua_floor {rnd}= d6_ + d8_ * (Y - 0.5) / X;\n\n"
        + "Expressions\n"
          "  eps_floor = hrua_floor;\n"
    )


def make_hrua_setup_template(N, K, n, fp):
    """
    Box only: absolute errors e6, e8 of the setup chain computing d6 and d8
    (hrua_setup_defs) over the box's integers -- the part of eps_floor the
    box floor query (make_hrua_floor_template) takes d6/d8 as given for.
    """
    rnd = FP_TO_FPTAYLOR_RND[fp]
    c = hrua_consts(N, K, n)
    return (
        "Variables\n"
        + ",\n".join(_box_var_lines(c, ("popsize", "mingoodbad", "m", "n"))) + ";\n\n"
        + "Definitions\n"
        + "\n".join(hrua_setup_defs(rnd, N, K, n)).rstrip(",") + ";\n\n"
        + "Expressions\n"
          "  e6 = d6_;\n"
          "  e8 = d8_;\n"
    )


def make_hrua_d10_template(fp, ranges):
    """
    Box only: absolute error e10 of d10 = sum of lgamma(B1..B4)
    [hypergeometric_hrua.c lines 91-92], each exact-integer argument ranging
    over hrua_box_arg_ranges -- computed once per parameter point, so the box
    accept query takes d10 as a Variable and the caller adds e10.
    """
    rnd = FP_TO_FPTAYLOR_RND[fp]
    defs = [loggam_defs(b, f"l{b}", rnd) for b in ("B1", "B2", "B3", "B4")]
    return (
        "Variables\n"
        + ",\n".join(interval_ivar(b, *ranges[b], kind="float64")
                     for b in ("B1", "B2", "B3", "B4")) + ";\n\n"
        + "Definitions\n"
        + "\n".join(line for d, _ in defs for line in d) + "\n"
        + f"  d10_ {rnd}= {' + '.join(name for _, name in defs)};\n\n"
        + "Expressions\n"
          "  e10 = d10_;\n"
    )


def make_hrua_accept_template(N, K, n, fp, x_lo, x_hi, z_lo, z_hi, d10=None,
                              ranges=None):
    """
    FPTaylor expression for eps_accept: absolute error of the acceptance
    test  2*log(X) <= T  written as a single expression

        2*log(X) - T,   T = d10 - (loggam(Z+1) + loggam(mingoodbad-Z+1)
                                   + loggam(m-Z+1) + loggam(maxgoodbad-m+Z+1))
                                             [hypergeometric_hrua.c line 117]

    Each loggam(x) is FPTaylor's native lgamma(x) directly (loggam_defs,
    dist_common.py). Z is restricted to [z_lo, z_hi] (from
    hrua_accept_z_range), where all four lgamma arguments stay > 0.

    For a box, the four arguments are independent exact (float64) Variables
    over hrua_box_arg_ranges and d10 a float64 Variable over its enclosure
    (make_hrua_d10_template bounds its own error, which the caller adds) --
    a sound relaxation, since every argument is exact int64 arithmetic in
    the C code, and --approx bounds each error term separately anyway.
    Tying them to Z and the box's integers instead measured >240s per query.
    """
    rnd = FP_TO_FPTAYLOR_RND[fp]
    log_rnd = ulp_rnd_op(rnd, "log")
    if _is_box(N, K, n):
        defs = [loggam_defs(a, f"l{a}", rnd) for a in ("A1", "A2", "A3", "A4")]
        return (
            "Variables\n"
            f"  real X in [{x_lo:.20e}, {x_hi:.20e}],\n"
            + ",\n".join(interval_ivar(a, *ranges[a], kind="float64")
                         for a in ("A1", "A2", "A3", "A4")) + ",\n"
            + interval_ivar("d10_", *d10, kind="float64") + ";\n\n"
            + "Definitions\n"
            + "\n".join(line for d, _ in defs for line in d) + "\n"
            + f"  log_x_ {log_rnd}= log(X),\n"
            + f"  hrua_accept {rnd}= 2.0 * log_x_ - d10_"
              f" + {' + '.join(name for _, name in defs)};\n\n"
            + "Expressions\n"
              "  eps_accept = hrua_accept;\n"
        )

    c   = hrua_consts(N, K, n)
    mgb, Mgb, m = c["mingoodbad"], c["maxgoodbad"], c["m"]

    defs_z,  name_z  = loggam_defs("Z1_", "lgz", rnd)
    defs_mz, name_mz = loggam_defs(f"{float(mgb):.1f} - Z_ + 1.0", "lgmz", rnd)
    defs_kz, name_kz = loggam_defs(f"{float(m):.1f} - Z_ + 1.0", "lgkz", rnd)
    defs_Mz, name_Mz = loggam_defs(f"{float(Mgb - m):.1f} + Z_ + 1.0", "lgMz", rnd)

    # W rather than Y is the second variable, and Z = W - f (see hrua_z_defs);
    # W in [z_lo + 1, z_hi] is what keeps Z inside the loggam domain.
    d9 = int(math.floor((m + 1) * (mgb + 1) / (c["popsize"] + 2)))
    # d10's four lgamma calls are represented as native FPTaylor calls rather
    # than precomputed Python literals, so each call's rounding is charged.
    defs_d9,  name_d9  = loggam_defs(f"{float(d9) + 1.0:.1f}", "d10a", rnd)
    defs_mgb, name_mgb = loggam_defs(f"{float(mgb - d9) + 1.0:.1f}", "d10b", rnd)
    defs_m,   name_m   = loggam_defs(f"{float(m - d9) + 1.0:.1f}", "d10c", rnd)
    defs_Mgb, name_Mgb = loggam_defs(f"{float(Mgb - m + d9) + 1.0:.1f}", "d10d", rnd)
    return (
        "Variables\n"
        f"  real X in [{x_lo:.20e}, {x_hi:.20e}],\n"
        f"  real W in [{z_lo + 1.0:.1f}, {z_hi:.1f}],\n"
        f"  real f in [0.0, 1.0];\n\n"
        + "Definitions\n"
        + "\n".join(defs_d9)  + "\n"
        + "\n".join(defs_mgb) + "\n"
        + "\n".join(defs_m)   + "\n"
        + "\n".join(defs_Mgb) + "\n"
        + f"  d10_ {rnd}= {name_d9} + {name_mgb} + {name_m} + {name_Mgb},\n"
        + "\n".join(hrua_z_defs()) + "\n"
        + "\n".join(defs_z)  + "\n"
        + "\n".join(defs_mz) + "\n"
        + "\n".join(defs_kz) + "\n"
        + "\n".join(defs_Mz) + "\n"
        + f"  log_x_ {log_rnd}= log(X),\n"
        + f"  hrua_accept {rnd}= 2.0 * log_x_ - d10_"
          f" + {name_z} + {name_mz} + {name_kz} + {name_Mz};\n\n"
        + "Expressions\n"
          "  eps_accept = hrua_accept;\n"
    )


def _widen(enc, err, rel=1e-12):
    """(lo, hi) widened by err plus a relative pad for Python's own
    evaluation of the enclosure."""
    lo, hi = enc
    pad = err + rel * max(abs(lo), abs(hi))
    return lo - pad, hi + pad


def _run_hrua_fptaylor(fptaylor, N, K, n, args, tag, inputs_dir, outputs_dir, env):
    """(eps_floor, eps_accept, tv) for the HRUA regime at (N, K, n);
    composes the ratio-of-uniforms rejection bound.

    W's declared range (hrua_accept_z_range) grows with N/K/n, so a flat
    --opt-x-abs-tol forces ever more splitting on it as the population
    scales up -- auto-derive a per-case value (~10x N) unless the user
    passed their own via --opt-x-abs-tol-vars (or its floor/accept
    variants).

    args.u_trunc plays the same role as binomial/poisson's --u-trunc: a
    floor on X (the ratio-of-uniforms proposal variable, playing the same
    role there as u/us does in BTRS/PTRS) below which the domain is
    clipped rather than analyzed, charged directly to TV as a flat amount
    -- same tradeoff as --v-trunc/--u-trunc elsewhere (see add_common_args
    in dist_common.py), not the tighter quadratic tail-probability
    estimate (hrua_xtail's docstring).

    N, K, n may be (lo, hi) intervals.  The box analysis splits each side
    into two small queries -- the once-per-point constants (d6/d8's setup
    chain, d10) bounded on their own and fed to the floor/accept queries as
    Variables over their enclosures, errors added -- see
    make_hrua_floor_template / make_hrua_accept_template.
    rou_proposal_deviation's factor is worst at d8's lo, reject_const at
    d8's hi times the box's largest modal probability (_hrua_modal_pmf_max).
    """
    label = _NKn_label(N, K, n)
    box = _is_box(N, K, n)
    fp, verbose = args.fp, args.verbose
    ratio_tol, bb_eval = args.bb_geometric_ratio_tol, args.bb_eval
    x_abs_tol, approx = args.opt_x_abs_tol, args.approx
    u_trunc = args.u_trunc
    auto_tol_vars = f"W={10.0 * iv(N)[1]:.6g}"
    floor_tol_vars = floor_x_abs_tol_vars(args)
    if floor_tol_vars is None:
        floor_tol_vars = auto_tol_vars
    accept_tol_vars = accept_x_abs_tol_vars(args)
    if accept_tol_vars is None:
        accept_tol_vars = auto_tol_vars
    c = hrua_consts(N, K, n)
    d8_lo, d8_hi = iv(c["d8"])

    xtail_star, _, _ = hrua_xtail(N, K, n)
    x_lo, x_hi = max(u_trunc, xtail_star), 1.0
    z_lo, z_hi = hrua_accept_z_range(N, K, n)
    if box:
        _check_hrua_setup_box(c)
        pmf_max, pmf_how = _hrua_modal_pmf_max(N, K, n)
        reject_const = 2.0 * d8_hi * pmf_max
    else:
        pmf_how = "exact"
        reject_const = 2.0 * c["d8"] * hrua_modal_pmf(N, K, n)
    vprint(verbose, f"hypergeometric HRUA {label}",
           d6=c["d6"], d7=c["d7"], d8=c["d8"], d11=c["d11"],
           x_lo=x_lo, x_hi=x_hi, z_lo=z_lo, z_hi=z_hi, u_trunc=u_trunc,
           reject_const=reject_const, modal_pmf=pmf_how)

    def query(what, text, tol_vars):
        in_path  = inputs_dir  / f"hypergeometric_hrua_{what}_{fp}_{tag}.txt"
        out_path = outputs_dir / f"hypergeometric_hrua_{what}_{fp}_{tag}.out"
        in_path.write_text(text)
        code, output = run_fptaylor_query(fptaylor, in_path, outputs_dir, env,
                                          ratio_tol, bb_eval, x_abs_tol, tol_vars, approx)
        out_path.write_text(output)
        if verbose >= 2:
            print(f"--- FPTaylor HRUA {what} ({label}) ---\n{output}")
        if code != 0:
            raise RuntimeError(f"FPTaylor HRUA {what} failed for {label}; see {out_path}")
        return extract_abs_errors_by_problem(output)

    if not box:
        # ---- floor ----
        floor_raw = query("floor", make_hrua_floor_template(N, K, n, fp, x_lo, x_hi),
                          floor_tol_vars)["eps_floor"]
        # ---- accept ----
        eps_accept = query("accept", make_hrua_accept_template(
            N, K, n, fp, x_lo, x_hi, z_lo, z_hi), accept_tol_vars)["eps_accept"]
    else:
        # ---- floor: setup-chain errors, then W over d6/d8's enclosures ----
        setup_vars = with_param_tols(floor_tol_vars, {_BOX_VARS[k]: c[k] for k in
                                                      ("popsize", "mingoodbad", "m", "n")})
        errs = query("setup", make_hrua_setup_template(N, K, n, fp), setup_vars)
        e6, e8 = errs["e6"], errs["e8"]
        d6, d8 = _widen(c["d6"], e6), _widen(c["d8"], e8)
        floor_raw = query("floor", make_hrua_floor_template(
            N, K, n, fp, x_lo, x_hi, d6=d6, d8=d8),
            with_param_tols(floor_tol_vars, {"d6_": d6, "d8_": d8}))["eps_floor"]
        # W = d6 + d8*(Y-0.5)/X: the computed d6/d8's own errors shift it by
        # at most e6 + e8*|Y-0.5|/X
        floor_raw += e6 + e8 * 0.5 / x_lo

        # ---- accept: d10's error, then the test over its enclosure ----
        ranges = hrua_box_arg_ranges(c, z_lo, z_hi)
        int_tols = ",".join(f"{k}=1" for k in ranges)       # exact integers
        e10 = query("d10", make_hrua_d10_template(fp, ranges), int_tols)["e10"]
        # lgamma is nondecreasing over the integers >= 1
        d10 = _widen((sum(math.lgamma(ranges[b][0]) for b in ("B1", "B2", "B3", "B4")),
                      sum(math.lgamma(ranges[b][1]) for b in ("B1", "B2", "B3", "B4"))),
                     e10, rel=1e-9)
        eps_accept = query("accept", make_hrua_accept_template(
            N, K, n, fp, x_lo, x_hi, z_lo, z_hi, d10=d10, ranges=ranges),
            with_param_tols(int_tols, {"d10_": d10}))["eps_accept"] + e10
    eps_floor = rou_proposal_deviation(floor_raw, d8_lo)

    # Charge the *actual* cutoff (xtail = max(u_trunc, xtail_star)), not the
    # raw u_trunc parameter -- when u_trunc doesn't bind (xtail_star is
    # already above it), the real excluded region is xtail_star-wide, and
    # charging only u_trunc would undercharge TV for the mass actually left
    # out of analysis. xtail (linear) is still a safe, looser-than-tight
    # upper bound on the true quadratic tail probability (see hrua_xtail's
    # docstring) for the xtail this small, so this stays conservative.
    tv = u_trunc + 2.0 * reject_const * eps_floor + acceptance_tv(eps_accept)
    return eps_floor, eps_accept, tv


# ---------------------------------------------------------------------------
# HYP FPTaylor template  (not _use_hrua)
# ---------------------------------------------------------------------------

def _make_hyp_template(N, K, n, fp):
    """
    FPTaylor input for (N, K, n) analysing the critical FP operation in
    random_hypergeometric_hyp (distributions/hypergeometric_hyp.c):

        d1 = bad + good - sample  =  (N-K) + K - n  =  N - n
        d2 = min(good, bad)       =  min(K, N-K)

        while (y > 0):
            u  = rk_double()
            y -= floor( u + (double)y / (double)(d1 + k) )
            k--
            if k == 0: break          # k loops over [sample, ..., 1]

      delta : abs error of  rnd64(u + rnd64(y / (d1 + k)))
                        vs  exact  u + y / (d1 + k)

      u in [0, 1),  y in [0, d2],  k in [1, sample]

    For a box, y's and k's ranges are the widest over it and d1 becomes a
    Variable over [N_lo - n_hi (>= 0), N_hi - n_lo].
    """
    rnd = FP_TO_FPTAYLOR_RND[fp]
    if _is_box(N, K, n):
        (N0, N1), (K0, K1), (n0, n1) = iv(N), iv(K), iv(n)
        d2, sample = min(K1, N1 - K0, N1 // 2), n1
        d1_var = ",\n" + interval_ivar("d1", max(0, N0 - n1), N1 - n0, kind="float64")
        d1_def = ""
    else:
        good, bad, sample = K, N - K, n
        d1     = bad + good - sample    # = N - n
        d2     = min(good, bad)         # = min(K, N-K)
        d1_var, d1_def = "", f"  d1 = {float(d1):.1f},\n"

    return (
        "Variables\n"
        f"  real u in [0.0, 1.0],\n"
        f"  real y in [0.0, {float(d2):.1f}],\n"
        f"  real k in [1.0, {float(sample):.1f}]{d1_var};\n\n"
        + "Definitions\n"
        + d1_def
        + f"  div_step {rnd}= y / (d1 + k),\n"
        f"  step     {rnd}= u + div_step;\n\n"
        + "Expressions\n"
        f"  delta = step;\n"
    )


def _compute_hyp_tv(n, delta):
    """tv from the loop's per-step floor-argument error: each of the (at
    most n) steps can flip one floor, reassigning up to 2*delta mass (a box
    passes its n_hi)."""
    return 2 * n * delta


def _run_hyp_fptaylor(fptaylor, N, K, n, args, tag, inputs_dir, outputs_dir, env):
    """(delta, tv) for the HYP regime at (N, K, n) -- points or intervals."""
    label = _NKn_label(N, K, n)
    fp, verbose = args.fp, args.verbose
    (N0, N1), (K0, K1), (n0, n1) = iv(N), iv(K), iv(n)
    vprint(verbose, f"hypergeometric HYP {label}",
           d1=(max(0, N0 - n1), N1 - n0) if _is_box(N, K, n) else N - n,
           d2=min(K1, N1 - K0) if _is_box(N, K, n) else min(K, N - K))

    hyp_input  = inputs_dir  / f"hypergeometric_hyp_{fp}_{tag}.txt"
    hyp_output = outputs_dir / f"hypergeometric_hyp_{fp}_{tag}.out"
    hyp_input.write_text(_make_hyp_template(N, K, n, fp))

    code, output = run_command([fptaylor, str(hyp_input)], cwd=ROOT, env=env)
    hyp_output.write_text(output)
    if verbose >= 2:
        print(f"--- FPTaylor HYP ({label}) ---\n{output}")
    if code != 0:
        raise RuntimeError(f"FPTaylor HYP failed for {label}; see {hyp_output}")

    abs_errors = extract_abs_errors_by_problem(output)
    if "delta" not in abs_errors:
        raise RuntimeError(f"{label}: could not parse absolute error for 'delta'")
    delta = abs_errors["delta"]
    return delta, _compute_hyp_tv(n1, delta)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _use_hrua(N, K, n):
    """
    numpy's regime dispatch (random_hypergeometric):
        (sample >= 10) && (sample <= good + bad - 10)  ->  HRUA
        otherwise                                      ->  HYP
    with good + bad = popsize = N.
    """
    good, bad = K, N - K
    sample = n
    return sample >= _HRUA_SWITCH and sample <= good + bad - _HRUA_SWITCH


def _is_degenerate(N, K, n):
    """Nothing to draw (n = 0) or no variance (K = 0 or K = N): the sampler
    returns a constant, so there is no FP error to analyse."""
    return n == 0 or min(K, N - K) == 0


def _NKn_label(N, K, n):
    if _is_box(N, K, n):
        return box_label({"N": N, "K": K, "n": n})
    return f"N={N} K={K} n={n}"


def safe_triple_name(N, K, n):
    return f"N{N}_K{K}_n{n}"


def _validate(N, K, n, loc=""):
    prefix = f"{loc}: " if loc else ""
    if N <= 0:
        raise ValueError(f"{prefix}N must be positive")
    if not (0 <= K <= N):
        raise ValueError(f"{prefix}K must be in [0, N]")
    if not (0 <= n <= N):
        raise ValueError(f"{prefix}n must be in [0, N]")


def read_NKn_triples(path):
    triples = []
    for lineno, line in enumerate(path.read_text().splitlines(), start=1):
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        tokens = line.split()
        if len(tokens) != 3:
            raise ValueError(f"{path}:{lineno}: expected 'N K n', got {line!r}")
        try:
            N, K, n = int(tokens[0]), int(tokens[1]), int(tokens[2])
        except ValueError as exc:
            raise ValueError(f"{path}:{lineno}: invalid (N, K, n) values") from exc
        _validate(N, K, n, f"{path}:{lineno}")
        triples.append((N, K, n))
    return triples


def _empty_row(N, K, n, fp):
    return {"N": N, "K": K, "n": n, "N_lo": "", "N_hi": "", "K_lo": "",
            "K_hi": "", "n_lo": "", "n_hi": "", "fp": fp, "regime": "",
            "delta": "", "eps_floor": "", "eps_accept": "", "tv": "",
            "n_boxes": "", "time_s": ""}


# ---------------------------------------------------------------------------
# Interval mode  (--N-range / --K-range / --n-range)
# ---------------------------------------------------------------------------

_REGIME_TAGS = {"degenerate": "degenerate", "hyp": "HYP", "hrua": "HRUA"}


def _box_regimes(box):
    """Regimes a box of integer (N, K, n) touches, over its valid points."""
    (N0, N1), (K0, K1), (n0, n1) = box["N"], box["K"], box["n"]
    s = _HRUA_SWITCH
    if n1 == 0 or K1 == 0 or (N0 == N1 and K0 == N0):
        return {"degenerate"}
    hrua = max(n0, s) <= min(n1, N1 - s)
    hyp = n0 < s or n1 > N0 - s
    return ({"hrua"} if hrua else set()) | ({"hyp"} if hyp else set())


def _split_at_switch(box):
    """Split n exactly at _HRUA_SWITCH, or at N - _HRUA_SWITCH when N is a
    point; otherwise bisect every axis."""
    (N0, N1), (n0, n1) = box["N"], box["n"]
    s = _HRUA_SWITCH
    if n0 < s <= n1:
        return [dict(box, n=(n0, s - 1)), dict(box, n=(s, n1))]
    if N0 == N1 and n0 <= N0 - s < n1:
        return [dict(box, n=(n0, N0 - s)), dict(box, n=(N0 - s + 1, n1))]
    return bisect_box(box, integer_axes=("N", "K", "n"))


def run_box(args, fptaylor, inputs_dir, outputs_dir, env, N_iv, K_iv, n_iv):
    """One row bounding TV over every valid (N, K, n) in the box."""
    start = time.perf_counter()
    box = {"N": N_iv, "K": K_iv, "n": n_iv}

    def analyse(sub, regime):
        N, K, n, tag = sub["N"], sub["K"], sub["n"], safe_box_name(sub)
        if regime == "degenerate":
            return {"delta": 0.0, "tv": 0.0}
        if regime == "hyp":
            delta, tv = _run_hyp_fptaylor(fptaylor, N, K, n, args, tag,
                                          inputs_dir, outputs_dir, env)
            return {"delta": delta, "tv": tv}
        eps_floor, eps_accept, tv = _run_hrua_fptaylor(
            fptaylor, N, K, n, args, tag, inputs_dir, outputs_dir, env)
        return {"eps_floor": eps_floor, "eps_accept": eps_accept, "tv": tv}

    fields = ("delta", "eps_floor", "eps_accept", "tv")
    results = analyse_param_box(box, _box_regimes, _split_at_switch, analyse,
                                args.split_depth, integer_axes=("N", "K", "n"),
                                verbose=args.verbose)
    worst = max_fields(results, fields)
    regimes = sorted({r["regime"] for r in results})

    row = _empty_row("", "", "", args.fp)
    row.update({"N_lo": N_iv[0], "N_hi": N_iv[1], "K_lo": K_iv[0], "K_hi": K_iv[1],
                "n_lo": n_iv[0], "n_hi": n_iv[1],
                "regime": "+".join(regimes), "n_boxes": len(results),
                "time_s": f"{elapsed_since(start):.6f}"})
    row.update({f: csv_num(worst[f]) for f in fields})
    print(f"{box_label(box)} [{'+'.join(_REGIME_TAGS[r] for r in regimes)}]"
          f" boxes={len(results)} delta={fmt_num(worst['delta'])}"
          f" eps_floor={fmt_num(worst['eps_floor'])}"
          f" eps_accept={fmt_num(worst['eps_accept'])} TV={fmt_num(worst['tv'])}"
          f" time={format_seconds(float(row['time_s']))}")
    return row


# ---------------------------------------------------------------------------
# Distribution interface
# ---------------------------------------------------------------------------

def add_args(parser):
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("input_file", nargs="?", type=Path,
                        help="File with (N K n) triples, one per line")
    source.add_argument("--N", type=int, default=None, dest="N_pop",
                        help="Population size (requires --K and --n)")
    source.add_argument("--N-range", nargs=2, type=int_or_float_str, default=None,
                        metavar=("NMIN", "NMAX"),
                        help="Interval mode: every integer N in [NMIN, NMAX] "
                             "(with --K/--K-range and --n/--n-range)")
    parser.add_argument("--K", type=int, default=None,
                        help="Number of success states in population")
    parser.add_argument("--K-range", nargs=2, type=int_or_float_str, default=None,
                        metavar=("KMIN", "KMAX"), help="Interval mode: K range")
    parser.add_argument("--n", type=int, default=None, dest="n_draw",
                        help="Number of draws")
    parser.add_argument("--n-range", nargs=2, type=int_or_float_str, default=None,
                        dest="n_draw_range", metavar=("NMIN", "NMAX"),
                        help="Interval mode: n (draws) range")


def default_out_dir(args):
    if any(getattr(args, a, None) is not None
           for a in ("N_range", "K_range", "n_draw_range")):
        return ROOT / "hypergeometric_runs_interval"
    lf = getattr(args, "input_file", None)
    if lf is None:
        return ROOT / "hypergeometric_runs"
    return ROOT / f"hypergeometric_runs_{lf.stem}"


def _box_axis(rng, point, name):
    if rng is not None:
        return parse_range(rng, name, lo_min=0, integer=True)
    if point is None:
        raise ValueError(f"interval mode needs {name} or its point value")
    return point, point


def run(args, fptaylor, inputs_dir, outputs_dir, env):
    if any(r is not None for r in (args.N_range, args.K_range, args.n_draw_range)):
        N_iv = _box_axis(args.N_range, args.N_pop, "--N-range")
        K_iv = _box_axis(args.K_range, args.K, "--K-range")
        n_iv = _box_axis(args.n_draw_range, args.n_draw, "--n-range")
        if N_iv[0] <= 0 or K_iv[0] > N_iv[1] or n_iv[0] > N_iv[1]:
            raise ValueError("interval mode: need N > 0 and some K, n <= N in the box")
        return [run_box(args, fptaylor, inputs_dir, outputs_dir, env, N_iv, K_iv, n_iv)]

    if args.N_pop is not None:
        if args.K is None or args.n_draw is None:
            raise ValueError("--K and --n are required when --N is given")
        _validate(args.N_pop, args.K, args.n_draw)
        triples = [(args.N_pop, args.K, args.n_draw)]
    else:
        triples = read_NKn_triples(args.input_file)
    if not triples:
        raise ValueError("no (N, K, n) triples found in input")

    rows = []
    for N, K, n in triples:
        start = time.perf_counter()
        tag = safe_triple_name(N, K, n)
        try:
            row = _empty_row(N, K, n, args.fp)

            # ---- degenerate: constant output, nothing to analyse ----
            if _is_degenerate(N, K, n):
                row.update({
                    "regime": "degenerate",
                    "delta": f"{0.0:.17e}",
                    "tv": f"{0.0:.17e}",
                    "time_s": f"{elapsed_since(start):.6f}",
                })
                rows.append(row)
                print(f"N={N} K={K} n={n} [degenerate] TV={0.0:.6e}"
                      f" time={format_seconds(float(row['time_s']))}")
                continue

            # ---- low range (HYP) ----
            if not _use_hrua(N, K, n):
                delta, tv = _run_hyp_fptaylor(
                    fptaylor, N, K, n, args, tag, inputs_dir, outputs_dir, env)

                row.update({
                    "regime": "hyp",
                    "delta": f"{delta:.17e}",
                    "tv": f"{tv:.17e}",
                    "time_s": f"{elapsed_since(start):.6f}",
                })
                rows.append(row)
                print(f"N={N} K={K} n={n} [HYP] delta={delta:.6e} TV={tv:.6e}"
                      f" time={format_seconds(float(row['time_s']))}")
                continue

            # ---- high range (HRUA) ----
            eps_floor, eps_accept, tv = _run_hrua_fptaylor(
                fptaylor, N, K, n, args, tag, inputs_dir, outputs_dir, env)

            row.update({
                "regime": "hrua",
                "eps_floor": f"{eps_floor:.17e}",
                "eps_accept": f"{eps_accept:.17e}",
                "tv": f"{tv:.17e}",
                "time_s": f"{elapsed_since(start):.6f}",
            })
            rows.append(row)
            print(f"N={N} K={K} n={n} [HRUA] eps_floor={eps_floor:.6e}"
                  f" eps_accept={eps_accept:.6e} TV={tv:.6e}"
                  f" time={format_seconds(float(row['time_s']))}")
        except Exception as exc:
            print(f"WARNING: skipping N={N} K={K} n={n}: {exc}")

    return rows


def write_plot(rows, plot_path, plot_components=False, plot_pgf=False):
    rows = [r for r in rows if r["n"] not in (None, "")]   # interval rows aren't points
    if not rows:
        print("Nothing to plot: interval-mode rows are not points on the n axis")
        return False
    points = sorted((int(r["n"]), float(r["tv"])) for r in rows)
    xs = [pt[0] for pt in points]
    series = [("TV", [pt[1] for pt in points], "^")]
    save_loglog_plot(xs, series, xlabel="n  (draws)", ylabel="error",
                     plot_path=plot_path, plot_pgf=plot_pgf)
