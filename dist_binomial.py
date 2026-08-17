"""
Binomial (legacy inversion) sampler FP-error analysis.
Follows the pattern in dist_geometric.py; called by main.py.
"""
import math
import sys
from pathlib import Path

import interval_error as ie
from dist_common import (
    ROOT, FP_TO_FPTAYLOR_RND,
    run_command, extract_deltas_by_problem, extract_abs_errors_by_problem,
    run_cire_llvm, extract_cire_abs_error,
    loggam_defs, loggam_ie, eps_logv, eps_logus, vprint,
    us_root, hormann_u_at, hormann_k_defs,
)

NAME = "binomial"
CSV_FIELDS = ["n", "p", "n_lo", "n_hi", "p_lo", "p_hi", "regime",
              "eps0", "eps1", "eps2", "eps_floor", "eps_accept", "tv"]

_BTRS_SWITCH = 30.0   # n*p threshold: inversion below, BTRS above


# ---------------------------------------------------------------------------
# FPTaylor template
# ---------------------------------------------------------------------------

def inversion_params(n, p):
    """(qn, z_lo, x_hi): the interval bounds the inversion templates use."""
    q = 1.0 - p
    qn_raw = math.exp(n * math.log(q))
    qn = max(qn_raw, sys.float_info.min)
    z_lo = max(min(qn_raw, math.exp(-22) / math.sqrt(2 * math.pi * n * p * q)),
               sys.float_info.min)
    x_hi = min(float(n), n * p + 10.0 * math.sqrt(n * p * q))
    return qn, z_lo, x_hi


def make_template(n, p, fp):
    """
    Single FPTaylor input for (n, p) with three expressions, one per
    elementary FP operation in legacy_random_binomial_inversion's sampling
    loop (distributions/binomial_legacy_inversion.c):

        qn = exp(n * log(q))                       (initial term, q = 1 - p)
        px = ((n - X + 1) * p * px) / (X * q)       i.e. px = z * (n-X+1)*p / (X*q),  z in (1e-6, 1)
        U -= px                <=>  sum += prod     sum in [qn, 1], prod in [0, 1]

      eps0 : rel. error of qn = exp(n * log(q))
      eps1 : rel. error of px = z * (n - X + 1) * p / (X * q)
      eps2 : rel. error of sum + prod
    """
    qn, z_lo, x_hi = inversion_params(n, p)
    rnd = FP_TO_FPTAYLOR_RND[fp]

    return (
        "Variables\n"
        f"  real z in [{z_lo:.20e}, 1.0],\n"
        f"  real X in [1.0, {x_hi:.1f}],\n"
        f"  real sum in [{qn:.20e}, 1.0],\n"
        f"  real prod in [0.0, 1.0];\n\n"
        + "Definitions\n"
        f"  n = {float(n):.1f},\n"
        f"  p = {p:.20e},\n"
        f"  q = 1.0 - p,\n"
        f"  qn_step  {rnd}= exp(n * log(q)),\n"
        f"  px_step  {rnd}= z * (n - X + 1) * p / (X * q),\n"
        f"  sum_step {rnd}= sum + prod;\n\n"
        + "Expressions\n"
        f"  eps0 = qn_step;\n"
        f"  eps1 = px_step;\n"
        f"  eps2 = sum_step;\n"
    )


# ---------------------------------------------------------------------------
# BTRS FPTaylor template  (n*p >= _BTRS_SWITCH)
# ---------------------------------------------------------------------------

def btrs_consts(n, p):
    """(spq, a, b, c): the setup constants btrs.c computes once per (n, p)."""
    spq = math.sqrt(n * p * (1.0 - p))
    b   = 1.15 + 2.53 * spq
    a   = -0.0873 + 0.0248 * b + 0.01 * p
    return spq, a, b, n * p + 0.5


def btrs_u_range(n, p, slack=1.0):
    """
    (u_lo, u_hi): the u values that can reach the acceptance test.

    btrs.c draws u in [-0.5, 0.5) and rejects outright unless
        k = floor(y(u)),   y(u) = (2*a/us + b)*u + c,   us = 0.5 - |u|
    lands in [0, n]                                  [btrs.c lines 58-64].
    Real and FP evaluation of y differ by at most eps_floor << 1, so floor
    can disagree by at most one integer: it is enough to enclose the u with
    k in [-1, n+1], i.e. y in [-1, n+2).  Outside that window both samplers
    reject, so those u contribute nothing to the total variation.

    Writing t = us, y = +-(a/t - 2*a + 0.5*b - b*t) + c, so each boundary is
    the positive root of  b*t^2 + beta*t - a = 0  with

        beta = 2*a - 0.5*b + (n + 1 + slack - c)     u > 0, y = n + 1 + slack
        beta = 2*a - 0.5*b + (c + slack)             u < 0, y = -slack

    In the BTRS regime a, b > 0, so y is monotone in t on each side of 0
    (dy/dt = -+(a/t^2 + b)) and each quadratic has exactly one positive root
    (see dist_common.us_root).
    """
    _, a, b, c = btrs_consts(n, p)
    if a <= 0.0:
        raise ValueError(f"BTRS shape constant a = {a:.6g} <= 0 "
                         f"(n*p*q = {n * p * (1.0 - p):.6g} too small); "
                         "the reachable u range is not a single interval")

    return (-(0.5 - us_root(a, b, c + slack)),
            0.5 - us_root(a, b, n + 1.0 + slack - c))


def btrs_u_at(n, p, y, consts=None):
    """The u with (2*a/us + b)*u + c = y (see dist_common.hormann_u_at)."""
    _, a, b, c = consts or btrs_consts(n, p)
    return hormann_u_at(a, b, c, y)


# k = floor(y) is modelled as y - f with f in [0, 1], so k > y_lo - 1: taking
# y_lo = 7 keeps k + 1 > 7, and y_hi = n - 6 keeps n - k + 1 >= 7, which is the
# domain of random_loggam's Stirling branch.  _K_SIGMAS additionally trims to
# the bulk of the distribution; everything outside is charged as k_tail.
#
# y_lo = 7 is the exact limit of that branch, not a margin: x >= 7 is what
# random_loggam requires [random_poisson_ptrs.c:44], and k + 1 > y_lo - 1 + 1
# = y_lo meets it with equality.  btrs_box_params already cuts at k + 1 >= 7;
# this makes the point path agree instead of being a k tighter.
_K_STIRLING_LO = 7.0
_K_STIRLING_HI_MARGIN = 6.0
_K_SIGMAS = 10.0

# v is rk_double(), i.e. a multiple of 2^-53 in [0, 1), so v is either 0 or at
# least 2^-53 -- the accept test never sees a v strictly between.  That makes
# 2^-53 the exact floor of the reachable domain of log(v), and the charged
# mass the quantisation of the v <= threshold test, one grid step.  The old
# 1e-10 was an arbitrary cut charging 10^6 times more than the sampler can
# actually lose here.
_VTAIL = 2.0 ** -53


def btrs_accept_k_range(n, p):
    """
    k window the accept query covers: the Stirling domain intersected with
    the +-10 sigma bulk.  The sigma trim is what the old template did
    implicitly; here it is charged for (see btrs_k_tail_prob).
    """
    spq = math.sqrt(n * p * (1.0 - p))
    k_lo = max(_K_STIRLING_LO, math.ceil(n * p - _K_SIGMAS * spq))
    k_hi = min(n - _K_STIRLING_HI_MARGIN, math.floor(n * p + _K_SIGMAS * spq))
    if k_lo >= k_hi:
        raise ValueError(
            f"empty accept window: k in [{k_lo:.6g}, {k_hi:.6g}] "
            f"(n*p = {n * p:.6g}); the accept analysis does not apply")
    return float(k_lo), float(k_hi)


def btrs_accept_u_range(n, p):
    """The u that map into btrs_accept_k_range."""
    consts = btrs_consts(n, p)
    k_lo, k_hi = btrs_accept_k_range(n, p)
    return btrs_u_at(n, p, k_lo, consts), btrs_u_at(n, p, k_hi, consts)


def btrs_k_tail_prob(n, p):
    """
    P(K outside the accept window), i.e. the output mass that query does not
    cover.  Draws proposing such a k are almost always rejected, so what
    matters is the *binomial* tail, not the (much larger) uniform measure of
    the excluded u.

    The lower tail is summed exactly: k_lo is pinned to the Stirling domain
    (_K_STIRLING_LO), so it is O(1) terms regardless of n, and the Chernoff
    bound is ~20x loose there -- 8.5e-6 vs 4.0e-7 at n=1000, p=0.03, which
    was the dominant term of the whole TV bound.

    The upper tail stays on the Chernoff-Hoeffding bound: it sits ~10 sigma
    out where the bound is already ~1e-15 and summing it would cost O(n)
    terms.  P(K >= k) <= exp(-n*D(k/n || p)) with D the Bernoulli KL
    divergence -- much tighter than Bernstein this far into the tail
    (8e-34 vs 1e-15 at n=1000, p=0.1).
    """
    k_lo, k_hi = btrs_accept_k_range(n, p)
    if not (k_lo < n * p < k_hi):
        raise ValueError(
            f"accept window [{k_lo:.6g}, {k_hi:.6g}] does not contain the "
            f"mode n*p = {n * p:.6g}; the accept analysis does not apply")

    def kl(a):
        d = a * math.log(a / p) if a > 0.0 else 0.0
        return d + ((1.0 - a) * math.log((1.0 - a) / (1.0 - p)) if a < 1.0 else 0.0)

    lower = binom_cdf_below(n, p, k_lo)
    upper = math.exp(-n * kl(k_hi / n))
    return lower + upper


def binom_cdf_below(n, p, k_lo):
    """
    P(K < k_lo) for K ~ Bin(n, p), summed term by term in log space.

    Only called with k_lo pinned to the Stirling domain, so the sum is a
    handful of terms.  The result is nudged up by a few ulp so it stays an
    upper bound on the exact tail despite the lgamma/exp rounding -- it is
    charged as probability mass, so erring high is the safe direction.
    """
    m = int(math.ceil(k_lo)) - 1          # largest integer k with k < k_lo
    if m < 0:
        return 0.0
    m = min(m, n)
    logp, logq = math.log(p), math.log1p(-p)
    lgn = math.lgamma(n + 1.0)
    total = math.fsum(
        math.exp(lgn - math.lgamma(k + 1.0) - math.lgamma(n - k + 1.0)
                 + k * logp + (n - k) * logq)
        for k in range(m + 1)
    )
    return total * (1.0 + 1e-12)


# ---------------------------------------------------------------------------
# Tail bound on the accept expression (interval arithmetic, not FPTaylor)
# ---------------------------------------------------------------------------
#
# The FPTaylor accept query covers k in the accept window only; the mass
# outside it used to be charged as a bare k_tail, which is a probability under
# the *ideal* binomial.  That is circular: bounding TV(ideal, fp) by a term
# that is itself only a bound on the fp sampler's tail up to TV assumes what
# is being proved.
#
# What breaks the circularity is that the accept test is a comparison in log
# space, log(v) <= A.  A sound absolute bound |A_fp - A| <= E therefore gives a
# *multiplicative* bound on the per-k acceptance probability, and after the
# rejection loop renormalises, a pointwise bound on the output law:
#
#     P_fp(K = k)  <=  exp(2E) * P_ideal(K = k)      for every k
#
# so  P_fp[K outside window] <= exp(2E) * k_tail.  E comes from FP analysis and
# k_tail from the binomial; neither depends on TV.
#
# E only has to keep exp(2E) an O(1) factor on a term of size ~1e-15, so the
# bound may be loose -- 0.1 is ample, 1.0 is survivable.  That is why interval
# arithmetic is the right tool: on this box (us down to us_min, k over the
# whole out-of-window range) FPTaylor's branch-and-bound is slow and returns
# 100%-suboptimal results, while interval propagation is cheap and cannot fail
# to terminate.

def _loggam_at(lo, hi, shift=0):
    """random_loggam on an exact-integer argument enclosure [lo, hi]."""
    if lo == hi and (lo == 1.0 or lo == 2.0):
        return ie.const(0.0)               # early return [random_poisson_ptrs.c:41]
    return loggam_ie(ie.var(lo, hi), shift=shift)


def _btrs_accept_ie(box, n_iv, p_iv, m_iv, fast, shift_k, shift_nk,
                    nk1_iv=None):
    """
    btrs.c's acceptance expression [btrs.c:88-90] in (enclosure, error)
    arithmetic.  `box` is ((k_lo, k_hi), (us_lo, us_hi)) -- the two dimensions
    bisect_max_err subdivides.

    k, m and n enter with err = 0 and so do the loggam arguments: k and m are
    int64_t casts and n - k + 1 / n - m + 1 are computed in integer arithmetic
    before the cast, so every one of them is an exactly representable double.
    us is exact too (Sterbenz on 0.5 - |u|, with u on the RNG grid).  The error
    being bounded is the one the accept arithmetic itself commits, given those
    inputs -- the error in *choosing* k is eps_floor's job, not this one.

    k and us are treated as independent here.  The reachable set is a curve in
    that rectangle, so the rectangle is a sound over-approximation, and it
    avoids reproducing hormann_k_defs' coupling for a bound that does not need
    the tightness.

    `nk1_iv` overrides the enclosure of n - k + 1.  It is needed wherever the
    caller enumerates by *offset* from n rather than by k: over a box in n,
    deriving n - k + 1 from independent n and k enclosures widens it to
    [n_lo - k_hi + 1, n_hi - k_lo + 1], which straddles zero as soon as the box
    is wider than the offset, and loggam's 1/x0 then has no sign.  The offset
    pins the true value -- n - k + 1 = j + 1 for every n -- so the caller
    passes it directly.
    """
    (k_lo, k_hi), (us_lo, us_hi) = box
    (n_lo, n_hi), (m_lo, m_hi) = n_iv, m_iv

    n, p = ie.var(*n_iv), ie.var(*p_iv)
    k, m, us = ie.var(k_lo, k_hi), ie.var(m_lo, m_hi), ie.var(us_lo, us_hi)
    one = ie.const(1.0)

    q   = ie.sub(one, p)
    spq = ie.isqrt(ie.mul(ie.mul(n, p), q))
    b   = ie.add(ie.const(1.15), ie.mul(ie.const(2.53), spq))
    a   = ie.add(ie.add(ie.const(-0.0873), ie.mul(ie.const(0.0248), b)),
                 ie.mul(ie.const(0.01), p))
    alpha = ie.mul(ie.add(ie.const(2.83), ie.div(ie.const(5.1), b)), spq)
    lpq   = ie.ilog(ie.div(p, q))

    # exact integer arguments, formed before any float op touches them
    h = ie.add(_loggam_at(m_lo + 1.0, m_hi + 1.0),
               _loggam_at(n_lo - m_hi + 1.0, n_hi - m_lo + 1.0))
    lgk  = _loggam_at(k_lo + 1.0, k_hi + 1.0, shift_k)
    nk1_lo, nk1_hi = nk1_iv if nk1_iv is not None else (n_lo - k_hi + 1.0,
                                                        n_hi - k_lo + 1.0)
    lgnk = _loggam_at(nk1_lo, nk1_hi, shift_nk)

    log_num = ie.add(a, ie.mul(b, ie.mul(us, us)))

    acc = ie.sub(ie.sub(h, lgk), lgnk)
    acc = ie.add(acc, ie.mul(ie.sub(k, m), lpq))
    acc = ie.sub(acc, ie.ilog(alpha))
    acc = ie.add(acc, ie.ilog(log_num))
    if not fast:
        acc = ie.sub(acc, ie.mul(ie.const(2.0), ie.ilog(us)))
    return acc


def btrs_accept_tail_eps(n_iv, p_iv, m_iv, k_win, us_min,
                         fast=False, target=0.05, max_splits=1024, verbose=0):
    """
    Sound bound E on the absolute error of the accept expression over every k
    the bulk FPTaylor query does not cover, so that k_tail can be charged as
    exp(2E) * k_tail instead of bare k_tail.

    Covers three regions, which differ only in which random_loggam branch runs:

      - Stirling tail:  k in [6, k_lo] and [k_hi, n_lo - 6], where both
        k + 1 >= 7 and n - k + 1 >= 7, so shift = 0 as in loggam_defs.
      - small k:        k = 0..5, where k + 1 < 7 triggers the argument
        reduction.  Six exact values, enumerated.
      - k near n:       n - k = 0..5, same branch on the other loggam.
        Enumerated by offset so it works when n is a box.

    Returns (E, n_boxes).  E is a sound upper bound whether or not `target`
    was reached; the split budget only controls how loose it is.
    """
    n_lo, n_hi = n_iv
    ak_lo, ak_hi = k_win
    us_iv = (us_min, 0.5)

    def run(k_iv, shift_k=0, shift_nk=0, nk1_iv=None):
        f = lambda bx: _btrs_accept_ie(bx, n_iv, p_iv, m_iv, fast,
                                       shift_k, shift_nk, nk1_iv)
        return ie.bisect_max_err(f, [(k_iv, us_iv)], target, max_splits)

    worst, boxes = 0.0, 0
    regions = []

    # Stirling-branch tail, both sides of the window.  k >= 6 is what makes
    # k + 1 >= 7; k <= n_lo - 6 does the same for n - k + 1 across the box.
    if ak_lo > 6.0:
        regions.append(("low", (6.0, ak_lo), 0, 0, None))
    if ak_hi < n_lo - 6.0:
        regions.append(("high", (ak_hi, n_lo - 6.0), 0, 0, None))
    # Argument-reduction branch: k + 1 in 1..6, i.e. k in 0..5.
    for i in range(6):
        regions.append((f"k={i}", (float(i), float(i)), 6 - i, 0, None))
    # ... and the mirror end, n - k + 1 in 1..6.  Enumerated by offset j = n - k
    # so that n - k + 1 stays pinned at j + 1 over a box in n; see the nk1_iv
    # note in _btrs_accept_ie.
    for j in range(6):
        regions.append((f"n-k={j}", (n_lo - j, n_hi - j), 0, 6 - j,
                        (j + 1.0, j + 1.0)))

    for tag, k_iv, sk, snk, nk1 in regions:
        if k_iv[0] > k_iv[1]:
            continue
        e, nb = run(k_iv, sk, snk, nk1)
        boxes += nb
        if verbose >= 2:
            print(f"    tail {tag:>8}: k in [{k_iv[0]:.6g}, {k_iv[1]:.6g}] "
                  f"E={e:.4g} ({nb} boxes)")
        worst = max(worst, e)
    return worst, boxes


def btrs_setup_defs(rnd, n_expr, p_expr, accept=False):
    """
    btrs.c's setup block [lines 48-51, 72-76] as FPTaylor Definitions.

    These are *derived* from n and p, not free inputs, so they have to be
    written as expressions: it keeps them correlated with each other and with
    u, and it charges the rounding of the setup arithmetic itself -- what
    dist_poisson_stable.py has to add by hand as _const_err.

    n_expr / p_expr are literals in point mode and variable names in box mode.
    """
    d = [f"  spq_   {rnd}= sqrt({n_expr} * {p_expr} * (1.0 - {p_expr})),",
         f"  b_     {rnd}= 1.15 + 2.53 * spq_,",
         f"  a_     {rnd}= -0.0873 + 0.0248 * b_ + 0.01 * {p_expr},",
         f"  c_     {rnd}= {n_expr} * {p_expr} + 0.5,"]
    if accept:
        d += [f"  alpha_ {rnd}= (2.83 + 5.1 / b_) * spq_,",
              f"  lpq_   {rnd}= log({p_expr} / (1.0 - {p_expr})),"]
    return d


def btrs_floor_defs(rnd):
    """us and the pre-floor value y [btrs.c lines 60-61], as written in C."""
    return [f"  us_    {rnd}= 0.5 - abs(u),",
            f"  y_     {rnd}= (2.0 * a_ / us_ + b_) * u + c_,"]


def btrs_k_defs(sign, n_expr, p_expr):
    """k = floor(y) for one sign of u (see dist_common.hormann_k_defs)."""
    return hormann_k_defs(
        sign,
        [f"  spqx_  = sqrt({n_expr} * {p_expr} * (1.0 - {p_expr})),",
         f"  bx_    = 1.15 + 2.53 * spqx_,",
         f"  ax_    = -0.0873 + 0.0248 * bx_ + 0.01 * {p_expr},",
         f"  cx_    = {n_expr} * {p_expr} + 0.5,"],
        "cx_")


def make_btrs_floor_template(n, p, fp, u_lo, u_hi):
    """
    FPTaylor expression for eps_floor: absolute error of
    (2*a/us + b)*u + c, with us = 0.5 - |u|          [btrs.c line 61]

    Only u is free; a, b, c are expressions in the literals n and p.
    """
    rnd = FP_TO_FPTAYLOR_RND[fp]
    return (
        "Variables\n"
        f"  real u in [{u_lo:.20e}, {u_hi:.20e}];\n\n"
        + "Definitions\n"
        + "\n".join(btrs_setup_defs(rnd, f"{float(n):.1f}", f"{p:.20e}")) + "\n"
        + "\n".join(btrs_floor_defs(rnd)).rstrip(",") + ";\n\n"
        + "Expressions\n"
          "  eps_floor = y_;\n"
    )


def make_btrs_accept_template(n, p, fp, u_lo, u_hi, sign, fast=False):
    """
    FPTaylor expression for eps_accept (excluding -log(v), see
    make_logv_template): absolute error of
    h - loggam(k+1) - loggam(n-k+1) + (k-m)*lpq
      - log(alpha) + log(a/us^2 + b),  with us = 0.5 - |u|   [btrs.c line 85]
    lgamma is approximated by inlining random_loggam's x>=7 Stirling branch.

    The free inputs are u and f only (see btrs_k_defs); m and h are
    integers/constants btrs.c derives once from n and p, so at a literal
    (n, p) they stay literals -- only the addition forming h is charged.
    sign selects which side of u = 0 this query covers, since k's definition
    needs it; the caller takes the max over the two.

    If fast is True, the -2*log(us_) term is omitted here and its error is
    computed separately (see make_btrs_logus_template) and summed in by the
    caller. This drops u as a shared variable between the two terms, which
    may yield a more conservative (looser) overall bound.
    """
    m   = int(math.floor((n + 1) * p))
    rnd = FP_TO_FPTAYLOR_RND[fp]

    defs_k,  name_k  = loggam_defs("k1_",  "lgk",  rnd)
    defs_nk, name_nk = loggam_defs("nk1_", "lgnk", rnd)
    log_us_term = "" if fast else " - 2.0 * log(us_)"

    return (
        "Variables\n"
        f"  real u in [{u_lo:.20e}, {u_hi:.20e}],\n"
        f"  real f in [0.0, 1.0];\n\n"
        + "Definitions\n"
        + "\n".join(btrs_setup_defs(rnd, f"{float(n):.1f}", f"{p:.20e}",
                                    accept=True)) + "\n"
        + f"  m_     = {float(m):.1f},\n"
        + f"  h_     {rnd}= {math.lgamma(m + 1):.20e}"
          f" + {math.lgamma(n - m + 1):.20e},\n"
        + f"  us_    {rnd}= 0.5 - abs(u),\n"
        + "\n".join(btrs_k_defs(sign, f"{float(n):.1f}", f"{p:.20e}")) + "\n"
        + f"  nk1_   = {float(n):.1f} - k_ + 1.0,\n"
        + "\n".join(defs_k)  + "\n"
        + "\n".join(defs_nk) + "\n"
        + f"  us_sq_      {rnd}= us_ * us_,\n"
        + f"  log_num_    {rnd}= a_ + b_ * us_sq_,\n"
        + f"  btrs_accept {rnd}= h_ - {name_k} - {name_nk}"
          f" + (k_ - m_) * lpq_ - log(alpha_) + log(log_num_){log_us_term};\n\n"
        + "Expressions\n"
          "  eps_accept = btrs_accept;\n"
    )


# ---------------------------------------------------------------------------
# Interval-parameter mode:  one query for a whole (n, p) box
# ---------------------------------------------------------------------------
#
# The point templates bake every setup constant in as a literal.  Here each
# one is promoted to an FPTaylor *variable* ranging over its enclosure across
# the box, so a single query bounds the error for every (n, p) in it.
#
# FPTaylor then maximises over the product of those intervals, which is a
# superset of the realizable (spq, a, b, c, ...) tuples -- sound, and only
# mildly loose: the error is driven by term magnitudes, and those peak at the
# same corner either way.  What it costs is correlation: (k, n-k) and (m, n)
# are no longer tied, which is why the box has to satisfy the validity
# constraints below (all arguments of the inlined loggam stay >= 7).

def btrs_box_params(n_lo, n_hi, p_lo, p_hi, slack=1.0):
    """
    Enclosure of every BTRS quantity over [n_lo, n_hi] x [p_lo, p_hi],
    as a dict of (lo, hi) pairs.  Each is obtained from the monotonicity of
    that quantity in n and p:

      npq   n up; p up to 1/2 then down     -> peak at p = 1/2 when in range
      b     up in spq                       alpha  up in spq (spq/b is up in spq)
      a     up in b and p                   lpq    up in p
      c     up in n and p                   m      up in n and p
      h     up in m and in n - m            (lgamma is up on [7, inf))
      us    up in a and b, down in gamma    (see dist_common.us_root)
    """
    q_lo, q_hi = 1.0 - p_hi, 1.0 - p_lo
    pq_ends = (p_lo * (1.0 - p_lo), p_hi * (1.0 - p_hi))
    pq_lo   = min(pq_ends)
    pq_hi   = 0.25 if p_lo <= 0.5 <= p_hi else max(pq_ends)

    spq_lo, spq_hi = math.sqrt(n_lo * pq_lo), math.sqrt(n_hi * pq_hi)
    b_lo,   b_hi   = 1.15 + 2.53 * spq_lo, 1.15 + 2.53 * spq_hi
    a_lo,   a_hi   = (-0.0873 + 0.0248 * b_lo + 0.01 * p_lo,
                      -0.0873 + 0.0248 * b_hi + 0.01 * p_hi)
    c_lo,   c_hi   = n_lo * p_lo + 0.5, n_hi * p_hi + 0.5
    alpha_lo       = (2.83 + 5.1 / b_lo) * spq_lo
    alpha_hi       = (2.83 + 5.1 / b_hi) * spq_hi
    lpq_lo, lpq_hi = math.log(p_lo / q_hi), math.log(p_hi / q_lo)

    m_lo = float(math.floor((n_lo + 1.0) * p_lo))
    m_hi = float(math.floor((n_hi + 1.0) * p_hi))
    k_lo = max(6.0, float(math.floor(m_lo - 10.0 * spq_hi)))
    k_hi = min(n_hi - 1.0, float(math.ceil(m_hi + 10.0 * spq_hi)))
    j_lo, j_hi = n_lo - k_hi, n_hi - k_lo          # j = n - k

    if a_lo <= 0.0:
        raise ValueError(f"BTRS shape constant a = {a_lo:.6g} <= 0 at the low "
                         f"corner (n*p*q = {n_lo * pq_lo:.6g} too small)")
    if k_hi <= k_lo:
        raise ValueError(f"empty k range [{k_lo:.0f}, {k_hi:.0f}] over the box")
    for name, lo in (("n - k", j_lo), ("m", m_lo), ("n - m", n_lo - m_hi)):
        if lo < 6.0:
            raise ValueError(
                f"box too wide: {name} reaches {lo:.6g} < 6, so the inlined "
                "Stirling branch of loggam (valid for x >= 7) does not cover "
                "it; narrow the n range or split the box")

    # u: t is smallest at (a_lo, b_lo, gamma_max), which widens the enclosure
    gam_hi = n_hi * (1.0 - p_lo) + 0.5 + slack     # max of n + 1 + slack - c
    gam_lo = n_hi * p_hi + 0.5 + slack             # max of c + slack
    us_pos, us_neg = us_root(a_lo, b_lo, gam_hi), us_root(a_lo, b_lo, gam_lo)

    # the accept query needs one k window valid across the box, and the u
    # window that maps into it for *every* (n, p) in the box -- so the widest
    # k_lo and narrowest k_hi, then the innermost u on each side (t largest,
    # i.e. a_hi, b_hi, gamma smallest)
    # unlike a point, a box takes the *whole* Stirling domain rather than a
    # +-10 sigma trim of it: one shared window has to serve every (n, p), and
    # intersecting the per-point sigma windows leaves so little that the
    # uncovered mass dominates (k_tail = 2e-4 instead of 1e-18 at
    # n in [900, 1100], p in [0.09, 0.11])
    ak_lo = _K_STIRLING_LO
    ak_hi = n_lo - _K_STIRLING_HI_MARGIN
    if not (ak_lo < n_lo * p_lo and n_hi * p_hi < ak_hi):
        raise ValueError(
            f"accept window k in [{ak_lo:.6g}, {ak_hi:.6g}] does not contain "
            f"n*p over [{n_lo * p_lo:.6g}, {n_hi * p_hi:.6g}]; narrow the box")
    # The window has to satisfy FPTaylor's *interval* bound on k, not just the
    # true k: with n a variable, IA evaluates n - k with the two occurrences of
    # n independent, and the slop from the mixed corners (a_hi/us against
    # -2*a_lo, and c_hi against n_lo) is enough to push n - k + 1 through zero
    # and trip the division-by-zero check.  Bisect on us for the largest
    # window whose IA bounds still land inside the loggam domain.
    def ia_k(t, upper):
        """IA bound on k = s*(a/us - 2a + 0.5b - b*us) + c - f at us = t."""
        if upper:
            return a_hi / t + 0.5 * b_hi - 2.0 * a_lo - b_lo * t + c_hi
        return -(a_hi / t + 0.5 * b_hi - 2.0 * a_lo - b_lo * t) + c_lo - 1.0

    def widest_us(upper, ok):
        lo, hi = 1e-12, 0.5           # ok(hi) holds: us = 0.5 is u = 0
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            if ok(ia_k(mid, upper)):
                hi = mid
            else:
                lo = mid
        return hi

    us_a_hi = widest_us(True,  lambda k: n_lo - k + 1.0 >= 7.0)
    us_a_lo = widest_us(False, lambda k: k + 1.0 >= 7.0)
    au_lo = -(0.5 - us_a_lo)
    au_hi = 0.5 - us_a_hi
    ak_lo = max(ak_lo, math.ceil(ia_k(us_a_lo, False)))
    ak_hi = min(ak_hi, math.floor(ia_k(us_a_hi, True)))

    # n*D(k/n || p) is increasing in n and, below the mean, in p -- so the
    # weakest lower-tail bound over the box is at (n_lo, p_lo) and the
    # weakest upper-tail one at (n_hi, p_hi)
    def kl(a, p):
        d = a * math.log(a / p) if a > 0.0 else 0.0
        return d + ((1.0 - a) * math.log((1.0 - a) / (1.0 - p)) if a < 1.0 else 0.0)

    k_tail = (math.exp(-n_lo * kl(ak_lo / n_lo, p_lo))
              + math.exp(-n_hi * kl(ak_hi / n_hi, p_hi)))

    return {
        "n": (n_lo, n_hi), "p": (p_lo, p_hi),
        "spq": (spq_lo, spq_hi), "a": (a_lo, a_hi), "b": (b_lo, b_hi),
        "c": (c_lo, c_hi), "alpha": (alpha_lo, alpha_hi),
        "lpq": (lpq_lo, lpq_hi), "m": (m_lo, m_hi),
        "h": (math.lgamma(m_lo + 1.0) + math.lgamma(n_lo - m_hi + 1.0),
              math.lgamma(m_hi + 1.0) + math.lgamma(n_hi - m_lo + 1.0)),
        "k": (k_lo, k_hi), "j": (j_lo, j_hi),
        "u": (-(0.5 - us_neg), 0.5 - us_pos),
        "us_min": (min(us_pos, us_neg), max(us_pos, us_neg)),
        "accept_k": (float(ak_lo), float(ak_hi)),
        "accept_u": (au_lo, au_hi),
        "k_tail": (k_tail, k_tail),
    }


def _var(name, box, key, last=False):
    lo, hi = box[key]
    return f"  real {name} in [{lo:.20e}, {hi:.20e}]{';' if last else ','}\n"


def make_btrs_floor_box_template(box, fp):
    """
    Interval-parameter form of make_btrs_floor_template: u, n and p are the
    only free inputs, and a, b, c stay expressions in n and p.
    """
    rnd = FP_TO_FPTAYLOR_RND[fp]
    return (
        "Variables\n"
        + _var("u", box, "u") + _var("n", box, "n")
        + _var("p", box, "p", last=True) + "\n"
        + "Definitions\n"
        + "\n".join(btrs_setup_defs(rnd, "n", "p")) + "\n"
        + "\n".join(btrs_floor_defs(rnd)).rstrip(",") + ";\n\n"
        + "Expressions\n"
          "  eps_floor = y_;\n"
    )


def make_btrs_accept_box_template(box, fp, sign, fast=False):
    """
    Interval-parameter form of make_btrs_accept_template: the free inputs are
    u, f, n, p and fm, where fm carries the floor in m = floor((n+1)*p) the
    same way f carries the one in k.

    h has to be built here rather than embedded, since m is no longer a
    literal; it uses the same inlined Stirling branch as the k terms, which
    is why the box has to keep m and n - m inside that domain.
    """
    rnd = FP_TO_FPTAYLOR_RND[fp]
    defs_k,  name_k  = loggam_defs("k1_",  "lgk",  rnd)
    defs_nk, name_nk = loggam_defs("nk1_", "lgnk", rnd)
    defs_m,  name_m  = loggam_defs("m1_",  "lgm",  rnd)
    defs_nm, name_nm = loggam_defs("nm1_", "lgnm", rnd)
    log_us_term = "" if fast else " - 2.0 * log(us_)"

    return (
        "Variables\n"
        + _var("u", box, "u") + _var("n", box, "n") + _var("p", box, "p")
        + "  real f in [0.0, 1.0],\n"
        + "  real fm in [0.0, 1.0];\n\n"
        + "Definitions\n"
        + "\n".join(btrs_setup_defs(rnd, "n", "p", accept=True)) + "\n"
        + "  m_     = (n + 1.0) * p - fm,\n"
        + "  m1_    = m_ + 1.0,\n"
        + "  nm1_   = n * (1.0 - p) - p + fm + 1.0,\n"   # = n - m_ + 1, one n
        + "\n".join(defs_m)  + "\n"
        + "\n".join(defs_nm) + "\n"
        + f"  h_     {rnd}= {name_m} + {name_nm},\n"
        + f"  us_    {rnd}= 0.5 - abs(u),\n"
        + "\n".join(btrs_k_defs(sign, "n", "p")) + "\n"
        # n - k + 1, but with n*(1 - p) kept as one product: writing it as
        # n - k_ mentions n twice (once directly, once inside c = n*p + 0.5),
        # and interval arithmetic cannot cancel the two, which drags the bound
        # through zero and trips FPTaylor's division-by-zero check on 1/(n-k+1)
        + "  nk1_   = n * (1.0 - p) - ys_ + f + 0.5,\n"
        + "\n".join(defs_k)  + "\n"
        + "\n".join(defs_nk) + "\n"
        + f"  us_sq_      {rnd}= us_ * us_,\n"
        + f"  log_num_    {rnd}= a_ + b_ * us_sq_,\n"
        + f"  btrs_accept {rnd}= h_ - {name_k} - {name_nk}"
          f" + (k_ - m_) * lpq_ - log(alpha_) + log(log_num_){log_us_term};\n\n"
        + "Expressions\n"
          "  eps_accept = btrs_accept;\n"
    )


def make_inversion_box_template(box, fp):
    """Interval-parameter form of make_template (inversion regime)."""
    rnd = FP_TO_FPTAYLOR_RND[fp]
    return (
        "Variables\n"
        f"  real z in [{box['z'][0]:.20e}, 1.0],\n"
        f"  real X in [1.0, {box['X'][1]:.20e}],\n"
        f"  real sum in [{box['qn'][0]:.20e}, 1.0],\n"
        f"  real prod in [0.0, 1.0],\n"
        + _var("n", box, "n") + _var("p", box, "p", last=True) + "\n"
        + "Definitions\n"
        f"  q = 1.0 - p,\n"
        f"  qn_step  {rnd}= exp(n * log(q)),\n"
        f"  px_step  {rnd}= z * (n - X + 1) * p / (X * q),\n"
        f"  sum_step {rnd}= sum + prod;\n\n"
        + "Expressions\n"
        f"  eps0 = qn_step;\n"
        f"  eps1 = px_step;\n"
        f"  eps2 = sum_step;\n"
    )


def inversion_box_params(n_lo, n_hi, p_lo, p_hi):
    """(qn, z, X, bound) enclosures for the inversion templates over a box."""
    tiny = sys.float_info.min
    npq_hi = n_hi * (0.25 if p_lo <= 0.5 <= p_hi
                     else max(p_lo * (1.0 - p_lo), p_hi * (1.0 - p_hi)))
    qn_lo = max(math.exp(n_hi * math.log(1.0 - p_hi)), tiny)
    qn_hi = max(math.exp(n_lo * math.log(1.0 - p_lo)), tiny)
    z_lo = max(min(qn_lo, math.exp(-22) / math.sqrt(2 * math.pi * npq_hi)), tiny)
    x_hi = min(float(n_hi), n_hi * p_hi + 10.0 * math.sqrt(npq_hi))
    return {
        "n": (n_lo, n_hi), "p": (p_lo, p_hi),
        "qn": (qn_lo, qn_hi), "z": (z_lo, 1.0), "X": (1.0, x_hi),
        "bound": (0.0, n_hi * p_hi + 10.0 * math.sqrt(npq_hi)),
    }


def _run_btrs_box(fptaylor, box, fp, tag, inputs_dir, outputs_dir, env,
                  verbose, fast=False):
    """(eps_floor, eps_accept, tv) valid for every (n, p) in the box."""
    vtail = _VTAIL
    us_min = box["us_min"][0]

    vprint(verbose, f"binomial BTRS box {_box_label(box)}",
           **{k: f"[{v[0]:.10g}, {v[1]:.10g}]" for k, v in box.items()},
           vtail=vtail)

    floor_input  = inputs_dir  / f"binomial_btrs_floor_{fp}_{tag}.txt"
    floor_output = outputs_dir / f"binomial_btrs_floor_{fp}_{tag}.out"
    floor_input.write_text(make_btrs_floor_box_template(box, fp))

    code, output = run_command([fptaylor, str(floor_input)], cwd=ROOT, env=env)
    floor_output.write_text(output)
    if verbose >= 2:
        print(f"--- FPTaylor BTRS box floor ---\n{output}")
    if code != 0:
        raise RuntimeError(f"FPTaylor BTRS box floor failed; see {floor_output}")

    eps_floor = 5 * extract_abs_errors_by_problem(output)["eps_floor"]

    au_lo, au_hi = box["accept_u"]
    accept_raw = 0.0
    for sign, lo, hi in ((-1, au_lo, 0.0), (+1, 0.0, au_hi)):
        s_tag = "m" if sign < 0 else "p"
        sbox = dict(box, u=(lo, hi))
        accept_input  = inputs_dir  / f"binomial_btrs_accept_{fp}_{tag}_{s_tag}.txt"
        accept_output = outputs_dir / f"binomial_btrs_accept_{fp}_{tag}_{s_tag}.out"
        accept_input.write_text(
            make_btrs_accept_box_template(sbox, fp, sign, fast=fast))

        code, output = run_command([fptaylor, str(accept_input)], cwd=ROOT, env=env)
        accept_output.write_text(output)
        if verbose >= 2:
            print(f"--- FPTaylor BTRS box accept s={sign:+d} ---\n{output}")
        if code != 0:
            raise RuntimeError(f"FPTaylor BTRS box accept s={sign:+d} failed; "
                               f"see {accept_output}")
        accept_raw = max(accept_raw,
                         extract_abs_errors_by_problem(output)["eps_accept"])

    # exp(2E) converts the ideal-binomial k_tail into a bound on the *fp*
    # sampler's out-of-window mass; see btrs_accept_tail_eps.
    tail_E, tail_boxes = btrs_accept_tail_eps(
        box["n"], box["p"], box["m"], box["accept_k"], us_min,
        fast=fast, verbose=verbose)
    k_tail_fp = math.exp(2.0 * tail_E) * box["k_tail"][0]
    vprint(verbose, "binomial BTRS box tail",
           tail_E=tail_E, tail_boxes=tail_boxes,
           k_tail=box["k_tail"][0], k_tail_fp=k_tail_fp)

    eps_accept = accept_raw + k_tail_fp + eps_logv(
        fptaylor, fp, vtail, inputs_dir, outputs_dir, env, verbose,
    )
    if fast:
        eps_accept += eps_logus(
            fptaylor, fp, us_min, inputs_dir, outputs_dir, env, verbose,
        )

    # accept_iter = alpha / (sqrt(2*pi)*spq) = (2.83 + 5.1/b) / sqrt(2*pi):
    # spq cancels, so the worst case over the box is at b_lo.
    accept_iter = (2.83 + 5.1 / box["b"][0]) / math.sqrt(2 * math.pi)
    tv = 2 * (eps_floor + vtail) * accept_iter + 2 * eps_accept / (1 - vtail)
    return eps_floor, eps_accept, tv


def _run_inversion_box(fptaylor, box, args, tag, inputs_dir, outputs_dir, env):
    """(eps0, eps1, eps2, tv) valid for every (n, p) in the box."""
    vprint(args.verbose, f"binomial inversion box {_box_label(box)}",
           **{k: f"[{v[0]:.10g}, {v[1]:.10g}]" for k, v in box.items()})

    input_path = inputs_dir / f"binomial_inversion_{args.fp}_{tag}.txt"
    input_path.write_text(make_inversion_box_template(box, args.fp))
    code, output = run_command(
        [fptaylor, "--rel-error", "true", str(input_path)], cwd=ROOT, env=env,
    )
    out_path = outputs_dir / f"binomial_inversion_{args.fp}_{tag}.out"
    out_path.write_text(output)
    if args.verbose >= 2:
        print(f"--- FPTaylor binomial_inversion box ---\n{output}")
    if code != 0:
        raise RuntimeError(f"FPTaylor failed for the box; see {out_path}")

    deltas = extract_deltas_by_problem(output, _box_label(box))
    eps0, eps1, eps2 = deltas["eps0"], deltas["eps1"], deltas["eps2"]
    tv = 0.5 * (eps0 + eps1 * box["p"][1] + eps2 * box["bound"][1])
    return eps0, eps1, eps2, tv


def _box_label(box):
    (n_lo, n_hi), (p_lo, p_hi) = box["n"], box["p"]
    return f"n in [{n_lo:.10g}, {n_hi:.10g}] p in [{p_lo:.10g}, {p_hi:.10g}]"


def safe_box_name(n_lo, n_hi, p_lo, p_hi):
    def fmt(v):
        return f"{v:.6g}".replace(".", "p").replace("-", "m").replace("+", "")
    return f"box_n{fmt(n_lo)}_{fmt(n_hi)}_p{fmt(p_lo)}_{fmt(p_hi)}"


def _run_btrs_fptaylor(fptaylor, n, p, fp, tag, inputs_dir, outputs_dir, env, verbose, fast=False):
    """Run FPTaylor for BTRS and return a partial row dict."""
    spq, a, b, c = btrs_consts(n, p)
    alpha = (2.83 + 5.1 / b) * spq
    u_lo, u_hi = btrs_u_range(n, p)
    us_min = min(0.5 - u_hi, 0.5 + u_lo)
    vtail = _VTAIL

    # the accept query only covers the u whose k lands in the accept window;
    # the rest of the output mass is charged as k_tail
    au_lo, au_hi = btrs_accept_u_range(n, p)
    k_lo, k_hi = btrs_accept_k_range(n, p)
    k_tail = btrs_k_tail_prob(n, p)
    vprint(verbose, f"binomial BTRS n={n} p={p}",
           spq=spq, a=a, b=b, c=c, alpha=alpha,
           u_lo=u_lo, u_hi=u_hi, us_min=us_min,
           accept_u_lo=au_lo, accept_u_hi=au_hi,
           k_lo=k_lo, k_hi=k_hi, k_tail=k_tail, vtail=vtail)

    floor_input  = inputs_dir  / f"binomial_btrs_floor_{fp}_{tag}.txt"
    floor_output = outputs_dir / f"binomial_btrs_floor_{fp}_{tag}.out"
    floor_input.write_text(make_btrs_floor_template(n, p, fp, u_lo, u_hi))

    code, output = run_command([fptaylor, str(floor_input)], cwd=ROOT, env=env)
    floor_output.write_text(output)
    if verbose >= 2:
        print(f"--- FPTaylor BTRS floor (n={n}, p={p}) ---\n{output}")
    if code != 0:
        raise RuntimeError(f"FPTaylor BTRS floor failed for n={n}, p={p}; see {floor_output}")

    # No u-tail probability to add: [u_lo, u_hi] is the reachable range, and
    # every u outside it is rejected by both the real and the FP sampler.
    eps_floor = 5 * extract_abs_errors_by_problem(output)["eps_floor"]

    # k's definition needs the sign of u, so the accept query runs once per side
    accept_raw = 0.0
    for sign, lo, hi in ((-1, au_lo, 0.0), (+1, 0.0, au_hi)):
        s_tag = "m" if sign < 0 else "p"
        accept_input  = inputs_dir  / f"binomial_btrs_accept_{fp}_{tag}_{s_tag}.txt"
        accept_output = outputs_dir / f"binomial_btrs_accept_{fp}_{tag}_{s_tag}.out"
        accept_input.write_text(
            make_btrs_accept_template(n, p, fp, lo, hi, sign, fast=fast))

        code, output = run_command([fptaylor, str(accept_input)], cwd=ROOT, env=env)
        accept_output.write_text(output)
        if verbose >= 2:
            print(f"--- FPTaylor BTRS accept s={sign:+d} (n={n}, p={p}) ---\n{output}")
        if code != 0:
            raise RuntimeError(f"FPTaylor BTRS accept s={sign:+d} failed for "
                               f"n={n}, p={p}; see {accept_output}")
        accept_raw = max(accept_raw,
                         extract_abs_errors_by_problem(output)["eps_accept"])

    # exp(2E) converts the ideal-binomial k_tail into a bound on the *fp*
    # sampler's out-of-window mass; see btrs_accept_tail_eps.
    m = math.floor((n + 1.0) * p)
    tail_E, tail_boxes = btrs_accept_tail_eps(
        (n, n), (p, p), (m, m), (k_lo, k_hi), us_min, fast=fast, verbose=verbose)
    k_tail_fp = math.exp(2.0 * tail_E) * k_tail
    vprint(verbose, f"binomial BTRS tail n={n} p={p}",
           tail_E=tail_E, tail_boxes=tail_boxes,
           k_tail=k_tail, k_tail_fp=k_tail_fp)

    eps_accept = accept_raw + eps_logv(
        fptaylor, fp, vtail, inputs_dir, outputs_dir, env, verbose,
    ) + k_tail_fp
    if fast:
        # the -2*log(us) query only sees us, so the smaller of the two tails
        # (a superset of the reachable us range) is the right bound to pass
        eps_accept += eps_logus(
            fptaylor, fp, us_min, inputs_dir, outputs_dir, env, verbose,
        )

    accept_iter = alpha / (math.sqrt(2 * math.pi) * spq)  # btrs is renormalized by the modal pmf f(m) = B(m) ~ 1/(sqrt(2*pi)*spq), so the per-iteration acceptance prob is 1/(alpha*f(m)).
    tv = 2 * (eps_floor + vtail) * accept_iter + 2 * eps_accept / (1 - vtail)
    return eps_floor, eps_accept, tv


# ---------------------------------------------------------------------------
# CIRE C code
# ---------------------------------------------------------------------------

_BINOM_C = """\
#include <math.h>
/* eps0: absolute error of exp(n * log(1-p)) */
double binom_eps0(double n, double p) { double q = 1.0 - p; return exp(n * log(q)); }
/* eps1: absolute error of z * (n - X + 1) * p / (X * (1-p)) */
double binom_eps1(double z, double X, double n, double p)
    { double q = 1.0 - p; return z * (n - X + 1.0) * p / (X * q); }
/* eps2: absolute error of sum + prod */
double binom_eps2(double s, double pr) { return s + pr; }
"""


def _run_cire(cire, n, p, args, inputs_dir, outputs_dir):
    """Return (eps0, eps1, eps2) relative errors via CIRE absolute errors."""
    q = 1.0 - p
    qn, z_lo, x_hi = inversion_params(n, p)

    tag = safe_pair_name(n, p)

    def _run(func, domains, label):
        rc, out = run_cire_llvm(
            cire, _BINOM_C, func, domains, tag, inputs_dir, outputs_dir,
            verbose=args.verbose,
        )
        if rc != 0:
            raise RuntimeError(f"CIRE failed for {label} (n={n}, p={p}); "
                               f"see outputs/{tag}_{func}.out")
        return extract_cire_abs_error(out, label)

    abs0 = _run("binom_eps0",
                [(float(n), float(n)), (p, p)],
                "eps0")
    abs1 = _run("binom_eps1",
                [(z_lo, 1.0), (1.0, x_hi),
                 (float(n), float(n)), (p, p)],
                "eps1")
    abs2 = _run("binom_eps2",
                [(qn, 1.0), (0.0, 1.0)],
                "eps2")

    # relative error = abs_error / lower_bound_of_exact_expression
    # eps0 lower bound: qn (the exact value, single-point expression)
    # eps1 lower bound: minimum of z*(n-X+1)*p/(X*q) at z=z_lo, X=x_hi
    # eps2 lower bound: qn (minimum of sum+prod = qn+0)
    eps1_lo = max(z_lo * (n - x_hi + 1.0) * p / (x_hi * q), sys.float_info.min)
    eps0 = abs0 / qn
    eps1 = abs1 / eps1_lo
    eps2 = abs2 / qn
    return eps0, eps1, eps2


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def safe_pair_name(n, p):
    p_str = f"{p:.6g}".replace(".", "p").replace("-", "m").replace("+", "")
    return f"n{n}_p{p_str}"


def read_np_pairs(path):
    pairs = []
    for lineno, line in enumerate(path.read_text().splitlines(), start=1):
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        tokens = line.split()
        if len(tokens) != 2:
            raise ValueError(f"{path}:{lineno}: expected 'n p', got {line!r}")
        try:
            n, p = int(tokens[0]), float(tokens[1])
        except ValueError as exc:
            raise ValueError(f"{path}:{lineno}: invalid (n, p) values") from exc
        if n <= 0:
            raise ValueError(f"{path}:{lineno}: n must be positive")
        if not (0 < p < 1):
            raise ValueError(f"{path}:{lineno}: p must be in (0, 1)")
        pairs.append((n, p))
    return pairs


# ---------------------------------------------------------------------------
# Distribution interface
# ---------------------------------------------------------------------------

def add_args(parser):
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("input_file", nargs="?", type=Path,
                        help="File with (n, p) pairs, one per line (format: 'n p')")
    source.add_argument("--n", type=int, default=None, help="Single n value")
    source.add_argument("--n-range", nargs=2, type=float, default=None,
                        metavar=("NMIN", "NMAX"),
                        help="Interval mode: bound the error over the whole "
                             "box n in [NMIN, NMAX] x p in [PMIN, PMAX] with "
                             "one FPTaylor query (requires --p-range)")
    parser.add_argument("--p-range", nargs=2, type=float, default=None,
                        metavar=("PMIN", "PMAX"),
                        help="Probability interval, required with --n-range")
    parser.add_argument("--p", type=float, default=None,
                        help="Probability p in (0,1), required with --n")
    parser.add_argument("--fast", action="store_true",
                        help="BTRS only: compute the -2*log(us) term of "
                             "eps_accept in a separate FPTaylor query and "
                             "sum it in, decoupling it from the shared "
                             "variable u. Faster, but may yield a more "
                             "conservative (looser) bound.")


def default_out_dir(args):
    backend = getattr(args, "backend", "fptaylor")
    if getattr(args, "n_range", None) is not None:
        return ROOT / f"binomial_runs_interval_{backend}"
    if getattr(args, "n", None) is not None:
        return ROOT / f"binomial_runs_{backend}"
    lf = getattr(args, "input_file", None)
    if lf is None:
        return ROOT / f"binomial_runs_{backend}"
    return ROOT / f"binomial_runs_{lf.stem}_{backend}"


def _run_box(args, fptaylor, inputs_dir, outputs_dir, env):
    """Interval mode: one row bounding the error over the whole (n, p) box."""
    n_lo, n_hi = args.n_range
    if args.p_range is None:
        raise ValueError("--p-range is required when --n-range is given")
    p_lo, p_hi = args.p_range
    if not (1 <= n_lo <= n_hi):
        raise ValueError("--n-range must satisfy 1 <= NMIN <= NMAX")
    if not (0 < p_lo <= p_hi < 1):
        raise ValueError("--p-range must satisfy 0 < PMIN <= PMAX < 1")
    if args.backend != "fptaylor":
        raise ValueError("interval mode is FPTaylor-only")

    tag = safe_box_name(n_lo, n_hi, p_lo, p_hi)
    row = {"n": "", "p": "",
           "n_lo": f"{n_lo:.17g}", "n_hi": f"{n_hi:.17g}",
           "p_lo": f"{p_lo:.17g}", "p_hi": f"{p_hi:.17g}"}

    # the sampler itself switches algorithm at n*p = 30, so the box has to sit
    # entirely on one side of it
    if n_lo * p_lo >= _BTRS_SWITCH:
        box = btrs_box_params(n_lo, n_hi, p_lo, p_hi)
        eps_floor, eps_accept, tv = _run_btrs_box(
            fptaylor, box, args.fp, tag, inputs_dir, outputs_dir, env,
            args.verbose, fast=args.fast,
        )
        row.update({"regime": "btrs-interval",
                    "eps0": "nan", "eps1": "nan", "eps2": "nan",
                    "eps_floor": f"{eps_floor:.17e}",
                    "eps_accept": f"{eps_accept:.17e}", "tv": f"{tv:.17e}"})
        print(f"{_box_label(box)} [BTRS] eps_floor={eps_floor:.6e}"
              f" eps_accept={eps_accept:.6e} TV={tv:.6e}")
    elif n_hi * p_hi < _BTRS_SWITCH:
        box = inversion_box_params(n_lo, n_hi, p_lo, p_hi)
        eps0, eps1, eps2, tv = _run_inversion_box(
            fptaylor, box, args, tag, inputs_dir, outputs_dir, env,
        )
        row.update({"regime": "inversion-interval",
                    "eps0": f"{eps0:.17e}", "eps1": f"{eps1:.17e}",
                    "eps2": f"{eps2:.17e}",
                    "eps_floor": "nan", "eps_accept": "nan", "tv": f"{tv:.17e}"})
        print(f"{_box_label(box)} eps0={eps0:.6e} eps1={eps1:.6e}"
              f" eps2={eps2:.6e} TV={tv:.6e}")
    else:
        raise ValueError(
            f"box straddles the n*p = {_BTRS_SWITCH:.0f} switch "
            f"(n*p spans [{n_lo * p_lo:.6g}, {n_hi * p_hi:.6g}]): the sampler "
            "uses inversion below it and BTRS above, so split the box there")
    return [row]


def run(args, fptaylor, inputs_dir, outputs_dir, env):
    if getattr(args, "n_range", None) is not None:
        try:
            return _run_box(args, fptaylor, inputs_dir, outputs_dir, env)
        except (ValueError, RuntimeError) as exc:
            raise SystemExit(f"error: {exc}")
    if getattr(args, "n", None) is not None:
        if args.p is None:
            raise ValueError("--p is required when --n is given")
        if args.n <= 0:
            raise ValueError("--n must be positive")
        if not (0 < args.p < 1):
            raise ValueError("--p must be in (0, 1)")
        pairs = [(args.n, args.p)]
    else:
        pairs = read_np_pairs(args.input_file)
    if not pairs:
        raise ValueError("no (n, p) pairs found in input")

    rows = []
    for n, p in pairs:
        tag = safe_pair_name(n, p)
        try:
            if n * p >= _BTRS_SWITCH:
                # ---- BTRS regime (FPTaylor only; CIRE not yet supported) ----
                eps_floor, eps_accept, tv = _run_btrs_fptaylor(
                    fptaylor, n, p, args.fp, tag, inputs_dir, outputs_dir,
                    env, args.verbose, fast=args.fast,
                )
                rows.append({
                    "n": n, "p": f"{p:.17g}", "regime": "btrs",
                    "eps0": "nan", "eps1": "nan", "eps2": "nan",
                    "eps_floor":  f"{eps_floor:.17e}",
                    "eps_accept": f"{eps_accept:.17e}",
                    "tv": f"{tv:.17e}",
                })
                print(f"n={n} p={p} [BTRS] eps_floor={eps_floor:.6e}"
                      f" eps_accept={eps_accept:.6e} TV={tv:.6e}")
            elif args.backend == "cire":
                # ---- inversion regime, CIRE ----
                qn, z_lo, x_hi = inversion_params(n, p)
                bound = n * p + 10.0 * math.sqrt(n * p * (1.0 - p))
                vprint(args.verbose, f"binomial inversion n={n} p={p}",
                       qn=qn, z_lo=z_lo, x_hi=x_hi, bound=bound)
                eps0, eps1, eps2 = _run_cire(fptaylor, n, p, args, inputs_dir, outputs_dir)
                tv = 0.5 * (eps0 + eps1 * p + eps2 * bound)
                rows.append({
                    "n": n, "p": f"{p:.17g}", "regime": "inversion",
                    "eps0": f"{eps0:.17e}", "eps1": f"{eps1:.17e}", "eps2": f"{eps2:.17e}",
                    "eps_floor": "nan", "eps_accept": "nan",
                    "tv": f"{tv:.17e}",
                })
                print(f"n={n} p={p} eps0={eps0:.6e} eps1={eps1:.6e} eps2={eps2:.6e} TV={tv:.6e}")
            else:
                # ---- inversion regime, FPTaylor ----
                qn, z_lo, x_hi = inversion_params(n, p)
                bound = n * p + 10.0 * math.sqrt(n * p * (1.0 - p))
                vprint(args.verbose, f"binomial inversion n={n} p={p}",
                       qn=qn, z_lo=z_lo, x_hi=x_hi, bound=bound)
                input_path = inputs_dir / f"binomial_inversion_{args.fp}_{tag}.txt"
                input_path.write_text(make_template(n, p, args.fp))
                code, output = run_command(
                    [fptaylor, "--rel-error", "true", str(input_path)],
                    cwd=ROOT, env=env,
                )
                out_path = outputs_dir / f"binomial_inversion_{args.fp}_{tag}.out"
                out_path.write_text(output)
                if args.verbose >= 2:
                    print(f"--- FPTaylor binomial_inversion (n={n}, p={p}) ---\n{output}")
                if code != 0:
                    raise RuntimeError(f"FPTaylor failed for n={n}, p={p}; see {out_path}")
                deltas = extract_deltas_by_problem(output, f"n={n} p={p}")
                eps0, eps1, eps2 = deltas["eps0"], deltas["eps1"], deltas["eps2"]
                tv = 0.5 * (eps0 + eps1 * p + eps2 * bound)
                rows.append({
                    "n": n, "p": f"{p:.17g}", "regime": "inversion",
                    "eps0": f"{eps0:.17e}", "eps1": f"{eps1:.17e}", "eps2": f"{eps2:.17e}",
                    "eps_floor": "nan", "eps_accept": "nan",
                    "tv": f"{tv:.17e}",
                })
                print(f"n={n} p={p} eps0={eps0:.6e} eps1={eps1:.6e} eps2={eps2:.6e} TV={tv:.6e}")
        except Exception as exc:
            print(f"WARNING: skipping n={n} p={p}: {exc}")

    return rows


def write_plot(rows, plot_path, plot_components=False, plot_pgf=False):
    import os, contextlib, math
    import numpy as np

    # interval-mode rows cover a box, not a point on the (n, np) grid
    rows = [r for r in rows if r.get("n") not in (None, "", "nan")]
    if not rows:
        print("Nothing to plot: interval-mode results are not points on the "
              "(n, np) grid")
        return False

    fields = [("eps0", "eps0"), ("eps2", "eps2"), ("TV", "tv")]
    if plot_components:
        fields = [("eps0", "eps0"), ("eps1", "eps1"), ("eps2", "eps2"), ("TV", "tv")]

    # Reparametrize: x = log2(n), y = log2(np) = ne - pe  (both integers).
    # This fills a dense rectangle instead of a thin diagonal band.
    ne_vals  = sorted({round(math.log2(float(r["n"]))) for r in rows})
    mnp_vals = sorted({round(math.log2(float(r["n"]) * float(r["p"]))) for r in rows})
    ne_idx   = {v: i for i, v in enumerate(ne_vals)}
    mnp_idx  = {v: i for i, v in enumerate(mnp_vals)}

    def make_grid(key):
        grid = np.full((len(mnp_vals), len(ne_vals)), np.nan)
        for r in rows:
            ne  = round(math.log2(float(r["n"])))
            mnp = round(math.log2(float(r["n"]) * float(r["p"])))
            v   = float(r[key])
            if math.isfinite(v) and v > 0:
                grid[mnp_idx[mnp], ne_idx[ne]] = math.log10(v)
        return grid

    with open(os.devnull, "w") as devnull, contextlib.redirect_stderr(devnull):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        flat_axes = axes.flat

        # y-axis tick labels: np = 2^mnp
        mnp_labels = [f"$2^{{{v}}}$" for v in mnp_vals]

        grids = [(label, make_grid(key)) for label, key in fields]
        vmin = min(np.nanmin(g) for _, g in grids)
        vmax = max(np.nanmax(g) for _, g in grids)

        for ax, (label, grid) in zip(flat_axes, grids):
            im = ax.pcolormesh(ne_vals, mnp_vals, grid,
                               cmap="viridis", vmin=vmin, vmax=vmax,
                               shading="nearest")
            fig.colorbar(im, ax=ax, label=f"log₁₀({label})")
            ax.set_xlabel("log₂(n)")
            ax.set_ylabel("np  (mean)")
            ax.set_yticks(mnp_vals)
            ax.set_yticklabels(mnp_labels)
            ax.set_title(label)

        for ax in list(flat_axes)[len(fields):]:
            ax.set_visible(False)

        fig.suptitle("Binomial FP error heatmap")
        plt.tight_layout()
        plt.savefig(plot_path, dpi=150)
        if plot_pgf:
            plt.savefig(plot_path.with_suffix(".pgf"), backend="pgf")
        plt.close()
