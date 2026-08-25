"""
Binomial sampler FP-error analysis: legacy inversion for n*p < _BTRS_SWITCH,
BTRS (Hormann transformed rejection, distributions/btrs.c) above it.

The BTRS bound covers the whole reachable u -- from u_lo, the u that maps to
k = 0, up to u_hi, the u that maps to k = n -- with one FPTaylor query per
side of u = 0 (see the "Sign-splitting of u" section); the 1/us conditioning
within a side is left to FPTaylor's own branch-and-bound splitting
(--opt bb --bb-split geometric, see dist_common.fptaylor_cmd).

Follows the pattern in dist_geometric.py; called by main.py.
"""
import math
import os
import shutil
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from dist_common import (
    ROOT, FP_TO_FPTAYLOR_RND,
    run_command, extract_deltas_by_problem, extract_abs_errors_by_problem,
    loggam_defs, eps_logv, eps_logus,
    transcendental_error_bound,
    vprint,
    fptaylor_cmd,
    us_root, hormann_u_at, hormann_k_defs,
    point_ivar,
)

NAME = "binomial"
CSV_FIELDS = ["n", "p", "n_lo", "n_hi", "p_lo", "p_hi", "regime",
              "eps0", "eps1", "eps2", "eps_floor", "eps_accept", "tv",
              "n_boxes"]

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
    One query per elementary FP op in legacy_random_binomial_inversion's loop
    (distributions/binomial_legacy_inversion.c):
      eps0 = qn = exp(n*log(q)), q = 1-p       eps1 = px = z*(n-X+1)*p/(X*q)
      eps2 = sum + prod  (sum in [qn,1], prod in [0,1])
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


def btrs_u_at(n, p, y, consts=None):
    """The u with (2*a/us + b)*u + c = y (see dist_common.hormann_u_at)."""
    _, a, b, c = consts or btrs_consts(n, p)
    return hormann_u_at(a, b, c, y)


# ---------------------------------------------------------------------------
# Sign-splitting of u
# ---------------------------------------------------------------------------
# btrs.c reaches u only through us = 0.5 - |u|, so u = 0 is a fold point where
# neither the accept template's sign-dependent k definition (btrs_k_defs) nor
# us itself is one-to-one.  Each side of u = 0 is therefore its own query; the
# 1/us blow-up as us -> 0 within a side is left to FPTaylor's own
# branch-and-bound splitting (--opt bb --bb-split geometric) rather than
# pre-shelled by binade in Python.

_K_SLACK = 1.0            # floor can disagree by one integer: y in [k-1, k+2]

# k1_ = k_ + 1 = (y_ - f) + 1 (btrs_k_defs) can enclose down to k_lo - _K_SLACK
# at a shell's low corner (y_'s slack-widened minimum, f at its max), which
# must stay > 0 for FPTaylor's lgamma; not a domain issue, just _K_SLACK/f
# pushing the enclosure below zero.  +1.0 is headroom past the break-even
# point (empirically sufficient); nk1_ mirrors this at the k-near-n end.
_K_BOUNDARY_MARGIN = _K_SLACK + 1.0


def y_window(k_lo, k_hi, slack=_K_SLACK):
    """
    The y range that covers every u the program can map to a k in [k_lo, k_hi].
    floor(y) = k means y in [k, k + 1), widened by `slack` on each side for the
    floor disagreement (see _K_SLACK).
    """
    return k_lo - slack, k_hi + 1.0 + slack


def shell_u(u_lo, u_hi, y_lo, y_hi):
    """
    Split [u_lo, u_hi] at u = 0 into (sign, lo, hi) pieces, one per side that
    is non-empty.  us = 0.5 - |u| is not one-to-one across u = 0, and the
    accept template needs the sign anyway (btrs_k_defs).  y_lo/y_hi are only
    for the error message.
    """
    if u_lo > u_hi:
        raise ValueError(f"empty u window for y in [{y_lo:.6g}, {y_hi:.6g}]")
    shells = []
    if u_lo < 0.0:
        shells.append((-1, u_lo, min(u_hi, 0.0)))
    if u_hi > 0.0:
        shells.append((+1, max(u_lo, 0.0), u_hi))
    return shells


def u_shells(n, p, y_lo, y_hi, consts=None):
    """shell_u at a literal (n, p); y is increasing in u, so the window inverts."""
    consts = consts or btrs_consts(n, p)
    return shell_u(btrs_u_at(n, p, y_lo, consts),
                   btrs_u_at(n, p, y_hi, consts), y_lo, y_hi)


def btrs_setup_defs(rnd, n_expr, p_expr, accept=False):
    """
    btrs.c's setup block [lines 48-51, 72-76] as FPTaylor Definitions.

    These are *derived* from n and p, not free inputs, so they have to be
    written as expressions: it keeps them correlated with each other and with
    u, and it charges the rounding of the setup arithmetic itself -- what
    dist_poisson_stable.py has to add by hand as _const_err.

    n_expr / p_expr are the FPTaylor source for n and p (literals here).
    """
    d = [f"  spq_   {rnd}= sqrt({n_expr} * {p_expr} * (1.0 - {p_expr})),",
         f"  b_     {rnd}= 1.15 + 2.53 * spq_,",
         f"  a_     {rnd}= -0.0873 + 0.0248 * b_ + 0.01 * {p_expr},",
         f"  c_     {rnd}= {n_expr} * {p_expr} + 0.5,"]
    if accept:
        d += [f"  alpha_ {rnd}= (2.83 + 5.1 / b_) * spq_,",
              f"  lpq_   {rnd}= log({p_expr} / (1.0 - {p_expr})),"]
    return d


def btrs_k_defs(sign, n_expr, p_expr):
    """k = floor(y) for one sign of u (see dist_common.hormann_k_defs)."""
    return hormann_k_defs(
        sign,
        [f"  spqx_  = sqrt({n_expr} * {p_expr} * (1.0 - {p_expr})),",
         f"  bx_    = 1.15 + 2.53 * spqx_,",
         f"  ax_    = -0.0873 + 0.0248 * bx_ + 0.01 * {p_expr},",
         f"  cx_    = {n_expr} * {p_expr} + 0.5,"],
        "cx_")


# --- template assembly -----------------------------------------------------
#
# Every BTRS query is one of two expressions -- eps_floor or eps_accept -- over
# some set of free inputs.  The point, per-k and interval-parameter forms differ
# only in which inputs are free and how k, m and h are spelled, so the two
# expressions are built once here and the callers supply just those pieces.

def _ivar(name, lo, hi, kind="real"):
    return f"  {kind} {name} in [{lo:.20e}, {hi:.20e}]"


def _template(var_lines, def_lines, expr):
    """Assemble one query; the last Variable and Definition get their ';'."""
    return ("Variables\n" + ",\n".join(var_lines) + ";\n\n"
            + "Definitions\n" + "\n".join(def_lines).rstrip(",") + ";\n\n"
            + "Expressions\n  " + expr + ";\n")


def make_floor_template(fp, var_lines, n_expr, p_expr):
    """
    eps_floor: absolute error of y = (2*a/us + b)*u + c, us = 0.5 - |u|
    [btrs.c:60-61].  a, b, c stay expressions in n and p so they remain
    correlated with each other and with u, and their own rounding is charged.
    """
    rnd = FP_TO_FPTAYLOR_RND[fp]
    floor_defs = [f"  us_    {rnd}= 0.5 - abs(u),",
                  f"  y_     {rnd}= (2.0 * a_ / us_ + b_) * u + c_,"]
    return _template(var_lines,
                     btrs_setup_defs(rnd, n_expr, p_expr) + floor_defs,
                     "eps_floor = y_")


def make_accept_template(fp, var_lines, n_expr, p_expr, mid, name_k, name_nk,
                         fast=False):
    """
    eps_accept: absolute error of
      h - loggam(k+1) - loggam(n-k+1) + (k-m)*lpq - log(alpha)
        + log(a + b*us^2) - 2*log(us)                          [btrs.c:85]
    excluding -log(v), which make_logv_template covers.

    `mid` supplies the variant-specific Definitions (us_, k, m, h and the two
    inlined random_loggam chains) between the shared setup block and the shared
    tail; name_k / name_nk name the two loggam results.

    With fast, -2*log(us) is dropped here and bounded by a separate eps_logus
    query.  That drops u as a variable shared between the two terms, so the sum
    may be looser.
    """
    rnd = FP_TO_FPTAYLOR_RND[fp]
    return _template(
        var_lines,
        btrs_setup_defs(rnd, n_expr, p_expr, accept=True) + mid + [
            f"  us_sq_      {rnd}= us_ * us_,",
            f"  log_num_    {rnd}= a_ + b_ * us_sq_,",
            f"  btrs_accept {rnd}= h_ - {name_k} - {name_nk}"
            f" + (k_ - m_) * lpq_ - log(alpha_) + log(log_num_)"
            f"{'' if fast else ' - 2.0 * log(us_)'},"],
        "eps_accept = btrs_accept")


def make_btrs_floor_template(n, p, fp, u_lo, u_hi):
    """
    Point form: only u varies; n and p are declared as Variables bracketing
    the exact n, p (see dist_common.exact_bracket) so a, b, c reference them by name
    instead of embedding the literal expression at every occurrence.
    """
    return make_floor_template(
        fp, [_ivar("u", u_lo, u_hi), point_ivar("n", n), point_ivar("p", p)],
        "n", "p")


def _point_hm_defs(n, p, rnd):
    """
    m and h as Definitions: m = floor((n+1)*p) is exact (unrounded) --
    n, p are exact literals here, so Python's (n+1)*p is the same IEEE-754
    double-precision operation the real sampler performs, no fm-style tie
    needed (unlike box mode, where m is a genuine enclosure over a range and
    can't be reduced to one Python float at all -- see btrs_box_m's absence
    and make_btrs_accept_box_template).  Tying m to (n, p) via fm here was
    tried and reverted: it adds a permanent width-1 dimension that, near the
    u ranges that already need heavy splitting (us -> 0), multiplies rather
    than adds to the search cost -- measured ~4.7x slower to reach the same
    suboptimality, with no correctness gain to justify it.

    h = lgamma(m+1) + lgamma(n-m+1) [btrs.c:77] is two independently-rounded
    lgamma calls plus a rounded addition -- what btrs.c does at runtime --
    via loggam_int_defs, same as k1_/nk1_.  Precomputing lgamma(m+1)/
    lgamma(n-m+1) in Python and charging only the addition's rounding would
    assume each lgamma() call is exact, which for m in the thousands is
    routinely on the order of the whole eps_accept bound.
    """
    m = int(math.floor((n + 1) * p))
    defs_m,  name_m  = loggam_int_defs(float(m) + 1.0,     "hm",  rnd)
    defs_nm, name_nm = loggam_int_defs(float(n - m) + 1.0, "hnm", rnd)
    return ([f"  m_     = {float(m):.1f},"]
            + defs_m + defs_nm
            + [f"  h_     {rnd}= {name_m} + {name_nm},"])


def make_btrs_accept_template(n, p, fp, u_lo, u_hi, sign, fast=False):
    """
    Point form, k tied to u: free inputs are u and f only (btrs_k_defs), and
    sign selects the side of u = 0 that k's definition needs.

    Valid for k in [_K_BOUNDARY_MARGIN, n - _K_BOUNDARY_MARGIN]: lgamma(x)
    is rigorous for every x > 0, but k1_ = k_ + 1's *enclosure* dips to <= 0
    closer than that to either end (see _K_BOUNDARY_MARGIN); the k outside
    that window go one at a time through make_btrs_accept_k_template.

    n and p are declared as Variables bracketing the exact n, p (see
    dist_common.exact_bracket) and referenced by name rather than re-embedded as
    literals at every occurrence.
    """
    rnd = FP_TO_FPTAYLOR_RND[fp]
    defs_k,  name_k  = loggam_defs("k1_",  "lgk",  rnd)
    defs_nk, name_nk = loggam_defs("nk1_", "lgnk", rnd)

    mid = (_point_hm_defs(n, p, rnd)
           + [f"  us_    {rnd}= 0.5 - abs(u),"]
           + btrs_k_defs(sign, "n", "p")
           + ["  nk1_   = n - k_ + 1.0,"]
           + defs_k + defs_nk)
    return make_accept_template(
        fp, [_ivar("u", u_lo, u_hi), point_ivar("n", n), point_ivar("p", p),
             "  real f in [0.0, 1.0]"],
        "n", "p", mid, name_k, name_nk, fast)


def make_btrs_accept_k_template(n, p, fp, u_lo, u_hi, k, fast=False):
    """
    Point form for one integer k, for the k the margin sweep can't cover (see
    _K_BOUNDARY_MARGIN): tying k to u via k_ = y_ - f would let k1_/nk1_'s
    enclosure reach <= 0, so k is a literal instead -- error-free, since k
    and n - k + 1 are formed in integer arithmetic and cast, same as btrs.c.
    This gives up btrs_k_defs's k-to-u coupling (u only ranges over the shell
    for this k); the error in *choosing* k is eps_floor's job either way.
    """
    rnd = FP_TO_FPTAYLOR_RND[fp]
    defs_k,  name_k  = loggam_int_defs(float(k) + 1.0,     "lgk",  rnd)
    defs_nk, name_nk = loggam_int_defs(float(n - k) + 1.0, "lgnk", rnd)

    mid = ([f"  k_     = {float(k):.1f},"]
           + _point_hm_defs(n, p, rnd)
           + [f"  us_    {rnd}= 0.5 - abs(u),"]
           + defs_k + defs_nk)
    return make_accept_template(
        fp, [_ivar("u", u_lo, u_hi), point_ivar("n", n), point_ivar("p", p)],
        "n", "p", mid, name_k, name_nk, fast)


def loggam_int_defs(x, prefix, rnd):
    """
    random_loggam(x) for an exact integer x >= 1.  random_loggam itself only
    takes lgamma(x) directly for x >= 7 -- below that it either early-returns
    0 for x in {1, 2} [random_poisson_ptrs.c:41] or reduces the argument up
    to x >= 7 first -- because its own Stirling series isn't valid below 7.
    FPTaylor's native lgamma(x) has no such restriction (rigorous for every
    x > 0, see FPTaylor/func.ml:lgamma_I), so there is nothing to special-case
    here: call it directly at x itself.
    Returns (lines, result_name) like loggam_defs.
    """
    return loggam_defs(f"{x:.1f}", prefix, rnd)


# ---------------------------------------------------------------------------
# Interval-parameter mode:  midpoint splitting of the (n, p) box
# ---------------------------------------------------------------------------
# Nothing is singular in n and p, so they get midpoint bisection into a grid of
# sub-boxes rather than binade shells; the bound is the max over sub-boxes.
# Within a sub-box they are FPTaylor variables, which costs correlation between
# their repeated occurrences.  The slop lands hardest on n - k + 1, which the
# inlined loggam needs a reciprocal and a logarithm of; naively it straddles
# zero once the box is wider than n - k and FPTaylor rejects the query.  Kept in
# check by (most important first) parametrizing each accept query by whichever
# of k and n - k is small (make_btrs_accept_box_template), minimising the
# occurrences of n and p, and --split-depth.

def split_box(n_iv, p_iv, depth):
    """
    The 4^depth sub-boxes of [n_lo, n_hi] x [p_lo, p_hi] obtained by bisecting
    each axis at its midpoint `depth` times.  Returned as (n_iv, p_iv) pairs.
    """
    boxes = [(n_iv, p_iv)]
    for _ in range(depth):
        boxes = [(sub_n, sub_p)
                 for n, p in boxes
                 for sub_n in _bisect(n)
                 for sub_p in _bisect(p)]
    return boxes


def _bisect(iv):
    lo, hi = iv
    mid = 0.5 * (lo + hi)
    return [(lo, hi)] if lo >= hi else [(lo, mid), (mid, hi)]


def btrs_box_consts(n_iv, p_iv):
    """
    Enclosures of btrs_consts over the box, as (lo, hi) pairs, from the
    monotonicity of each quantity: spq and b are increasing in n and in
    p*(1-p) (which peaks at p = 1/2), a is increasing in b and in p, and
    c = n*p + 0.5 is increasing in both.
    """
    (n_lo, n_hi), (p_lo, p_hi) = n_iv, p_iv
    pq_ends = (p_lo * (1.0 - p_lo), p_hi * (1.0 - p_hi))
    pq_lo   = min(pq_ends)
    pq_hi   = 0.25 if p_lo <= 0.5 <= p_hi else max(pq_ends)

    spq = (math.sqrt(n_lo * pq_lo), math.sqrt(n_hi * pq_hi))
    b   = (1.15 + 2.53 * spq[0], 1.15 + 2.53 * spq[1])
    a   = (-0.0873 + 0.0248 * b[0] + 0.01 * p_lo,
           -0.0873 + 0.0248 * b[1] + 0.01 * p_hi)
    c   = (n_lo * p_lo + 0.5, n_hi * p_hi + 0.5)
    return spq, a, b, c


def box_u_at(box_consts, y):
    """
    Enclosure of {u : y_{n,p}(u) = y} over the box: the union of btrs_u_at's
    single root at each (n, p).  us_root grows with a and b and shrinks with
    gamma = |y - c|, c = n*p + 0.5, so the widest u comes from (a_lo, b_lo,
    gamma_hi).  Which side of u = 0 depends on where y falls relative to c;
    inside the enclosure of c the union straddles it.
    """
    _, a, b, c = box_consts
    if y >= c[1]:                                  # above c for every (n, p)
        gam = (y - c[1], y - c[0])
        return (0.5 - us_root(a[1], b[1], gam[0]),
                0.5 - us_root(a[0], b[0], gam[1]))
    if y <= c[0]:                                  # below c for every (n, p)
        gam = (c[0] - y, c[1] - y)
        return (-(0.5 - us_root(a[0], b[0], gam[1])),
                -(0.5 - us_root(a[1], b[1], gam[0])))
    gam_hi = max(c[1] - y, y - c[0])
    t = us_root(a[0], b[0], gam_hi)
    return (-(0.5 - t), 0.5 - t)


def box_u_shells(box_consts, y_lo, y_hi):
    """shell_u over a parameter box: the widest u window, split at u = 0."""
    return shell_u(box_u_at(box_consts, y_lo)[0],
                   box_u_at(box_consts, y_hi)[1], y_lo, y_hi)


def make_btrs_floor_box_template(box, fp):
    """Interval-parameter form: u, n and p are the only free inputs."""
    return make_floor_template(fp, [_ivar("u", *box["u"]), _ivar("n", *box["n"]),
                                    _ivar("p", *box["p"])], "n", "p")


def make_btrs_accept_box_template(box, fp, low, x_iv, fast=False):
    """
    Interval-parameter form of the accept template, parametrized by whichever
    of k and j = n - k is the *small* one:

        low=True   x = k,      k + 1 = x + 1,  n - k + 1 = n - x + 1
        low=False  x = n - k,  k + 1 = n - x + 1,  n - k + 1 = x + 1

    n - x + 1 carries the full width of the box (n and x can't cancel);
    parametrizing by the small side keeps that wide argument on the large
    loggam (harmless there) and leaves the small argument exact.  Point mode
    doesn't need this since n is a literal.

    x_iv is x's enclosure; a degenerate (v, v) is emitted as a literal
    (loggam_int_defs) rather than an FPTaylor variable.  Unlike the point
    form, k is not tied to u here (no f): the caller pairs an x enclosure
    with the u window mapping into it, a sound over-approximation of the
    reachable (x, u) curve.  Tying k to u live via btrs_k_defs instead (as
    the point form does) forces FPTaylor to re-derive a division (a/us)
    with n, p *as ranges* inside the query; division amplifies interval
    width multiplicatively, and the resulting margin needed to keep k1_/
    nk1_ positive explodes non-linearly with the box's width (measured:
    2.6 -> 9.65 -> 78.7 -> 642 -> unsatisfiable as p's width alone grew
    from 0 to 0.01) rather than scaling gently -- so this shelled form,
    which never lets u and n/p interact through that division inside the
    query, is the one that actually works for a box wide enough to matter.

    m is tied to (n, p) the same way k is tied to u: m_ = (n+1)*p - fm,
    fm in [0, 1), encoding floor((n+1)*p) exactly (m is one integer, computed
    once, same as k_ and j_ above -- no rounding).  This is safe where tying
    k to u was not: (n+1)*p is a product of two *positive* ranges, which
    interval arithmetic bounds exactly (no amplification), unlike a/us's
    division by something that can shrink toward zero.  It also means m_
    never appears in FPTaylor's own per-variable domain-size check (only
    declared Variables do) -- fm in [0,1) is narrow regardless of the box's
    width, whereas m_ declared directly could be thousands wide, so this
    removes what was often the single most expensive dimension to narrow.
    h = lgamma(m+1) + lgamma(n-m+1) is computed from m *inside* the query
    (loggam_defs), same as before.
    """
    rnd = FP_TO_FPTAYLOR_RND[fp]
    x_lo, x_hi = x_iv
    literal = x_lo == x_hi
    x_ref = "x_" if literal else "x"
    small = f"{x_ref} + 1.0"

    if low:
        kdefs = [f"  k_     = {x_ref},", f"  k1_    = {small},",
                 "  nk1_   = n - k_ + 1.0,"]
        small_pre, big_pre, big_expr, small_expr = "lgk", "lgnk", "nk1_", "k1_"
    else:
        kdefs = [f"  j_     = {x_ref},", f"  nk1_   = {small},",
                 "  k_     = n - j_,", "  k1_    = k_ + 1.0,"]
        small_pre, big_pre, big_expr, small_expr = "lgnk", "lgk", "k1_", "nk1_"

    if literal:
        defs_small, name_small = loggam_int_defs(x_lo + 1.0, small_pre, rnd)
    else:
        defs_small, name_small = loggam_defs(small_expr, small_pre, rnd)
    defs_big, name_big = loggam_defs(big_expr, big_pre, rnd)
    defs_hm,  name_hm  = loggam_defs("m_ + 1.0", "hm", rnd)
    defs_hnm, name_hnm = loggam_defs("n - m_ + 1.0", "hnm", rnd)

    var_lines = [_ivar("u", *box["u"]), _ivar("n", *box["n"]),
                 _ivar("p", *box["p"]), "  real fm in [0.0, 1.0]"]
    if not literal:
        var_lines.append(_ivar("x", *box["x"]))

    mid = ([f"  us_    {rnd}= 0.5 - abs(u),"]
           + ([f"  x_     = {x_lo:.1f},"] if literal else [])
           + kdefs
           + ["  m_     = (n + 1.0) * p - fm,"]
           + defs_small + defs_big
           + defs_hm + defs_hnm
           + [f"  h_     {rnd}= {name_hm} + {name_hnm},"])
    return make_accept_template(
        fp, var_lines, "n", "p", mid,
        name_small if low else name_big,
        name_big if low else name_small, fast)


def make_inversion_box_template(box, fp):
    """Interval-parameter form of make_template (inversion regime)."""
    rnd = FP_TO_FPTAYLOR_RND[fp]
    return (
        "Variables\n"
        f"  real z in [{box['z'][0]:.20e}, 1.0],\n"
        f"  real X in [1.0, {box['X'][1]:.20e}],\n"
        f"  real sum in [{box['qn'][0]:.20e}, 1.0],\n"
        f"  real prod in [0.0, 1.0],\n"
        + _ivar("n", *box["n"]) + ",\n"
        + _ivar("p", *box["p"]) + ";\n\n"
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


# ---------------------------------------------------------------------------
# Running one FPTaylor query per box
# ---------------------------------------------------------------------------

def _fptaylor_max(fptaylor, boxes, expr, inputs_dir, outputs_dir, env,
                  verbose, jobs, ratio_tol, bb_eval=False, x_abs_tol=None):
    """
    Run one FPTaylor query per box and return (max_error, n_boxes).

    `boxes` is a list of (label, stem, template_text): label goes in the log,
    stem names the input/output files.

    bb_eval=False (default) compiles each query with --opt bb under a fixed
    filename relative to the working directory regardless of --tmp-base-dir
    [FPTaylor/opt_basic_bb.ml], so concurrent bb queries race and clobber
    each other's compiled program -- `jobs` is capped to 1 worker.
    bb_eval=True uses the interpreted --opt bb-eval instead: no compile step,
    no race, `jobs` workers run concurrently.
    """
    def one(box):
        label, stem, text = box
        in_path  = inputs_dir  / f"{stem}.txt"
        out_path = outputs_dir / f"{stem}.out"
        in_path.write_text(text)
        # each query gets its own scratch dir; see dist_common.fptaylor_cmd
        work = Path(tempfile.mkdtemp(prefix="fpt_", dir=outputs_dir))
        try:
            code, output = run_command(
                fptaylor_cmd(fptaylor, in_path, work, ratio_tol, bb_eval, x_abs_tol),
                cwd=ROOT, env=env)
        finally:
            shutil.rmtree(work, ignore_errors=True)
        out_path.write_text(output)
        if verbose >= 2:
            print(f"--- FPTaylor {stem} ---\n{output}")
        if code != 0:
            raise RuntimeError(f"FPTaylor failed on {label}; see {out_path}")
        errors = extract_abs_errors_by_problem(output)
        if expr not in errors:
            # FPTaylor exits 0 but reports nothing when the box leaves the
            # domain of a subexpression (a log or a division by an interval
            # straddling zero), so a missing bound has to be an error too
            raise RuntimeError(f"FPTaylor reported no {expr} bound on {label}; "
                               f"see {out_path}")
        return errors[expr]

    with ThreadPoolExecutor(max_workers=jobs if bb_eval else 1) as pool:
        errors = list(pool.map(one, boxes))

    worst = 0.0
    for (label, _, _), err in zip(boxes, errors):
        if verbose >= 1:
            print(f"    {label:<34} {expr} = {err:.6e}")
        worst = max(worst, err)
    return worst, len(boxes)


def _shell_label(sign, u_lo, u_hi):
    us_lo, us_hi = 0.5 - max(abs(u_lo), abs(u_hi)), 0.5 - min(abs(u_lo), abs(u_hi))
    return f"u{'<' if sign < 0 else '>'}0 us in [{us_lo:.3e}, {us_hi:.3e}]"


def _shell_stem(prefix, fp, tag, index, sign):
    return f"{prefix}_{fp}_{tag}_{'m' if sign < 0 else 'p'}{index:02d}"


def _btrs_floor_boxes(n, p, fp, tag, consts):
    """
    One box per side of u = 0 that btrs.c does not reject outright, i.e. the u
    it maps to some k in [0, n]        [btrs.c:62-64].
    """
    y_lo, y_hi = y_window(0.0, float(n))
    boxes = []
    for i, (sign, u_lo, u_hi) in enumerate(u_shells(n, p, y_lo, y_hi, consts)):
        boxes.append((f"floor {_shell_label(sign, u_lo, u_hi)}",
                      _shell_stem("binomial_btrs_floor", fp, tag, i, sign),
                      make_btrs_floor_template(n, p, fp, u_lo, u_hi)))
    return boxes


def _btrs_accept_boxes(n, p, fp, tag, consts, fast):
    """
    Boxes covering the accept expression for every k in [0, n], in two parts:

      - k in [_K_BOUNDARY_MARGIN, n - _K_BOUNDARY_MARGIN]: one query per side
        of u = 0 covers a whole run of k, with k tied to u by btrs_k_defs.
      - the boundary k outside that window (k1_ or nk1_'s enclosure would
        dip to <= 0 there, see _K_BOUNDARY_MARGIN): one query each, with k
        as a literal (make_btrs_accept_k_template).
    """
    boxes = []
    margin = _K_BOUNDARY_MARGIN
    y_lo, y_hi = y_window(margin, n - margin)
    for i, (sign, u_lo, u_hi) in enumerate(u_shells(n, p, y_lo, y_hi, consts)):
        boxes.append((f"accept {_shell_label(sign, u_lo, u_hi)}",
                      _shell_stem("binomial_btrs_accept", fp, tag, i, sign),
                      make_btrs_accept_template(n, p, fp, u_lo, u_hi, sign,
                                                fast=fast)))

    edge = math.ceil(margin)
    for k in (list(range(edge)) + [n - j for j in range(edge)]):
        ky_lo, ky_hi = y_window(k, k)
        u_lo = btrs_u_at(n, p, ky_lo, consts)
        u_hi = btrs_u_at(n, p, ky_hi, consts)
        boxes.append((f"accept k={k}",
                      f"binomial_btrs_accept_{fp}_{tag}_k{k}",
                      make_btrs_accept_k_template(n, p, fp, u_lo, u_hi, k,
                                                  fast=fast)))
    return boxes


def combine_tv(fptaylor, fp, eps_floor, accept_raw, accept_iter, us_lo, us_hi,
               args, inputs_dir, outputs_dir, env, ledger=False):
    """
    (eps_accept, tv, n_extra): fold the per-query maxima into the TV bound.

    eps_accept still owes -log(v) (and, with --fast, the split-out
    -2*log(us)) before it's complete.  us_lo/us_hi are how far short of the
    full u in [-0.5, 0.5) each side's box falls; below args.u_trunc, that
    shortfall of u mass is charged to TV directly (real and FP samplers may
    disagree on it, same as a sampler that disagrees on a positive-
    probability event):

        tv = 2*eps_floor*accept_iter + 2*eps_accept
             + max(0, u_trunc - us_lo) + max(0, u_trunc - us_hi)

    ledger=True charges -log(v) via transcendental_error_bound instead of a
    query, and skips -2*log(us) entirely; unused by any current caller, kept
    for a path that ledgers log/exp rather than inlining them.
    """
    us_min = min(us_lo, us_hi)
    if ledger:
        logv = transcendental_error_bound("log", 1.0, fp)
        n_extra = 0
    else:
        logv, n_extra = eps_logv(fptaylor, fp, args.u_trunc, inputs_dir, outputs_dir,
                                 env, args.verbose, args.bb_geometric_ratio_tol,
                                 args.bb_eval, args.opt_x_abs_tol)
    eps_accept = accept_raw + logv
    if args.fast and not ledger:
        # the -2*log(us) query only sees us, so the smallest reachable us (a
        # superset of every shell's us range) is the right bound to pass
        logus, n_logus = eps_logus(fptaylor, fp, us_min, inputs_dir,
                                   outputs_dir, env, args.verbose,
                                   args.bb_geometric_ratio_tol,
                                   args.bb_eval, args.opt_x_abs_tol)
        eps_accept += logus
        n_extra += n_logus
    u_excess = max(0.0, args.u_trunc - us_lo) + max(0.0, args.u_trunc - us_hi)
    tv = 2 * eps_floor * accept_iter + 2 * eps_accept + u_excess
    return eps_accept, tv, n_extra


def _run_btrs_fptaylor(fptaylor, n, p, args, tag, inputs_dir, outputs_dir, env):
    """(eps_floor, eps_accept, tv, n_boxes) for the BTRS regime at (n, p)."""
    fp, verbose = args.fp, args.verbose
    consts = spq, a, b, c = btrs_consts(n, p)
    if a <= 0.0:
        raise ValueError(f"BTRS shape constant a = {a:.6g} <= 0 "
                         f"(n*p*q = {n * p * (1.0 - p):.6g} too small); "
                         "the reachable u range is not a single interval")
    if n < 2 * _K_BOUNDARY_MARGIN:
        raise ValueError(f"n = {n} is too small to leave any k in "
                         f"[_K_BOUNDARY_MARGIN, n - _K_BOUNDARY_MARGIN] "
                         "for the accept margin sweep to cover")
    alpha  = (2.83 + 5.1 / b) * spq
    fy_lo, fy_hi = y_window(0.0, float(n))
    u_lo, u_hi = btrs_u_at(n, p, fy_lo, consts), btrs_u_at(n, p, fy_hi, consts)
    us_lo, us_hi = 0.5 + u_lo, 0.5 - u_hi
    vprint(verbose, f"binomial BTRS n={n} p={p}",
           spq=spq, a=a, b=b, c=c, alpha=alpha,
           u_lo=u_lo, u_hi=u_hi, us_lo=us_lo, us_hi=us_hi, u_trunc=args.u_trunc)

    floor_boxes  = _btrs_floor_boxes(n, p, fp, tag, consts)
    accept_boxes = _btrs_accept_boxes(n, p, fp, tag, consts, args.fast)

    # k-tail (k outside [0, n]) is rejected identically by both samplers, so
    # it costs nothing; but the u the boxes above don't reach beyond u_trunc
    # on either side is charged to TV in combine_tv (see us_lo/us_hi there).
    floor_raw, n_floor = _fptaylor_max(
        fptaylor, floor_boxes, "eps_floor", inputs_dir, outputs_dir, env,
        verbose, args.jobs, args.bb_geometric_ratio_tol,
        args.bb_eval, args.opt_x_abs_tol)
    eps_floor = 5 * floor_raw

    accept_raw, n_accept = _fptaylor_max(
        fptaylor, accept_boxes, "eps_accept", inputs_dir, outputs_dir, env,
        verbose, args.jobs, args.bb_geometric_ratio_tol,
        args.bb_eval, args.opt_x_abs_tol)

    # btrs is renormalized by the modal pmf f(m) = B(m) ~ 1/(sqrt(2*pi)*spq),
    # so the per-iteration acceptance probability is 1/(alpha*f(m)).
    accept_iter = alpha / (math.sqrt(2 * math.pi) * spq)
    eps_accept, tv, n_extra = combine_tv(
        fptaylor, fp, eps_floor, accept_raw, accept_iter, us_lo, us_hi, args,
        inputs_dir, outputs_dir, env)
    vprint(verbose, f"binomial BTRS boxes n={n} p={p}",
           floor_boxes=n_floor, accept_boxes=n_accept, logv_boxes=n_extra,
           floor_raw=floor_raw, accept_raw=accept_raw)
    return eps_floor, eps_accept, tv, n_floor + n_accept + n_extra



def btrs_box_k_shells(n_iv, p_iv):
    """
    (low, high): enclosures covering every k in [0, n] for every n in the box.
    `low` encloses k, `high` the offset j = n - k; a k is covered if it
    appears in either.  Each is one query over the whole range of x + 1, with
    the 1/(x+1) and log(x+1) conditioning left to FPTaylor's own
    branch-and-bound splitting (--opt bb --bb-split geometric).

    The halves must meet (k_mid + j_mid >= n_hi - 1); too-wide a box raises,
    telling the caller to bump --split-depth.  A half that collapses to an
    empty range (e.g. the other half alone already reaches k = 0) is omitted.
    """
    n_lo, n_hi = n_iv
    j_mid = n_lo
    k_mid = min(j_mid, n_hi - 1.0 - j_mid)
    if k_mid + j_mid < n_hi - 1.0:
        raise ValueError(
            f"n in [{n_lo:.6g}, {n_hi:.6g}] is too wide to cover k in [0, n]: "
            f"the low half reaches k = {k_mid:.6g} and the high half n - k = "
            f"{j_mid:.6g}, which do not meet; raise --split-depth")

    def shells(x_mid):
        return [(0.0, x_mid)] if x_mid > 0.0 else []

    return shells(k_mid), shells(j_mid)


def _btrs_sub_box_boxes(n_iv, p_iv, fp, tag, fast):
    """
    (floor_boxes, accept_boxes) for one sub-box of the parameter grid: the same
    coverage as the point-mode sweeps (_btrs_floor_boxes, _btrs_accept_boxes),
    with n and p as FPTaylor variables and every u window widened to hold for
    every (n, p) in the sub-box.

    The accept sweep is split in k (or in the offset n - k) rather than tied
    to u, and each k range is paired with the u window (split at u = 0) that
    maps into it; see make_btrs_accept_box_template.
    """
    (n_lo, n_hi), (p_lo, p_hi) = n_iv, p_iv
    consts = btrs_box_consts(n_iv, p_iv)
    _, a, _, _ = consts
    if a[0] <= 0.0:
        raise ValueError(f"BTRS shape constant a reaches {a[0]:.6g} <= 0 at the "
                         f"low corner of n in [{n_lo:.6g}, {n_hi:.6g}], "
                         f"p in [{p_lo:.6g}, {p_hi:.6g}]")
    sub = safe_box_name(n_lo, n_hi, p_lo, p_hi)
    floor_boxes, accept_boxes = [], []

    for i, (sign, u_lo, u_hi) in enumerate(
            box_u_shells(consts, *y_window(0.0, n_hi))):
        box = {"u": (u_lo, u_hi), "n": n_iv, "p": p_iv}
        floor_boxes.append((f"floor {_shell_label(sign, u_lo, u_hi)}",
                            _shell_stem("binomial_btrs_floor", fp,
                                        f"{tag}_{sub}", i, sign),
                            make_btrs_floor_box_template(box, fp)))

    low, high = btrs_box_k_shells(n_iv, p_iv)
    for is_low, x_shells in ((True, low), (False, high)):
        side = "k" if is_low else "n-k"
        for x_lo, x_hi in x_shells:
            # the k these x cover, and hence the y and the u that reach them
            k_iv = (x_lo, x_hi) if is_low else (n_lo - x_hi, n_hi - x_lo)
            y_lo, y_hi = y_window(*k_iv)
            u_win = box_u_shells(consts, y_lo, y_hi)
            for i, (sign, u_lo, u_hi) in enumerate(u_win):
                box = {"u": (u_lo, u_hi), "n": n_iv, "p": p_iv,
                       "x": (x_lo, x_hi)}
                x_tag = (f"{x_lo:.0f}" if x_lo == x_hi
                         else f"{x_lo:.0f}-{x_hi:.0f}")
                accept_boxes.append((
                    f"accept {side}={x_tag} {_shell_label(sign, u_lo, u_hi)}",
                    f"binomial_btrs_accept_{fp}_{tag}_{sub}_"
                    f"{'k' if is_low else 'j'}{x_tag}_"
                    f"{'m' if sign < 0 else 'p'}{i:02d}",
                    make_btrs_accept_box_template(box, fp, is_low, (x_lo, x_hi),
                                                  fast=fast)))
    return floor_boxes, accept_boxes


def _run_btrs_box(fptaylor, n_iv, p_iv, args, tag, inputs_dir, outputs_dir, env):
    """
    (eps_floor, eps_accept, tv, n_boxes) valid for every (n, p) in the box.

    The parameter box is bisected at its midpoints into 4^--split-depth
    sub-boxes; each is swept exactly as a point is, and the max over sub-boxes
    is the bound for the whole box.
    """
    fp, verbose = args.fp, args.verbose
    _, n_hi = n_iv

    sub_boxes = split_box(n_iv, p_iv, args.split_depth)
    vprint(verbose, f"binomial BTRS box {_box_label({'n': n_iv, 'p': p_iv})}",
           split_depth=args.split_depth, sub_boxes=len(sub_boxes),
           u_trunc=args.u_trunc)

    eps_floor = eps_accept_raw = 0.0
    n_boxes = 0
    for sub_n, sub_p in sub_boxes:
        floor_boxes, accept_boxes = _btrs_sub_box_boxes(
            sub_n, sub_p, fp, tag, args.fast)
        if verbose >= 1:
            print(f"  sub-box {_box_label({'n': sub_n, 'p': sub_p})}"
                  f" ({len(floor_boxes)} floor + {len(accept_boxes)} accept)")

        floor_raw, n_floor = _fptaylor_max(
            fptaylor, floor_boxes, "eps_floor", inputs_dir, outputs_dir, env,
            verbose, args.jobs, args.bb_geometric_ratio_tol,
            args.bb_eval, args.opt_x_abs_tol)
        accept_raw, n_accept = _fptaylor_max(
            fptaylor, accept_boxes, "eps_accept", inputs_dir, outputs_dir, env,
            verbose, args.jobs, args.bb_geometric_ratio_tol,
            args.bb_eval, args.opt_x_abs_tol)

        eps_floor = max(eps_floor, 5 * floor_raw)
        eps_accept_raw = max(eps_accept_raw, accept_raw)
        n_boxes += n_floor + n_accept

    # widest u reached anywhere in the box, on each side, for the split-out
    # -2*log(us) query and for the u_trunc excess in combine_tv
    consts = btrs_box_consts(n_iv, p_iv)
    fy_lo, fy_hi = y_window(0.0, n_hi)
    us_lo = 0.5 + box_u_at(consts, fy_lo)[0]
    us_hi = 0.5 - box_u_at(consts, fy_hi)[1]

    # accept_iter = alpha / (sqrt(2*pi)*spq) = (2.83 + 5.1/b) / sqrt(2*pi):
    # spq cancels, so the worst case over the box is at b_lo.
    accept_iter = (2.83 + 5.1 / consts[2][0]) / math.sqrt(2 * math.pi)
    eps_accept, tv, n_extra = combine_tv(
        fptaylor, fp, eps_floor, eps_accept_raw, accept_iter, us_lo, us_hi, args,
        inputs_dir, outputs_dir, env)
    return eps_floor, eps_accept, tv, n_boxes + n_extra


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
                        help="Interval mode (inversion regime only): bound the "
                             "error over the whole box n in [NMIN, NMAX] x "
                             "p in [PMIN, PMAX] with one FPTaylor query "
                             "(requires --p-range)")
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
    parser.add_argument("--split-depth", type=int, default=1,
                        help="Interval mode: how many times to bisect each of "
                             "n and p at its midpoint, giving 4^DEPTH sub-boxes "
                             "(default: 1). Sampler variables (u, v) are left "
                             "to FPTaylor's own --bb-split geometric splitting "
                             "instead; only the parameters are split here.")
    parser.add_argument("--jobs", "-j", type=int, default=os.cpu_count() or 1,
                        help="Currently unused: FPTaylor queries always run "
                             "with --opt bb, which is unsafe to run "
                             "concurrently (see dist_common.fptaylor_cmd), so "
                             "BTRS queries are forced to a single worker "
                             "regardless of this value")


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

    if args.split_depth < 0:
        raise ValueError("--split-depth must be >= 0")

    # the sampler itself switches algorithm at n*p = 30, so the box has to sit
    # entirely on one side of it
    if n_lo * p_lo >= _BTRS_SWITCH:
        eps_floor, eps_accept, tv, n_boxes = _run_btrs_box(
            fptaylor, (n_lo, n_hi), (p_lo, p_hi), args, tag,
            inputs_dir, outputs_dir, env,
        )
        row.update({"regime": "btrs-interval",
                    "eps0": "nan", "eps1": "nan", "eps2": "nan",
                    "eps_floor": f"{eps_floor:.17e}",
                    "eps_accept": f"{eps_accept:.17e}", "tv": f"{tv:.17e}",
                    "n_boxes": n_boxes})
        print(f"{_box_label({'n': (n_lo, n_hi), 'p': (p_lo, p_hi)})} [BTRS]"
              f" boxes={n_boxes} eps_floor={eps_floor:.6e}"
              f" eps_accept={eps_accept:.6e} TV={tv:.6e}")
        return [row]

    if n_hi * p_hi >= _BTRS_SWITCH:
        raise ValueError(
            f"box straddles the n*p = {_BTRS_SWITCH:.0f} switch "
            f"(n*p spans [{n_lo * p_lo:.6g}, {n_hi * p_hi:.6g}]): the sampler "
            "uses inversion below it and BTRS above, so split the box there")

    box = inversion_box_params(n_lo, n_hi, p_lo, p_hi)
    eps0, eps1, eps2, tv = _run_inversion_box(
        fptaylor, box, args, tag, inputs_dir, outputs_dir, env,
    )
    row.update({"regime": "inversion-interval",
                "eps0": f"{eps0:.17e}", "eps1": f"{eps1:.17e}",
                "eps2": f"{eps2:.17e}",
                "eps_floor": "nan", "eps_accept": "nan", "tv": f"{tv:.17e}",
                "n_boxes": 1})
    print(f"{_box_label(box)} eps0={eps0:.6e} eps1={eps1:.6e}"
          f" eps2={eps2:.6e} TV={tv:.6e}")
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
                eps_floor, eps_accept, tv, n_boxes = _run_btrs_fptaylor(
                    fptaylor, n, p, args, tag, inputs_dir, outputs_dir, env,
                )
                rows.append({
                    "n": n, "p": f"{p:.17g}", "regime": "btrs",
                    "eps0": "nan", "eps1": "nan", "eps2": "nan",
                    "eps_floor":  f"{eps_floor:.17e}",
                    "eps_accept": f"{eps_accept:.17e}",
                    "tv": f"{tv:.17e}", "n_boxes": n_boxes,
                })
                print(f"n={n} p={p} [BTRS] boxes={n_boxes}"
                      f" eps_floor={eps_floor:.6e}"
                      f" eps_accept={eps_accept:.6e} TV={tv:.6e}")
            else:
                # ---- inversion regime, (FPTaylor only; CIRE not yet supported) ----
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
                    "tv": f"{tv:.17e}", "n_boxes": 1,
                })
                print(f"n={n} p={p} eps0={eps0:.6e} eps1={eps1:.6e} eps2={eps2:.6e} TV={tv:.6e}")
        except Exception as exc:
            print(f"WARNING: skipping n={n} p={p}: {exc}")

    return rows


def write_plot(rows, plot_path, plot_components=False, plot_pgf=False):
    import contextlib
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
