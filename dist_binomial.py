"""
Binomial sampler FP-error analysis.
High range (n*p >= _BTRS_SWITCH) follows the BTRS algorithm in
distributions/btrs.c (Hormann transformed rejection) and mirrors the PTRS
analysis in dist_poisson.py (eps_floor / eps_accept split, shared -log(v) and
-2*log(us) helpers, --fast flag). Low range (n*p < _BTRS_SWITCH) follows the
legacy inversion loop in distributions/binomial_legacy_inversion.c. Both
regimes analyse the p the sampler actually runs on, min(p, 1-p) (sampler_p).

Every runner takes n and p either as points or as (lo, hi) intervals
(interval mode, --n-range / --p-range): see dist_common's "Interval (box)
mode" section.  With p <= 1/2 every BTRS and inversion constant is monotone
in both n and p, so the worst case over a box is always one of its two
corners, (n_lo, p_lo) or (n_hi, p_hi).
"""
import math
import sys
import time
from pathlib import Path

from dist_common import (
    ROOT, FP_TO_FPTAYLOR_RND,
    run_command, extract_deltas_by_problem, extract_abs_errors_by_problem,
    loggam_defs, eps_logv, eps_logus, run_fptaylor_query,
    ulp_rnd_op,
    iv, is_point, param_ivar, interval_ivar,
    us_root, hormann_u_at, hormann_proposal_deviation, acceptance_tv,
    vprint, elapsed_since, format_seconds,
    floor_x_abs_tol_vars, accept_x_abs_tol_vars,
    dist_switch,
    BoxTooWide, analyse_param_box, max_fields, parse_range, bisect_box,
    safe_box_name, box_label, csv_num, fmt_num, with_param_tols,
    int_or_float_str,
)

NAME = "binomial"
CSV_FIELDS = ["n", "p", "n_lo", "n_hi", "p_lo", "p_hi", "fp", "regime",
              "eps0", "eps1", "eps2", "eps_floor", "eps_accept", "tv",
              "n_boxes", "time_s"]

# n*p threshold: inversion below, BTRS above -- overridable via
# fptaylor_settings.toml's [binomial].switch (dist_common.dist_switch).
_BTRS_SWITCH = dist_switch(NAME, 30.0)


# ---------------------------------------------------------------------------
# BTRS FPTaylor templates  (n*p >= _BTRS_SWITCH)
# ---------------------------------------------------------------------------

def btrs_consts(n, p):
    """(spq, a, b, c): the setup constants btrs.c computes once per (n, p).
    For interval n and/or p, each is an (lo, hi) enclosure: with p <= 1/2,
    spq, b, a and c all increase with both n and p, so the enclosure is the
    (n_lo, p_lo) and (n_hi, p_hi) corners."""
    if isinstance(n, tuple) or isinstance(p, tuple):
        (n_lo, n_hi), (p_lo, p_hi) = iv(n), iv(p)
        return tuple(zip(btrs_consts(n_lo, p_lo), btrs_consts(n_hi, p_hi)))
    spq = math.sqrt(n * p * (1.0 - p))
    b   = 1.15 + 2.53 * spq
    a   = -0.0873 + 0.0248 * b + 0.01 * p
    return spq, a, b, n * p + 0.5


def btrs_setup_defs(rnd, n_expr, p_expr, accept=False):
    """
    btrs.c's setup block [lines 48-51, 72-76] as FPTaylor Definitions.
    These are derived from (n, p), not free inputs, so writing them as
    expressions keeps them correlated and charges the rounding of the setup
    arithmetic itself.
    """
    d = [f"  spq_   {rnd}= sqrt({n_expr} * {p_expr} * (1.0 - {p_expr})),",
         f"  b_     {rnd}= 1.15 + 2.53 * spq_,",
         f"  a_     {rnd}= -0.0873 + 0.0248 * b_ + 0.01 * {p_expr},",
         f"  c_     {rnd}= {n_expr} * {p_expr} + 0.5,"]
    if accept:
        d += [f"  alpha_ {rnd}= (2.83 + 5.1 / b_) * spq_,",
              f"  lpq_   {ulp_rnd_op(rnd, 'log')}= log({p_expr} / (1.0 - {p_expr})),"]
    return d


def btrs_accept_k_range(n):
    """
    k window the accept query covers: the sampler's entire support [0, n]
    (for an interval n, [0, n_hi]).  Unlike PTRS
    (dist_poisson.ptrs_accept_k_range), no boundary margin is needed: k1_ =
    k+1 and nk1_ = n-k+1 are exact functions of the declared k, so for a
    point n they stay in [1, n+1] -- safely > 0 -- across the whole range.
    """
    return 0.0, float(iv(n)[1])


def btrs_accept_k_parts(n):
    """
    [(param, lo, hi)]: the accept queries covering every k in [0, n], for
    every n.  A point n is one query over k in [0, n].  For an interval n,
    decoupling k from n would let nk1_ = n - k + 1 reach n_lo - n_hi + 1 <= 0
    (lgamma's pole), so k is covered by two queries instead:
      ("k", 0, n_hi - 1 - n_lo): k itself small, nk1_ >= n_lo - K + 1 > 0;
      ("j", 0, n_lo):            j = n - k small, k1_ = n - j + 1 >= 1.
    Every integer k in [0, n] is in one of them (k <= K, or n - k <= n_lo),
    as long as K <= n_lo, i.e. n_hi <= 2*n_lo + 1 -- a wider box raises
    BoxTooWide and gets bisected.
    """
    n_lo, n_hi = iv(n)
    if n_lo == n_hi:
        return [("k", 0.0, float(n_lo))]
    k_top = float(n_hi - 1 - n_lo)
    if k_top > n_lo:
        raise BoxTooWide(f"n in [{n_lo}, {n_hi}] spans more than a factor 2: "
                         "the k/j accept split can't cover k in [0, n]")
    return [("k", 0.0, k_top), ("j", 0.0, float(n_lo))]


def btrs_u_at(n, p, y, consts=None):
    """The u with (2*a/us + b)*u + c = y (see dist_common.hormann_u_at)."""
    _, a, b, c = consts or btrs_consts(n, p)
    return hormann_u_at(a, b, c, y)


def box_u_at(box_consts, y):
    """
    Enclosure (lo, hi) of {u : y_{n,p}(u) = y} over a box, from btrs_consts'
    enclosures: us_root grows with a and b and shrinks with gamma = |y - c|
    (dist_common.us_root), so the extreme u's come from opposite corners.
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


def btrs_u_range(n, p, k_lo, k_hi, consts):
    """[u_lo, u_hi]: every u whose y floors into [k_lo, k_hi] (y_window),
    for every (n, p) in the box."""
    y_lo, y_hi = y_window(k_lo, k_hi)
    if isinstance(consts[0], tuple):
        return box_u_at(consts, y_lo)[0], box_u_at(consts, y_hi)[1]
    return btrs_u_at(n, p, y_lo, consts), btrs_u_at(n, p, y_hi, consts)


_K_SLACK = 1.0            # floor can disagree by one integer: y in [k-1, k+2]


def y_window(k_lo, k_hi, slack=_K_SLACK):
    """y range covering every u mapping to a k in [k_lo, k_hi] (floor(y)=k
    means y in [k, k+1)), widened by `slack` for floor disagreement."""
    return k_lo - slack, k_hi + 1.0 + slack


def clip_u_trunc(u_lo, u_hi, u_trunc):
    """Clip [u_lo, u_hi] so us=0.5-|u| never dips below u_trunc; return
    (u_lo, u_hi, excess), excess being the trimmed u-probability mass
    (charged to TV by the caller, like v_trunc).  Same role as
    dist_poisson.clip_u_trunc, but BTRS charges the actual trimmed mass
    rather than a flat 2*u_trunc.  A window entirely beyond the boundary
    clips to empty (u_lo > u_hi)."""
    excess = 0.0
    lo_edge = u_trunc - 0.5
    if u_lo < lo_edge:
        excess += min(u_hi, lo_edge) - u_lo
        u_lo = lo_edge
    hi_edge = 0.5 - u_trunc
    if u_hi > hi_edge:
        excess += u_hi - max(u_lo, hi_edge)
        u_hi = hi_edge
    return u_lo, u_hi, excess


def make_btrs_floor_template(n, p, fp, u_lo, u_hi):
    """
    FPTaylor expression for eps_floor: absolute error of
    (2*a/us + b)*u + c, with us = 0.5 - |u|   [btrs.c lines 60-61]

    Only u is free; n and p are declared as Variables bracketing their exact
    values (dist_common.exact_bracket) so a, b, c reference them by name
    instead of re-embedding the literals at every occurrence -- or, for
    intervals, ranging over them (dist_common.param_ivar).  u runs over its
    full signed range in one query: the expression is a smooth function of u
    across u = 0, so nothing here needs splitting by sign.
    """
    rnd = FP_TO_FPTAYLOR_RND[fp]
    return (
        "Variables\n"
        f"  real u in [{u_lo:.20e}, {u_hi:.20e}],\n"
        + param_ivar("n", n) + ",\n"
        + param_ivar("p", p) + ";\n\n"
        + "Definitions\n"
        + "\n".join(btrs_setup_defs(rnd, "n", "p")) + "\n"
        + f"  us_    {rnd}= 0.5 - abs(u),\n"
        + f"  btrs_floor {rnd}= (2.0 * a_ / us_ + b_) * u + c_;\n\n"
        + "Expressions\n"
          "  eps_floor = btrs_floor;\n"
    )


def btrs_m_range(n, p):
    """
    Box-mode enclosure [m_lo, m_hi] of BTRS's floor-encoded m
    (m_ = (n+1)*p - fm, fm in [0, 1); see make_btrs_accept_template) over
    the box: (n+1)*p is increasing in both n and p (p > 0), so the corners
    give (n_lo+1)*p_lo and (n_hi+1)*p_hi; widened by 1 below for the floor
    itself (m <= (n+1)*p < m+1) and a further relative 1e-9 pad for this
    function's own float64 rounding.

    Raises BoxTooWide if the box is wide enough that this enclosure's
    upper end would reach n_lo + 1 -- the point at which n - m_ + 1 (hnm's
    lgamma argument) could go non-positive for some point in the box.
    """
    n_lo, n_hi = iv(n)
    p_lo, p_hi = iv(p)
    lo = (n_lo + 1.0) * p_lo - 1.0
    hi = (n_hi + 1.0) * p_hi
    pad = 1e-9 * max(abs(lo), abs(hi), 1.0)
    lo, hi = lo - pad, hi + pad
    if n_lo - hi + 1.0 <= 0.0:
        raise BoxTooWide(f"n in [{n_lo}, {n_hi}] p in [{p_lo}, {p_hi}]: "
                         "m's box-wide range reaches n_lo -- n - m_ + 1 "
                         "(hnm's lgamma argument) would go non-positive")
    return lo, hi


def make_btrs_accept_template(n, p, fp, u_lo, u_hi, k_lo, k_hi, fast=False,
                              param="k"):
    """
    FPTaylor expression for eps_accept (excluding -log(v), see
    dist_common.make_logv_template): absolute error of
    h - loggam(k+1) - loggam(n-k+1) + (k-m)*lpq - log(alpha)
    + log(a/us^2 + b), with us = 0.5 - |u|   [btrs.c line 85, rearranged so
    that log(v) is alone on the left-hand side].
    loggam is FPTaylor's native lgamma directly (loggam_defs, dist_common.py).
    log(a/us^2 + b) is rewritten as log(a + b*us^2) - 2*log(us) to avoid
    forming 1/us^2 directly when us is small.

    m is tied to (n, p) via a floor encoding fm (m_ = (n+1)*p - fm, fm in
    [0, 1)); h = lgamma(m+1) + lgamma(n-m+1) [btrs.c line 77] is derived
    from m_ inside the query. For a point (n, p), m_ is computed inside
    FPTaylor rather than precomputed in Python, which can't guarantee
    matching the compiled sampler's rounding bit-for-bit -- exact, and
    fast, since m_ = (n+1)*p - fm is then linear in fm alone.

    For a box, (n+1)*p is a genuine product of two ranging Variables, and
    feeding that bilinear term through *two* lgamma calls (h_'s own
    anti-correlated pair, since m_+1 and n-m_+1 always sum to n+2) measured
    at >90s without converging. Declaring m_ directly as its own Variable
    over a Python-computed enclosure (btrs_m_range) instead -- the same
    reparametrization already used for k below -- removes that bilinear
    coupling (n - m_ + 1 becomes linear in two independent Variables) and
    measured at ~18s for the same query. This is a sound relaxation for the
    same reason k's is: m_ is a *deterministic* function of (n, p), not a
    per-draw random quantity needing its own error term, so its box-wide
    enclosure can be computed once, outside FPTaylor, like any other
    derived-constant enclosure in this file.

    k is declared directly as a Variable over [k_lo, k_hi] (from
    btrs_accept_k_range) rather than derived here from u via the y = f(u)
    map: k and u are only jointly reachable through the sampler's *exact*
    floor relationship, and eps_floor already bounds any disagreement about
    which k a given u floors to. Re-deriving k from u inside this template
    would needlessly propagate u's own error through that derivative-large
    map into loggam(k+1)/loggam(n-k+1), inflating eps_accept and
    double-counting what eps_floor already covers. Declaring k directly, and
    letting u range over its own full (both-signs) interval only for
    us_/log_num_, is a sound relaxation -- the same reparametrization used
    for PTRS's k (dist_poisson.make_ptrs_accept_template) and HRUA's Z
    (dist_hypergeometric.hrua_z_defs).

    param="j" declares j = n - k over [k_lo, k_hi] instead (the high half of
    an interval n's support, see btrs_accept_k_parts).

    If fast is True, the -2*log(us_) term is omitted here and its error is
    computed separately (see dist_common.make_logus_template) and summed in
    by the caller. This drops u as a shared variable between the two terms,
    which may yield a more conservative (looser) overall bound.
    """
    rnd = FP_TO_FPTAYLOR_RND[fp]
    defs_hm,  name_hm  = loggam_defs("m_ + 1.0", "hm", rnd)
    defs_hnm, name_hnm = loggam_defs("n - m_ + 1.0", "hnm", rnd)
    defs_k,   name_k   = loggam_defs("k1_",  "lgk",  rnd)
    defs_nk,  name_nk  = loggam_defs("nk1_", "lgnk", rnd)
    log_rnd = ulp_rnd_op(rnd, "log")
    log_us_def  = "" if fast else f"  log_us_ {log_rnd}= log(us_),\n"
    log_us_term = "" if fast else " - 2.0 * log_us_"
    if param == "k":
        k_defs, k_ref = "  k1_    = k + 1.0,\n  nk1_   = n - k + 1.0,\n", "k"
    else:
        k_defs, k_ref = ("  k_     = n - j,\n  k1_    = k_ + 1.0,\n"
                         "  nk1_   = j + 1.0,\n"), "k_"

    if not is_point(n) or not is_point(p):
        m_var_line = interval_ivar("m_", *btrs_m_range(n, p), kind="real")
        m_def_line = ""
    else:
        m_var_line = "  real fm in [0.0, 1.0]"
        m_def_line = "  m_     = (n + 1.0) * p - fm,\n"

    return (
        "Variables\n"
        f"  real u in [{u_lo:.20e}, {u_hi:.20e}],\n"
        f"  real {param} in [{k_lo:.20e}, {k_hi:.20e}],\n"
        + param_ivar("n", n) + ",\n"
        + param_ivar("p", p) + ",\n"
        + m_var_line + ";\n\n"
        + "Definitions\n"
        + "\n".join(btrs_setup_defs(rnd, "n", "p", accept=True)) + "\n"
        + m_def_line
        + "\n".join(defs_hm + defs_hnm) + "\n"
        + f"  h_     {rnd}= {name_hm} + {name_hnm},\n"
        + f"  us_    {rnd}= 0.5 - abs(u),\n"
        + k_defs
        + "\n".join(defs_k + defs_nk) + "\n"
        + f"  us_sq_      {rnd}= us_ * us_,\n"
        + f"  log_num_    {rnd}= a_ + b_ * us_sq_,\n"
        + f"  log_alpha_  {log_rnd}= log(alpha_),\n"
        + f"  log_lognum_ {log_rnd}= log(log_num_),\n"
        + log_us_def
        + f"  btrs_accept {rnd}= h_ - {name_k} - {name_nk}"
          f" + ({k_ref} - m_) * lpq_ - log_alpha_ + log_lognum_{log_us_term};\n\n"
        + "Expressions\n"
          "  eps_accept = btrs_accept;\n"
    )


def _alias_tol(tol_vars, src, dst):
    """tol_vars with dst given src's tolerance, if src has one and dst none."""
    parts = dict(kv.split("=", 1) for kv in (tol_vars or "").split(",") if "=" in kv)
    if src in parts and dst not in parts:
        return f"{tol_vars},{dst}={parts[src]}"
    return tol_vars


def _run_btrs_fptaylor(fptaylor, n, p, args, tag, inputs_dir, outputs_dir, env):
    """(eps_floor, eps_accept, tv) for the BTRS regime at (n, p).

    n and/or p may be (lo, hi) intervals.  The Hormann factor
    1 + 3/(4a+b), accept_iter and the a > 0 check are worst at the
    (n_lo, p_lo) corner, where a and b are smallest; the u window is the
    union over the box (box_u_at)."""
    label = _np_label(n, p)
    (n_lo, n_hi), (p_lo, p_hi) = iv(n), iv(p)
    fp, verbose = args.fp, args.verbose
    fast, v_trunc = args.fast, args.v_trunc
    ratio_tol, bb_eval = args.bb_geometric_ratio_tol, args.bb_eval
    x_abs_tol, approx = args.opt_x_abs_tol, args.approx
    u_trunc = args.u_trunc
    floor_tol_vars = with_param_tols(floor_x_abs_tol_vars(args), {"n": n, "p": p})
    accept_tol_vars = with_param_tols(accept_x_abs_tol_vars(args), {"n": n, "p": p})
    if u_trunc is None or not (0.0 <= u_trunc < 0.5):
        raise ValueError("BTRS requires --u-trunc with 0 <= u_trunc < 0.5")
    consts = btrs_consts(n, p)
    spq, a, b, c = btrs_consts(n_lo, p_lo)
    if a <= 0.0:
        raise ValueError(f"BTRS shape constant a = {a:.6g} <= 0 "
                         f"(n*p*q = {n_lo * p_lo * (1.0 - p_lo):.6g} too small); "
                         "the reachable u range is not a single interval")
    alpha = (2.83 + 5.1 / b) * spq
    accept_iter = alpha / (math.sqrt(2 * math.pi) * spq)

    # k is declared directly over its own interval (btrs_accept_k_range),
    # decoupled from u -- see make_btrs_accept_template's docstring -- and
    # both queries run over the one u window reaching every k in it (the
    # sampler's whole support), clipped by u_trunc; the clipped-off mass is
    # charged to TV as u_excess.
    k_lo, k_hi = btrs_accept_k_range(n)
    u_lo, u_hi = btrs_u_range(n, p, k_lo, k_hi, consts)
    us_min = max(min(0.5 + u_lo, 0.5 - u_hi), u_trunc)   # smallest reachable us
    u_lo, u_hi, u_excess = clip_u_trunc(u_lo, u_hi, u_trunc)
    if u_lo > u_hi:
        raise ValueError(f"{label}: u-range emptied by u_trunc={u_trunc}")
    k_parts = btrs_accept_k_parts(n)
    vprint(verbose, f"binomial BTRS {label}",
           spq=spq, a=a, b=b, c=c, alpha=alpha,
           u_lo=u_lo, u_hi=u_hi, k_lo=k_lo, k_hi=k_hi, us_min=us_min,
           u_excess=u_excess, v_trunc=v_trunc, u_trunc=u_trunc,
           **({} if len(k_parts) == 1 else {"k_parts": k_parts}))

    # ---- floor ----
    floor_input  = inputs_dir  / f"binomial_btrs_floor_{fp}_{tag}.txt"
    floor_output = outputs_dir / f"binomial_btrs_floor_{fp}_{tag}.out"
    floor_input.write_text(make_btrs_floor_template(n, p, fp, u_lo, u_hi))

    code, output = run_fptaylor_query(fptaylor, floor_input, outputs_dir, env,
                                       ratio_tol, bb_eval, x_abs_tol, floor_tol_vars, approx)
    floor_output.write_text(output)
    if verbose >= 2:
        print(f"--- FPTaylor BTRS floor ({label}) ---\n{output}")
    if code != 0:
        raise RuntimeError(f"FPTaylor BTRS floor failed for {label}; see {floor_output}")

    floor_raw = extract_abs_errors_by_problem(output)["eps_floor"]
    eps_floor = hormann_proposal_deviation(floor_raw, a, b)

    # ---- accept: max over the k parts (one for a point n) ----
    accept_raw = 0.0
    for param, x_lo, x_hi in k_parts:
        suffix = "" if len(k_parts) == 1 else f"_{param}"
        accept_input  = inputs_dir  / f"binomial_btrs_accept_{fp}_{tag}{suffix}.txt"
        accept_output = outputs_dir / f"binomial_btrs_accept_{fp}_{tag}{suffix}.out"
        accept_input.write_text(
            make_btrs_accept_template(n, p, fp, u_lo, u_hi, x_lo, x_hi,
                                      fast=fast, param=param))

        code, output = run_fptaylor_query(fptaylor, accept_input, outputs_dir,
                                           env, ratio_tol, bb_eval, x_abs_tol,
                                           _alias_tol(accept_tol_vars, "k", param),
                                           approx)
        accept_output.write_text(output)
        if verbose >= 2:
            print(f"--- FPTaylor BTRS accept{suffix} ({label}) ---\n{output}")
        if code != 0:
            raise RuntimeError(f"FPTaylor BTRS accept{suffix} failed for "
                               f"{label}; see {accept_output}")
        accept_raw = max(accept_raw,
                         extract_abs_errors_by_problem(output)["eps_accept"])

    logv, _ = eps_logv(
        fptaylor, fp, v_trunc, inputs_dir, outputs_dir, env, verbose, ratio_tol,
        bb_eval, x_abs_tol, accept_tol_vars, approx=approx,
    )
    eps_accept = accept_raw + logv
    if fast:
        # the -2*log(us) query only sees us, so the smallest reachable us is
        # the right lower bound to pass
        logus, _ = eps_logus(
            fptaylor, fp, us_min, inputs_dir, outputs_dir, env, verbose, ratio_tol,
            bb_eval, x_abs_tol, accept_tol_vars, approx=approx,
        )
        eps_accept += logus

    tv = (2.0 * eps_floor * accept_iter
          + acceptance_tv(eps_accept)
          + u_excess + v_trunc)
    return eps_floor, eps_accept, tv


# ---------------------------------------------------------------------------
# Inversion FPTaylor template  (n*p < _BTRS_SWITCH)
# ---------------------------------------------------------------------------

def inversion_params(n, p):
    """(qn, z_lo, x_hi): the interval bounds the inversion template uses.
    For a box, evaluate at its (n_hi, p_hi) corner: qn and z_lo shrink and
    x_hi grows with both n and p (p <= 1/2), so that corner's ranges
    contain every other point's."""
    q = 1.0 - p
    qn_raw = math.exp(n * math.log(q))
    qn = max(qn_raw, sys.float_info.min)
    z_lo = max(min(qn_raw, math.exp(-22) / math.sqrt(2 * math.pi * n * p * q)),
               sys.float_info.min)
    x_hi = min(float(n), n * p + 10.0 * math.sqrt(n * p * q))
    return qn, z_lo, x_hi


def _make_inversion_template(n, p, fp):
    """One query per FP op in legacy_random_binomial_inversion's loop
    (distributions/binomial_legacy_inversion.c): eps0=qn, eps1=px, eps2=sum+prod.
    n and p are exact literals for a point, Variables over intervals."""
    (n_lo, n_hi), (p_lo, p_hi) = iv(n), iv(p)
    qn, z_lo, x_hi = inversion_params(n_hi, p_hi)
    rnd = FP_TO_FPTAYLOR_RND[fp]
    np_vars, np_defs = "", ""
    if n_lo == n_hi:
        np_defs += f"  n = {float(n):.1f},\n"
    else:
        np_vars += ",\n" + interval_ivar("n", n_lo, n_hi, kind="float64")
    if p_lo == p_hi:
        np_defs += f"  p = {p:.20e},\n"
    else:
        np_vars += ",\n" + interval_ivar("p", p_lo, p_hi, kind="float64")

    return (
        "Variables\n"
        f"  real z in [{z_lo:.20e}, 1.0],\n"
        f"  real X in [1.0, {x_hi:.1f}],\n"
        f"  real sum in [{qn:.20e}, 1.0],\n"
        f"  real prod in [0.0, 1.0]{np_vars};\n\n"
        + "Definitions\n"
        + np_defs
        + f"  q = 1.0 - p,\n"
        f"  qn_step  {rnd}= exp(n * log(q)),\n"
        f"  px_step  {rnd}= z * (n - X + 1) * p / (X * q),\n"
        f"  sum_step {rnd}= sum + prod;\n\n"
        + "Expressions\n"
        f"  eps0 = qn_step;\n"
        f"  eps1 = px_step;\n"
        f"  eps2 = sum_step;\n"
    )


def _compute_inversion_tv(n, p, eps0, eps1, eps2):
    """tv from the inversion loop's per-op relative errors: eps1 is charged
    per step on p, eps2 on up to `bound` summation steps (both increasing in
    n and p, so a box passes its (n_hi, p_hi) corner)."""
    bound = n * p + 10.0 * math.sqrt(n * p * (1.0 - p))
    return 0.5 * (eps0 + eps1 * p + eps2 * bound)


def _run_inversion_fptaylor(fptaylor, n, p, args, tag, inputs_dir, outputs_dir, env):
    """(eps0, eps1, eps2, tv) for the inversion regime at (n, p) -- points or
    (lo, hi) intervals."""
    label = _np_label(n, p)
    (n_lo, n_hi), (p_lo, p_hi) = iv(n), iv(p)
    fp, verbose = args.fp, args.verbose
    qn, z_lo, x_hi = inversion_params(n_hi, p_hi)
    if x_hi > n_lo:
        # n - X + 1 must stay > 0 for every (n, X) the box pairs up
        raise BoxTooWide(f"{label}: inversion's X range reaches n_lo")
    vprint(verbose, f"binomial inversion {label}", qn=qn, z_lo=z_lo, x_hi=x_hi)

    inv_input  = inputs_dir  / f"binomial_inversion_{fp}_{tag}.txt"
    inv_output = outputs_dir / f"binomial_inversion_{fp}_{tag}.out"
    inv_input.write_text(_make_inversion_template(n, p, fp))

    code, output = run_command(
        [fptaylor, "--rel-error", "true", str(inv_input)], cwd=ROOT, env=env)
    inv_output.write_text(output)
    if verbose >= 2:
        print(f"--- FPTaylor inversion ({label}) ---\n{output}")
    if code != 0:
        raise RuntimeError(f"FPTaylor inversion failed for {label}; see {inv_output}")

    deltas = extract_deltas_by_problem(output, label)
    eps0, eps1, eps2 = deltas["eps0"], deltas["eps1"], deltas["eps2"]
    return eps0, eps1, eps2, _compute_inversion_tv(n_hi, p_hi, eps0, eps1, eps2)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def sampler_p(p):
    """The p the sampler actually runs on: btrs.c / the inversion loop both
    draw with min(p, 1-p) and reflect the result."""
    return 1.0 - p if p > 0.5 else p


def sampler_p_interval(p_lo, p_hi):
    """sampler_p's image of [p_lo, p_hi]: the reflection folds it at 1/2."""
    vals = [sampler_p(p_lo), sampler_p(p_hi)]
    if p_lo <= 0.5 <= p_hi:
        vals.append(0.5)
    return min(vals), max(vals)


def _use_btrs(n, p):
    """BTRS above the n*p switch, inversion below; p is sampler_p(p)."""
    return n * p >= _BTRS_SWITCH


def _np_label(n, p):
    if isinstance(n, tuple) or isinstance(p, tuple):
        return box_label({"n": n, "p": p})
    return f"n={n} p={p}"


def _fmt_signed(v):
    return f"{v:.6g}".replace(".", "p").replace("-", "m").replace("+", "")


def safe_pair_name(n, p):
    return f"n{n}_p{_fmt_signed(p)}"


def _validate(n, p, loc=""):
    prefix = f"{loc}: " if loc else ""
    if n <= 0:
        raise ValueError(f"{prefix}n must be positive")
    if not (0 < p < 1):
        raise ValueError(f"{prefix}p must be in (0, 1)")


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
        _validate(n, p, f"{path}:{lineno}")
        pairs.append((n, p))
    return pairs


def _empty_row(n, p, fp):
    return {"n": n, "p": "" if p == "" else f"{p:.17g}",
            "n_lo": "", "n_hi": "", "p_lo": "", "p_hi": "", "fp": fp,
            "regime": "", "eps0": "", "eps1": "", "eps2": "",
            "eps_floor": "", "eps_accept": "", "tv": "", "n_boxes": "",
            "time_s": ""}


# ---------------------------------------------------------------------------
# Interval mode  (--n-range / --p-range)
# ---------------------------------------------------------------------------

_REGIME_TAGS = {"inversion": "inversion", "btrs": "BTRS"}


def _box_regimes(box):
    """Regimes a box of (n, sampler p) touches: n*p is increasing in both."""
    (n_lo, n_hi), (p_lo, p_hi) = box["n"], box["p"]
    return (({"inversion"} if n_lo * p_lo < _BTRS_SWITCH else set())
            | ({"btrs"} if n_hi * p_hi >= _BTRS_SWITCH else set()))


def _split_at_switch(box):
    """Split towards the n*p = _BTRS_SWITCH hyperbola: exactly at it when one
    axis is a point, else by bisecting both axes."""
    (n_lo, n_hi), (p_lo, p_hi) = box["n"], box["p"]
    if p_lo == p_hi:
        n_star = math.ceil(_BTRS_SWITCH / p_lo)
        if n_lo < n_star <= n_hi:
            return [dict(box, n=(n_lo, n_star - 1)), dict(box, n=(n_star, n_hi))]
    if n_lo == n_hi:
        p_star = _BTRS_SWITCH / n_lo
        if p_lo < p_star <= p_hi:
            return [dict(box, p=(p_lo, math.nextafter(p_star, 0.0))),
                    dict(box, p=(p_star, p_hi))]
    return bisect_box(box, integer_axes=("n",))


def run_box(args, fptaylor, inputs_dir, outputs_dir, env, n_iv, p_iv):
    """One row bounding TV over every (n, p) in n_iv x p_iv."""
    start = time.perf_counter()
    box = {"n": n_iv, "p": sampler_p_interval(*p_iv)}

    def analyse(sub, regime):
        n, p, tag = sub["n"], sub["p"], safe_box_name(sub)
        if regime == "inversion":
            eps0, eps1, eps2, tv = _run_inversion_fptaylor(
                fptaylor, n, p, args, tag, inputs_dir, outputs_dir, env)
            return {"eps0": eps0, "eps1": eps1, "eps2": eps2, "tv": tv}
        eps_floor, eps_accept, tv = _run_btrs_fptaylor(
            fptaylor, n, p, args, tag, inputs_dir, outputs_dir, env)
        return {"eps_floor": eps_floor, "eps_accept": eps_accept, "tv": tv}

    fields = ("eps0", "eps1", "eps2", "eps_floor", "eps_accept", "tv")
    results = analyse_param_box(box, _box_regimes, _split_at_switch, analyse,
                                args.split_depth, integer_axes=("n",),
                                verbose=args.verbose)
    worst = max_fields(results, fields)
    regimes = sorted({r["regime"] for r in results})

    row = _empty_row("", "", args.fp)
    row.update({"n_lo": n_iv[0], "n_hi": n_iv[1],
                "p_lo": f"{p_iv[0]:.17g}", "p_hi": f"{p_iv[1]:.17g}",
                "regime": "+".join(regimes), "n_boxes": len(results),
                "time_s": f"{elapsed_since(start):.6f}"})
    row.update({f: csv_num(worst[f]) for f in fields})
    label = box_label({"n": n_iv, "p": p_iv})
    if box["p"] != p_iv:
        label += f" (sampler uses p in [{box['p'][0]:.10g}, {box['p'][1]:.10g}])"
    print(f"{label} [{'+'.join(_REGIME_TAGS[r] for r in regimes)}]"
          f" boxes={len(results)} eps0={fmt_num(worst['eps0'])}"
          f" eps1={fmt_num(worst['eps1'])} eps2={fmt_num(worst['eps2'])}"
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
                        help="File with (n, p) pairs, one per line (format: 'n p')")
    source.add_argument("--n", type=int, default=None,
                        help="Single n value (requires --p or --p-range)")
    source.add_argument("--n-range", nargs=2, type=int_or_float_str, default=None,
                        metavar=("NMIN", "NMAX"),
                        help="Interval mode: one TV bound valid for every "
                             "integer n in [NMIN, NMAX] (with --p or --p-range)")
    p_source = parser.add_mutually_exclusive_group()
    p_source.add_argument("--p", type=float, default=None,
                          help="Probability p in (0,1), required with --n")
    p_source.add_argument("--p-range", nargs=2, type=float, default=None,
                          metavar=("PMIN", "PMAX"),
                          help="Interval mode: every p in [PMIN, PMAX] "
                               "(with --n or --n-range)")
    parser.add_argument("--fast", action="store_true",
                        help="BTRS only: compute the -2*log(us) term of "
                             "eps_accept in a separate FPTaylor query and "
                             "sum it in, decoupling it from the shared "
                             "variable u. Faster, but may yield a more "
                             "conservative (looser) bound.")


def default_out_dir(args):
    if getattr(args, "n_range", None) is not None or getattr(args, "p_range", None) is not None:
        return ROOT / "binomial_runs_interval"
    lf = getattr(args, "input_file", None)
    if lf is None:
        return ROOT / "binomial_runs"
    return ROOT / f"binomial_runs_{lf.stem}"


def run(args, fptaylor, inputs_dir, outputs_dir, env):
    if args.n_range is not None or args.p_range is not None:
        if args.input_file is not None:
            raise ValueError("--p-range needs --n or --n-range, not an input file")
        n_iv = (parse_range(args.n_range, "--n-range", lo_min=1, integer=True)
                if args.n_range is not None else (args.n, args.n))
        p_iv = (parse_range(args.p_range, "--p-range")
                if args.p_range is not None else (args.p, args.p))
        if None in n_iv or None in p_iv:
            raise ValueError("interval mode needs n (--n/--n-range) and p (--p/--p-range)")
        for n, p in ((n_iv[0], p_iv[0]), (n_iv[1], p_iv[1])):
            _validate(n, p, "interval mode")
        return [run_box(args, fptaylor, inputs_dir, outputs_dir, env, n_iv, p_iv)]

    if args.n is not None:
        if args.p is None:
            raise ValueError("--p is required when --n is given")
        _validate(args.n, args.p)
        pairs = [(args.n, args.p)]
    else:
        pairs = read_np_pairs(args.input_file)
    if not pairs:
        raise ValueError("no (n, p) pairs found in input")

    rows = []
    for n, p in pairs:
        start = time.perf_counter()
        analysis_p = sampler_p(p)
        tag = safe_pair_name(n, p)
        label = f"n={n} p={p}"
        if analysis_p != p:
            label += f" (sampler uses p={analysis_p:.17g})"
        try:
            row = _empty_row(n, p, args.fp)

            # ---- low range (inversion) ----
            if not _use_btrs(n, analysis_p):
                eps0, eps1, eps2, tv = _run_inversion_fptaylor(
                    fptaylor, n, analysis_p, args, tag, inputs_dir, outputs_dir, env)

                row.update({
                    "regime": "inversion",
                    "eps0": f"{eps0:.17e}",
                    "eps1": f"{eps1:.17e}",
                    "eps2": f"{eps2:.17e}",
                    "tv": f"{tv:.17e}",
                    "time_s": f"{elapsed_since(start):.6f}",
                })
                rows.append(row)
                print(f"{label} [inversion] eps0={eps0:.6e} eps1={eps1:.6e}"
                      f" eps2={eps2:.6e} TV={tv:.6e}"
                      f" time={format_seconds(float(row['time_s']))}")
                continue

            # ---- high range (BTRS) ----
            eps_floor, eps_accept, tv = _run_btrs_fptaylor(
                fptaylor, n, analysis_p, args, tag, inputs_dir, outputs_dir, env)

            row.update({
                "regime": "btrs",
                "eps_floor": f"{eps_floor:.17e}",
                "eps_accept": f"{eps_accept:.17e}",
                "tv": f"{tv:.17e}",
                "time_s": f"{elapsed_since(start):.6f}",
            })
            rows.append(row)
            print(f"{label} [BTRS] eps_floor={eps_floor:.6e}"
                  f" eps_accept={eps_accept:.6e} TV={tv:.6e}"
                  f" time={format_seconds(float(row['time_s']))}")
        except Exception as exc:
            print(f"WARNING: skipping n={n} p={p}: {exc}")

    return rows


def write_plot(rows, plot_path, plot_components=False, plot_pgf=False):
    import contextlib
    import os
    import numpy as np

    rows = [r for r in rows if r["n"] not in (None, "")]   # interval rows aren't points
    if not rows:
        print("Nothing to plot: interval-mode rows are not points on the (n, np) grid")
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
            if not r[key]:           # field not produced by this row's regime
                continue
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
