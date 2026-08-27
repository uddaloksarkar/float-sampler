"""Binomial sampler FP-error analysis: legacy inversion for n*p < _BTRS_SWITCH,
BTRS (Hormann transformed rejection, distributions/btrs.c) above it."""
import math
import os
import shutil
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from dist_common import (
    ROOT, FP_TO_FPTAYLOR_RND,
    run_command, extract_deltas_by_problem, extract_abs_errors_by_problem,
    loggam_defs, eps_logv, eps_logus, ulp_rnd_op,
    vprint,
    fptaylor_cmd,
    us_root, hormann_u_at, hormann_k_defs,
    point_ivar,
    floor_x_abs_tol_vars, accept_x_abs_tol_vars,
    hormann_proposal_deviation, acceptance_tv,
    elapsed_since, format_seconds,
    dist_switch,
)

NAME = "binomial"
CSV_FIELDS = ["n", "p", "n_lo", "n_hi", "p_lo", "p_hi", "regime",
              "eps0", "eps1", "eps2", "eps_floor", "eps_accept", "tv",
              "n_boxes", "time_s"]

# n*p threshold: inversion below, BTRS above -- overridable via
# fptaylor_settings.toml's [binomial].switch (dist_common.dist_switch).
_BTRS_SWITCH = dist_switch(NAME, 30.0)


def sampler_p(p):
    return 1.0 - p if p > 0.5 else p


def sampler_p_interval(p_lo, p_hi):
    vals = [sampler_p(p_lo), sampler_p(p_hi)]
    if p_lo <= 0.5 <= p_hi:
        vals.append(0.5)
    return min(vals), max(vals)


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
    """One query per FP op in legacy_random_binomial_inversion's loop
    (distributions/binomial_legacy_inversion.c): eps0=qn, eps1=px, eps2=sum+prod."""
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


_K_SLACK = 1.0            # floor can disagree by one integer: y in [k-1, k+2]


def y_window(k_lo, k_hi, slack=_K_SLACK):
    """y range covering every u mapping to a k in [k_lo, k_hi] (floor(y)=k
    means y in [k, k+1)), widened by `slack` for floor disagreement."""
    return k_lo - slack, k_hi + 1.0 + slack


def shell_u(u_lo, u_hi, y_lo, y_hi):
    """Split [u_lo, u_hi] at u=0 (us=0.5-|u| isn't 1:1 across it, and
    btrs_k_defs needs the sign); y_lo/y_hi only for the error message."""
    if u_lo > u_hi:
        raise ValueError(f"empty u window for y in [{y_lo:.6g}, {y_hi:.6g}]")
    shells = []
    if u_lo < 0.0:
        shells.append((-1, u_lo, min(u_hi, 0.0)))
    if u_hi > 0.0:
        shells.append((+1, max(u_lo, 0.0), u_hi))
    return shells


def clip_u_trunc(u_lo, u_hi, u_trunc):
    """Clip [u_lo, u_hi] so us=0.5-|u| never dips below u_trunc; return
    (u_lo, u_hi, excess), excess being the trimmed u-probability mass
    (charged to TV by the caller, like v_trunc).  A box entirely beyond the
    boundary clips to empty (u_lo > u_hi); callers must skip that query."""
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


def u_shells(n, p, y_lo, y_hi, consts=None, u_trunc=0.0):
    """shell_u at a literal (n, p); y is increasing in u, so it inverts.
    Returns (shells, excess); see clip_u_trunc."""
    consts = consts or btrs_consts(n, p)
    u_lo, u_hi, excess = clip_u_trunc(
        btrs_u_at(n, p, y_lo, consts), btrs_u_at(n, p, y_hi, consts), u_trunc)
    if u_lo > u_hi:
        return [], excess
    return shell_u(u_lo, u_hi, y_lo, y_hi), excess


def btrs_setup_defs(rnd, n_expr, p_expr, accept=False):
    """btrs.c's setup block [lines 48-51, 72-76] as FPTaylor Definitions --
    kept as expressions (not free inputs) so their own rounding is charged
    and they stay correlated with u and each other."""
    d = [f"  spq_   {rnd}= sqrt({n_expr} * {p_expr} * (1.0 - {p_expr})),",
         f"  b_     {rnd}= 1.15 + 2.53 * spq_,",
         f"  a_     {rnd}= -0.0873 + 0.0248 * b_ + 0.01 * {p_expr},",
         f"  c_     {rnd}= {n_expr} * {p_expr} + 0.5,"]
    if accept:
        d += [f"  alpha_ {rnd}= (2.83 + 5.1 / b_) * spq_,",
              f"  lpq_   {ulp_rnd_op(rnd, 'log')}= log({p_expr} / (1.0 - {p_expr})),"]
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


def _ivar(name, lo, hi, kind="real"):
    return f"  {kind} {name} in [{lo:.20e}, {hi:.20e}]"


def _template(var_lines, def_lines, expr):
    """Assemble one query; the last Variable and Definition get their ';'."""
    return ("Variables\n" + ",\n".join(var_lines) + ";\n\n"
            + "Definitions\n" + "\n".join(def_lines).rstrip(",") + ";\n\n"
            + "Expressions\n  " + expr + ";\n")


def make_floor_template(fp, var_lines, n_expr, p_expr):
    """eps_floor: absolute error of y=(2*a/us+b)*u+c, us=0.5-|u| [btrs.c:60-61]."""
    rnd = FP_TO_FPTAYLOR_RND[fp]
    floor_defs = [f"  us_    {rnd}= 0.5 - abs(u),",
                  f"  y_     {rnd}= (2.0 * a_ / us_ + b_) * u + c_,"]
    return _template(var_lines,
                     btrs_setup_defs(rnd, n_expr, p_expr) + floor_defs,
                     "eps_floor = y_")


def make_accept_template(fp, var_lines, n_expr, p_expr, mid, name_k, name_nk,
                         fast=False):
    """eps_accept: absolute error of h - loggam(k+1) - loggam(n-k+1)
    + (k-m)*lpq - log(alpha) + log(a+b*us^2) - 2*log(us) [btrs.c:85],
    excluding -log(v) (make_logv_template).  `mid` supplies the
    variant-specific Definitions between the shared setup and tail; with
    fast, -2*log(us) is dropped here for a separate (cheaper, looser)
    eps_logus query instead."""
    rnd = FP_TO_FPTAYLOR_RND[fp]
    log_rnd = ulp_rnd_op(rnd, "log")
    log_us_def = [] if fast else [f"  log_us_ {log_rnd}= log(us_),"]
    log_us_term = "" if fast else " - 2.0 * log_us_"
    return _template(
        var_lines,
        btrs_setup_defs(rnd, n_expr, p_expr, accept=True) + mid + [
            f"  us_sq_      {rnd}= us_ * us_,",
            f"  log_num_    {rnd}= a_ + b_ * us_sq_,",
            f"  log_alpha_  {log_rnd}= log(alpha_),",
            f"  log_lognum_ {log_rnd}= log(log_num_),",
        ] + log_us_def + [
            f"  btrs_accept {rnd}= h_ - {name_k} - {name_nk}"
            f" + (k_ - m_) * lpq_ - log_alpha_ + log_lognum_"
            f"{log_us_term},"],
        "eps_accept = btrs_accept")


def make_btrs_floor_template(n, p, fp, u_lo, u_hi):
    """Point form: only u varies; n, p are Variables bracketing the exact
    values (dist_common.exact_bracket), referenced by name."""
    return make_floor_template(
        fp, [_ivar("u", u_lo, u_hi), point_ivar("n", n), point_ivar("p", p)],
        "n", "p")


def _point_hm_defs(rnd):
    """m tied to (n, p) via a floor-encoding fm (m_=(n+1)*p-fm, fm in [0,1)),
    computed inside FPTaylor rather than precomputed in Python, which can't
    guarantee matching the compiled sampler's rounding bit-for-bit; h =
    lgamma(m+1)+lgamma(n-m+1) [btrs.c:77] is likewise derived from m_ inside
    the query rather than precomputed, since each lgamma() call's own error
    can be as large as the whole eps_accept bound for m in the thousands."""
    defs_hm,  name_hm  = loggam_defs("m_ + 1.0", "hm", rnd)
    defs_hnm, name_hnm = loggam_defs("n - m_ + 1.0", "hnm", rnd)
    return (["  m_     = (n + 1.0) * p - fm,"]
            + defs_hm + defs_hnm
            + [f"  h_     {rnd}= {name_hm} + {name_hnm},"])


def make_btrs_accept_template(n, p, fp, u_lo, u_hi, k_lo, k_hi, fast=False):
    """Point form. k is declared directly as a Variable over [k_lo, k_hi]
    (the full [0, n] in practice -- k1_ = k+1 and nk1_ = n-k+1 are exact
    functions of the declared k, so they stay in [1, n+1], safely > 0,
    across that whole range) rather than derived here from u via the
    y = f(u) map: k and u are only jointly reachable through the sampler's
    *exact* floor relationship, and eps_floor already bounds any
    disagreement about which k a given u floors to. Re-deriving k from u
    inside this template would needlessly propagate u's own error through
    that derivative-large map into loggam(k+1)/loggam(n-k+1), inflating
    eps_accept and double-counting what eps_floor already covers (see the
    old hormann_k_defs docstring in dist_common.py). Declaring k directly,
    and letting u range over its own full (both-signs) interval only for
    us_/log_num_, is a sound relaxation -- the same reparametrization used
    for PTRS's k (dist_poisson.py) and HRUA's W
    (dist_hypergeometric.hrua_z_defs) -- and drops the sign split entirely,
    since us_ = 0.5 - abs(u) no longer needs one."""
    rnd = FP_TO_FPTAYLOR_RND[fp]
    defs_k,  name_k  = loggam_defs("k1_",  "lgk",  rnd)
    defs_nk, name_nk = loggam_defs("nk1_", "lgnk", rnd)

    mid = (_point_hm_defs(rnd)
           + [f"  us_    {rnd}= 0.5 - abs(u),",
              "  k_     = k,",
              "  k1_    = k + 1.0,",
              "  nk1_   = n - k + 1.0,"]
           + defs_k + defs_nk)
    return make_accept_template(
        fp, [_ivar("u", u_lo, u_hi), _ivar("k", k_lo, k_hi),
             point_ivar("n", n), point_ivar("p", p),
             "  real fm in [0.0, 1.0]"],
        "n", "p", mid, name_k, name_nk, fast)


def loggam_int_defs(x, prefix, rnd):
    """Native FPTaylor lgamma for an exact integer x >= 1."""
    return loggam_defs(f"{x:.1f}", prefix, rnd)


def split_box(n_iv, p_iv, depth):
    """The 4^depth (n_iv, p_iv) sub-boxes from bisecting each axis at its
    midpoint `depth` times."""
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
    """Enclosures of btrs_consts over the box, from the monotonicity of
    each quantity (spq, b increasing in n and p*(1-p); a increasing in b, p;
    c=n*p+0.5 increasing in both)."""
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
    """Enclosure of {u : y_{n,p}(u) = y} over the box: the union of
    btrs_u_at's root at each (n, p); widest u comes from (a_lo, b_lo,
    gamma_hi) since us_root grows with a, b and shrinks with gamma=|y-c|."""
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


def box_u_shells(box_consts, y_lo, y_hi, u_trunc=0.0):
    """shell_u over a parameter box: the widest u window, split at u=0.
    Returns (shells, excess); see clip_u_trunc."""
    u_lo, u_hi, excess = clip_u_trunc(
        box_u_at(box_consts, y_lo)[0], box_u_at(box_consts, y_hi)[1], u_trunc)
    if u_lo > u_hi:
        return [], excess
    return shell_u(u_lo, u_hi, y_lo, y_hi), excess


def make_btrs_floor_box_template(box, fp):
    """Interval-parameter form: u, n and p are the only free inputs."""
    return make_floor_template(fp, [_ivar("u", *box["u"]), _ivar("n", *box["n"]),
                                    _ivar("p", *box["p"])], "n", "p")


def make_btrs_accept_box_template(box, fp, low, x_iv, fast=False):
    """Interval-parameter accept form, parametrized by whichever of k and
    j=n-k is small (low=True: x=k; low=False: x=n-k) so the box-width-sized
    argument n-x+1 lands on the large loggam and the small one stays exact.
    k is not tied to u here (unlike the point form): the caller instead
    pairs an x enclosure with the u window mapping into it, since deriving
    k live via a/us with n, p as ranges makes the k1_/nk1_ margin explode
    non-linearly with box width.  m is tied to (n,p) via m_=(n+1)*p-fm
    (safe: a product of positive ranges, unlike a/us's division)."""
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


def _run_inversion_fptaylor(fptaylor, template, label, tag, args,
                            inputs_dir, outputs_dir, env, p_bound, bound):
    """Run one inversion-regime FPTaylor query; return (eps0, eps1, eps2, tv),
    tv = 0.5*(eps0 + eps1*p_bound + eps2*bound)."""
    input_path = inputs_dir / f"binomial_inversion_{args.fp}_{tag}.txt"
    input_path.write_text(template)
    code, output = run_command(
        [fptaylor, "--rel-error", "true", str(input_path)], cwd=ROOT, env=env)
    out_path = outputs_dir / f"binomial_inversion_{args.fp}_{tag}.out"
    out_path.write_text(output)
    if args.verbose >= 2:
        print(f"--- FPTaylor binomial_inversion {label} ---\n{output}")
    if code != 0:
        raise RuntimeError(f"FPTaylor failed for {label}; see {out_path}")

    deltas = extract_deltas_by_problem(output, label)
    eps0, eps1, eps2 = deltas["eps0"], deltas["eps1"], deltas["eps2"]
    tv = 0.5 * (eps0 + eps1 * p_bound + eps2 * bound)
    return eps0, eps1, eps2, tv


def _run_inversion_box(fptaylor, box, args, tag, inputs_dir, outputs_dir, env):
    """(eps0, eps1, eps2, tv) valid for every (n, p) in the box."""
    vprint(args.verbose, f"binomial inversion box {_box_label(box)}",
           **{k: f"[{v[0]:.10g}, {v[1]:.10g}]" for k, v in box.items()})
    return _run_inversion_fptaylor(
        fptaylor, make_inversion_box_template(box, args.fp), _box_label(box),
        tag, args, inputs_dir, outputs_dir, env, box["p"][1], box["bound"][1])


def _box_label(box):
    (n_lo, n_hi), (p_lo, p_hi) = box["n"], box["p"]
    return f"n in [{n_lo:.10g}, {n_hi:.10g}] p in [{p_lo:.10g}, {p_hi:.10g}]"


def _fmt_signed(v):
    return f"{v:.6g}".replace(".", "p").replace("-", "m").replace("+", "")


def safe_box_name(n_lo, n_hi, p_lo, p_hi):
    return (f"box_n{_fmt_signed(n_lo)}_{_fmt_signed(n_hi)}"
            f"_p{_fmt_signed(p_lo)}_{_fmt_signed(p_hi)}")


def _fptaylor_max(fptaylor, boxes, expr, inputs_dir, outputs_dir, env,
                  verbose, jobs, ratio_tol, bb_eval=False, x_abs_tol=None,
                  x_abs_tol_vars=None, approx=True):
    """Run one FPTaylor query per (label, stem, template_text) box; return
    (max_error, n_boxes).  bb_eval=False compiles under a fixed filename
    regardless of --tmp-base-dir, so concurrent queries race -- jobs is
    capped to 1; bb_eval=True (interpreted, no compile step) runs `jobs`
    concurrently."""
    def one(box):
        label, stem, text = box
        in_path  = inputs_dir  / f"{stem}.txt"
        out_path = outputs_dir / f"{stem}.out"
        in_path.write_text(text)
        work = Path(tempfile.mkdtemp(prefix="fpt_", dir=outputs_dir))
        try:
            code, output = run_command(
                fptaylor_cmd(fptaylor, in_path, work, ratio_tol, bb_eval,
                            x_abs_tol, x_abs_tol_vars, approx),
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
            # FPTaylor exits 0 but reports nothing if the box leaves a
            # subexpression's domain (log/division straddling zero)
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


def _floor_shell_boxes(shells, fp, tag, template_fn):
    """(label, stem, template) triples, one per u=0 side, from `shells`;
    template_fn(u_lo, u_hi) builds the query text for one side."""
    return [(f"floor {_shell_label(sign, u_lo, u_hi)}",
            _shell_stem("binomial_btrs_floor", fp, tag, i, sign),
            template_fn(u_lo, u_hi))
            for i, (sign, u_lo, u_hi) in enumerate(shells)]


def _btrs_floor_boxes(n, p, fp, tag, consts, u_trunc=0.0):
    """One box per side of u=0 covering every k in [0, n] [btrs.c:62-64];
    --u-trunc clips each outer edge (clip_u_trunc). Returns (boxes, excess)."""
    y_lo, y_hi = y_window(0.0, float(n))
    shells, excess = u_shells(n, p, y_lo, y_hi, consts, u_trunc)
    boxes = _floor_shell_boxes(
        shells, fp, tag,
        lambda u_lo, u_hi: make_btrs_floor_template(n, p, fp, u_lo, u_hi))
    return boxes, excess


def _btrs_accept_boxes(n, p, fp, tag, consts, fast, u_trunc=0.0):
    """One box covering the accept expression for every k in [0, n], k
    declared directly (see make_btrs_accept_template).  --u-trunc clips or
    drops it; returns (boxes, excess).

    k1_ = k+1 and nk1_ = n-k+1 are exact (unrounded) functions of the
    directly-declared k, so they stay in [1, n+1] -- safely > 0 -- across
    the whole k in [0, n] range with no separate boundary-k handling
    needed (unlike the old u-derived k_, which needed padding/slack and a
    margin to keep its *enclosure* off zero; see make_btrs_accept_template).
    k no longer depends on u's sign either (declared directly, not derived
    via the y = f(u) map), so unlike the floor boxes this is a single box,
    not one per sign -- us_ = 0.5 - abs(u) is already sign-symmetric."""
    y_lo, y_hi = y_window(0.0, float(n))
    u_lo, u_hi, excess = clip_u_trunc(
        btrs_u_at(n, p, y_lo, consts), btrs_u_at(n, p, y_hi, consts), u_trunc)
    if u_lo > u_hi:
        return [], excess
    us_lo, us_hi = 0.5 - max(abs(u_lo), abs(u_hi)), 0.5 - min(abs(u_lo), abs(u_hi))
    boxes = [(f"accept k in [0, {n}] us in [{us_lo:.3e}, {us_hi:.3e}]",
             f"binomial_btrs_accept_{fp}_{tag}_main",
             make_btrs_accept_template(n, p, fp, u_lo, u_hi, 0.0, float(n),
                                       fast=fast))]
    return boxes, excess


def combine_tv(fptaylor, fp, eps_floor, accept_raw, accept_iter, us_lo, us_hi,
               u_excess, args, inputs_dir, outputs_dir, env):
    """Fold proposal, acceptance, and explicit seed exclusions into TV."""
    us_min = min(us_lo, us_hi)
    logv, n_extra = eps_logv(fptaylor, fp, args.v_trunc, inputs_dir, outputs_dir,
                             env, args.verbose, args.bb_geometric_ratio_tol,
                             args.bb_eval, args.opt_x_abs_tol, accept_x_abs_tol_vars(args),
                             args.approx)
    eps_accept = accept_raw + logv
    if args.fast:
        # the -2*log(us) query only sees us, so the smallest reachable us (a
        # superset of every shell's us range) is the right bound to pass
        logus, n_logus = eps_logus(fptaylor, fp, us_min, inputs_dir,
                                   outputs_dir, env, args.verbose,
                                   args.bb_geometric_ratio_tol,
                                   args.bb_eval, args.opt_x_abs_tol, accept_x_abs_tol_vars(args),
                                   args.approx)
        eps_accept += logus
        n_extra += n_logus
    tv = (2.0 * eps_floor * accept_iter
          + acceptance_tv(eps_accept)
          + u_excess + args.v_trunc)
    return eps_accept, tv, n_extra


def _run_floor_accept(fptaylor, floor_boxes, accept_boxes, args,
                      inputs_dir, outputs_dir, env):
    """Run one floor/accept box set through _fptaylor_max; return
    (floor_raw, accept_raw, n_floor, n_accept)."""
    floor_raw, n_floor = _fptaylor_max(
        fptaylor, floor_boxes, "eps_floor", inputs_dir, outputs_dir, env,
        args.verbose, args.jobs, args.bb_geometric_ratio_tol,
        args.bb_eval, args.opt_x_abs_tol, floor_x_abs_tol_vars(args), args.approx)
    accept_raw, n_accept = _fptaylor_max(
        fptaylor, accept_boxes, "eps_accept", inputs_dir, outputs_dir, env,
        args.verbose, args.jobs, args.bb_geometric_ratio_tol,
        args.bb_eval, args.opt_x_abs_tol, accept_x_abs_tol_vars(args), args.approx)
    return floor_raw, accept_raw, n_floor, n_accept


def _run_btrs_fptaylor(fptaylor, n, p, args, tag, inputs_dir, outputs_dir, env):
    """(eps_floor, eps_accept, tv, n_boxes) for the BTRS regime at (n, p)."""
    fp, verbose = args.fp, args.verbose
    consts = spq, a, b, c = btrs_consts(n, p)
    if a <= 0.0:
        raise ValueError(f"BTRS shape constant a = {a:.6g} <= 0 "
                         f"(n*p*q = {n * p * (1.0 - p):.6g} too small); "
                         "the reachable u range is not a single interval")
    alpha  = (2.83 + 5.1 / b) * spq
    fy_lo, fy_hi = y_window(0.0, float(n))
    u_lo, u_hi = btrs_u_at(n, p, fy_lo, consts), btrs_u_at(n, p, fy_hi, consts)
    us_lo, us_hi = 0.5 + u_lo, 0.5 - u_hi
    vprint(verbose, f"binomial BTRS n={n} p={p}",
           spq=spq, a=a, b=b, c=c, alpha=alpha,
           u_lo=u_lo, u_hi=u_hi, us_lo=us_lo, us_hi=us_hi,
           v_trunc=args.v_trunc, u_trunc=args.u_trunc)

    floor_boxes,  floor_excess  = _btrs_floor_boxes(n, p, fp, tag, consts,
                                                    args.u_trunc)
    accept_boxes, accept_excess = _btrs_accept_boxes(n, p, fp, tag, consts,
                                                      args.fast, args.u_trunc)
    # same underlying trimmed region measured two ways -- max, not sum
    u_excess = max(floor_excess, accept_excess)
    us_lo, us_hi = max(us_lo, args.u_trunc), max(us_hi, args.u_trunc)

    floor_raw, accept_raw, n_floor, n_accept = _run_floor_accept(
        fptaylor, floor_boxes, accept_boxes, args, inputs_dir, outputs_dir, env)
    eps_floor = hormann_proposal_deviation(floor_raw, a, b)

    accept_iter = alpha / (math.sqrt(2 * math.pi) * spq)
    eps_accept, tv, n_extra = combine_tv(
        fptaylor, fp, eps_floor, accept_raw, accept_iter, us_lo, us_hi,
        u_excess, args, inputs_dir, outputs_dir, env)
    vprint(verbose, f"binomial BTRS boxes n={n} p={p}",
           floor_boxes=n_floor, accept_boxes=n_accept, logv_boxes=n_extra,
           floor_raw=floor_raw, accept_raw=accept_raw)
    return eps_floor, eps_accept, tv, n_floor + n_accept + n_extra



def btrs_box_k_shells(n_iv, p_iv):
    """(low, high): enclosures covering every k in [0, n] for every n in the
    box (low encloses k, high the offset j=n-k).  Raises if the halves
    don't meet (k_mid + j_mid >= n_hi - 1); caller should bump --split-depth."""
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


def _btrs_sub_box_boxes(n_iv, p_iv, fp, tag, fast, u_trunc=0.0):
    """(floor_boxes, accept_boxes, floor_excess, accept_excess) for one
    sub-box: same coverage as the point-mode sweeps, with n, p as FPTaylor
    variables (see make_btrs_accept_box_template).  floor_excess/
    accept_excess are kept separate since they trim overlapping u-mass."""
    (n_lo, n_hi), (p_lo, p_hi) = n_iv, p_iv
    consts = btrs_box_consts(n_iv, p_iv)
    _, a, _, _ = consts
    if a[0] <= 0.0:
        raise ValueError(f"BTRS shape constant a reaches {a[0]:.6g} <= 0 at the "
                         f"low corner of n in [{n_lo:.6g}, {n_hi:.6g}], "
                         f"p in [{p_lo:.6g}, {p_hi:.6g}]")
    sub = safe_box_name(n_lo, n_hi, p_lo, p_hi)
    accept_boxes = []
    accept_excess = 0.0

    floor_shells, floor_excess = box_u_shells(consts, *y_window(0.0, n_hi), u_trunc)
    floor_boxes = _floor_shell_boxes(
        floor_shells, fp, f"{tag}_{sub}",
        lambda u_lo, u_hi: make_btrs_floor_box_template(
            {"u": (u_lo, u_hi), "n": n_iv, "p": p_iv}, fp))

    low, high = btrs_box_k_shells(n_iv, p_iv)
    for is_low, x_shells in ((True, low), (False, high)):
        side = "k" if is_low else "n-k"
        for x_lo, x_hi in x_shells:
            k_iv = (x_lo, x_hi) if is_low else (n_lo - x_hi, n_hi - x_lo)
            y_lo, y_hi = y_window(*k_iv)
            u_win, win_excess = box_u_shells(consts, y_lo, y_hi, u_trunc)
            accept_excess += win_excess
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
    return floor_boxes, accept_boxes, floor_excess, accept_excess


def _run_btrs_box(fptaylor, n_iv, p_iv, args, tag, inputs_dir, outputs_dir, env):
    """(eps_floor, eps_accept, tv, n_boxes) valid for every (n, p) in the
    box: bisected into 4^--split-depth sub-boxes, max taken over all."""
    fp, verbose = args.fp, args.verbose
    _, n_hi = n_iv

    sub_boxes = split_box(n_iv, p_iv, args.split_depth)
    vprint(verbose, f"binomial BTRS box {_box_label({'n': n_iv, 'p': p_iv})}",
           split_depth=args.split_depth, sub_boxes=len(sub_boxes),
           v_trunc=args.v_trunc, u_trunc=args.u_trunc)

    eps_floor = eps_accept_raw = floor_excess = accept_excess = 0.0
    n_boxes = 0
    for sub_n, sub_p in sub_boxes:
        floor_boxes, accept_boxes, sub_floor_excess, sub_accept_excess = \
            _btrs_sub_box_boxes(sub_n, sub_p, fp, tag, args.fast, args.u_trunc)
        if verbose >= 1:
            print(f"  sub-box {_box_label({'n': sub_n, 'p': sub_p})}"
                  f" ({len(floor_boxes)} floor + {len(accept_boxes)} accept)")

        floor_raw, accept_raw, n_floor, n_accept = _run_floor_accept(
            fptaylor, floor_boxes, accept_boxes, args, inputs_dir, outputs_dir, env)

        _, a_iv, b_iv, _ = btrs_box_consts(sub_n, sub_p)
        eps_floor = max(
            eps_floor,
            hormann_proposal_deviation(floor_raw, a_iv[0], b_iv[0]),
        )
        eps_accept_raw = max(eps_accept_raw, accept_raw)
        floor_excess = max(floor_excess, sub_floor_excess)
        accept_excess = max(accept_excess, sub_accept_excess)
        n_boxes += n_floor + n_accept

    u_excess = max(floor_excess, accept_excess)

    consts = btrs_box_consts(n_iv, p_iv)
    fy_lo, fy_hi = y_window(0.0, n_hi)
    us_lo = max(0.5 + box_u_at(consts, fy_lo)[0], args.u_trunc)
    us_hi = max(0.5 - box_u_at(consts, fy_hi)[1], args.u_trunc)

    # spq cancels in alpha/(sqrt(2pi)*spq), so the box's worst case is at b_lo
    accept_iter = (2.83 + 5.1 / consts[2][0]) / math.sqrt(2 * math.pi)
    eps_accept, tv, n_extra = combine_tv(
        fptaylor, fp, eps_floor, eps_accept_raw, accept_iter, us_lo, us_hi,
        u_excess, args, inputs_dir, outputs_dir, env)
    return eps_floor, eps_accept, tv, n_boxes + n_extra


def safe_pair_name(n, p):
    return f"n{n}_p{_fmt_signed(p)}"


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


def _btrs_fields(eps_floor, eps_accept, tv, n_boxes, regime):
    return {"regime": regime, "eps0": "nan", "eps1": "nan", "eps2": "nan",
            "eps_floor": f"{eps_floor:.17e}", "eps_accept": f"{eps_accept:.17e}",
            "tv": f"{tv:.17e}", "n_boxes": n_boxes}


def _inversion_fields(eps0, eps1, eps2, tv, n_boxes, regime):
    return {"regime": regime,
            "eps0": f"{eps0:.17e}", "eps1": f"{eps1:.17e}", "eps2": f"{eps2:.17e}",
            "eps_floor": "nan", "eps_accept": "nan",
            "tv": f"{tv:.17e}", "n_boxes": n_boxes}


def _print_btrs(label, n_boxes, eps_floor, eps_accept, tv):
    print(f"{label} [BTRS] boxes={n_boxes} eps_floor={eps_floor:.6e}"
          f" eps_accept={eps_accept:.6e} TV={tv:.6e}")


def _print_inversion(label, eps0, eps1, eps2, tv):
    print(f"{label} eps0={eps0:.6e} eps1={eps1:.6e} eps2={eps2:.6e} TV={tv:.6e}")


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

    analysis_p_lo, analysis_p_hi = sampler_p_interval(p_lo, p_hi)
    tag = safe_box_name(n_lo, n_hi, p_lo, p_hi)
    row = {"n": "", "p": "",
           "n_lo": f"{n_lo:.17g}", "n_hi": f"{n_hi:.17g}",
           "p_lo": f"{p_lo:.17g}", "p_hi": f"{p_hi:.17g}"}
    start = time.perf_counter()

    if args.split_depth < 0:
        raise ValueError("--split-depth must be >= 0")

    if n_lo * analysis_p_lo >= _BTRS_SWITCH:
        eps_floor, eps_accept, tv, n_boxes = _run_btrs_box(
            fptaylor, (n_lo, n_hi), (analysis_p_lo, analysis_p_hi), args, tag,
            inputs_dir, outputs_dir, env,
        )
        row.update(_btrs_fields(eps_floor, eps_accept, tv, n_boxes, "btrs-interval"))
        row["time_s"] = f"{elapsed_since(start):.6f}"
        _print_btrs(_box_label({"n": (n_lo, n_hi), "p": (p_lo, p_hi)}),
                   n_boxes, eps_floor, eps_accept, tv)
        return [row]

    if n_hi * analysis_p_hi >= _BTRS_SWITCH:
        raise ValueError(
            f"box straddles the n*p = {_BTRS_SWITCH:.0f} switch "
            f"(n*min(p,1-p) spans "
            f"[{n_lo * analysis_p_lo:.6g}, {n_hi * analysis_p_hi:.6g}]): the sampler "
            "uses inversion below it and BTRS above, so split the box there")

    box = inversion_box_params(n_lo, n_hi, analysis_p_lo, analysis_p_hi)
    eps0, eps1, eps2, tv = _run_inversion_box(
        fptaylor, box, args, tag, inputs_dir, outputs_dir, env,
    )
    row.update(_inversion_fields(eps0, eps1, eps2, tv, 1, "inversion-interval"))
    row["time_s"] = f"{elapsed_since(start):.6f}"
    _print_inversion(_box_label(box), eps0, eps1, eps2, tv)
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
        start = time.perf_counter()
        analysis_p = sampler_p(p)
        tag = safe_pair_name(n, p)
        try:
            base_row = {"n": n, "p": f"{p:.17g}"}
            label = f"n={n} p={p}"
            if analysis_p != p:
                label += f" (sampler uses p={analysis_p:.17g})"
            if n * analysis_p >= _BTRS_SWITCH:
                eps_floor, eps_accept, tv, n_boxes = _run_btrs_fptaylor(
                    fptaylor, n, analysis_p, args, tag, inputs_dir, outputs_dir, env,
                )
                base_row.update(_btrs_fields(eps_floor, eps_accept, tv, n_boxes, "btrs"))
                base_row["time_s"] = f"{elapsed_since(start):.6f}"
                rows.append(base_row)
                _print_btrs(label, n_boxes, eps_floor, eps_accept, tv)
            else:
                qn, z_lo, x_hi = inversion_params(n, analysis_p)
                bound = n * analysis_p + 10.0 * math.sqrt(n * analysis_p * (1.0 - analysis_p))
                vprint(args.verbose, f"binomial inversion n={n} p={p}",
                       qn=qn, z_lo=z_lo, x_hi=x_hi, bound=bound)
                eps0, eps1, eps2, tv = _run_inversion_fptaylor(
                    fptaylor, make_template(n, analysis_p, args.fp), label, tag, args,
                    inputs_dir, outputs_dir, env, analysis_p, bound)
                base_row.update(_inversion_fields(eps0, eps1, eps2, tv, 1, "inversion"))
                base_row["time_s"] = f"{elapsed_since(start):.6f}"
                rows.append(base_row)
                _print_inversion(label, eps0, eps1, eps2, tv)
        except Exception as exc:
            print(f"WARNING: skipping n={n} p={p}: {exc}")

    return rows


def write_plot(rows, plot_path, plot_components=False, plot_pgf=False):
    import contextlib
    import numpy as np

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
