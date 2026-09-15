"""
Poisson sampler FP-error analysis.
High range (lambda >= SWITCH) follows the PTRS algorithm in
distributions/random_poisson_ptrs.c and mirrors the BTRS analysis in
dist_binomial.py (eps_floor / eps_accept split, shared -log(v) and
-2*log(us) helpers, --fast flag).
"""
import math
import time
from pathlib import Path

from analyticError import FP_BETA, SWITCH, computeDeltaHighRange, computeDeltaLowRange
from dist_common import (
    ROOT, FP_TO_FPTAYLOR_RND,
    run_command, extract_abs_errors_by_problem,
    save_loglog_plot,
    loggam_defs, eps_logv, eps_logus, run_fptaylor_query,
    ulp_rnd_op,
    iv, param_ivar, interval_ivar,
    hormann_proposal_deviation, acceptance_tv,
    vprint, elapsed_since, format_seconds,
    floor_x_abs_tol_vars, accept_x_abs_tol_vars,
    analyse_param_box, max_fields, parse_range, safe_box_name, box_label,
    csv_num, fmt_num, with_param_tols,
)

NAME = "poisson"
CSV_FIELDS = ["lambda", "lambda_lo", "lambda_hi", "fp", "regime",
              "eps_floor", "eps_accept", "tv", "ref_tv", "n_boxes", "time_s"]


# ---------------------------------------------------------------------------
# PTRS FPTaylor templates  (lambda >= SWITCH)
# ---------------------------------------------------------------------------

def ptrs_consts(lam):
    """(slam, a, b, c): the setup constants random_poisson_ptrs computes once."""
    slam = math.sqrt(lam)
    b = 0.931 + 2.53 * slam
    a = -0.059 + 0.02483 * b
    return slam, a, b, lam + 0.43


def ptrs_setup_defs(rnd, lam_expr, accept=False):
    """
    random_poisson_ptrs's setup block [lines 77-82] as FPTaylor Definitions.
    These are derived from lambda, not free inputs, so writing them as
    expressions keeps them correlated and charges the rounding of the setup
    arithmetic itself.
    """
    d = [f"  slam_ {rnd}= sqrt({lam_expr}),",
         f"  b_    {rnd}= 0.931 + 2.53 * slam_,",
         f"  a_    {rnd}= -0.059 + 0.02483 * b_,",
         f"  c_    {rnd}= {lam_expr} + 0.43,"]
    if accept:
        d += [f"  ialp_ {rnd}= 1.1239 + 1.1328 / (b_ - 3.4),",
              f"  llam_ {ulp_rnd_op(rnd, 'log')}= log({lam_expr}),"]
    return d


# k1_ = k_ + 1 = (y_ - f) + 1, f in [0, 1]: k1_'s enclosure dips as low as
# k_lo - 1 at f = 1, so k_lo >= 1 is what keeps k1_ > 0 -- a hard
# domain-validity floor (loggam(k1_) needs k1_ > 0), not an approximation
# of excluded tail mass.
_K_BOUNDARY_MARGIN = 1.0

def ptrs_accept_k_range(lam, u_trunc):
    """
    k window the accept query covers: the entire domain reachable out to
    the u_trunc cutoff, not a fixed sigma window -- no excluded-tail mass
    to separately bound this way, since u_trunc already accounts for it
    (same as the floor query, see clip_u_trunc's docstring).

    y(u) = (2*a/us + b)*u + c [random_poisson_ptrs.c line 89] is strictly
    increasing in u (see hormann_u_at), so the maximum reachable k is
    floor(y) at u's own positive truncation boundary, u = 0.5 - u_trunc
    (us = u_trunc there). The minimum is clamped to _K_BOUNDARY_MARGIN: y
    goes negative well before u reaches -0.5, which is unphysical.
    """
    _, a, b, c = ptrs_consts(lam)
    u_hi = 0.5 - u_trunc
    y_hi = (2.0 * a / u_trunc + b) * u_hi + c
    k_lo, k_hi = _K_BOUNDARY_MARGIN, math.floor(y_hi)
    if k_lo >= k_hi:
        raise ValueError(f"empty accept window k in [{k_lo:.6g}, {k_hi:.6g}] "
                         f"for lambda={lam:.6g}, u_trunc={u_trunc:.3g}")
    return float(k_lo), float(k_hi)


def clip_u_trunc(u_lo, u_hi, u_trunc):
    """Clip [u_lo, u_hi] to keep us = 0.5 - |u| >= u_trunc; same role as
    dist_binomial.clip_u_trunc (same name, same purpose), but PTRS charges
    u_trunc to TV as a flat amount (see _run_ptrs_fptaylor) rather than the
    actual trimmed-mass excess BTRS computes, so unlike BTRS's version this
    one has no excess to return."""
    edge_lo = -0.5 + u_trunc
    edge_hi = 0.5 - u_trunc
    return max(u_lo, edge_lo), min(u_hi, edge_hi)


def make_ptrs_floor_template(lam, fp, utail):
    """
    FPTaylor expression for eps_floor: absolute error of
    (2*a/us + b)*u + (lambda + 0.43), with us = 0.5 - |u|
    [random_poisson_ptrs.c line 89]

    Only u is free; lambda is declared as a Variable bracketing its exact
    value (dist_common.exact_bracket) so a, b, c reference it by name
    instead of re-embedding the literal at every occurrence -- or, for an
    (lo, hi) interval, ranging over it (dist_common.param_ivar).
    """
    rnd = FP_TO_FPTAYLOR_RND[fp]
    return (
        "Variables\n"
        f"  real u in [{-(0.5 - utail):.20e}, {0.5 - utail:.20e}],\n"
        + param_ivar("lam", lam) + ";\n\n"
        + "Definitions\n"
        + "\n".join(ptrs_setup_defs(rnd, "lam")) + "\n"
        + f"  us_   {rnd}= 0.5 - abs(u),\n"
        + f"  ptrs_floor {rnd}= (2.0 * a_ / us_ + b_) * u + c_;\n\n"
        + "Expressions\n"
          "  eps_floor = ptrs_floor;\n"
    )


def make_ptrs_accept_template(lam, fp, u_lo, u_hi, k_lo, k_hi, fast=False):
    """
    FPTaylor expression for eps_accept (excluding -log(v), see
    dist_common.make_logv_template): absolute error of
    -lambda + k*log(lambda) - loggam(k+1) - log(invalpha) + log(a/us^2 + b),
    with us = 0.5 - |u|   [random_poisson_ptrs.c lines 98-99, rearranged so
    that log(v) is alone on the left-hand side].
    loggam(k+1) is FPTaylor's native lgamma(k1_) directly (loggam_defs,
    dist_common.py).
    log(a/us^2 + b) is rewritten as log(a + b*us^2) - 2*log(us) to avoid
    forming 1/us^2 directly when us is small (see dist_binomial.py).

    k is declared directly as a Variable over [k_lo, k_hi] (from
    ptrs_accept_k_range) rather than derived here from u via the y = f(u)
    map: k and u are only jointly reachable through the *exact* (unrounded)
    floor relationship the sampler enforces, and eps_floor already bounds
    any disagreement about which k a given u floors to. Re-deriving k from
    u inside this template would needlessly propagate u's own error through
    that derivative-~1e6 map into loggam(k+1), inflating eps_accept and
    double-counting what eps_floor already covers (see the old
    hormann_k_defs docstring in dist_common.py). Declaring k directly, and
    letting u range over its own full (both-signs) interval only for
    us_/log_num_, is a sound relaxation -- exactly the reparametrization
    used for hypergeometric's W (see dist_hypergeometric.hrua_z_defs) --
    and confirmed by direct comparison to give a *tighter* bound here too
    (single query, no domain errors, ~28% smaller eps_accept at
    lambda=40 than the old per-sign derivation).

    If fast is True, the -2*log(us_) term is omitted here and its error is
    computed separately (see dist_common.make_logus_template) and summed in
    by the caller. This drops u as a shared variable between the two terms,
    which may yield a more conservative (looser) overall bound.
    """
    rnd = FP_TO_FPTAYLOR_RND[fp]
    defs_k, name_k = loggam_defs("k1_", "lgk", rnd)
    log_rnd = ulp_rnd_op(rnd, "log")
    log_us_def  = "" if fast else f"  log_us_ {log_rnd}= log(us_),\n"
    log_us_term = "" if fast else " - 2.0 * log_us_"

    return (
        "Variables\n"
        f"  real u in [{u_lo:.20e}, {u_hi:.20e}],\n"
        f"  real k in [{k_lo:.20e}, {k_hi:.20e}],\n"
        + param_ivar("lam", lam) + ";\n\n"
        + "Definitions\n"
        + "\n".join(ptrs_setup_defs(rnd, "lam", accept=True)) + "\n"
        + f"  us_    {rnd}= 0.5 - abs(u),\n"
        + f"  k1_    = k + 1.0,\n"
        + "\n".join(defs_k) + "\n"
        + f"  us_sq_      {rnd}= us_ * us_,\n"
        + f"  log_num_    {rnd}= a_ + b_ * us_sq_,\n"
        + f"  log_ialp_   {log_rnd}= log(ialp_),\n"
        + f"  log_lognum_ {log_rnd}= log(log_num_),\n"
        + log_us_def
        + f"  ptrs_accept {rnd}= -lam + k * llam_ - {name_k}"
          f" - log_ialp_ + log_lognum_{log_us_term};\n\n"
        + "Expressions\n"
          "  eps_accept = ptrs_accept;\n"
    )


def _run_ptrs_fptaylor(fptaylor, lam, args, tag, inputs_dir, outputs_dir, env):
    """(eps_floor, eps_accept, tv) for the PTRS regime at lambda; composes
    the theorem 2.7 bound.

    lam may be an (lo, hi) interval: the queries then range over it, and
    the constants TV multiplies by take their worst case over it -- slam, a,
    b, c all increase with lambda, so invalpha and the Hormann factor
    1 + 3/(4a+b) are largest at lo, while the k window is widest at hi.
    """
    label = _lam_label(lam)
    lam_lo, lam_hi = iv(lam)
    fp, verbose = args.fp, args.verbose
    fast, v_trunc = args.fast, args.v_trunc
    ratio_tol, bb_eval = args.bb_geometric_ratio_tol, args.bb_eval
    x_abs_tol, approx = args.opt_x_abs_tol, args.approx
    u_trunc = args.u_trunc
    floor_tol_vars = with_param_tols(floor_x_abs_tol_vars(args), {"lam": lam})
    accept_tol_vars = with_param_tols(accept_x_abs_tol_vars(args), {"lam": lam})
    if u_trunc is None or not (0.0 < u_trunc < 0.5):
        raise ValueError("PTRS requires --u-trunc with 0 < u_trunc < 0.5")
    slam, a, b, c = ptrs_consts(lam_lo)
    invalpha = 1.1239 + 1.1328 / (b - 3.4)

    # k is declared directly over its own interval (ptrs_accept_k_range),
    # decoupled from u -- see make_ptrs_accept_template's docstring -- so u
    # no longer needs a sign-specific derivation and both sides run as one
    # query over u's full (both-signs), u_trunc-truncated range: the
    # distribution's entire domain out to that cutoff, same as the floor
    # query, with no separate tail-probability correction needed.
    k_lo, k_hi = ptrs_accept_k_range(lam_hi, u_trunc)
    u_lo, u_hi = clip_u_trunc(-0.5, 0.5, u_trunc)
    if u_lo > u_hi:
        raise ValueError(f"{label}: u-range emptied by u_trunc={u_trunc}")
    vprint(verbose, f"poisson PTRS {label}",
           slam=slam, a=a, b=b, c=c, invalpha=invalpha,
           u_lo=u_lo, u_hi=u_hi, k_lo=k_lo, k_hi=k_hi,
           v_trunc=v_trunc, u_trunc=u_trunc)

    # ---- floor ----
    floor_input  = inputs_dir  / f"poisson_ptrs_floor_{fp}_{tag}.txt"
    floor_output = outputs_dir / f"poisson_ptrs_floor_{fp}_{tag}.out"
    floor_input.write_text(make_ptrs_floor_template(lam, fp, u_trunc))

    code, output = run_fptaylor_query(fptaylor, floor_input, outputs_dir, env,
                                       ratio_tol, bb_eval, x_abs_tol, floor_tol_vars, approx)
    floor_output.write_text(output)
    if verbose >= 2:
        print(f"--- FPTaylor PTRS floor ({label}) ---\n{output}")
    if code != 0:
        raise RuntimeError(f"FPTaylor PTRS floor failed for {label}; see {floor_output}")

    floor_raw = extract_abs_errors_by_problem(output)["eps_floor"]
    eps_floor = hormann_proposal_deviation(floor_raw, a, b)

    # ---- accept ----
    accept_input  = inputs_dir  / f"poisson_ptrs_accept_{fp}_{tag}.txt"
    accept_output = outputs_dir / f"poisson_ptrs_accept_{fp}_{tag}.out"
    accept_input.write_text(
        make_ptrs_accept_template(lam, fp, u_lo, u_hi, k_lo, k_hi, fast=fast))

    code, output = run_fptaylor_query(fptaylor, accept_input, outputs_dir,
                                       env, ratio_tol, bb_eval, x_abs_tol,
                                       accept_tol_vars, approx)
    accept_output.write_text(output)
    if verbose >= 2:
        print(f"--- FPTaylor PTRS accept ({label}) ---\n{output}")
    if code != 0:
        raise RuntimeError(f"FPTaylor PTRS accept failed for "
                           f"{label}; see {accept_output}")
    accept_raw = extract_abs_errors_by_problem(output)["eps_accept"]

    logv, _ = eps_logv(
        fptaylor, fp, v_trunc, inputs_dir, outputs_dir, env, verbose, ratio_tol,
        bb_eval, x_abs_tol, accept_tol_vars, approx=approx,
    )
    eps_accept = accept_raw + logv
    if fast:
        logus, _ = eps_logus(
            fptaylor, fp, u_trunc, inputs_dir, outputs_dir, env, verbose, ratio_tol,
            bb_eval, x_abs_tol, accept_tol_vars, approx=approx,
        )
        eps_accept += logus

    tv = (2.0 * u_trunc + v_trunc
          + 2.0 * invalpha * eps_floor
          + acceptance_tv(eps_accept))
    return eps_floor, eps_accept, tv


# ---------------------------------------------------------------------------
# Low-range FPTaylor templates  (lambda < SWITCH)
# ---------------------------------------------------------------------------

def _low_range_k_star(lam):
    """Knuth's worst-case iteration count K* = lambda + 10*sqrt(lambda); for
    an (lo, hi) interval, at hi -- the product error only grows with the
    number of multiplications, so the longest chain covers every lambda."""
    lam_hi = iv(lam)[1]
    return int(lam_hi + 10 * math.sqrt(lam_hi))


def _low_range_lambda_lines(lam):
    """(Variables lines, Definitions lines) introducing `lambda`: an exact
    literal for a point (the input string, verbatim), a Variable over an
    (lo, hi) interval."""
    if isinstance(lam, tuple):
        return [interval_ivar("lambda", *lam, kind="float64")], []
    return [], [f"  lambda = {lam}"]


def _make_low_range_template(lam, fp):
    """lam: the input string (a point) or an (lo, hi) interval."""
    k_star = _low_range_k_star(lam if isinstance(lam, tuple) else float(lam))
    rnd = FP_TO_FPTAYLOR_RND[fp]
    lam_vars, lam_defs = _low_range_lambda_lines(lam)
    var_lines = [f"  real u_{i} in [0, 1]" for i in range(1, k_star + 1)] + lam_vars
    def_lines = (
        lam_defs + [f"  L = {rnd}(exp(-lambda))", f"  p_1 = {rnd}(u_1)"]
        + [f"  p_{i} = {rnd}(p_{i-1} * u_{i})" for i in range(2, k_star + 1)]
    )
    return (
        "Variables\n" + ",\n".join(var_lines) + ";\n\n"
        + "Definitions\n" + ",\n".join(def_lines) + ";\n\n"
        + "Expressions\n"
        + f"  L_compute = L;\n"
        + f"  prod_compute = p_{k_star};\n"
    )


def _make_log_low_range_template(lam, fp):
    """lam: the input string (a point) or an (lo, hi) interval."""
    k_star = _low_range_k_star(lam if isinstance(lam, tuple) else float(lam))
    rnd = FP_TO_FPTAYLOR_RND[fp]
    lam_vars, lam_defs = _low_range_lambda_lines(lam)
    lam_ref = "lambda" if isinstance(lam, tuple) else lam
    var_lines = [f"  real u_{i} in [1e-300, 1]" for i in range(1, k_star + 1)] + lam_vars
    def_lines = (
        lam_defs
        + [f"  lambda_fp {rnd}= {lam_ref}",
           f"  logp_1 = {rnd}(log(u_1))"]
        + [f"  logp_{i} = {rnd}(logp_{i-1} + {rnd}(log(u_{i})))"
           for i in range(2, k_star + 1)]
    )
    return (
        "Variables\n" + ",\n".join(var_lines) + ";\n\n"
        + "Definitions\n" + ",\n".join(def_lines) + ";\n\n"
        + "Expressions\n"
        + f"  log_prod_compute = logp_{k_star};\n"
        + f"  lambda_fp_compute = lambda_fp;\n"
    )


def _run_low_range_fptaylor(fptaylor, lam, args, tag, inputs_dir, outputs_dir, env):
    """tv for the low range at lambda -- `lam` is the input string (embedded
    verbatim as the template's literal) or an (lo, hi) interval, for which
    TV is taken at hi (the smallest e^-lambda the error is divided by).
    Also used by dist_poisson_stable."""
    label = _lam_label(lam)
    fp, verbose = args.fp, args.verbose
    lam_hi = iv(lam)[1] if isinstance(lam, tuple) else float(lam)
    vprint(verbose, f"poisson low range {label}", k_star=_low_range_k_star(lam_hi))

    lr_input  = inputs_dir  / f"low_range_{fp}_lam_{tag}.txt"
    lr_output = outputs_dir / f"low_range_{fp}_lam_{tag}.out"
    if args.use_log:
        lr_input.write_text(_make_log_low_range_template(lam, fp))
    else:
        lr_input.write_text(_make_low_range_template(lam, fp))

    code, output = run_command([fptaylor, str(lr_input)], cwd=ROOT, env=env)
    lr_output.write_text(output)
    if verbose >= 2:
        print(f"--- FPTaylor low range ({label}) ---\n{output}")
    if code != 0:
        raise RuntimeError(f"FPTaylor low range failed for {label}; see {lr_output}")

    errs = extract_abs_errors_by_problem(output)
    if args.use_log:
        missing = {"log_prod_compute", "lambda_fp_compute"} - errs.keys()
        if missing:
            raise RuntimeError(f"could not parse low-range errors for {', '.join(sorted(missing))}")
        _, tv = _compute_log_low_range_delta(
            errs["lambda_fp_compute"], errs["log_prod_compute"])
    else:
        missing = {"L_compute", "prod_compute"} - errs.keys()
        if missing:
            raise RuntimeError(f"could not parse low-range errors for {', '.join(sorted(missing))}")
        _, _, tv = _compute_low_range_delta(
            lam_hi, errs["L_compute"], errs["prod_compute"])
    return tv


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def safe_lambda_name(lam):
    return lam.replace("+", "").replace("-", "m").replace(".", "p").replace("E", "e")


def read_lambdas(path):
    lambdas = []
    for lineno, line in enumerate(path.read_text().splitlines(), start=1):
        line = line.split("#", 1)[0].replace(",", " ").strip()
        if not line:
            continue
        for token in line.split():
            try:
                lam = float(token)
            except ValueError as exc:
                raise ValueError(f"{path}:{lineno}: invalid lambda {token!r}") from exc
            if lam <= 0:
                raise ValueError(f"{path}:{lineno}: lambda must be positive")
            lambdas.append(token)
    return lambdas


def _compute_low_range_delta(lam, l_compute_error, prod_compute_error):
    l_value = math.exp(-lam)
    E = prod_compute_error + l_compute_error
    delta = math.inf if E >= l_value else 2 * E / (l_value - E)
    return l_value, E, delta


def _compute_log_low_range_delta(lambda_fp_error, log_prod_error):
    E = lambda_fp_error + log_prod_error
    return E, 2 * E


def _lam_label(lam):
    """'lambda=50' for a point, 'lambda in [30, 100]' for an interval."""
    if isinstance(lam, tuple):
        return f"lambda in [{lam[0]:.10g}, {lam[1]:.10g}]"
    return f"lambda={lam}"


def _empty_row(lam, fp):
    return {"lambda": lam, "lambda_lo": "", "lambda_hi": "", "fp": fp,
            "regime": "", "eps_floor": "", "eps_accept": "", "tv": "",
            "ref_tv": "", "n_boxes": "", "time_s": ""}


# ---------------------------------------------------------------------------
# Interval mode  (--lam-range)
# ---------------------------------------------------------------------------

_LOW_TOP = math.nextafter(SWITCH, 0.0)     # largest lambda in the low range


def _box_regimes(box):
    lo, hi = box["lam"]
    return ({"low"} if lo < SWITCH else set()) | ({"ptrs"} if hi >= SWITCH else set())


def _split_at_switch(box):
    lo, hi = box["lam"]
    return [{"lam": (lo, _LOW_TOP)}, {"lam": (SWITCH, hi)}]


def _ref_tv(lam_iv, regime, fp):
    """analyticError's reference bound at the box's endpoints (indicative
    only: the analytic formulas are not maximised over the box)."""
    ref = (computeDeltaLowRange if regime == "low"
           else lambda lam, beta: computeDeltaHighRange(lam, beta)[0])
    return max(ref(lam, FP_BETA[fp]) for lam in iv(lam_iv))


def run_box(args, fptaylor, inputs_dir, outputs_dir, env, box, ptrs_runner=None,
            ptrs_regime=("ptrs", "PTRS")):
    """One row bounding TV over every lambda in box["lam"], via
    dist_common.analyse_param_box: split at SWITCH, then the low-range and
    PTRS runners over each piece.  ptrs_runner / ptrs_regime (CSV name,
    printed tag) let dist_poisson_stable reuse this with its own PTRS
    analysis."""
    ptrs_runner = ptrs_runner or _run_ptrs_fptaylor
    start = time.perf_counter()

    def analyse(sub, regime):
        lam, tag = sub["lam"], safe_box_name(sub)
        if regime == "low":
            tv = _run_low_range_fptaylor(fptaylor, lam, args, tag, inputs_dir, outputs_dir, env)
            return {"tv": tv, "ref_tv": _ref_tv(lam, regime, args.fp)}
        eps_floor, eps_accept, tv = ptrs_runner(
            fptaylor, lam, args, tag, inputs_dir, outputs_dir, env)
        return {"eps_floor": eps_floor, "eps_accept": eps_accept, "tv": tv,
                "ref_tv": _ref_tv(lam, regime, args.fp)}

    results = analyse_param_box(box, _box_regimes, _split_at_switch, analyse,
                                args.split_depth, verbose=args.verbose)
    worst = max_fields(results, ("eps_floor", "eps_accept", "tv", "ref_tv"))
    names = {"low": ("low", "low"), "ptrs": ptrs_regime}
    regimes = sorted({r["regime"] for r in results})

    row = _empty_row("", args.fp)
    row.update({
        "lambda_lo": f"{box['lam'][0]:.17g}",
        "lambda_hi": f"{box['lam'][1]:.17g}",
        "regime": "+".join(names[r][0] for r in regimes),
        "eps_floor": csv_num(worst["eps_floor"]),
        "eps_accept": csv_num(worst["eps_accept"]),
        "tv": csv_num(worst["tv"]),
        "ref_tv": csv_num(worst["ref_tv"]),
        "n_boxes": len(results),
        "time_s": f"{elapsed_since(start):.6f}",
    })
    print(f"{box_label(box)} [{'+'.join(names[r][1] for r in regimes)}]"
          f" boxes={len(results)} eps_floor={fmt_num(worst['eps_floor'])}"
          f" eps_accept={fmt_num(worst['eps_accept'])} TV={fmt_num(worst['tv'])}"
          f" ref_TV={fmt_num(worst['ref_tv'])}"
          f" time={format_seconds(float(row['time_s']))}")
    return row


def lam_range_box(args):
    lo, hi = parse_range(args.lam_range, "--lam-range")
    if lo <= 0:
        raise ValueError("--lam-range: lambda must be positive")
    return {"lam": (lo, hi)}


# ---------------------------------------------------------------------------
# Distribution interface
# ---------------------------------------------------------------------------

def add_args(parser):
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("lambda_file", nargs="?", type=Path,
                        help="File with lambda values, one or more per line")
    source.add_argument("--lam", type=float, default=None,
                        help="Single lambda value")
    source.add_argument("--lam-range", nargs=2, type=float, default=None,
                        metavar=("LMIN", "LMAX"),
                        help="Interval mode: one TV bound valid for every "
                             "lambda in [LMIN, LMAX] (see --split-depth)")
    parser.add_argument("--use-log", action="store_true",
                        help="Use log-space template for low-range lambdas")
    parser.add_argument("--fast", action="store_true",
                        help="PTRS only: compute the -2*log(us) term of "
                             "eps_accept in a separate FPTaylor query and "
                             "sum it in, decoupling it from the shared "
                             "variable u. Faster, but may yield a more "
                             "conservative (looser) bound.")


def default_out_dir(args):
    if getattr(args, "lam_range", None) is not None:
        return ROOT / "poisson_runs_interval"
    lf = getattr(args, "lambda_file", None)
    if lf is None:
        return ROOT / "poisson_runs"
    return ROOT / f"poisson_runs_{lf.stem}"


def run(args, fptaylor, inputs_dir, outputs_dir, env):
    if args.lam_range is not None:
        return [run_box(args, fptaylor, inputs_dir, outputs_dir, env, lam_range_box(args))]
    lambdas = [str(args.lam)] if args.lam is not None else read_lambdas(args.lambda_file)

    rows = []
    for lam in lambdas:
        start = time.perf_counter()
        lam_float = float(lam)
        tag = safe_lambda_name(lam)
        try:
            row = _empty_row(lam, args.fp)

            # ---- low range ----
            if lam_float < SWITCH:
                tv = _run_low_range_fptaylor(
                    fptaylor, lam, args, tag, inputs_dir, outputs_dir, env)
                ref_tv = computeDeltaLowRange(lam_float, FP_BETA[args.fp])

                row.update({
                    "regime": "low",
                    "tv": f"{tv:.17e}",
                    "ref_tv": f"{ref_tv:.17e}",
                    "time_s": f"{elapsed_since(start):.6f}",
                })
                rows.append(row)
                print(f"lambda={lam} [low] TV={tv:.6e} ref_TV={ref_tv:.6e}"
                      f" time={format_seconds(float(row['time_s']))}")
                continue

            # ---- high range (PTRS) ----
            eps_floor, eps_accept, tv = _run_ptrs_fptaylor(
                fptaylor, lam_float, args, tag, inputs_dir, outputs_dir, env)
            ref_tv = computeDeltaHighRange(lam_float, FP_BETA[args.fp])[0]

            row.update({
                "regime": "ptrs",
                "eps_floor": f"{eps_floor:.17e}",
                "eps_accept": f"{eps_accept:.17e}",
                "tv": f"{tv:.17e}",
                "ref_tv": f"{ref_tv:.17e}",
                "time_s": f"{elapsed_since(start):.6f}",
            })
            rows.append(row)
            print(f"lambda={lam} [PTRS] eps_floor={eps_floor:.6e}"
                  f" eps_accept={eps_accept:.6e} TV={tv:.6e} ref_TV={ref_tv:.6e}"
                  f" time={format_seconds(float(row['time_s']))}")
        except Exception as exc:
            print(f"WARNING: skipping lambda={lam}: {exc}")

    return rows


def write_plot(rows, plot_path, plot_components=False, plot_pgf=False):
    rows = [r for r in rows if r["lambda"]]          # interval rows aren't points
    if not rows:
        print("Nothing to plot: interval-mode rows are not points on the lambda axis")
        return False
    points = []
    for row in rows:
        points.append((float(row["lambda"]), float(row["tv"]), float(row["ref_tv"])))
    points.sort(key=lambda r: r[0])
    xs = [r[0] for r in points]
    series = [
        ("TV (computed)", [r[1] for r in points], "^"),
        ("TV (analyticError)", [r[2] for r in points], "x"),
    ]
    save_loglog_plot(xs, series, xlabel="lambda", ylabel="error",
                     plot_path=plot_path, plot_pgf=plot_pgf, ylim_top=0.9)
