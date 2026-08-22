"""
Poisson sampler FP-error analysis.
High range (lambda >= SWITCH) follows the PTRS algorithm in
distributions/random_poisson_ptrs.c and mirrors the BTRS analysis in
dist_binomial.py (eps_floor / eps_accept split, shared -log(v) and
-2*log(us) helpers, --fast flag).
"""
import math
import shutil
import tempfile
from pathlib import Path

from analyticError import FP_BETA, SWITCH, computeDeltaHighRange, computeDeltaLowRange
from dist_common import (
    ROOT, FP_TO_FPTAYLOR_RND,
    run_command, extract_abs_errors_by_problem,
    save_loglog_plot,
    loggam_defs, eps_logv, eps_logus, fptaylor_cmd,
    hormann_u_at, hormann_k_defs,
)

NAME = "poisson"
CSV_FIELDS = ["lambda", "fp", "regime", "eps_floor", "eps_accept", "tv", "ref_tv"]


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
              f"  llam_ {rnd}= log({lam_expr}),"]
    return d


def ptrs_k_defs(sign, lam_expr):
    """k = floor(y) for one sign of u (see dist_common.hormann_k_defs)."""
    return hormann_k_defs(
        sign,
        [f"  slamx_ = sqrt({lam_expr}),",
         f"  bx_    = 0.931 + 2.53 * slamx_,",
         f"  ax_    = -0.059 + 0.02483 * bx_,",
         f"  cx_    = {lam_expr} + 0.43,"],
        "cx_")


# k = floor(y) is modelled as y - f with f in [0, 1], so the window's lower end
# has to be 7 for k + 1 > 7 (random_loggam's Stirling branch needs x >= 7).
# 7 is the exact limit of that branch, not a margin -- the previous 8 charged
# a whole extra k of Poisson mass as k_tail for nothing.
_K_STIRLING_LO = 7.0
_K_SIGMAS = 10.0

def ptrs_accept_k_range(lam):
    """k window the accept query covers: Stirling domain, +-10 sigma bulk."""
    slam = math.sqrt(lam)
    k_lo = max(_K_STIRLING_LO, math.ceil(lam - _K_SIGMAS * slam))
    k_hi = math.floor(lam + _K_SIGMAS * slam)
    if k_lo >= k_hi:
        raise ValueError(f"empty accept window k in [{k_lo:.6g}, {k_hi:.6g}] "
                         f"for lambda={lam:.6g}")
    return float(k_lo), float(k_hi)


def ptrs_accept_u_range(lam):
    """The u that map into ptrs_accept_k_range."""
    _, a, b, c = ptrs_consts(lam)
    k_lo, k_hi = ptrs_accept_k_range(lam)
    return hormann_u_at(a, b, c, k_lo), hormann_u_at(a, b, c, k_hi)


def poisson_cdf_below(lam, k_lo):
    """
    P(K < k_lo) for K ~ Poisson(lam), summed term by term in log space.

    Only called with k_lo pinned to the Stirling domain (_K_STIRLING_LO), so
    this is a handful of terms whatever lam is.  Nudged up by a few ulp to
    stay an upper bound on the exact tail despite lgamma/exp rounding: it is
    charged as probability mass, so erring high is the safe direction.
    """
    m = int(math.ceil(k_lo)) - 1          # largest integer k with k < k_lo
    if m < 0:
        return 0.0
    loglam = math.log(lam)
    total = math.fsum(
        math.exp(-lam + k * loglam - math.lgamma(k + 1.0))
        for k in range(m + 1)
    )
    return total * (1.0 + 1e-12)


def ptrs_k_tail_prob(lam):
    """
    P(K outside the accept window): the output mass that query does not cover.

    The lower tail is summed exactly -- k_lo is pinned to the Stirling domain,
    so it is O(1) terms, and the Chernoff bound is badly loose that close to
    the mode (at lambda=30 it gives 1.1e-5 for a tail that is really 1.2e-6),
    which made it the dominant term of the whole TV bound.

    The upper tail stays on the Chernoff bound, using the Poisson rate
    function h(t) = t*log(t) - t + 1, i.e. P(K >= x) <= exp(-lam*h(x/lam)):
    it sits ~10 sigma out where the bound is already negligible and summing
    it would cost O(lam) terms.  Bernstein is far too crude this far out --
    at lambda=50 it gives 2e-6 for a tail that is really ~1e-12, which would
    then dominate the whole bound.
    """
    k_lo, k_hi = ptrs_accept_k_range(lam)
    if not (k_lo < lam < k_hi):
        raise ValueError(f"accept window [{k_lo:.6g}, {k_hi:.6g}] does not "
                         f"contain lambda={lam:.6g}")

    def h(t):
        return t * math.log(t) - t + 1.0 if t > 0.0 else 1.0

    return poisson_cdf_below(lam, k_lo) + math.exp(-lam * h(k_hi / lam))


def make_ptrs_floor_template(lam, fp, utail):
    """
    FPTaylor expression for eps_floor: absolute error of
    (2*a/us + b)*u + (lambda + 0.43), with us = 0.5 - |u|
    [random_poisson_ptrs.c line 89]

    Only u is free; a, b, c are expressions in the literal lambda.
    """
    rnd = FP_TO_FPTAYLOR_RND[fp]
    return (
        "Variables\n"
        f"  real u in [{-(0.5 - utail):.20e}, {0.5 - utail:.20e}];\n\n"
        + "Definitions\n"
        + "\n".join(ptrs_setup_defs(rnd, f"{lam:.20e}")) + "\n"
        + f"  us_   {rnd}= 0.5 - abs(u),\n"
        + f"  ptrs_floor {rnd}= (2.0 * a_ / us_ + b_) * u + c_;\n\n"
        + "Expressions\n"
          "  eps_floor = ptrs_floor;\n"
    )


def make_ptrs_accept_template(lam, fp, u_lo, u_hi, sign, fast=False):
    """
    FPTaylor expression for eps_accept (excluding -log(v), see
    dist_common.make_logv_template): absolute error of
    -lambda + k*log(lambda) - loggam(k+1) - log(invalpha) + log(a/us^2 + b),
    with us = 0.5 - |u|   [random_poisson_ptrs.c lines 98-99, rearranged so
    that log(v) is alone on the left-hand side].
    lgamma is approximated by inlining random_loggam's x>=7 Stirling branch.
    log(a/us^2 + b) is rewritten as log(a + b*us^2) - 2*log(us) to avoid
    forming 1/us^2 directly when us is small (see dist_binomial.py).

    If fast is True, the -2*log(us_) term is omitted here and its error is
    computed separately (see dist_common.make_logus_template) and summed in
    by the caller. This drops u as a shared variable between the two terms,
    which may yield a more conservative (looser) overall bound.
    """
    rnd = FP_TO_FPTAYLOR_RND[fp]
    defs_k, name_k = loggam_defs("k1_", "lgk", rnd)
    log_us_term = "" if fast else " - 2.0 * log(us_)"

    return (
        "Variables\n"
        f"  real u in [{u_lo:.20e}, {u_hi:.20e}],\n"
        f"  real f in [0.0, 1.0];\n\n"
        + "Definitions\n"
        + "\n".join(ptrs_setup_defs(rnd, f"{lam:.20e}", accept=True)) + "\n"
        + f"  us_    {rnd}= 0.5 - abs(u),\n"
        + "\n".join(ptrs_k_defs(sign, f"{lam:.20e}")) + "\n"
        + "\n".join(defs_k) + "\n"
        + f"  us_sq_      {rnd}= us_ * us_,\n"
        + f"  log_num_    {rnd}= a_ + b_ * us_sq_,\n"
        + f"  ptrs_accept {rnd}= -{lam:.20e} + k_ * llam_ - {name_k}"
          f" - log(ialp_) + log(log_num_){log_us_term};\n\n"
        + "Expressions\n"
          "  eps_accept = ptrs_accept;\n"
    )


def _run_fptaylor_query(fptaylor, input_path, outputs_dir, env, ratio_tol,
                        bb_eval=False, x_abs_tol=None):
    """Run one FPTaylor query and return output.

    PTRS queries run one per call site, not concurrently, so the compile race
    documented in dist_common.fptaylor_cmd does not apply here even with the
    default --opt bb backend.
    """
    work = Path(tempfile.mkdtemp(prefix="fpt_", dir=outputs_dir))
    try:
        return run_command(
            fptaylor_cmd(fptaylor, input_path, work, ratio_tol, bb_eval, x_abs_tol),
            cwd=ROOT, env=env)
    finally:
        shutil.rmtree(work, ignore_errors=True)


def _run_ptrs_fptaylor(fptaylor, lam, fp, tag, inputs_dir, outputs_dir, env, verbose,
                       fast=False, u_trunc=2.0 ** -53, ratio_tol=2.0,
                       bb_eval=False, x_abs_tol=None):
    """Run FPTaylor for PTRS and return (eps_floor, eps_accept, tv)."""
    _, a, b, _ = ptrs_consts(lam)
    invalpha = 1.1239 + 1.1328 / (b - 3.4)
    utail = 1e-3

    floor_input  = inputs_dir  / f"poisson_ptrs_floor_{fp}_{tag}.txt"
    floor_output = outputs_dir / f"poisson_ptrs_floor_{fp}_{tag}.out"
    floor_input.write_text(make_ptrs_floor_template(lam, fp, utail))

    code, output = _run_fptaylor_query(fptaylor, floor_input, outputs_dir, env,
                                       ratio_tol, bb_eval, x_abs_tol)
    floor_output.write_text(output)
    if verbose >= 2:
        print(f"--- FPTaylor PTRS floor (lambda={lam}) ---\n{output}")
    if code != 0:
        raise RuntimeError(f"FPTaylor PTRS floor failed for lambda={lam}; see {floor_output}")

    t_eta = (2.0 * a / utail + b) * (0.5 - utail)
    s_eta = t_eta - 0.5
    u_tail_prob = 2.0 * math.exp(-s_eta ** 2 / (2.0 * (lam + s_eta / 3.0)))

    eps_floor = 5 * extract_abs_errors_by_problem(output)["eps_floor"] + u_tail_prob

    # k's definition needs the sign of u, so the accept query runs once per side
    au_lo, au_hi = ptrs_accept_u_range(lam)
    k_tail = ptrs_k_tail_prob(lam)
    accept_raw = 0.0
    for sign, lo, hi in ((-1, au_lo, 0.0), (+1, 0.0, au_hi)):
        s_tag = "m" if sign < 0 else "p"
        accept_input  = inputs_dir  / f"poisson_ptrs_accept_{fp}_{tag}_{s_tag}.txt"
        accept_output = outputs_dir / f"poisson_ptrs_accept_{fp}_{tag}_{s_tag}.out"
        accept_input.write_text(
            make_ptrs_accept_template(lam, fp, lo, hi, sign, fast=fast))

        code, output = _run_fptaylor_query(fptaylor, accept_input, outputs_dir,
                                           env, ratio_tol, bb_eval, x_abs_tol)
        accept_output.write_text(output)
        if verbose >= 2:
            print(f"--- FPTaylor PTRS accept s={sign:+d} (lambda={lam}) ---\n{output}")
        if code != 0:
            raise RuntimeError(f"FPTaylor PTRS accept s={sign:+d} failed for "
                               f"lambda={lam}; see {accept_output}")
        accept_raw = max(accept_raw,
                         extract_abs_errors_by_problem(output)["eps_accept"])

    eps_accept = accept_raw + k_tail + eps_logv(
        fptaylor, fp, u_trunc, inputs_dir, outputs_dir, env, verbose, ratio_tol,
        bb_eval, x_abs_tol,
    )[0]
    if fast:
        eps_accept += eps_logus(
            fptaylor, fp, utail, inputs_dir, outputs_dir, env, verbose, ratio_tol,
            bb_eval, x_abs_tol,
        )[0]

    accept_iter = invalpha
    tv = 2 * eps_floor * accept_iter + 2 * eps_accept + u_trunc
    return eps_floor, eps_accept, tv


# ---------------------------------------------------------------------------
# Low-range FPTaylor templates  (lambda < SWITCH)
# ---------------------------------------------------------------------------

def _make_low_range_template(lam_str, fp):
    lam = float(lam_str)
    k_star = int(lam + 10 * math.sqrt(lam))
    rnd = FP_TO_FPTAYLOR_RND[fp]
    var_lines = [f"  real u_{i} in [0, 1]" for i in range(1, k_star + 1)]
    def_lines = (
        [f"  lambda = {lam_str}", f"  L = {rnd}(exp(-lambda))", f"  p_1 = {rnd}(u_1)"]
        + [f"  p_{i} = {rnd}(p_{i-1} * u_{i})" for i in range(2, k_star + 1)]
    )
    return (
        "Variables\n" + ",\n".join(var_lines) + ";\n\n"
        + "Definitions\n" + ",\n".join(def_lines) + ";\n\n"
        + "Expressions\n"
        + f"  L_compute = L;\n"
        + f"  prod_compute = p_{k_star};\n"
    )


def _make_log_low_range_template(lam_str, fp):
    lam = float(lam_str)
    k_star = int(lam + 10 * math.sqrt(lam))
    rnd = FP_TO_FPTAYLOR_RND[fp]
    var_lines = [f"  real u_{i} in [1e-300, 1]" for i in range(1, k_star + 1)]
    def_lines = (
        [f"  lambda = {lam_str}",
         f"  lambda_fp {rnd}= {lam_str}",
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


def _empty_row(lam, fp):
    return {"lambda": lam, "fp": fp, "regime": "",
            "eps_floor": "", "eps_accept": "", "tv": "", "ref_tv": ""}


# ---------------------------------------------------------------------------
# Distribution interface
# ---------------------------------------------------------------------------

def add_args(parser):
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("lambda_file", nargs="?", type=Path,
                        help="File with lambda values, one or more per line")
    source.add_argument("--lam", type=float, default=None,
                        help="Single lambda value")
    parser.add_argument("--use-log", action="store_true",
                        help="Use log-space template for low-range lambdas")
    parser.add_argument("--fast", action="store_true",
                        help="PTRS only: compute the -2*log(us) term of "
                             "eps_accept in a separate FPTaylor query and "
                             "sum it in, decoupling it from the shared "
                             "variable u. Faster, but may yield a more "
                             "conservative (looser) bound.")


def default_out_dir(args):
    lf = getattr(args, "lambda_file", None)
    if lf is None:
        return ROOT / "poisson_runs"
    return ROOT / f"poisson_runs_{lf.stem}"


def run(args, fptaylor, inputs_dir, outputs_dir, env):
    lambdas = [str(args.lam)] if args.lam is not None else read_lambdas(args.lambda_file)

    rows = []
    for lam in lambdas:
        lam_float = float(lam)
        tag = safe_lambda_name(lam)
        try:
            # ---- low range ----
            if lam_float < SWITCH:
                lr_input = inputs_dir / f"low_range_{args.fp}_lam_{tag}.txt"
                if args.use_log:
                    lr_input.write_text(_make_log_low_range_template(lam, args.fp))
                else:
                    lr_input.write_text(_make_low_range_template(lam, args.fp))

                code, output = run_command([fptaylor, str(lr_input)], cwd=ROOT, env=env)
                out_path = outputs_dir / f"low_range_{args.fp}_lam_{tag}.out"
                out_path.write_text(output)
                if args.verbose >= 2:
                    print(f"--- FPTaylor low range (lambda={lam}) ---\n{output}")
                if code != 0:
                    raise RuntimeError(f"FPTaylor low range failed for lambda={lam}; see {out_path}")

                errs = extract_abs_errors_by_problem(output)
                if args.use_log:
                    missing = {"log_prod_compute", "lambda_fp_compute"} - errs.keys()
                    if missing:
                        raise RuntimeError(f"could not parse low-range errors for {', '.join(sorted(missing))}")
                    _, low_tv = _compute_log_low_range_delta(
                        errs["lambda_fp_compute"], errs["log_prod_compute"]
                    )
                else:
                    missing = {"L_compute", "prod_compute"} - errs.keys()
                    if missing:
                        raise RuntimeError(f"could not parse low-range errors for {', '.join(sorted(missing))}")
                    _, _, low_tv = _compute_low_range_delta(
                        lam_float, errs["L_compute"], errs["prod_compute"]
                    )

                ref_tv = computeDeltaLowRange(lam_float, FP_BETA[args.fp])
                row = _empty_row(lam, args.fp)
                row.update({"regime": "low", "tv": f"{low_tv:.17e}", "ref_tv": f"{ref_tv:.17e}"})
                rows.append(row)
                print(f"lambda={lam} [low] TV={row['tv']} ref_TV={row['ref_tv']}")
                continue

            # ---- high range (PTRS) ----
            eps_floor, eps_accept, tv = _run_ptrs_fptaylor(
                fptaylor, lam_float, args.fp, tag, inputs_dir, outputs_dir,
                env, args.verbose, fast=args.fast, u_trunc=args.u_trunc,
                ratio_tol=args.bb_geometric_ratio_tol,
                bb_eval=args.bb_eval, x_abs_tol=args.opt_x_abs_tol,
            )
            ref_tv = computeDeltaHighRange(lam_float, FP_BETA[args.fp])[0]

            row = _empty_row(lam, args.fp)
            row.update({
                "regime": "ptrs",
                "eps_floor": f"{eps_floor:.17e}",
                "eps_accept": f"{eps_accept:.17e}",
                "tv": f"{tv:.17e}",
                "ref_tv": f"{ref_tv:.17e}",
            })
            rows.append(row)
            print(f"lambda={lam} [PTRS] eps_floor={eps_floor:.6e}"
                  f" eps_accept={eps_accept:.6e} TV={tv:.6e} ref_TV={ref_tv:.6e}")
        except Exception as exc:
            print(f"WARNING: skipping lambda={lam}: {exc}")

    return rows


def write_plot(rows, plot_path, plot_components=False, plot_pgf=False):
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
