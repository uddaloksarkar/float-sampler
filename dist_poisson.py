"""
Poisson sampler FP-error analysis.
High range (lambda >= SWITCH) follows the PTRS algorithm in
distributions/random_poisson_ptrs.c and mirrors the BTRS analysis in
dist_binomial.py (eps_floor / eps_accept split, shared -log(v) and
-2*log(us) helpers, --fast flag).
"""
import math
from pathlib import Path

from analyticError import FP_BETA, SWITCH, computeDeltaHighRange, computeDeltaLowRange
from dist_common import (
    ROOT, FP_TO_FPTAYLOR_RND,
    run_command, extract_abs_errors_by_problem,
    save_loglog_plot,
    loggam_defs, eps_logv, eps_logus,
)

NAME = "poisson"
CSV_FIELDS = ["lambda", "fp", "regime", "eps_floor", "eps_accept", "tv", "ref_tv"]


# ---------------------------------------------------------------------------
# PTRS FPTaylor templates  (lambda >= SWITCH)
# ---------------------------------------------------------------------------

def make_ptrs_floor_template(lam, fp, utail):
    """
    FPTaylor expression for eps_floor: absolute error of
    (2*a/us + b)*u + (lambda + 0.43), with us = 0.5 - |u|
    [random_poisson_ptrs.c line 89]
    """
    slam = math.sqrt(lam)
    b = 0.931 + 2.53 * slam
    a = -0.059 + 0.02483 * b
    c = lam + 0.43
    rnd = FP_TO_FPTAYLOR_RND[fp]

    return (
        "Variables\n"
        f"  real u in [{-(0.5 - utail):.20e}, {0.5 - utail:.20e}];\n\n"
        "Definitions\n"
        f"  a_  = {a:.20e},\n"
        f"  b_  = {b:.20e},\n"
        f"  c_  = {c:.20e},\n"
        f"  us_ {rnd}= 0.5 - abs(u),\n"
        f"  ptrs_floor {rnd}= (2.0 * a_ / us_ + b_) * u + c_;\n\n"
        "Expressions\n"
        "  eps_floor = ptrs_floor;\n"
    )


def make_ptrs_accept_template(lam, fp, utail, fast=False):
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
    slam = math.sqrt(lam)
    b = 0.931 + 2.53 * slam
    a = -0.059 + 0.02483 * b
    invalpha = 1.1239 + 1.1328 / (b - 3.4)
    loglam = math.log(lam)
    rnd = FP_TO_FPTAYLOR_RND[fp]

    # k_lo >= 6 so that k+1 >= 7 (Stirling branch valid)
    k_lo = float(max(6, int(lam - 10 * slam)))
    k_hi = float(int(math.ceil(lam + 10 * slam)))

    defs_k, name_k = loggam_defs("k + 1.0", "lgk", rnd)

    log_us_term = "" if fast else " - 2.0 * log(us_)"

    return (
        "Variables\n"
        f"  real u in [{-(0.5 - utail):.20e}, {0.5 - utail:.20e}],\n"
        f"  real k in [{k_lo:.1f}, {k_hi:.1f}];\n\n"
        "Definitions\n"
        f"  a_        = {a:.20e},\n"
        f"  b_        = {b:.20e},\n"
        f"  lam_      = {lam:.20e},\n"
        f"  loglam_   = {loglam:.20e},\n"
        f"  invalpha_ = {invalpha:.20e},\n"
        + "\n".join(defs_k) + "\n"
        + f"  us_         {rnd}= 0.5 - abs(u),\n"
        + f"  us_sq_      {rnd}= us_ * us_,\n"
        + f"  log_num_    {rnd}= a_ + b_ * us_sq_,\n"
        + f"  ptrs_accept {rnd}= -lam_ + k * loglam_ - {name_k}"
          f" - log(invalpha_) + log(log_num_){log_us_term};\n\n"
        + "Expressions\n"
          "  eps_accept = ptrs_accept;\n"
    )


def _run_ptrs_fptaylor(fptaylor, lam, fp, tag, inputs_dir, outputs_dir, env, verbose, fast=False):
    """Run FPTaylor for PTRS and return (eps_floor, eps_accept, tv)."""
    slam = math.sqrt(lam)
    b = 0.931 + 2.53 * slam
    invalpha = 1.1239 + 1.1328 / (b - 3.4)
    utail = 1e-5
    vtail = 1e-10

    floor_input  = inputs_dir  / f"poisson_ptrs_floor_{fp}_{tag}.txt"
    floor_output = outputs_dir / f"poisson_ptrs_floor_{fp}_{tag}.out"
    floor_input.write_text(make_ptrs_floor_template(lam, fp, utail))

    code, output = run_command([fptaylor, str(floor_input)], cwd=ROOT, env=env)
    floor_output.write_text(output)
    if verbose:
        print(f"--- FPTaylor PTRS floor (lambda={lam}) ---\n{output}")
    if code != 0:
        raise RuntimeError(f"FPTaylor PTRS floor failed for lambda={lam}; see {floor_output}")

    eps_floor = 5 * extract_abs_errors_by_problem(output)["eps_floor"] + 2 * utail

    accept_input  = inputs_dir  / f"poisson_ptrs_accept_{fp}_{tag}.txt"
    accept_output = outputs_dir / f"poisson_ptrs_accept_{fp}_{tag}.out"
    accept_input.write_text(make_ptrs_accept_template(lam, fp, utail, fast=fast))

    code, output = run_command([fptaylor, str(accept_input)], cwd=ROOT, env=env)
    accept_output.write_text(output)
    if verbose:
        print(f"--- FPTaylor PTRS accept (lambda={lam}) ---\n{output}")
    if code != 0:
        raise RuntimeError(f"FPTaylor PTRS accept failed for lambda={lam}; see {accept_output}")

    eps_accept = extract_abs_errors_by_problem(output)["eps_accept"] + eps_logv(
        fptaylor, fp, vtail, inputs_dir, outputs_dir, env, verbose,
    )
    if fast:
        eps_accept += eps_logus(
            fptaylor, fp, utail, inputs_dir, outputs_dir, env, verbose,
        )

    accept_iter = invalpha 
    tv = 2 * (eps_floor + vtail) * accept_iter + 2 * eps_accept / (1 - vtail)
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
                if args.verbose:
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
                env, args.verbose, fast=args.fast,
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
