"""
Poisson sampler FP-error analysis.
Refactored from fpsampler.py; called by main.py.
"""
import math
from pathlib import Path

from analyticError import FP_BETA, SWITCH, computeDeltaHighRange, computeDeltaLowRange
from dist_common import (
    ROOT, FP_TO_FPTAYLOR_RND,
    find_gelpia, run_command,
    extract_abs_error, extract_abs_errors_by_problem,
    save_loglog_plot,
)

NAME = "poisson"
CSV_FIELDS = ["lambda", "fp", "regime", "delta_e", "delta_h", "total_error", "tv"]

MIN_LOWER_RE = __import__("re").compile(r"Minimum lower bound\s+([-+\deE.]+)")

# ---------------------------------------------------------------------------
# FPTaylor templates
# ---------------------------------------------------------------------------

DELTA_E_TEMPLATE = """Variables
  real u in [0.45, 0.49],
  real floor_err in [-1, 0];

Definitions
  lambda = {lam},
  pi = 3.14159265358979323846,
  c = lambda + 0.445,
  b = 0.931 + 2.53 * sqrt(lambda),
  a = -0.059 + 0.02483 * b,
  alpha = 1.1239 + 1.1328 / (b - 3.4),
  K_real = ((2 * a) / (0.5 - abs(u)) + b) * u + c,
  K = K_real + floor_err,
  lr {rnd}= -lambda + K * log(lambda) - K * log(K) + K
            - 0.5 * log(2 * pi * K) - log(u) - log(alpha);

Expressions
  delta_e = lr;
"""

DELTA_K_TEMPLATE = """Variables
  real u in [0.45, 0.49],
  real floor_err in [-1, 0];

Definitions
  lambda = {lam},
  c = lambda + 0.445,
  b = 0.931 + 2.53 * sqrt(lambda),
  a = -0.059 + 0.02483 * b,
  K_real = ((2 * a) / (0.5 - abs(u)) + b) * u + c,
  K {rnd}= K_real + floor_err;

Expressions
  delta_k = K;
"""


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


def _write_fptaylor_input(template, lam, fp, path):
    path.write_text(template.format(lam=lam, rnd=FP_TO_FPTAYLOR_RND[fp]))


def _compute_low_range_delta(lam, l_compute_error, prod_compute_error):
    l_value = math.exp(-lam)
    E = prod_compute_error + l_compute_error
    delta = math.inf if E >= l_value else 2 * E / (l_value - E)
    return l_value, E, delta


def _compute_log_low_range_delta(lambda_fp_error, log_prod_error):
    E = lambda_fp_error + log_prod_error
    return E, 2 * E


def _write_gelpia_h_query(lam, args, path):
    path.write_text("\n".join([
        f"# --input-epsilon {args.gelpia_input_epsilon:e}",
        f"# --output-epsilon {args.gelpia_output_epsilon:e}",
        f"# --output-epsilon-relative {args.gelpia_output_epsilon_relative:e}",
        f"# --timeout {args.gelpia_timeout}",
        f"# --max-iters {args.gelpia_max_iters}",
        "",
        f"[{args.u_lo:.20e}, {args.u_hi:.20e}] u;",
        "",
        "(((-0.059 + (0.02483 * (0.931 + (2.53 * sqrt({lam}))))) / "
        "((0.5 - abs(u)) * (0.5 - abs(u)))) + "
        "(0.931 + (2.53 * sqrt({lam}))));".format(lam=f"{float(lam):.20e}"),
        "",
    ]))


def _empty_row(lam, fp):
    return {"lambda": lam, "fp": fp, "regime": "",
            "delta_e": "", "delta_h": "", "total_error": "", "tv": ""}


# ---------------------------------------------------------------------------
# Distribution interface
# ---------------------------------------------------------------------------

def add_args(parser):
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("lambda_file", nargs="?", type=Path,
                        help="File with lambda values, one or more per line")
    source.add_argument("--lam", type=float, default=None,
                        help="Single lambda value")
    parser.add_argument("--gelpia", default=None,
                        help="Path to Gelpia executable")
    parser.add_argument("--u-lo", type=float, default=0.45)
    parser.add_argument("--u-hi", type=float, default=0.49)
    parser.add_argument("--use-log", action="store_true",
                        help="Use log-space template for low-range lambdas")
    parser.add_argument("--gelpia-input-epsilon", type=float, default=1e-8)
    parser.add_argument("--gelpia-output-epsilon", type=float, default=1e-8)
    parser.add_argument("--gelpia-output-epsilon-relative", type=float, default=1e-8)
    parser.add_argument("--gelpia-timeout", type=int, default=60)
    parser.add_argument("--gelpia-max-iters", type=int, default=50000)


def default_out_dir(args):
    lf = getattr(args, "lambda_file", None)
    if lf is None:
        return ROOT / "total_error_runs_lam"
    return ROOT / f"total_error_runs_{lf.stem}"


def run(args, fptaylor, inputs_dir, outputs_dir, env):
    lambdas = [str(args.lam)] if args.lam is not None else read_lambdas(args.lambda_file)

    if args.u_lo < 0 or args.u_hi >= 0.5 or args.u_lo > args.u_hi:
        raise ValueError("u interval must satisfy 0 <= u-lo <= u-hi < 0.5")

    needs_gelpia = any(float(lam) >= SWITCH for lam in lambdas)
    gelpia = find_gelpia(getattr(args, "gelpia", None)) if needs_gelpia else None
    if needs_gelpia and not gelpia:
        raise RuntimeError("Gelpia not found; pass --gelpia or set $GELPIA/$GELPIA_PATH")

    rows = []
    for lam in lambdas:
        lam_float = float(lam)
        tag = safe_lambda_name(lam)

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
                _, low_delta = _compute_log_low_range_delta(
                    errs["lambda_fp_compute"], errs["log_prod_compute"]
                )
            else:
                missing = {"L_compute", "prod_compute"} - errs.keys()
                if missing:
                    raise RuntimeError(f"could not parse low-range errors for {', '.join(sorted(missing))}")
                _, _, low_delta = _compute_low_range_delta(
                    lam_float, errs["L_compute"], errs["prod_compute"]
                )

            tv = computeDeltaLowRange(lam_float, FP_BETA[args.fp])
            row = _empty_row(lam, args.fp)
            row.update({"regime": "low", "total_error": f"{low_delta:.17e}", "tv": f"{tv:.17e}"})
            rows.append(row)
            print(f"lambda={lam} Total={row['total_error']} TV={row['tv']}")
            continue

        # ---- high range ----
        de_input = inputs_dir / f"delta_e_{args.fp}_lam_{tag}.txt"
        dk_input = inputs_dir / f"delta_k_{args.fp}_lam_{tag}.txt"
        h_query  = inputs_dir / f"h_min_lam_{tag}.dop"
        _write_fptaylor_input(DELTA_E_TEMPLATE, lam, args.fp, de_input)
        _write_fptaylor_input(DELTA_K_TEMPLATE, lam, args.fp, dk_input)
        _write_gelpia_h_query(lam, args, h_query)

        de_code, de_output = run_command([fptaylor, str(de_input)], cwd=ROOT, env=env)
        de_out = outputs_dir / f"delta_e_{args.fp}_lam_{tag}.out"
        de_out.write_text(de_output)
        if args.verbose:
            print(f"--- FPTaylor DeltaE (lambda={lam}) ---\n{de_output}")
        if de_code != 0:
            raise RuntimeError(f"FPTaylor DeltaE failed for lambda={lam}; see {de_out}")
        delta_e = extract_abs_error(de_output, "DeltaE")

        dk_code, dk_output = run_command([fptaylor, str(dk_input)], cwd=ROOT, env=env)
        dk_out = outputs_dir / f"delta_k_{args.fp}_lam_{tag}.out"
        dk_out.write_text(dk_output)
        if args.verbose:
            print(f"--- FPTaylor DeltaK (lambda={lam}) ---\n{dk_output}")
        if dk_code != 0:
            raise RuntimeError(f"FPTaylor DeltaK failed for lambda={lam}; see {dk_out}")
        delta_k = extract_abs_error(dk_output, "DeltaK")

        h_code, h_output = run_command([gelpia, "--mode=min", str(h_query)], cwd=ROOT)
        h_out = outputs_dir / f"h_min_lam_{tag}.out"
        h_out.write_text(h_output)
        if args.verbose:
            print(f"--- Gelpia h_min (lambda={lam}) ---\n{h_output}")
        if h_code != 0:
            raise RuntimeError(f"Gelpia h_min failed for lambda={lam}; see {h_out}")
        m = MIN_LOWER_RE.search(h_output)
        if not m:
            raise RuntimeError(f"could not parse Gelpia h_min for lambda={lam}")
        h_min_lower = float(m.group(1))

        b = 0.931 + 2.53 * math.sqrt(lam_float)
        alpha = 1.1239 + 1.1328 / (b - 3.4)
        delta_h = delta_k * alpha * (lam_float + math.sqrt(lam_float)) / h_min_lower
        total_error = delta_e + delta_h
        tv = computeDeltaHighRange(lam_float, FP_BETA[args.fp])[0]

        row = _empty_row(lam, args.fp)
        row.update({
            "regime": "high",
            "delta_e": f"{delta_e:.17e}",
            "delta_h": f"{delta_h:.17e}",
            "total_error": f"{total_error:.17e}",
            "tv": f"{tv:.17e}",
        })
        rows.append(row)
        print(f"lambda={lam} DeltaE={row['delta_e']} DeltaH={row['delta_h']} "
              f"Total={row['total_error']} TV={row['tv']}")

    return rows


def write_plot(rows, plot_path, plot_components=False, plot_pgf=False):
    points = []
    for row in rows:
        if row["regime"] == "low":
            points.append((float(row["lambda"]), None, None,
                           float(row["total_error"]), float(row["tv"])))
        else:
            points.append((float(row["lambda"]), float(row["delta_e"]),
                           float(row["delta_h"]), float(row["total_error"]),
                           float(row["tv"])))
    points.sort(key=lambda r: r[0])
    xs = [r[0] for r in points]
    series = []
    if plot_components:
        series += [
            ("DeltaE", [r[1] for r in points], "o"),
            ("DeltaH", [r[2] for r in points], "s"),
        ]
    series += [
        ("total_error", [r[3] for r in points], "^"),
        ("TV (analyticError)", [r[4] for r in points], "x"),
    ]
    save_loglog_plot(xs, series, xlabel="lambda", ylabel="error",
                     plot_path=plot_path, plot_pgf=plot_pgf, ylim_top=0.9)
