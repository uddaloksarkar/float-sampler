#!/usr/bin/env python3
import argparse
import contextlib
import csv
import math
import os
import re
import shutil
import subprocess
from pathlib import Path

from analyticError import FP_BETA, SWITCH, computeDeltaHighRange, computeDeltaLowRange


ROOT = Path(__file__).resolve().parent
ABS_ERROR_RE = re.compile(r"Absolute error [^:]*:\s*([-+\deE.]+)")
MIN_LOWER_RE = re.compile(r"Minimum lower bound\s+([-+\deE.]+)")
FP_TO_FPTAYLOR_RND = {
    "fp32": "rnd32",
    "fp64": "rnd64",
    "fp128": "rnd128",
}


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


def make_low_range_template(lam_str, fp):
    lam = float(lam_str)
    k_star = int(lam + 10 * math.sqrt(lam))
    rnd = FP_TO_FPTAYLOR_RND[fp]

    var_lines = [f"  real u_{i} in [0, 1]" for i in range(1, k_star + 1)]
    def_lines = (
        [f"  lambda = {lam_str}", f"  L = {rnd}(exp(-lambda))", f"  p_1 = {rnd}(u_1)"]
        + [f"  p_{i} = {rnd}(p_{i-1} * u_{i})" for i in range(2, k_star + 1)]
    )
    return (
        "Variables\n"
        + ",\n".join(var_lines) + ";\n\n"
        + "Definitions\n"
        + ",\n".join(def_lines) + ";\n\n"
        + "Expressions\n"
        + f"  L_compute = L;\n"
        + f"  prod_compute = p_{k_star};\n"
    )


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
                raise ValueError(f"{path}:{lineno}: lambda must be positive, got {token!r}")
            lambdas.append(token)
    return lambdas


def safe_lambda_name(lam):
    return lam.replace("+", "").replace("-", "m").replace(".", "p").replace("E", "e")


def find_fptaylor(explicit):
    if explicit:
        return explicit
    env = os.environ.get("FPTAYLOR")
    if env:
        return env
    local = ROOT / "FPTaylor" / "fptaylor"
    if local.exists():
        return str(local)
    return shutil.which("fptaylor")


def find_gelpia(explicit):
    if explicit:
        return explicit
    env = os.environ.get("GELPIA")
    if env:
        return env
    gelpia_path = os.environ.get("GELPIA_PATH")
    if gelpia_path:
        return str(Path(gelpia_path) / "bin" / "gelpia")
    local = ROOT / "gelpia" / "bin" / "gelpia"
    if local.exists():
        return str(local)
    local = ROOT / "FPTaylor" / "gelpia" / "bin" / "gelpia"
    if local.exists():
        return str(local)
    return shutil.which("gelpia")


def run_command(cmd, cwd=None, env=None):
    proc = subprocess.run(
        cmd,
        cwd=cwd,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    return proc.returncode, proc.stdout


def extract_abs_error(output, label):
    match = ABS_ERROR_RE.search(output)
    if not match:
        raise RuntimeError(f"could not parse {label} FPTaylor absolute error")
    return float(match.group(1))


def extract_abs_errors_by_problem(output):
    errors = {}
    current = None
    for line in output.splitlines():
        problem_match = re.match(r"Problem:\s*(\S+)", line)
        if problem_match:
            current = problem_match.group(1)
            continue
        if current is None:
            continue
        error_match = re.match(r"Absolute error [^:]*:\s*([-+\deE.]+)", line)
        if error_match:
            errors[current] = float(error_match.group(1))
            current = None
    return errors


def extract_h_min_lower(output):
    match = MIN_LOWER_RE.search(output)
    if not match:
        raise RuntimeError("could not parse Gelpia h_min_lower")
    return float(match.group(1))


def write_fptaylor_input(template, lam, fp, path):
    path.write_text(template.format(lam=lam, rnd=FP_TO_FPTAYLOR_RND[fp]))


def compute_low_range_delta(lam, l_compute_error, prod_compute_error):
    l_value = math.exp(-lam)
    E = prod_compute_error + l_compute_error
    if E >= l_value:
        delta = math.inf
    else:
        delta = 2 * E / (l_value - E)
    return l_value, E, delta


def write_gelpia_h_query(lam, args, path):
    path.write_text(
        "\n".join([
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
        ])
    )


def compute_tv(lam, fp):
    beta = FP_BETA[fp]
    if lam < SWITCH:
        return computeDeltaLowRange(lam, beta)
    return computeDeltaHighRange(lam, beta)[0]


def empty_row(lam, fp):
    return {
        "lambda": lam,
        "fp": fp,
        "regime": "",
        "delta_e": "",
        "delta_k": "",
        "h_min_lower": "",
        "delta_h": "",
        "total_error": "",
        "low_l_value": "",
        "low_l_compute_error": "",
        "low_prod_compute_error": "",
        "low_err": "",
        "low_delta": "",
        "compute_delta_low_range": "",
        "tv": "",
        "low_range_input": "",
        "delta_e_input": "",
        "delta_k_input": "",
        "h_query": "",
        "low_range_output": "",
        "delta_e_output": "",
        "delta_k_output": "",
        "h_output": "",
    }


def write_plot(rows, plot_path, plot_components=False, plot_pgf=False):
    points = []
    for row in rows:
        if row["regime"] == "low":
            points.append((
                float(row["lambda"]),
                None,
                None,
                float(row["low_delta"]),
                float(row["compute_delta_low_range"]),
            ))
        else:
            points.append((
                float(row["lambda"]),
                float(row["delta_e"]),
                float(row["delta_h"]),
                float(row["total_error"]),
                float(row["tv"]),
            ))
    if not points:
        raise ValueError("no rows available to plot")

    mpl_cache = plot_path.parent / ".matplotlib"
    xdg_cache = plot_path.parent / ".cache"
    mpl_cache.mkdir(parents=True, exist_ok=True)
    xdg_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_cache))
    os.environ.setdefault("XDG_CACHE_HOME", str(xdg_cache))

    with open(os.devnull, "w") as devnull, contextlib.redirect_stderr(devnull):
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        points.sort(key=lambda row: row[0])
        xs = [row[0] for row in points]
        series = []
        if plot_components:
            series += [
                ("DeltaE", [row[1] for row in points], "o"),
                ("DeltaH", [row[2] for row in points], "s"),
            ]
        series += [
            ("Total/new low delta", [row[3] for row in points], "^"),
            ("analyticError.py", [row[4] for row in points], "x"),
        ]

        plt.figure(figsize=(7, 4.5))
        for label, ys, marker in series:
            series_points = [
                (x, y) for x, y in zip(xs, ys)
                if y is not None and math.isfinite(y) and y > 0
            ]
            if not series_points:
                continue
            series_xs, series_ys = zip(*series_points)
            plt.loglog(series_xs, series_ys, marker=marker, label=label)
        plt.xlabel("lambda")
        plt.ylabel("error")
        plt.ylim(top=0.9)
        plt.grid(True, which="both", alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(plot_path)
        if plot_pgf:
            pgf_path = plot_path.with_suffix(".pgf")
            plt.savefig(pgf_path, backend="pgf")
        plt.close()


def default_out_dir(lambda_file):
    if lambda_file is None:
        return ROOT / "total_error_runs_lam"
    return ROOT / f"total_error_runs_{lambda_file.stem}"


def main():
    parser = argparse.ArgumentParser(
        description="Compute DeltaE + DeltaH for each lambda using FPTaylor and Gelpia."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("lambda_file", nargs="?", type=Path,
                        help="File with lambda values, one or more per line")
    source.add_argument("--lam", type=float, default=None,
                        help="Single lambda value")
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="Output directory (default: total_error_runs_<lambda-file-stem>)")
    parser.add_argument("--fptaylor", default=None,
                        help="Path to FPTaylor executable")
    parser.add_argument("--gelpia", default=None,
                        help="Path to Gelpia executable")
    parser.add_argument("--fp", choices=("fp32", "fp64", "fp128"), default="fp64",
                        help="Floating-point format for FPTaylor (default: fp64)")
    parser.add_argument("--u-lo", type=float, default=0.45)
    parser.add_argument("--u-hi", type=float, default=0.49)
    parser.add_argument("--plot", action="store_true",
                        help="Plot total error and TV")
    parser.add_argument("--plot-components", action="store_true",
                        help="Include DeltaE and DeltaH series in the plot (requires --plot)")
    parser.add_argument("--plot-pgf", action="store_true",
                        help="Also save the plot in PGF format alongside the PNG")
    parser.add_argument("--plot-file", type=Path, default=None,
                        help="Plot output path (default: <out-dir>/total_error_vs_lambda.png)")
    parser.add_argument("--gelpia-input-epsilon", type=float, default=1e-8)
    parser.add_argument("--gelpia-output-epsilon", type=float, default=1e-8)
    parser.add_argument("--gelpia-output-epsilon-relative", type=float, default=1e-8)
    parser.add_argument("--gelpia-timeout", type=int, default=60)
    parser.add_argument("--gelpia-max-iters", type=int, default=50000)
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Print FPTaylor and Gelpia output to stdout")
    args = parser.parse_args()

    if args.lam is not None and args.lam <= 0:
        parser.error("--lam must be positive")
    if args.u_lo < 0 or args.u_hi >= 0.5 or args.u_lo > args.u_hi:
        parser.error("u interval must satisfy 0 <= --u-lo <= --u-hi < 0.5")

    lambdas = [str(args.lam)] if args.lam is not None else read_lambdas(args.lambda_file)
    if not lambdas:
        parser.error(f"{args.lambda_file} did not contain any lambda values")

    fptaylor = find_fptaylor(args.fptaylor)
    if not fptaylor:
        parser.error("FPTaylor executable not found; pass --fptaylor or set $FPTAYLOR")
    needs_gelpia = any(float(lam) >= SWITCH for lam in lambdas)
    gelpia = find_gelpia(args.gelpia) if needs_gelpia else None
    if needs_gelpia and not gelpia:
        parser.error("Gelpia executable not found; pass --gelpia or set $GELPIA/$GELPIA_PATH")

    out_dir = (args.out_dir or default_out_dir(args.lambda_file)).resolve()
    inputs_dir = out_dir / "inputs"
    outputs_dir = out_dir / "outputs"
    inputs_dir.mkdir(parents=True, exist_ok=True)
    outputs_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env.setdefault("FPTAYLOR_BASE", str(ROOT / "FPTaylor"))

    rows = []
    for lam in lambdas:
        lam_float = float(lam)
        tag = safe_lambda_name(lam)

        if lam_float < SWITCH:
            low_range_input = inputs_dir / f"low_range_{args.fp}_lam_{tag}.txt"
            low_range_input.write_text(make_low_range_template(lam, args.fp))

            low_code, low_output = run_command([fptaylor, str(low_range_input)], cwd=ROOT, env=env)
            low_output_path = outputs_dir / f"low_range_{args.fp}_lam_{tag}.out"
            low_output_path.write_text(low_output)
            if args.verbose:
                print(f"--- FPTaylor low range (lambda={lam}) ---\n{low_output}")
            if low_code != 0:
                raise RuntimeError(f"FPTaylor low range failed for lambda={lam}; see {low_output_path}")

            low_errors = extract_abs_errors_by_problem(low_output)
            missing = {"L_compute", "prod_compute"} - low_errors.keys()
            if missing:
                names = ", ".join(sorted(missing))
                raise RuntimeError(f"could not parse low-range FPTaylor errors for {names}")

            l_error = low_errors["L_compute"]
            prod_error = low_errors["prod_compute"]
            l_value, low_err, low_delta = compute_low_range_delta(
                lam_float, l_error, prod_error
            )
            compute_delta_low = computeDeltaLowRange(lam_float, FP_BETA[args.fp])

            row = empty_row(lam, args.fp)
            row.update({
                "regime": "low",
                "total_error": f"{low_delta:.17e}",
                "low_l_value": f"{l_value:.17e}",
                "low_l_compute_error": f"{l_error:.17e}",
                "low_prod_compute_error": f"{prod_error:.17e}",
                "low_err": f"{low_err:.17e}",
                "low_delta": f"{low_delta:.17e}",
                "compute_delta_low_range": f"{compute_delta_low:.17e}",
                "tv": f"{compute_delta_low:.17e}",
                "low_range_input": str(low_range_input),
                "low_range_output": str(low_output_path),
            })
            rows.append(row)

            print(
                f"lambda={lam} Delta={row['low_delta']} "
                f"ComputeDeltaLowRange={row['compute_delta_low_range']}"
            )
            continue

        delta_e_input = inputs_dir / f"delta_e_{args.fp}_lam_{tag}.txt"
        delta_k_input = inputs_dir / f"delta_k_{args.fp}_lam_{tag}.txt"
        h_query = inputs_dir / f"h_min_lam_{tag}.dop"
        write_fptaylor_input(DELTA_E_TEMPLATE, lam, args.fp, delta_e_input)
        write_fptaylor_input(DELTA_K_TEMPLATE, lam, args.fp, delta_k_input)
        write_gelpia_h_query(lam, args, h_query)

        delta_e_code, delta_e_output = run_command([fptaylor, str(delta_e_input)], cwd=ROOT, env=env)
        delta_e_output_path = outputs_dir / f"delta_e_{args.fp}_lam_{tag}.out"
        delta_e_output_path.write_text(delta_e_output)
        if args.verbose:
            print(f"--- FPTaylor DeltaE (lambda={lam}) ---\n{delta_e_output}")
        if delta_e_code != 0:
            raise RuntimeError(f"FPTaylor DeltaE failed for lambda={lam}; see {delta_e_output_path}")
        delta_e = extract_abs_error(delta_e_output, "DeltaE")

        delta_k_code, delta_k_output = run_command([fptaylor, str(delta_k_input)], cwd=ROOT, env=env)
        delta_k_output_path = outputs_dir / f"delta_k_{args.fp}_lam_{tag}.out"
        delta_k_output_path.write_text(delta_k_output)
        if args.verbose:
            print(f"--- FPTaylor DeltaK (lambda={lam}) ---\n{delta_k_output}")
        if delta_k_code != 0:
            raise RuntimeError(f"FPTaylor DeltaK failed for lambda={lam}; see {delta_k_output_path}")
        delta_k = extract_abs_error(delta_k_output, "DeltaK")

        h_code, h_output = run_command([gelpia, "--mode=min", str(h_query)], cwd=ROOT)
        h_output_path = outputs_dir / f"h_min_lam_{tag}.out"
        h_output_path.write_text(h_output)
        if args.verbose:
            print(f"--- Gelpia h_min (lambda={lam}) ---\n{h_output}")
        if h_code != 0:
            raise RuntimeError(f"Gelpia h_min failed for lambda={lam}; see {h_output_path}")
        h_min_lower = extract_h_min_lower(h_output)

        b = 0.931 + 2.53 * math.sqrt(lam_float)
        alpha = 1.1239 + 1.1328 / (b - 3.4)
        delta_h = delta_k * alpha * (lam_float + math.sqrt(lam_float)) / h_min_lower
        total_error = delta_e + delta_h
        tv = compute_tv(lam_float, args.fp)
        row = empty_row(lam, args.fp)
        row.update({
            "regime": "high",
            "delta_e": f"{delta_e:.17e}",
            "delta_k": f"{delta_k:.17e}",
            "h_min_lower": f"{h_min_lower:.17e}",
            "delta_h": f"{delta_h:.17e}",
            "total_error": f"{total_error:.17e}",
            "tv": f"{tv:.17e}",
            "delta_e_input": str(delta_e_input),
            "delta_k_input": str(delta_k_input),
            "h_query": str(h_query),
            "delta_e_output": str(delta_e_output_path),
            "delta_k_output": str(delta_k_output_path),
            "h_output": str(h_output_path),
        })
        rows.append(row)

        print(
            f"lambda={lam} DeltaE={row['delta_e']} DeltaH={row['delta_h']} "
            f"Total={row['total_error']} TV={row['tv']}"
        )

    summary_path = out_dir / "summary.csv"
    with summary_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=empty_row("", args.fp).keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote summary: {summary_path}")

    if args.plot:
        plot_path = (args.plot_file or (out_dir / "total_error_vs_lambda.png")).resolve()
        plot_path.parent.mkdir(parents=True, exist_ok=True)
        write_plot(rows, plot_path, plot_components=args.plot_components, plot_pgf=args.plot_pgf)
        print(f"Wrote plot: {plot_path}")


if __name__ == "__main__":
    main()
