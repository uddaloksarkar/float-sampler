#!/usr/bin/env python3
import argparse
import contextlib
import csv
import os
import re
import shutil
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parent
ABS_ERROR_RE = re.compile(r"Absolute error [^:]*:\s*([-+\deE.]+)")
FP_TO_FPTAYLOR_RND = {
    "fp32": "rnd32",
    "fp64": "rnd64",
    "fp128": "rnd128",
}


FPTAYLOR_TEMPLATE = """Variables
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
  hk_error = K;
"""


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


def default_out_dir(lambda_file):
    return ROOT / f"hk_error_runs_{lambda_file.stem}"


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


def extract_abs_error(text):
    match = ABS_ERROR_RE.search(text)
    return match.group(1) if match else ""


def write_fptaylor_input(lam, fp, out_dir):
    name = f"poisson_hk_{fp}_lam_{safe_lambda_name(lam)}.txt"
    path = out_dir / name
    path.write_text(FPTAYLOR_TEMPLATE.format(lam=lam, rnd=FP_TO_FPTAYLOR_RND[fp]))
    return path


def write_plot(rows, plot_path):
    points = [
        (float(row["lambda"]), float(row["fptaylor_abs_error"]))
        for row in rows
        if row["fptaylor_abs_error"]
    ]
    if not points:
        raise ValueError("no numeric FPTaylor errors available to plot")

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

        xs, ys = zip(*sorted(points))
        plt.figure(figsize=(6, 4))
        plt.loglog(xs, ys, marker="o", label="FPTaylor K absolute error")
        plt.xlabel("lambda")
        plt.ylabel("error in K")
        plt.grid(True, which="both", alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(plot_path)
        plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Run FPTaylor for the K computation over each lambda in a file."
    )
    parser.add_argument("lambda_file", type=Path, help="File with lambda values, one or more per line")
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="Directory for generated inputs and outputs (default: hk_error_runs_<lambda-file-stem>)")
    parser.add_argument("--fptaylor", default=None,
                        help="Path to FPTaylor executable (default: $FPTAYLOR, ./FPTaylor/fptaylor, or PATH)")
    parser.add_argument("--fp", choices=("fp32", "fp64", "fp128"), default="fp64",
                        help="Floating-point format for FPTaylor input (default: fp64)")
    parser.add_argument("--plot", action="store_true",
                        help="Plot FPTaylor K errors across lambda values")
    parser.add_argument("--plot-file", type=Path, default=None,
                        help="Plot output path (default: <out-dir>/hk_error_vs_lambda.png)")
    args = parser.parse_args()

    lambdas = read_lambdas(args.lambda_file)
    if not lambdas:
        parser.error(f"{args.lambda_file} did not contain any lambda values")

    fptaylor = find_fptaylor(args.fptaylor)
    if not fptaylor:
        parser.error("FPTaylor executable not found; pass --fptaylor, set $FPTAYLOR, or build ./FPTaylor/fptaylor")

    out_dir = (args.out_dir or default_out_dir(args.lambda_file)).resolve()
    inputs_dir = out_dir / "fptaylor_inputs"
    outputs_dir = out_dir / "outputs"
    inputs_dir.mkdir(parents=True, exist_ok=True)
    outputs_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env.setdefault("FPTAYLOR_BASE", str(ROOT / "FPTaylor"))

    rows = []
    for lam in lambdas:
        tag = safe_lambda_name(lam)
        fptaylor_input = write_fptaylor_input(lam, args.fp, inputs_dir)
        fptaylor_cmd = [fptaylor, str(fptaylor_input)]
        fptaylor_code, fptaylor_output = run_command(fptaylor_cmd, cwd=ROOT, env=env)
        fptaylor_output_path = outputs_dir / f"fptaylor_hk_{args.fp}_lam_{tag}.out"
        fptaylor_output_path.write_text(fptaylor_output)
        fptaylor_error = extract_abs_error(fptaylor_output)

        rows.append({
            "lambda": lam,
            "fp": args.fp,
            "fptaylor_rnd": FP_TO_FPTAYLOR_RND[args.fp],
            "fptaylor_exit": fptaylor_code,
            "fptaylor_abs_error": fptaylor_error,
            "fptaylor_input": str(fptaylor_input),
            "fptaylor_output": str(fptaylor_output_path),
        })

        print(f"lambda={lam} hk_abs_error={fptaylor_error or 'NA'}")

    summary_path = out_dir / "summary.csv"
    with summary_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote summary: {summary_path}")

    if args.plot:
        plot_path = (args.plot_file or (out_dir / "hk_error_vs_lambda.png")).resolve()
        plot_path.parent.mkdir(parents=True, exist_ok=True)
        write_plot(rows, plot_path)
        print(f"Wrote plot: {plot_path}")


if __name__ == "__main__":
    main()
