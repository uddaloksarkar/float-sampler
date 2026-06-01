#!/usr/bin/env python3
import argparse
import csv
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ABS_ERROR_RE = re.compile(r"Absolute error \(exact\)[^:]*:\s*([-+\deE.]+)")
BOUNDS_LO_RE = re.compile(r"Bounds \(without rounding\):\s*\[([-+\deE.]+),")
FP_TO_FPTAYLOR_RND = {
    "fp32": "rnd32",
    "fp64": "rnd64",
    "fp128": "rnd128",
}


def make_binom_template(n, p, fp):
    """
    Single FPTaylor input for a (n, p) pair with two expressions:
      delta_1 : max_{z in [q^n,1], k in [1,n]} relative error of z*(n-k+1)*p/(k*q)
      eps_exp : relative error of exp(n*log(q))  [init probability q^n]
    """
    q = 1.0 - p
    z_lo = max(q ** n, sys.float_info.min)
    rnd = FP_TO_FPTAYLOR_RND[fp]

    return (
        "Variables\n"
        f"  real z in [{z_lo:.20e}, 1.0],\n"
        f"  real k in [1.0, {float(n):.1f}];\n\n"
        + "Definitions\n"
        f"  n = {float(n):.1f},\n"
        f"  p = {p:.20e},\n"
        f"  q = {q:.20e},\n"
        f"  ratio {rnd}= z * (n - k + 1) * p / (k * q),\n"
        f"  init  {rnd}= exp(n * log(q));\n\n"
        + "Expressions\n"
        f"  delta_1 = ratio;\n"
        f"  eps_exp = init;\n"
    )


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
            n = int(tokens[0])
            p = float(tokens[1])
        except ValueError as exc:
            raise ValueError(f"{path}:{lineno}: invalid (n, p) values") from exc
        if n <= 0:
            raise ValueError(f"{path}:{lineno}: n must be positive")
        if not (0 < p < 1):
            raise ValueError(f"{path}:{lineno}: p must be in (0, 1)")
        pairs.append((n, p))
    return pairs


def safe_pair_name(n, p):
    p_str = f"{p:.6g}".replace(".", "p").replace("-", "m").replace("+", "")
    return f"n{n}_p{p_str}"


def find_fptaylor(explicit):
    if explicit:
        return explicit
    env_val = os.environ.get("FPTAYLOR")
    if env_val:
        return env_val
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


def extract_deltas_by_problem(output, label):
    """
    Parse per-problem delta = abs_error / lower_bound from a FPTaylor run
    that may contain multiple Problem sections.

    delta is a valid upper bound on relative error: |fl(e)-e|/|e| ≤ abs/min|e|.
    FPTaylor's --rel-error warns "close to zero" and skips its own relative-error
    output, so we derive it from the absolute-error and bounds it always computes.
    """
    deltas = {}
    for section in re.split(r"(?=^-{10,})", output, flags=re.MULTILINE):
        m = re.search(r"Problem:\s*(\S+)", section)
        if not m:
            continue
        name = m.group(1)
        abs_m = ABS_ERROR_RE.search(section)
        lo_m = BOUNDS_LO_RE.search(section)
        if not abs_m:
            raise RuntimeError(f"{label}: could not parse absolute error for '{name}'")
        if not lo_m:
            raise RuntimeError(f"{label}: could not parse bounds for '{name}'")
        abs_error = float(abs_m.group(1))
        lower_bound = float(lo_m.group(1))
        if lower_bound <= 0:
            raise RuntimeError(
                f"{label} '{name}': lower bound is non-positive ({lower_bound})"
            )
        deltas[name] = abs_error / lower_bound
    return deltas


def default_out_dir(input_file):
    if input_file is None:
        return ROOT / "binom_runs"
    return ROOT / f"binom_runs_{input_file.stem}"


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Compute TV bound for a binomial sampler via FPTaylor relative-error analysis.\n"
            "\n"
            "For each (n, p) pair, computes:\n"
            "  delta = max_{k in [n], z in [q^n, 1]} |fl(z*(n-k+1)*p/(k*q)) - exact| / |exact|\n"
            "  TV    = n*p * delta\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "input_file", nargs="?", type=Path,
        help="File with (n, p) pairs, one per line (format: 'n p')",
    )
    source.add_argument("--n", type=int, default=None, help="Single n value")
    parser.add_argument("--p", type=float, default=None,
                        help="Probability p in (0,1), required with --n")
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="Output directory (default: binom_runs[_<stem>]/)")
    parser.add_argument("--fptaylor", default=None,
                        help="Path to FPTaylor executable")
    parser.add_argument("--fp", choices=("fp32", "fp64", "fp128"), default="fp64",
                        help="Floating-point format (default: fp64)")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Print FPTaylor output to stdout")
    args = parser.parse_args()

    if args.n is not None:
        if args.p is None:
            parser.error("--p is required when --n is given")
        if args.n <= 0:
            parser.error("--n must be positive")
        if not (0 < args.p < 1):
            parser.error("--p must be in (0, 1)")
        pairs = [(args.n, args.p)]
        input_file = None
    else:
        input_file = args.input_file
        pairs = read_np_pairs(input_file)
    if not pairs:
        parser.error("no (n, p) pairs found in input")

    fptaylor = find_fptaylor(args.fptaylor)
    if not fptaylor:
        parser.error("FPTaylor not found; pass --fptaylor or set $FPTAYLOR")

    out_dir = (args.out_dir or default_out_dir(input_file)).resolve()
    inputs_dir = out_dir / "inputs"
    outputs_dir = out_dir / "outputs"
    inputs_dir.mkdir(parents=True, exist_ok=True)
    outputs_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env.setdefault("FPTAYLOR_BASE", str(ROOT / "FPTaylor"))

    rows = []
    for n, p in pairs:
        q = 1.0 - p
        mean = n * p
        tag = safe_pair_name(n, p)

        input_path = inputs_dir / f"binom_{args.fp}_{tag}.txt"
        input_path.write_text(make_binom_template(n, p, args.fp))

        code, output = run_command(
            [fptaylor, "--rel-error", "true", str(input_path)],
            cwd=ROOT,
            env=env,
        )
        out_path = outputs_dir / f"binom_{args.fp}_{tag}.out"
        out_path.write_text(output)
        if args.verbose:
            print(f"--- FPTaylor (n={n}, p={p}) ---\n{output}")
        if code != 0:
            raise RuntimeError(f"FPTaylor failed for n={n}, p={p}; see {out_path}")

        deltas = extract_deltas_by_problem(output, f"n={n} p={p}")
        delta_1 = deltas["delta_1"]
        eps_exp = deltas["eps_exp"]

        tv = 0.5 * mean * delta_1 + 0.5 * eps_exp

        rows.append({
            "n": n,
            "p": f"{p:.17g}",
            "delta_1": f"{delta_1:.17e}",
            "eps_exp": f"{eps_exp:.17e}",
            "tv": f"{tv:.17e}",
        })
        print(f"n={n} p={p} delta_1={delta_1:.6e} eps_exp={eps_exp:.6e} TV={tv:.6e}")

    summary_path = out_dir / "summary.csv"
    with summary_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["n", "p", "delta_1", "eps_exp", "tv"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote summary: {summary_path}")


if __name__ == "__main__":
    main()
