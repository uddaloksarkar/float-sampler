#!/usr/bin/env python3
import argparse
import re
import shutil
import subprocess
from pathlib import Path
import os


ROOT = Path(__file__).resolve().parent
MIN_LOWER_RE = re.compile(r"Minimum lower bound\s+([-+\deE.]+)")
MIN_UPPER_RE = re.compile(r"Minimum upper bound\s+([-+\deE.]+)")


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


def gelpia_number(x):
    return f"{x:.20e}"


def build_query(args):
    lines = [
        f"# --input-epsilon {args.input_epsilon:e}",
        f"# --output-epsilon {args.output_epsilon:e}",
        f"# --output-epsilon-relative {args.output_epsilon_relative:e}",
        f"# --timeout {args.timeout}",
        f"# --max-iters {args.max_iters}",
        "",
        f"[{gelpia_number(args.u_lo)}, {gelpia_number(args.u_hi)}] u;",
    ]

    if args.lam is None:
        lines.append(
            f"[{gelpia_number(args.lambda_lo)}, {gelpia_number(args.lambda_hi)}] lam;"
        )
        lam = "lam"
    else:
        lam = gelpia_number(args.lam)

    sqrt_lam = f"sqrt({lam})"
    b = f"(0.931 + (2.53 * {sqrt_lam}))"
    a = f"(-0.059 + (0.02483 * {b}))"
    denom = "(0.5 - abs(u))"
    h = f"(({a} / ({denom} * {denom})) + {b})"
    lines.extend(["", f"{h};"])
    return "\n".join(lines) + "\n"


def parse_minimum(output):
    lower = MIN_LOWER_RE.search(output)
    upper = MIN_UPPER_RE.search(output)
    return (
        lower.group(1) if lower else "",
        upper.group(1) if upper else "",
    )


def main():
    parser = argparse.ArgumentParser(
        description="Use Gelpia to minimize h = a/(0.5 - abs(u))^2 + b."
    )
    parser.add_argument("--gelpia", default=None,
                        help="Path to Gelpia executable (default: $GELPIA, $GELPIA_PATH/bin/gelpia, or PATH)")
    parser.add_argument("--lam", type=float, default=None,
                        help="Fixed lambda. If omitted, --lambda-lo/--lambda-hi define a lambda interval.")
    parser.add_argument("--lambda-lo", type=float, default=100.0,
                        help="Lower lambda bound when --lam is omitted (default: 100)")
    parser.add_argument("--lambda-hi", type=float, default=100000.0,
                        help="Upper lambda bound when --lam is omitted (default: 100000)")
    parser.add_argument("--u-lo", type=float, default=0.45,
                        help="Lower u bound (default: 0.45)")
    parser.add_argument("--u-hi", type=float, default=0.49,
                        help="Upper u bound (default: 0.49)")
    parser.add_argument("--out", type=Path, default=ROOT / "gelpia_h_min.txt",
                        help="Gelpia query file to write (default: gelpia_h_min.txt)")
    parser.add_argument("--write-only", action="store_true",
                        help="Only write the Gelpia query; do not run Gelpia")
    parser.add_argument("--input-epsilon", type=float, default=1e-8)
    parser.add_argument("--output-epsilon", type=float, default=1e-8)
    parser.add_argument("--output-epsilon-relative", type=float, default=1e-8)
    parser.add_argument("--timeout", type=int, default=60,
                        help="Gelpia timeout in seconds (default: 60)")
    parser.add_argument("--max-iters", type=int, default=50000)
    args = parser.parse_args()

    if args.lam is not None and args.lam <= 0:
        parser.error("--lam must be positive")
    if args.lambda_lo <= 0 or args.lambda_hi <= 0 or args.lambda_lo > args.lambda_hi:
        parser.error("lambda interval must satisfy 0 < --lambda-lo <= --lambda-hi")
    if args.u_lo < 0 or args.u_hi >= 0.5 or args.u_lo > args.u_hi:
        parser.error("u interval must satisfy 0 <= --u-lo <= --u-hi < 0.5")

    query = build_query(args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(query)
    print(f"Wrote Gelpia query: {args.out}")

    if args.write_only:
        return

    gelpia = find_gelpia(args.gelpia)
    if not gelpia:
        parser.error("Gelpia executable not found; pass --gelpia or set $GELPIA/$GELPIA_PATH")

    proc = subprocess.run(
        [gelpia, "--mode=min", str(args.out)],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    print(proc.stdout, end="")
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)

    lower, upper = parse_minimum(proc.stdout)
    if lower or upper:
        print(f"h_min_lower={lower or 'NA'} h_min_upper={upper or 'NA'}")


if __name__ == "__main__":
    main()
