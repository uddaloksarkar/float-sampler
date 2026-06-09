"""
Hypergeometric distribution FP-error analysis.
Follows the pattern in dist_geometric.py; called by main.py.
"""
from pathlib import Path

from dist_common import (
    ROOT, FP_TO_FPTAYLOR_RND,
    run_command, extract_abs_errors_by_problem,
    save_loglog_plot,
)

NAME = "hypergeometric"
CSV_FIELDS = ["N", "K", "n", "delta", "tv"]


# ---------------------------------------------------------------------------
# FPTaylor template
# ---------------------------------------------------------------------------

def make_template(N, K, n, fp):
    """
    FPTaylor input for (N, K, n) analysing the critical FP operation in
    random_hypergeometric_hyp (distributions/hypergeometric_hyp.c):

        d1 = bad + good - sample  =  (N-K) + K - n  =  N - n
        d2 = min(good, bad)       =  min(K, N-K)

        while (y > 0):
            u  = rk_double()
            y -= floor( u + (double)y / (double)(d1 + k) )
            k--
            if k == 0: break          # k loops over [sample, ..., 1]

      delta : abs error of  rnd64(u + rnd64(y / (d1 + k)))
                        vs  exact  u + y / (d1 + k)

      u in [0, 1),  y in [0, d2],  k in [1, sample]
    """
    good   = K
    bad    = N - K
    sample = n
    d1     = bad + good - sample    # = N - n
    d2     = min(good, bad)         # = min(K, N-K)
    rnd    = FP_TO_FPTAYLOR_RND[fp]

    return (
        "Variables\n"
        f"  real u in [0.0, 1.0],\n"
        f"  real y in [0.0, {float(d2):.1f}],\n"
        f"  real k in [1.0, {float(sample):.1f}];\n\n"
        + "Definitions\n"
        f"  d1 = {float(d1):.1f},\n"
        f"  div_step {rnd}= y / (d1 + k),\n"
        f"  step     {rnd}= u + div_step;\n\n"
        + "Expressions\n"
        f"  delta = step;\n"
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def safe_triple_name(N, K, n):
    return f"N{N}_K{K}_n{n}"


def read_Nkn_triples(path):
    triples = []
    for lineno, line in enumerate(path.read_text().splitlines(), start=1):
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        tokens = line.split()
        if len(tokens) != 3:
            raise ValueError(f"{path}:{lineno}: expected 'N K n', got {line!r}")
        try:
            N, K, n = int(tokens[0]), int(tokens[1]), int(tokens[2])
        except ValueError as exc:
            raise ValueError(f"{path}:{lineno}: invalid (N, K, n) values") from exc
        _validate(N, K, n, f"{path}:{lineno}")
        triples.append((N, K, n))
    return triples


def _validate(N, K, n, loc=""):
    prefix = f"{loc}: " if loc else ""
    if N <= 0:
        raise ValueError(f"{prefix}N must be positive")
    if not (0 <= K <= N):
        raise ValueError(f"{prefix}K must be in [0, N]")
    if not (0 <= n <= N):
        raise ValueError(f"{prefix}n must be in [0, N]")


# ---------------------------------------------------------------------------
# Distribution interface
# ---------------------------------------------------------------------------

def add_args(parser):
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("input_file", nargs="?", type=Path,
                        help="File with (N K n) triples, one per line")
    source.add_argument("--N", type=int, default=None, dest="N_pop",
                        help="Population size")
    parser.add_argument("--K", type=int, default=None,
                        help="Number of success states in population")
    parser.add_argument("--n", type=int, default=None, dest="n_draw",
                        help="Number of draws")


def default_out_dir(args):
    lf = getattr(args, "input_file", None)
    if lf is None:
        return ROOT / "hypergeometric_runs"
    return ROOT / f"hypergeometric_runs_{lf.stem}"


def run(args, fptaylor, inputs_dir, outputs_dir, env):
    if getattr(args, "N_pop", None) is not None:
        if args.K is None or args.n_draw is None:
            raise ValueError("--K and --n are required when --N is given")
        N, K, n = args.N_pop, args.K, args.n_draw
        _validate(N, K, n)
        triples = [(N, K, n)]
    else:
        triples = read_Nkn_triples(args.input_file)
    if not triples:
        raise ValueError("no (N, K, n) triples found in input")

    rows = []
    for N, K, n in triples:
        sample = n
        d2 = min(K, N - K)

        # degenerate: nothing to draw or no variance
        if sample == 0 or d2 == 0:
            delta, tv = 0.0, 0.0
            rows.append({
                "N": N, "K": K, "n": n,
                "delta": f"{delta:.17e}",
                "tv":    f"{tv:.17e}",
            })
            print(f"N={N} K={K} n={n} delta={delta:.6e} TV={tv:.6e}")
            continue

        try:
            tag = safe_triple_name(N, K, n)
            input_path = inputs_dir / f"hypergeometric_{args.fp}_{tag}.txt"
            input_path.write_text(make_template(N, K, n, args.fp))

            code, output = run_command(
                [fptaylor, str(input_path)],
                cwd=ROOT, env=env,
            )
            out_path = outputs_dir / f"hypergeometric_{args.fp}_{tag}.out"
            out_path.write_text(output)
            if args.verbose:
                print(f"--- FPTaylor hypergeometric (N={N} K={K} n={n}) ---\n{output}")
            if code != 0:
                raise RuntimeError(
                    f"FPTaylor failed for N={N} K={K} n={n}; see {out_path}")

            abs_errors = extract_abs_errors_by_problem(output)
            if "delta" not in abs_errors:
                raise RuntimeError(
                    f"N={N} K={K} n={n}: could not parse absolute error for 'delta'")
            delta = abs_errors["delta"]
            tv    = 2 * sample * delta

            rows.append({
                "N": N, "K": K, "n": n,
                "delta": f"{delta:.17e}",
                "tv":    f"{tv:.17e}",
            })
            print(f"N={N} K={K} n={n} delta={delta:.6e} TV={tv:.6e}")
        except Exception as exc:
            print(f"WARNING: skipping N={N} K={K} n={n}: {exc}")

    return rows


def write_plot(rows, plot_path, plot_components=False, plot_pgf=False):
    xs = [int(r["n"]) for r in rows]
    series = [("TV", [float(r["tv"]) for r in rows], "^")]
    save_loglog_plot(xs, series, xlabel="n  (draws)", ylabel="error",
                     plot_path=plot_path, plot_pgf=plot_pgf)
