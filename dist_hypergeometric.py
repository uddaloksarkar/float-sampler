"""
Hypergeometric distribution FP-error analysis.
Follows the pattern in dist_geometric.py; called by main.py.

Two regimes, matching numpy's dispatch:
  10 <= sample <= good + bad - 10 : HRUA (ratio-of-uniforms,
                                    distributions/hypergeometric_hrua.c)
  otherwise                       : HYP  (inversion-style loop,
                                    distributions/hypergeometric_hyp.c)
"""
import math
from pathlib import Path

from dist_common import (
    ROOT, FP_TO_FPTAYLOR_RND,
    run_command, extract_abs_errors_by_problem,
    save_loglog_plot, loggam_defs,
)

NAME = "hypergeometric"
CSV_FIELDS = ["N", "K", "n", "regime", "delta", "eps_w", "eps_accept", "tv"]

_HRUA_SWITCH = 10      # see _use_hrua()

_D1 = 1.7155277699214135   # 2*sqrt(2/e)
_D2 = 0.8989161620588988   # 3 - 2*sqrt(3/e)


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
# HRUA FPTaylor templates  (sample > _HRUA_SWITCH)
# ---------------------------------------------------------------------------

def _hrua_constants(N, K, n):
    """
    The d4..d11 setup constants of random_hypergeometric_hrua
    (distributions/hypergeometric_hrua.c lines 81-93), computed in exact
    Python arithmetic.
    """
    good, bad  = K, N - K
    mingoodbad = min(good, bad)
    maxgoodbad = max(good, bad)
    popsize    = good + bad
    m          = min(n, popsize - n)

    d4  = mingoodbad / popsize
    d5  = 1.0 - d4
    d6  = m * d4 + 0.5
    d7  = math.sqrt((popsize - m) * n * d4 * d5 / (popsize - 1) + 0.5)
    d8  = _D1 * d7 + _D2
    d9  = int(math.floor((m + 1) * (mingoodbad + 1) / (popsize + 2)))
    d10 = (math.lgamma(d9 + 1) + math.lgamma(mingoodbad - d9 + 1)
           + math.lgamma(m - d9 + 1) + math.lgamma(maxgoodbad - m + d9 + 1))
    d11 = min(min(m, mingoodbad) + 1.0, math.floor(d6 + 16 * d7))

    return dict(mingoodbad=mingoodbad, maxgoodbad=maxgoodbad, popsize=popsize,
                m=m, d6=d6, d7=d7, d8=d8, d10=d10, d11=d11)


def hrua_xtail(N, K, n, ulps=8.0):
    """
    The X cutoff that minimizes the total eps_w error budget.

    Restricting X to [xtail, 1] trades two terms against each other.  A draw
    survives the fast rejection  (W < 0) || (W >= d11)  iff

        -d6*X/d8  <=  Y - 0.5  <  (d11 - d6)*X/d8            [line 99-105]

    so P(survive | X) = min(1, d11*X/d8) -- linear in X, never zero.  Cutting
    the domain at xtail therefore discards surviving draws with probability

        eps_tail(x) = int_0^x d11*t/d8 dt = d11 * x^2 / (2*d8)     (quadratic)

    while the analysis error grows as the worst case |W| = d6 + d8/(2x):

        eps_w(x) ~ rho * d8 / (2x),  rho ~ `ulps` * 2^-53           (hyperbolic)

    rho is the measured relative error of the FPTaylor bound, flat at ~6e-16
    (about 5 ulps) over 24 decades of xtail; `ulps` defaults to 8 for margin.
    Minimizing the sum gives x* = (rho * d8^2 / (2*d11))^(1/3).

    The hyperbolic term is an artifact of the analysis box: FPTaylor treats X
    and Y as independent and so evaluates at |Y-0.5| = 0.5, where the real
    algorithm would already have rejected the draw.  A joint constraint on
    (X, Y) would remove it, at which point xtail could go much lower.

    Returns (xtail, eps_w_estimate, eps_tail).
    """
    c = _hrua_constants(N, K, n)
    d6, d8, d11 = c["d6"], c["d8"], c["d11"]
    rho = ulps * 2.0 ** -53
    xtail = (rho * d8 * d8 / (2.0 * d11)) ** (1.0 / 3.0)
    return xtail, rho * (d6 + d8 / (2.0 * xtail)), d11 * xtail ** 2 / (2.0 * d8)


def make_hrua_w_template(N, K, n, fp, xtail, xhi=1.0):
    """
    FPTaylor expression for eps_w: absolute error of the candidate

        W = d6 + d8 * (Y - 0.5) / X          [hypergeometric_hrua.c line 99]

    X, Y are independent uniforms; X is restricted to [xtail, xhi] because
    W blows up as X -> 0 (those draws are rejected by the W >= d11 test).

    xhi exists so logbb.log_bb can ask for a sub-box of X: FPTaylor's bound
    here degrades with the endpoint ratio xhi/xtail, so the domain has to be
    split geometrically once xtail drops much below 1e-3.
    """
    c   = _hrua_constants(N, K, n)
    rnd = FP_TO_FPTAYLOR_RND[fp]

    return (
        "Variables\n"
        f"  real X in [{xtail:.20e}, {xhi:.20e}],\n"
        f"  real Y in [0.0, 1.0];\n\n"
        "Definitions\n"
        f"  d6_ = {c['d6']:.20e},\n"
        f"  d8_ = {c['d8']:.20e},\n"
        f"  w_step {rnd}= d6_ + d8_ * (Y - 0.5) / X;\n\n"
        "Expressions\n"
        "  eps_w = w_step;\n"
    )


def make_hrua_accept_template(N, K, n, fp, xtail, xhi=1.0):
    """
    FPTaylor expression for eps_accept: absolute error of the acceptance
    test  2*log(X) <= T  written as a single expression

        2*log(X) - T,   T = d10 - (loggam(Z+1) + loggam(mingoodbad-Z+1)
                                   + loggam(m-Z+1) + loggam(maxgoodbad-m+Z+1))
                                             [hypergeometric_hrua.c line 117]

    lgamma is approximated by inlining random_loggam's x >= 7 Stirling
    branch, so Z is restricted to the range where all four arguments are
    >= 7 (same restriction as the BTRS/PTRS analyses).
    """
    c   = _hrua_constants(N, K, n)
    rnd = FP_TO_FPTAYLOR_RND[fp]
    mgb, Mgb, m = c["mingoodbad"], c["maxgoodbad"], c["m"]

    # Z = floor(W) in [0, d11 - 1]; narrow to keep every loggam argument >= 7
    z_lo = float(max(0, 6, 6 - (Mgb - m)))
    z_hi = float(min(int(c["d11"]) - 1, mgb - 6, m - 6))
    if z_lo > z_hi:
        raise RuntimeError(
            f"N={N} K={K} n={n}: no Z range with all loggam arguments >= 7 "
            f"(z_lo={z_lo}, z_hi={z_hi}); HRUA analysis not applicable")

    defs_z,  name_z  = loggam_defs("Z + 1.0", "lgz", rnd)
    defs_mz, name_mz = loggam_defs(f"{float(mgb):.1f} - Z + 1.0", "lgmz", rnd)
    defs_kz, name_kz = loggam_defs(f"{float(m):.1f} - Z + 1.0", "lgkz", rnd)
    defs_Mz, name_Mz = loggam_defs(f"{float(Mgb - m):.1f} + Z + 1.0", "lgMz", rnd)

    return (
        "Variables\n"
        f"  real X in [{xtail:.20e}, {xhi:.20e}],\n"
        f"  real Z in [{z_lo:.1f}, {z_hi:.1f}];\n\n"
        "Definitions\n"
        f"  d10_ = {c['d10']:.20e},\n"
        + "\n".join(defs_z)  + "\n"
        + "\n".join(defs_mz) + "\n"
        + "\n".join(defs_kz) + "\n"
        + "\n".join(defs_Mz) + "\n"
        + f"  hrua_accept {rnd}= 2.0 * log(X) - d10_"
          f" + {name_z} + {name_mz} + {name_kz} + {name_Mz};\n\n"
        + "Expressions\n"
          "  eps_accept = hrua_accept;\n"
    )


def _run_hrua_fptaylor(fptaylor, N, K, n, fp, tag, inputs_dir, outputs_dir, env, verbose):
    """Run both HRUA FPTaylor queries and return (eps_w, eps_accept, tv)."""
    xtail = 1e-4

    w_input  = inputs_dir  / f"hypergeometric_hrua_w_{fp}_{tag}.txt"
    w_output = outputs_dir / f"hypergeometric_hrua_w_{fp}_{tag}.out"
    w_input.write_text(make_hrua_w_template(N, K, n, fp, xtail))

    code, output = run_command([fptaylor, str(w_input)], cwd=ROOT, env=env)
    w_output.write_text(output)
    if verbose >= 2:
        print(f"--- FPTaylor HRUA W (N={N} K={K} n={n}) ---\n{output}")
    if code != 0:
        raise RuntimeError(f"FPTaylor HRUA W failed for N={N} K={K} n={n}; see {w_output}")
    eps_w = extract_abs_errors_by_problem(output)["eps_w"]

    accept_input  = inputs_dir  / f"hypergeometric_hrua_accept_{fp}_{tag}.txt"
    accept_output = outputs_dir / f"hypergeometric_hrua_accept_{fp}_{tag}.out"
    accept_input.write_text(make_hrua_accept_template(N, K, n, fp, xtail))

    code, output = run_command([fptaylor, str(accept_input)], cwd=ROOT, env=env)
    accept_output.write_text(output)
    if verbose >= 2:
        print(f"--- FPTaylor HRUA accept (N={N} K={K} n={n}) ---\n{output}")
    if code != 0:
        raise RuntimeError(f"FPTaylor HRUA accept failed for N={N} K={K} n={n}; see {accept_output}")
    eps_accept = extract_abs_errors_by_problem(output)["eps_accept"]

    tv = 2 * (eps_w + 1.5 * eps_accept)
    return eps_w, eps_accept, tv


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _use_hrua(N, K, n):
    """
    numpy's regime dispatch (random_hypergeometric):
        (sample >= 10) && (sample <= good + bad - 10)  ->  HRUA
        otherwise                                      ->  HYP
    with good + bad = popsize = N.
    """
    good, bad = K, N - K
    sample = n
    return sample >= _HRUA_SWITCH and sample <= good + bad - _HRUA_SWITCH


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
                "N": N, "K": K, "n": n, "regime": "degenerate",
                "delta": f"{delta:.17e}",
                "tv":    f"{tv:.17e}",
            })
            print(f"N={N} K={K} n={n} delta={delta:.6e} TV={tv:.6e}")
            continue

        try:
            tag = safe_triple_name(N, K, n)

            # ---- HRUA regime ----
            if _use_hrua(N, K, n):
                eps_w, eps_accept, tv = _run_hrua_fptaylor(
                    fptaylor, N, K, n, args.fp, tag, inputs_dir, outputs_dir,
                    env, args.verbose,
                )
                rows.append({
                    "N": N, "K": K, "n": n, "regime": "hrua",
                    "delta": "nan",
                    "eps_w":      f"{eps_w:.17e}",
                    "eps_accept": f"{eps_accept:.17e}",
                    "tv":         f"{tv:.17e}",
                })
                print(f"N={N} K={K} n={n} [HRUA] eps_w={eps_w:.6e}"
                      f" eps_accept={eps_accept:.6e} TV={tv:.6e}")
                continue

            # ---- HYP regime ----
            input_path = inputs_dir / f"hypergeometric_{args.fp}_{tag}.txt"
            input_path.write_text(make_template(N, K, n, args.fp))

            code, output = run_command(
                [fptaylor, str(input_path)],
                cwd=ROOT, env=env,
            )
            out_path = outputs_dir / f"hypergeometric_{args.fp}_{tag}.out"
            out_path.write_text(output)
            if args.verbose >= 2:
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
                "N": N, "K": K, "n": n, "regime": "hyp",
                "delta": f"{delta:.17e}",
                "eps_w": "nan", "eps_accept": "nan",
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
