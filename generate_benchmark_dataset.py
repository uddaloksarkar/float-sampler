#!/usr/bin/env python3
"""
Generate 1000-case benchmark parameter files for poisson, binomial, and
hypergeometric, spanning a wide log-scale range of fp64-representable
values -- including values deliberately near the extreme end of what each
sampler can take (n/N/lambda up to ~1e15, close to 2^53 where fp64 stops
representing every integer exactly; p down to ~1e-15).

Each file is written in that distribution's own native input-file format
(read_lambdas / read_np_pairs / read_Nkn_triples), one case per line, ready
for creator.sh to iterate over directly.

Sampling is log-uniform across the stated range (so every order of
magnitude gets roughly equal representation, not just the top end where a
linear-uniform draw would put almost everything) plus a fixed set of exact
endpoint/extreme values forced into the mix so the true extremes are
always covered, not just probabilistically likely.

Usage:
  python3 generate_benchmark_dataset.py [--seed 0] [--n-cases 1000] [--out-dir .]
"""
import argparse
import random
import sys
from pathlib import Path

# 2^53: largest integer fp64 represents exactly; n/N/lambda extremes are
# kept at or below this so "extreme" doesn't silently mean "not exactly
# representable" for reasons unrelated to the sampler's own analysis.
_MAX_EXACT_INT = 2 ** 53


def log_uniform(rng, lo, hi):
    import math
    return math.exp(rng.uniform(math.log(lo), math.log(hi)))


def poisson_cases(rng, n_cases):
    """lambda > 0, spanning low-range (<30) and PTRS high-range, up to
    extreme (~9e15, near _MAX_EXACT_INT)."""
    anchors = [
        1e-300, 1e-100, 1e-15, 1e-10, 1e-6, 1e-3, 1e-2, 1e-1,   # near-zero rates
        0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0, 20.0, 25.0,    # low range, sparse
        28.0, 29.0, 29.9, 29.99, 29.999,                          # approaching SWITCH
        30.0,                                                     # exactly at SWITCH
        30.001, 30.01, 30.1, 31.0, 35.0, 40.0, 50.0,               # just above SWITCH
        75.0, 100.0, 200.0, 500.0,                                 # moderate PTRS
        1e3, 3e3, 1e4, 3e4, 1e5, 3e5, 1e6, 3e6,                    # mid log-scale
        1e7, 1e8, 1e9, 1e10, 1e11, 1e12, 1e13, 1e14,               # high log-scale
        float(_MAX_EXACT_INT) / 2, float(_MAX_EXACT_INT) - 1.0,    # near the exact-int edge
        float(_MAX_EXACT_INT),
    ]
    n_random = n_cases - len(anchors)
    lambdas = list(anchors) + [log_uniform(rng, 1e-2, float(_MAX_EXACT_INT))
                               for _ in range(n_random)]
    rng.shuffle(lambdas)
    return [f"{lam:.10g}" for lam in lambdas]


def binomial_cases(rng, n_cases):
    """(n, p): n a positive int, p in (0,1); n*p spans both the inversion
    (<30) and BTRS (>=30) regimes, with n up to extreme (~9e15) and p down
    to extreme (~1e-15) or up to extreme (~1-1e-15)."""
    m = _MAX_EXACT_INT
    anchors = [
        # --- inversion regime (n*p < 30), small/moderate n across p ---
        (1, 0.5), (2, 0.5), (5, 0.9), (10, 0.5), (30, 0.99),
        (100, 0.1), (100, 0.29), (300, 0.09), (1000, 0.01),
        (1000, 0.001), (10000, 0.001), (100000, 0.0001),
        # --- n*p astride the BTRS switch, from both sides ---
        (100, 0.29), (100, 0.30), (100, 0.31),
        (1000, 0.029), (1000, 0.030), (1000, 0.031),
        # --- BTRS regime, modest scale, p across the board ---
        (100, 0.5), (10900, 0.1), (10900, 0.5), (10900, 0.9),
        (1000, 0.5), (1000, 0.9), (1000, 0.99),
        # --- p near 0 or near 1 at several n scales ---
        (100, 1e-9), (100, 1.0 - 1e-9),
        (1e6, 1e-9), (1e6, 1.0 - 1e-9),
        (1e9, 1e-12), (1e9, 1.0 - 1e-12),
        (35, 1.0 - 1e-2),
        # --- n across the full log-scale at a fixed, well-behaved p ---
        (10, 0.5), (100, 0.5), (1000, 0.5), (1e4, 0.5), (1e5, 0.5),
        (1e6, 0.5), (1e7, 0.5), (1e8, 0.5), (1e9, 0.5), (1e10, 0.5),
        (1e11, 0.5), (1e12, 0.5), (1e13, 0.5), (1e14, 0.5),
        # --- large but not extreme n (1e11-1e14, well below 2^53~9e15),
        # full spread of p -- distinct from both the "modest n" BTRS tier
        # above and the "n near 2^53" extreme tier below ---
        (1e11, 1e-9), (1e11, 1e-3), (1e11, 0.1), (1e11, 0.9), (1e11, 1.0 - 1e-9),
        (1e12, 1e-10), (1e12, 1e-4), (1e12, 0.1), (1e12, 0.9), (1e12, 1.0 - 1e-10),
        (1e13, 1e-11), (1e13, 1e-5), (1e13, 0.1), (1e13, 0.9), (1e13, 1.0 - 1e-11),
        (1e14, 1e-12), (1e14, 1e-6), (1e14, 0.1), (1e14, 0.9), (1e14, 1.0 - 1e-12),
        # --- extreme n (near 2^53), full spread of p ---
        (m, 1e-15), (m, 1e-9), (m, 1e-3), (m, 0.1), (m, 0.5),
        (m, 0.9), (m, 1.0 - 1e-9), (m, 1.0 - 1e-15),
        (m - 1, 0.5), (m // 2, 0.5),
    ]
    cases = list(anchors)
    while len(cases) < n_cases:
        n = max(2, int(round(log_uniform(rng, 2, float(_MAX_EXACT_INT)))))
        p = log_uniform(rng, 1e-15, 0.999999999999999)
        if rng.random() < 0.5:
            p = 1.0 - p  # also cover p close to 1, not just close to 0
        if 0 < p < 1 and n >= 1:
            cases.append((n, p))
    rng.shuffle(cases)
    return [f"{int(round(n))} {p:.17g}" for n, p in cases[:n_cases]]


def hypergeometric_cases(rng, n_cases):
    """(N, K, n): population N up to extreme (~1e13 -- kept a bit below
    binomial/poisson's ceiling since HRUA's combinatorial d9/d10 terms are
    more sensitive at the very top of fp64's exact-integer range), K and n
    each a random fraction of N so both stay valid (0 <= K,n <= N) and the
    sample is a meaningful fraction of the population."""
    n_max = 10 ** 13
    anchors = [
        # --- HYP regime (sample or good+bad-sample < _HRUA_SWITCH=10) ---
        (20, 10, 1), (20, 10, 2), (20, 10, 19), (20, 10, 20),
        (1000, 500, 3), (1000, 500, 997), (1000, 500, 1000),
        (1e6, 5e5, 5), (1e6, 5e5, 999996),
        # --- modest HRUA regime, K/n balanced or skewed ---
        (100, 40, 30), (1000, 300, 200), (20, 10, 10),
        (1000, 100, 500), (1000, 900, 500), (1000, 500, 100), (1000, 500, 900),
        (10000, 3000, 4000),
        # --- K or n at the boundary (0 or N) ---
        (1000, 0, 500), (1000, 1000, 500), (1000, 500, 0), (1000, 500, 1000),
        (1e6, 0, 100), (1e6, 1000000, 100),
        # --- N across the full log-scale, K/n a fixed balanced fraction ---
        (100, 40, 30), (1000, 400, 300), (1e4, 4000, 3000),
        (1e5, 40000, 30000), (1e6, 400000, 300000), (1e7, 4000000, 3000000),
        (1e8, 4e7, 3e7), (1e9, 4e8, 3e8), (1e10, 4e9, 3e9),
        # --- large but not extreme N (1e9-1e12), full spread of K/n skew ---
        (1e9, 1, 1), (1e9, 999999999, 999999999), (1e9, 1e6, 1e3), (1e9, 1e3, 1e6),
        (1e10, 5, 5), (1e10, 5000000, 3000000), (1e10, 9999999995, 9999999995),
        (1e11, 100, 100), (1e11, 5e10, 5e10), (1e11, 1e5, 1e8),
        (1e12, 1000, 1000), (1e12, 6e11, 4e11), (1e12, 1e9, 1e6),
        # --- extreme population (near n_max), varied K/n skew ---
        (n_max, n_max // 3, n_max // 4),                # balanced
        (n_max, 1, 1),                                  # extreme, tiny K/n
        (n_max, n_max - 1, n_max - 1),                  # extreme, K/n near N
        (n_max, 1, n_max - 1),                           # tiny K, huge n
        (n_max, n_max - 1, 1),                           # huge K, tiny n
        (n_max, n_max // 2, n_max // 2),                 # K=N/2 exactly
    ]
    cases = list(anchors)
    while len(cases) < n_cases:
        N = max(20, int(round(log_uniform(rng, 20, float(n_max)))))
        K = max(0, min(N, int(round(N * rng.uniform(0.001, 0.999)))))
        n = max(0, min(N, int(round(N * rng.uniform(0.001, 0.999)))))
        cases.append((N, K, n))
    rng.shuffle(cases)
    return [f"{int(round(N))} {int(round(K))} {int(round(n))}" for N, K, n in cases[:n_cases]]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-cases", type=int, default=1000)
    ap.add_argument("--out-dir", type=Path, default=Path("."))
    args = ap.parse_args()

    rng = random.Random(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    files = {
        "benchmark_poisson_lambdas.txt": poisson_cases(rng, args.n_cases),
        "benchmark_binomial_np.txt": binomial_cases(rng, args.n_cases),
        "benchmark_hypergeometric_NKn.txt": hypergeometric_cases(rng, args.n_cases),
    }
    for name, lines in files.items():
        path = args.out_dir / name
        path.write_text("\n".join(lines) + "\n")
        print(f"wrote {len(lines)} cases to {path}", file=sys.stderr)


if __name__ == "__main__":
    main()
