# distributions

Reference C implementations of discrete distribution samplers, extracted from
[NumPy](https://github.com/numpy/numpy) (commit `605940366983132c73dfac6e1620f2f63a551fca`).
These serve as the finite-precision samplers whose statistical distance from the
ideal is analysed by the tools in the parent directory.

## Files

| File | Distribution | Algorithm | Regime |
|---|---|---|---|
| `random_poisson_mult.c` | Poisson | `random_poisson_mult` — Knuth multiplication method | λ < 10 |
| `random_poisson_ptrs.c` | Poisson | `random_poisson_ptrs` — PTRS, Transformed Rejection with Squeeze (Hormann 1993) | λ ≥ 10 |
| `btpe.c` | Binomial | BTPE — Binomial, Triangular, Parallelogram, Exponential (Kachitvichyanukul & Schmeiser 1988) | n·p ≥ 30 |
| `binomial_legacy_inversion.c` | Binomial | Legacy inversion (`legacy_random_binomial_inversion`, cached constants) | n·p ≤ 30 |
| `btrs.c` | Binomial | `binomialvariate` — BTRS, Transformed Rejection with Squeeze (Hörmann), with BG geometric fallback (Devroye); ported from [CPython's `random.py`](https://github.com/python/cpython/blob/main/Lib/random.py#L789) | n·p < 10 → BG, else BTRS |
| `multinomial_legacy.c` | Multinomial | `legacy_random_multinomial` — per-coordinate legacy binomial draws (inversion/BTPE chain) | — |
| `zipf.c` | Zipf | `legacy_random_zipf` — rejection sampling | — |
| `hypergeometric_hyp.c` | Hypergeometric | `random_hypergeometric_hyp` — direct urn sampling | sample ≤ 10 |
| `hypergeometric_hrua.c` | Hypergeometric | `random_hypergeometric_hrua` — HRUA* (ratio-of-uniforms) | sample > 10 |
| `geometric_inversion.c` | Geometric | `legacy_geometric_inversion` — CDF inversion | p < 1/3 |
| `geometric_search.c` | Geometric | `random_geometric_search` — direct CDF search | p ≥ 1/3 |

## References

- Knuth, D. E. (1969). *Seminumerical Algorithms*. The Art of Computer Programming, Vol. 2.
- Hormann, W. (1993). The Transformed Rejection Method for Generating Poisson Random Variables. *Insurance: Mathematics and Economics*, 12, 39–45.
- Kachitvichyanukul, V. and Schmeiser, B. W. (1988). Binomial Random Variate Generation. *Communications of the ACM*, 31(2), 216–222.