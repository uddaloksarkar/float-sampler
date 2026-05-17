# distributions

Reference C implementations of discrete distribution samplers, extracted from
[NumPy](https://github.com/numpy/numpy) (commit `605940366983132c73dfac6e1620f2f63a551fca`).
These serve as the finite-precision samplers whose statistical distance from the
ideal is analysed by the tools in the parent directory.

## Files

| File | Distribution | Algorithm | Regime |
|---|---|---|---|
| `poisson_knuth.c` | Poisson | Knuth multiplication method | λ < 30 |
| `ptrs.c` | Poisson | PTRS — Transformed Rejection with Squeeze (Hormann 1993) | λ ≥ 30 |
| `btpe.c` | Binomial | BTPE — Binomial, Triangular, Parallelogram, Exponential (Kachitvichyanukul & Schmeiser 1988) | n·p ≥ 30 |
| `binomial_inv.c` | Binomial | Inversion (sequential CDF search) | n·p < 30 |

## References

- Knuth, D. E. (1969). *Seminumerical Algorithms*. The Art of Computer Programming, Vol. 2.
- Hormann, W. (1993). The Transformed Rejection Method for Generating Poisson Random Variables. *Insurance: Mathematics and Economics*, 12, 39–45.
- Kachitvichyanukul, V. and Schmeiser, B. W. (1988). Binomial Random Variate Generation. *Communications of the ACM*, 31(2), 216–222.