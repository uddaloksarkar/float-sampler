/*
 * Poisson random variate generation via PTRS
 * (Transformed Rejection with Squeeze).
 *
 * Extracted from:
 *   numpy/random/src/distributions/distributions.c
 *   commit 605940366983132c73dfac6e1620f2f63a551fca
 *
 * Original algorithm: W. Hormann (1993).
 *   "The Transformed Rejection Method for Generating Poisson Random Variables."
 *   Insurance: Mathematics and Economics, 12, 39-45.
 *
 * Used when lam >= 10.
 */

#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>

static double rk_double(void *state) {
    (void)state;
    return (double)rand() / ((double)RAND_MAX + 1.0);
}

/*
 * random_poisson_ptrs
 *
 * Parameters
 *   rstate : opaque RNG state
 *   lam    : Poisson mean (lam >= 10)
 *
 * Returns a Poisson(lam) variate.
 *
 * Algorithm outline (Hormann 1993, Table 2):
 *   T1. Draw U ~ Uniform(-0.5, 0.5), V ~ Uniform(0, 1).
 *   T2. us = 0.5 - |U|; propose k = floor((2a/us + b)*U + lam + 0.43).
 *   T3. Fast accept: us >= 0.07 and V <= vr.
 *   T4. Reject if k < 0.
 *   T5. Log-transform V: V <- log(V * invalpha / (a/us^2 + b)).
 *   T6. Accept if V <= k*log(lam) - lam - log(k!)   [exact Poisson log-PMF].
 */
int64_t random_poisson_ptrs(void *rstate, double lam) {
    int64_t k;
    double U, V, slam, loglam, b, a, invalpha, vr, us;

    slam     = sqrt(lam);
    loglam   = log(lam);
    b        = 0.931 + 2.53 * slam;
    a        = -0.059 + 0.02483 * b;
    invalpha = 1.1239 + 1.1328 / (b - 3.4);
    vr       = 0.9277 - 3.6224 / (b - 2.0);

    for (;;) {
        U  = rk_double(rstate) - 0.5;
        V  = rk_double(rstate);
        us = 0.5 - fabs(U);

        k = (int64_t)(floor((2.0 * a / us + b) * U + lam + 0.43));
        if (k < 0) continue;

        /* Fast squeeze acceptance */
        if (us >= 0.07 && V <= vr) return k;

        /* Log-ratio acceptance against exact Poisson PMF */
        V = log(V * invalpha / (a / (us * us) + b));
        if (V <= k * loglam - lam - lgamma((double)(k + 1)))
            return k;
    }
}

/* ---- Test harness ---- */
#define N_SAMPLES 200000
int main(void) {
    const double lam = 50.0;   /* lam >= 10: PTRS regime */
    srand(42);
    double sum = 0.0, sum2 = 0.0;
    for (int i = 0; i < N_SAMPLES; i++) {
        double x = (double)random_poisson_ptrs(NULL, lam);
        sum += x; sum2 += x * x;
    }
    double mean = sum / N_SAMPLES;
    double var  = sum2 / N_SAMPLES - mean * mean;
    printf("random_poisson_ptrs  lam=%.1f\n", lam);
    printf("  empirical mean = %.4f   (theory = %.4f)\n", mean, lam);
    printf("  empirical var  = %.4f   (theory = %.4f)\n", var,  lam);
    return 0;
}