/*
 * Poisson random variate generation via the multiplication method (Knuth).
 *
 * Extracted from:
 *   numpy/random/src/distributions/distributions.c
 *   commit 605940366983132c73dfac6e1620f2f63a551fca
 *
 * Reference: Knuth, D. E. (1969). Seminumerical Algorithms.
 *   The Art of Computer Programming, Vol. 2.
 *
 * Used when lam < 10.
 */

#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>

static double rk_double(void *state) {
    (void)state;
    return (double)rand() / ((double)RAND_MAX + 1.0);
}

int64_t random_poisson_mult(void *rstate, double lam) {
    int64_t X;
    double prod, U, enlam;

    enlam = exp(-lam);
    X = 0;
    prod = 1.0;
    while (1) {
        U = rk_double(rstate);
        prod *= U;
        if (prod > enlam) {
            X += 1;
        } else {
            return X;
        }
    }
}

/* ---- Test harness ---- */
#define N_SAMPLES 200000
int main(void) {
    const double lam = 4.5;   /* lam < 10: mult regime */
    srand(42);
    double sum = 0.0, sum2 = 0.0;
    for (int i = 0; i < N_SAMPLES; i++) {
        double x = (double)random_poisson_mult(NULL, lam);
        sum += x; sum2 += x * x;
    }
    double mean = sum / N_SAMPLES;
    double var  = sum2 / N_SAMPLES - mean * mean;
    printf("random_poisson_mult  lam=%.1f\n", lam);
    printf("  empirical mean = %.4f   (theory = %.4f)\n", mean, lam);
    printf("  empirical var  = %.4f   (theory = %.4f)\n", var,  lam);
    return 0;
}