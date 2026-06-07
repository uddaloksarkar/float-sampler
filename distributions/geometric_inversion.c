/*
 * Legacy geometric random variate generation via inversion of the CDF.
 *
 * Extracted from:
 *   numpy/random/src/legacy/legacy-distributions.c
 *   commit 605940366983132c73dfac6e1620f2f63a551fca
 *   (function legacy_geometric_inversion)
 *
 * Used when p < 1/3 (legacy_random_geometric dispatch).
 *
 * Note: npy_log1p is NumPy's portability macro for log1p; it is used here
 * directly since this file has no dependency on npy_math.h.
 */

#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>

static double rk_double(void *state) {
    (void)state;
    return (double)rand() / ((double)RAND_MAX + 1.0);
}

long legacy_geometric_inversion(void *rstate, double p) {
  return (long)ceil(log1p(-rk_double(rstate)) / log(1 - p));
}

/* ---- Test harness ---- */
#define N_SAMPLES 200000
int main(void) {
    const double p = 0.1;   /* p < 1/3: inversion regime */
    srand(42);
    double sum = 0.0, sum2 = 0.0;
    for (int i = 0; i < N_SAMPLES; i++) {
        double x = (double)legacy_geometric_inversion(NULL, p);
        sum += x; sum2 += x * x;
    }
    double mean = sum / N_SAMPLES;
    double var  = sum2 / N_SAMPLES - mean * mean;
    printf("legacy_geometric_inversion  p=%.2f\n", p);
    printf("  empirical mean = %.4f   (theory = %.4f)\n", mean, 1.0 / p);
    printf("  empirical var  = %.4f   (theory = %.4f)\n", var,  (1.0 - p) / (p * p));
    return 0;
}
