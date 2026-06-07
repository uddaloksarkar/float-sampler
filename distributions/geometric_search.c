/*
 * Geometric random variate generation via direct CDF search.
 *
 * Extracted from:
 *   numpy/random/src/distributions/distributions.c
 *   commit 605940366983132c73dfac6e1620f2f63a551fca
 *   (function random_geometric_search)
 *
 * Used when p >= 1/3 (legacy_random_geometric dispatch); this is the
 * "search" counterpart to legacy_geometric_inversion.
 */

#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>

static double rk_double(void *state) {
    (void)state;
    return (double)rand() / ((double)RAND_MAX + 1.0);
}

int64_t random_geometric_search(void *rstate, double p) {
  double U;
  int64_t X;
  double sum, prod, q;

  X = 1;
  sum = prod = p;
  q = 1.0 - p;
  U = rk_double(rstate);
  while (U > sum) {
    prod *= q;
    sum += prod;
    X++;
  }
  return X;
}

/* ---- Test harness ---- */
#define N_SAMPLES 200000
int main(void) {
    const double p = 0.6;   /* p >= 1/3: search regime */
    srand(42);
    double sum = 0.0, sum2 = 0.0;
    for (int i = 0; i < N_SAMPLES; i++) {
        double x = (double)random_geometric_search(NULL, p);
        sum += x; sum2 += x * x;
    }
    double mean = sum / N_SAMPLES;
    double var  = sum2 / N_SAMPLES - mean * mean;
    printf("random_geometric_search  p=%.2f\n", p);
    printf("  empirical mean = %.4f   (theory = %.4f)\n", mean, 1.0 / p);
    printf("  empirical var  = %.4f   (theory = %.4f)\n", var,  (1.0 - p) / (p * p));
    return 0;
}
