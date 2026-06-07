/*
 * Zipf random variate generation via rejection sampling.
 *
 * Extracted from:
 *   numpy/random/src/legacy/legacy-distributions.c
 *   commit 605940366983132c73dfac6e1620f2f63a551fca
 *   (function legacy_random_zipf)
 */

#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>

static double rk_double(void *state) {
    (void)state;
    return (double)rand() / ((double)RAND_MAX + 1.0);
}

int64_t legacy_random_zipf(void *rstate, double a) {
  double am1, b;

  am1 = a - 1.0;
  b = pow(2.0, am1);
  while (1) {
    double T, U, V, X;

    U = 1.0 - rk_double(rstate);
    V = rk_double(rstate);
    X = floor(pow(U, -1.0 / am1));
    /*
     * The real result may be above what can be represented in a signed
     * long. Since this is a straightforward rejection algorithm, we can
     * just reject this value. This function then models a Zipf
     * distribution truncated to sys.maxint.
     */
    if (X > (double)INT64_MAX || X < 1.0) {
      continue;
    }

    T = pow(1.0 + 1.0 / X, am1);
    if (V * X * (T - 1.0) / (b - 1.0) <= T / b) {
      return (int64_t)X;
    }
  }
}

/* ---- Test harness ---- */
#define N_SAMPLES 200000
int main(void) {
    const double a = 2.5;
    srand(42);
    double sum = 0.0, sum2 = 0.0;
    for (int i = 0; i < N_SAMPLES; i++) {
        double x = (double)legacy_random_zipf(NULL, a);
        sum += x; sum2 += x * x;
    }
    double mean = sum / N_SAMPLES;
    double var  = sum2 / N_SAMPLES - mean * mean;
    printf("legacy_random_zipf  a=%.2f\n", a);
    printf("  empirical mean = %.4f\n", mean);
    printf("  empirical var  = %.4f\n", var);
    return 0;
}
