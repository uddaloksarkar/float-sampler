/*
 * Hypergeometric random variate generation via direct urn sampling (HYP).
 *
 * Extracted from:
 *   numpy/random/src/legacy/legacy-distributions.c
 *   commit 605940366983132c73dfac6e1620f2f63a551fca
 *   (function random_hypergeometric_hyp)
 *
 * Used when 0 < sample <= 10.
 */

#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>

static double rk_double(void *state) {
    (void)state;
    return (double)rand() / ((double)RAND_MAX + 1.0);
}

#define MIN(x, y) (((x) < (y)) ? x : y)

int64_t random_hypergeometric_hyp(void *rstate,
                                  int64_t good,
                                  int64_t bad,
                                  int64_t sample) {
  int64_t d1, d2, k, y, z;
  double u;

  d1 = bad + good - sample;
  d2 = MIN(bad, good);

  y = d2;
  k = sample;
  while (y > 0) {
    u = rk_double(rstate);
    y -= (int64_t)floor(u + (double)y / (double)(d1 + k));
    k--;
    if (k == 0)
      break;
  }
  z = d2 - y;
  if (good > bad)
    z = sample - z;
  return z;
}

/* ---- Test harness ---- */
#define N_SAMPLES 200000
int main(void) {
    const int64_t good = 8, bad = 6, sample = 5;   /* sample <= 10: HYP regime */
    srand(42);
    double sum = 0.0, sum2 = 0.0;
    for (int i = 0; i < N_SAMPLES; i++) {
        double x = (double)random_hypergeometric_hyp(NULL, good, bad, sample);
        sum += x; sum2 += x * x;
    }
    double mean = sum / N_SAMPLES;
    double var  = sum2 / N_SAMPLES - mean * mean;
    double popsize = good + bad;
    double theory_mean = sample * (double)good / popsize;
    double theory_var  = theory_mean * (bad / popsize) * (popsize - sample) / (popsize - 1.0);
    printf("random_hypergeometric_hyp  good=%lld  bad=%lld  sample=%lld\n",
           (long long)good, (long long)bad, (long long)sample);
    printf("  empirical mean = %.4f   (theory = %.4f)\n", mean, theory_mean);
    printf("  empirical var  = %.4f   (theory = %.4f)\n", var,  theory_var);
    return 0;
}
