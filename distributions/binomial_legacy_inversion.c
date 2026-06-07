/*
 * Legacy binomial random variate generation via inversion (sequential CDF
 * search), preserved for RandomState stream compatibility.
 *
 * Extracted from:
 *   numpy/random/src/legacy/legacy-distributions.c
 *   commit 605940366983132c73dfac6e1620f2f63a551fca
 *
 * Used when n*p <= 30 (small expected count). Differs from the modern
 * random_binomial_inversion in its qn formula (exp(n*log(q)) vs.
 * exp(n*log1p(-p))) and in caching the per-(n,p) constants in `binomial`
 * across calls.
 */

#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>

static double rk_double(void *state) {
    (void)state;
    return (double)rand() / ((double)RAND_MAX + 1.0);
}

/* Subset of NumPy's binomial_t holding the fields legacy_random_binomial_inversion caches. */
typedef struct {
    int     has_binomial;
    double  psave;
    int64_t nsave;
    double  r;
    double  q;
    double  c;
    int64_t m;
} binomial_t;

int64_t legacy_random_binomial_inversion(void *rstate, int64_t n, double p,
                                         binomial_t *binomial)
{
  double q, qn, np, px, U;
  int64_t X, bound;

  if (!(binomial->has_binomial) || (binomial->nsave != n) ||
      (binomial->psave != p)) {
    binomial->nsave = n;
    binomial->psave = p;
    binomial->has_binomial = 1;
    binomial->q = q = 1.0 - p;
    binomial->r = qn = exp(n * log(q));
    binomial->c = np = n * p;
    binomial->m = bound = (int64_t)fmin((double)n, np + 10.0 * sqrt(np * q + 1));
  } else {
    q = binomial->q;
    qn = binomial->r;
    np = binomial->c;
    bound = binomial->m;
  }
  X = 0;
  px = qn;
  U = rk_double(rstate);
  while (U > px) {
    X++;
    if (X > bound) {
      X = 0;
      px = qn;
      U = rk_double(rstate);
    } else {
      U -= px;
      px = ((n - X + 1) * p * px) / (X * q);
    }
  }
  return X;
}

/* ---- Test harness ---- */
#define N_SAMPLES 200000
int main(void) {
    const int64_t n = 50;
    const double  p = 0.3;   /* n*p = 15 <= 30: inversion regime */
    binomial_t binomial = {0};
    srand(42);
    double sum = 0.0, sum2 = 0.0;
    for (int i = 0; i < N_SAMPLES; i++) {
        double x = (double)legacy_random_binomial_inversion(NULL, n, p, &binomial);
        sum += x; sum2 += x * x;
    }
    double mean = sum / N_SAMPLES;
    double var  = sum2 / N_SAMPLES - mean * mean;
    printf("legacy_random_binomial_inversion  n=%lld  p=%.2f\n", (long long)n, p);
    printf("  empirical mean = %.4f   (theory = %.4f)\n", mean, n * p);
    printf("  empirical var  = %.4f   (theory = %.4f)\n", var,  n * p * (1.0 - p));
    return 0;
}
