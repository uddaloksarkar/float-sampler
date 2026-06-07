/*
 * Hypergeometric random variate generation via HRUA*
 * (H-ypergeometric via Ratio-of-Uniforms with Acceptance/rejection squeezes).
 *
 * Extracted from:
 *   numpy/random/src/legacy/legacy-distributions.c
 *   commit 605940366983132c73dfac6e1620f2f63a551fca
 *   (function random_hypergeometric_hrua, plus its random_loggam helper
 *   from numpy/random/src/distributions/distributions.c)
 *
 * Used when sample > 10.
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
#define MAX(x, y) (((x) > (y)) ? x : y)

/*
 * log-gamma function to support some of these distributions. The
 * algorithm comes from SPECFUN by Shanjie Zhang and Jianming Jin and their
 * book "Computation of Special Functions", 1996, John Wiley & Sons, Inc.
 */
static double random_loggam(double x) {
  double x0, x2, lg2pi, gl, gl0;
  int64_t k, n;

  static double a[10] = {8.333333333333333e-02, -2.777777777777778e-03,
                         7.936507936507937e-04, -5.952380952380952e-04,
                         8.417508417508418e-04, -1.917526917526918e-03,
                         6.410256410256410e-03, -2.955065359477124e-02,
                         1.796443723688307e-01, -1.39243221690590e+00};

  if ((x == 1.0) || (x == 2.0)) {
    return 0.0;
  } else if (x < 7.0) {
    n = (int64_t)(7 - x);
  } else {
    n = 0;
  }
  x0 = x + n;
  x2 = (1.0 / x0) * (1.0 / x0);
  /* log(2 * M_PI) */
  lg2pi = 1.8378770664093453e+00;
  gl0 = a[9];
  for (k = 8; k >= 0; k--) {
    gl0 *= x2;
    gl0 += a[k];
  }
  gl = gl0 / x0 + 0.5 * lg2pi + (x0 - 0.5) * log(x0) - x0;
  if (x < 7.0) {
    for (k = 1; k <= n; k++) {
      gl -= log(x0 - 1.0);
      x0 -= 1.0;
    }
  }
  return gl;
}

/* D1 = 2*sqrt(2/e) */
/* D2 = 3 - 2*sqrt(3/e) */
#define D1 1.7155277699214135
#define D2 0.8989161620588988
int64_t random_hypergeometric_hrua(void *rstate,
                                   int64_t good,
                                   int64_t bad,
                                   int64_t sample) {
  int64_t mingoodbad, maxgoodbad, popsize, m, d9;
  double d4, d5, d6, d7, d8, d10, d11;
  int64_t Z;
  double T, W, X, Y;

  mingoodbad = MIN(good, bad);
  popsize = good + bad;
  maxgoodbad = MAX(good, bad);
  m = MIN(sample, popsize - sample);
  d4 = ((double)mingoodbad) / popsize;
  d5 = 1.0 - d4;
  d6 = m * d4 + 0.5;
  d7 = sqrt((double)(popsize - m) * sample * d4 * d5 / (popsize - 1) + 0.5);
  d8 = D1 * d7 + D2;
  d9 = (int64_t)floor((double)(m + 1) * (mingoodbad + 1) / (popsize + 2));
  d10 = (random_loggam(d9 + 1) + random_loggam(mingoodbad - d9 + 1) +
         random_loggam(m - d9 + 1) + random_loggam(maxgoodbad - m + d9 + 1));
  d11 = MIN(MIN(m, mingoodbad) + 1.0, floor(d6 + 16 * d7));
  /* 16 for 16-decimal-digit precision in D1 and D2 */

  while (1) {
    X = rk_double(rstate);
    Y = rk_double(rstate);
    W = d6 + d8 * (Y - 0.5) / X;

    /* fast rejection: */
    if ((W < 0.0) || (W >= d11))
      continue;

    Z = (int64_t)floor(W);
    T = d10 - (random_loggam(Z + 1) + random_loggam(mingoodbad - Z + 1) +
               random_loggam(m - Z + 1) + random_loggam(maxgoodbad - m + Z + 1));

    /* fast acceptance: */
    if ((X * (4.0 - X) - 3.0) <= T)
      break;

    /* fast rejection: */
    if (X * (X - T) >= 1)
      continue;
    /* log(0.0) is ok here, since always accept */
    if (2.0 * log(X) <= T)
      break; /* acceptance */
  }

  /* this is a correction to HRUA* by Ivan Frohne in rv.py */
  if (good > bad)
    Z = m - Z;

  /* another fix from rv.py to allow sample to exceed popsize/2 */
  if (m < sample)
    Z = good - Z;

  return Z;
}
#undef D1
#undef D2

/* ---- Test harness ---- */
#define N_SAMPLES 200000
int main(void) {
    const int64_t good = 80, bad = 60, sample = 30;   /* sample > 10: HRUA regime */
    srand(42);
    double sum = 0.0, sum2 = 0.0;
    for (int i = 0; i < N_SAMPLES; i++) {
        double x = (double)random_hypergeometric_hrua(NULL, good, bad, sample);
        sum += x; sum2 += x * x;
    }
    double mean = sum / N_SAMPLES;
    double var  = sum2 / N_SAMPLES - mean * mean;
    double popsize = good + bad;
    double theory_mean = sample * (double)good / popsize;
    double theory_var  = theory_mean * (bad / popsize) * (popsize - sample) / (popsize - 1.0);
    printf("random_hypergeometric_hrua  good=%lld  bad=%lld  sample=%lld\n",
           (long long)good, (long long)bad, (long long)sample);
    printf("  empirical mean = %.4f   (theory = %.4f)\n", mean, theory_mean);
    printf("  empirical var  = %.4f   (theory = %.4f)\n", var,  theory_var);
    return 0;
}
