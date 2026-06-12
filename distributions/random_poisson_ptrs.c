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

/*
 * The transformed rejection method for generating Poisson random variables
 * W. Hoermann
 * Insurance: Mathematics and Economics 12, 39-45 (1993)
 */
#define LS2PI 0.91893853320467267
#define TWELFTH 0.083333333333333333333333
int64_t random_poisson_ptrs(void *rstate, double lam) {
  int64_t k;
  double U, V, slam, loglam, a, b, invalpha, vr, us;

  slam = sqrt(lam);
  loglam = log(lam);
  b = 0.931 + 2.53 * slam;
  a = -0.059 + 0.02483 * b;
  invalpha = 1.1239 + 1.1328 / (b - 3.4);
  vr = 0.9277 - 3.6224 / (b - 2);

  while (1) {
    U = rk_double(rstate) - 0.5;
    V = rk_double(rstate);
    us = 0.5 - fabs(U);
    k = (int64_t)floor((2 * a / us + b) * U + lam + 0.43);
    if ((us >= 0.07) && (V <= vr)) {
      return k;
    }
    if ((k < 0) || ((us < 0.013) && (V > us))) {
      continue;
    }
    /* log(V) == log(0.0) ok here */
    /* if U==0.0 so that us==0.0, log is ok since always returns */
    /* a / (us * us) + b is rewritten as (a + b * us * us) / (us * us)
     * to avoid forming 1 / us^2 directly when us is small; the
     * resulting -log(us * us) = -2.0 * log(us) is folded into the
     * comparison in log space instead. */
    double us2 = us * us;
    if (log(V) <= -lam + (double)k * loglam - random_loggam((double)k + 1)
                  - log(invalpha) + log(a + b * us2) - 2.0 * log(us)) {
      return k;
    }
  }
}
#undef LS2PI
#undef TWELFTH

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