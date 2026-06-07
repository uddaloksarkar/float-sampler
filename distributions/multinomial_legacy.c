/*
 * Legacy multinomial random variate generation, preserved for RandomState
 * stream compatibility.
 *
 * Extracted from:
 *   numpy/random/src/legacy/legacy-distributions.c
 *   commit 605940366983132c73dfac6e1620f2f63a551fca
 *   (functions legacy_random_multinomial, legacy_random_binomial,
 *   legacy_random_binomial_original, legacy_random_binomial_btpe,
 *   legacy_random_binomial_inversion)
 *
 * legacy_random_multinomial draws each coordinate via legacy_random_binomial,
 * which dispatches to the inversion or BTPE samplers depending on n*p. The
 * full chain is reproduced here (rather than calling out) since bug fixes to
 * random_binomial would otherwise change the legacy RandomState stream.
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

/* Mirrors NumPy's binomial_t: per-(n,p) constants cached across calls by both
 * the inversion and BTPE samplers. */
typedef struct {
    int     has_binomial;
    double  psave;
    int64_t nsave;
    double  r;
    double  q;
    double  fm;
    int64_t m;
    double  p1;
    double  xm;
    double  xl;
    double  xr;
    double  c;
    double  laml;
    double  lamr;
    double  p2;
    double  p3;
    double  p4;
} binomial_t;

static int64_t legacy_random_binomial_inversion(void *rstate, int64_t n, double p,
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

/*
 * BTPE implementation preserved for compatibility. The last two error terms of
 * the Stirling approximation are incorrectly added
 */
static int64_t legacy_random_binomial_btpe(void *rstate,
                                           int64_t n,
                                           double p,
                                           binomial_t *binomial) {
  double r, q, fm, p1, xm, xl, xr, c, laml, lamr, p2, p3, p4;
  double a, u, v, s, F, rho, t, A, nrq, x1, x2, f1, f2, z, z2, w, w2, x;
  int64_t m, y, k, i;

  if (!(binomial->has_binomial) || (binomial->nsave != n) ||
      (binomial->psave != p)) {
    /* initialize */
    binomial->nsave = n;
    binomial->psave = p;
    binomial->has_binomial = 1;
    binomial->r = r = MIN(p, 1.0 - p);
    binomial->q = q = 1.0 - r;
    binomial->fm = fm = n * r + r;
    binomial->m = m = (int64_t)floor(binomial->fm);
    binomial->p1 = p1 = floor(2.195 * sqrt(n * r * q) - 4.6 * q) + 0.5;
    binomial->xm = xm = m + 0.5;
    binomial->xl = xl = xm - p1;
    binomial->xr = xr = xm + p1;
    binomial->c = c = 0.134 + 20.5 / (15.3 + m);
    a = (fm - xl) / (fm - xl * r);
    binomial->laml = laml = a * (1.0 + a / 2.0);
    a = (xr - fm) / (xr * q);
    binomial->lamr = lamr = a * (1.0 + a / 2.0);
    binomial->p2 = p2 = p1 * (1.0 + 2.0 * c);
    binomial->p3 = p3 = p2 + c / laml;
    binomial->p4 = p4 = p3 + c / lamr;
  } else {
    r = binomial->r;
    q = binomial->q;
    fm = binomial->fm;
    m = binomial->m;
    p1 = binomial->p1;
    xm = binomial->xm;
    xl = binomial->xl;
    xr = binomial->xr;
    c = binomial->c;
    laml = binomial->laml;
    lamr = binomial->lamr;
    p2 = binomial->p2;
    p3 = binomial->p3;
    p4 = binomial->p4;
  }

/* sigh ... */
Step10:
  nrq = n * r * q;
  u = rk_double(rstate) * p4;
  v = rk_double(rstate);
  if (u > p1)
    goto Step20;
  y = (int64_t)floor(xm - p1 * v + u);
  goto Step60;

Step20:
  if (u > p2)
    goto Step30;
  x = xl + (u - p1) / c;
  v = v * c + 1.0 - fabs(m - x + 0.5) / p1;
  if (v > 1.0)
    goto Step10;
  y = (int64_t)floor(x);
  goto Step50;

Step30:
  if (u > p3)
    goto Step40;
  y = (int64_t)floor(xl + log(v) / laml);
  /* Reject if v==0.0 since previous cast is undefined */
  if ((y < 0) || (v == 0.0))
    goto Step10;
  v = v * (u - p2) * laml;
  goto Step50;

Step40:
  y = (int64_t)floor(xr - log(v) / lamr);
  /* Reject if v==0.0 since previous cast is undefined */
  if ((y > n) || (v == 0.0))
    goto Step10;
  v = v * (u - p3) * lamr;

Step50:
  k = llabs(y - m);
  if ((k > 20) && (k < ((nrq) / 2.0 - 1)))
    goto Step52;

  s = r / q;
  a = s * (n + 1);
  F = 1.0;
  if (m < y) {
    for (i = m + 1; i <= y; i++) {
      F *= (a / i - s);
    }
  } else if (m > y) {
    for (i = y + 1; i <= m; i++) {
      F /= (a / i - s);
    }
  }
  if (v > F)
    goto Step10;
  goto Step60;

Step52:
  rho =
      (k / (nrq)) * ((k * (k / 3.0 + 0.625) + 0.16666666666666666) / nrq + 0.5);
  t = -k * k / (2 * nrq);
  /* log(0.0) ok here */
  A = log(v);
  if (A < (t - rho))
    goto Step60;
  if (A > (t + rho))
    goto Step10;

  x1 = (double)y + 1;
  f1 = (double)m + 1;
  z = (double)n + 1 - (double)m;
  w = (double)n - (double)y + 1;
  x2 = x1 * x1;
  f2 = f1 * f1;
  z2 = z * z;
  w2 = w * w;
  /* The last two terms are subtracted in the corrected version */
  if (A > (xm * log(f1 / x1) + (n - m + 0.5) * log(z / w) +
           (y - m) * log(w * r / (x1 * q)) +
           (13680. - (462. - (132. - (99. - 140. / f2) / f2) / f2) / f2) / f1 /
               166320. +
           (13680. - (462. - (132. - (99. - 140. / z2) / z2) / z2) / z2) / z /
               166320. +
           (13680. - (462. - (132. - (99. - 140. / x2) / x2) / x2) / x2) / x1 /
               166320. +
           (13680. - (462. - (132. - (99. - 140. / w2) / w2) / w2) / w2) / w /
               166320.)) {
    goto Step10;
  }

Step60:
  if (p > 0.5) {
    y = n - y;
  }

  return y;
}

static int64_t legacy_random_binomial_original(void *rstate,
                                               double p,
                                               int64_t n,
                                               binomial_t *binomial) {
  double q;

  if (p <= 0.5) {
    if (p * n <= 30.0) {
      return legacy_random_binomial_inversion(rstate, n, p, binomial);
    } else {
      return legacy_random_binomial_btpe(rstate, n, p, binomial);
    }
  } else {
    q = 1.0 - p;
    if (q * n <= 30.0) {
      return n - legacy_random_binomial_inversion(rstate, n, q, binomial);
    } else {
      return n - legacy_random_binomial_btpe(rstate, n, q, binomial);
    }
  }
}

static int64_t legacy_random_binomial(void *rstate, double p,
                                      int64_t n, binomial_t *binomial) {
  return (int64_t) legacy_random_binomial_original(rstate, p, n, binomial);
}

void legacy_random_multinomial(void *rstate, int64_t n,
                               int64_t *mnix, double *pix, int64_t d,
                               binomial_t *binomial) {
  /*
   * Mirrors random_multinomial but dispatches to legacy_random_binomial,
   * since bug fixes to random_binomial would otherwise change the
   * RandomState stream.
   */
  double remaining_p = 1.0;
  int64_t j;
  int64_t dn = n;
  for (j = 0; j < (d - 1); j++) {
    mnix[j] = (int64_t)legacy_random_binomial(
        rstate, pix[j] / remaining_p, dn, binomial);
    dn = dn - mnix[j];
    if (dn <= 0) {
      break;
    }
    remaining_p -= pix[j];
  }
  if (dn > 0) {
      mnix[d - 1] = dn;
  }
}

/* ---- Test harness ---- */
#define N_SAMPLES 50000
int main(void) {
    const int64_t n = 100;
    const int64_t d = 4;
    double pix[4] = {0.1, 0.2, 0.3, 0.4};
    int64_t mnix[4];
    binomial_t binomial = {0};
    srand(42);

    double sum[4] = {0}, sum2[4] = {0};
    for (int i = 0; i < N_SAMPLES; i++) {
        legacy_random_multinomial(NULL, n, mnix, pix, d, &binomial);
        int64_t check = 0;
        for (int j = 0; j < d; j++) {
            sum[j]  += mnix[j];
            sum2[j] += (double)mnix[j] * mnix[j];
            check   += mnix[j];
        }
        if (check != n) {
            printf("ERROR: counts do not sum to n (got %lld)\n", (long long)check);
            return 1;
        }
    }
    printf("legacy_random_multinomial  n=%lld  d=%lld\n", (long long)n, (long long)d);
    for (int j = 0; j < d; j++) {
        double mean = sum[j] / N_SAMPLES;
        double var  = sum2[j] / N_SAMPLES - mean * mean;
        printf("  bin %d: empirical mean = %.4f (theory = %.4f)   "
               "empirical var = %.4f (theory = %.4f)\n",
               j, mean, n * pix[j], var, n * pix[j] * (1.0 - pix[j]));
    }
    return 0;
}
