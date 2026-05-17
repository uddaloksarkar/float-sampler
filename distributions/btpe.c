/*
 * Binomial random variate generation via BTPE algorithm.
 *
 * Extracted from:
 *   numpy/random/src/legacy/legacy-distributions.c
 *   commit 605940366983132c73dfac6e1620f2f63a551fca
 *
 * Original algorithm: Kachitvichyanukul, V. and Schmeiser, B. W. (1988).
 *   "Binomial Random Variate Generation."
 *   Communications of the ACM, 31(2), 216-222.
 *
 * BTPE = Binomial, Triangular, Parallelogram, Exponential.
 * Used when n*p >= 30 (otherwise inversion is preferred).
 */

#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>

static double rk_double(void *state) {
    (void)state;
    return (double)rand() / ((double)RAND_MAX + 1.0);
}

int64_t binomial_btpe(void *rstate, int64_t n, double p) {
    double r, q, fm, p1, xm, xl, xr, c, laml, lamr, p2, p3, p4;
    double a, u, v, s, F, rho, t, A, nrq, x1, x2, f1, f2, z, z2, w, w2, x;
    int64_t m, y, k, i;

    r    = (p < 0.5) ? p : 1.0 - p;
    q    = 1.0 - r;
    fm   = n * r + r;
    m    = (int64_t)fm;
    p1   = floor(2.195 * sqrt(n * r * q) - 4.6 * q) + 0.5;
    xm   = m + 0.5;
    xl   = xm - p1;
    xr   = xm + p1;
    c    = 0.134 + 20.5 / (15.3 + m);
    a    = (fm - xl) / (fm - xl * r);
    laml = a * (1.0 + a / 2.0);
    a    = (xr - fm) / (xr * q);
    lamr = a * (1.0 + a / 2.0);
    p2   = p1 * (1.0 + 2.0 * c);
    p3   = p2 + c / laml;
    p4   = p3 + c / lamr;
    nrq  = n * r * q;

    for (;;) {
        u = rk_double(rstate) * p4;
        v = rk_double(rstate);

        /* Triangular region */
        if (u <= p1) {
            y = (int64_t)(xm - p1 * v + u);
            goto finish;
        }

        /* Parallelogram region */
        if (u <= p2) {
            x = xl + (u - p1) / c;
            v = v * c + 1.0 - fabs(m - x + 0.5) / p1;
            if (v > 1.0 || v <= 0.0) continue;
            y = (int64_t)x;
        } else if (u <= p3) {
            /* Left exponential tail */
            y = (int64_t)(xl + log(v) / laml);
            if (y < 0) continue;
            v *= (u - p2) * laml;
        } else {
            /* Right exponential tail */
            y = (int64_t)(xr - log(v) / lamr);
            if (y > n) continue;
            v *= (u - p3) * lamr;
        }

        k = llabs(y - m);

        if (k <= 20 || k >= nrq / 2.0 - 1.0) {
            s = r / q;
            a = s * (n + 1);
            F = 1.0;
            if (m < y)
                for (i = m + 1; i <= y; i++) F *= (a / i - s);
            else
                for (i = y + 1; i <= m; i++) F /= (a / i - s);
            if (v > F) continue;
            goto finish;
        }

        rho = (k / nrq) * ((k * (k / 3.0 + 0.625) + 0.1666666666666) / nrq + 0.5);
        t   = -(k * k) / (2.0 * nrq);
        A   = log(v);
        if (A < t - rho) goto finish;
        if (A > t + rho) continue;

        x1 = y + 1;  f1 = m + 1;  z = n + 1 - m;  w = n - y + 1.0;
        x2 = x1*x1;  f2 = f1*f1;  z2 = z*z;        w2 = w*w;

        if (A > xm * log(f1/x1)
                + (n - m + 0.5) * log(z/w)
                + (y - m) * log(w * r / (x1 * q))
                + (13860.-(462.-(132.-(99.-140./f2)/f2)/f2)/f2)/f1/166320.
                + (13860.-(462.-(132.-(99.-140./z2)/z2)/z2)/z2)/z /166320.
                + (13860.-(462.-(132.-(99.-140./x2)/x2)/x2)/x2)/x1/166320.
                + (13860.-(462.-(132.-(99.-140./w2)/w2)/w2)/w2)/w /166320.)
            continue;

    finish:
        return (p > 0.5) ? n - y : y;
    }
}

/* ---- Test harness ---- */
#define N_SAMPLES 200000
int main(void) {
    const int64_t n = 200;
    const double  p = 0.4;   /* n*p = 80 >= 30: BTPE regime */
    srand(42);
    double sum = 0.0, sum2 = 0.0;
    for (int i = 0; i < N_SAMPLES; i++) {
        double x = (double)binomial_btpe(NULL, n, p);
        sum += x; sum2 += x * x;
    }
    double mean = sum / N_SAMPLES;
    double var  = sum2 / N_SAMPLES - mean * mean;
    printf("binomial_btpe  n=%lld  p=%.2f\n", (long long)n, p);
    printf("  empirical mean = %.4f   (theory = %.4f)\n", mean, n * p);
    printf("  empirical var  = %.4f   (theory = %.4f)\n", var,  n * p * (1.0 - p));
    return 0;
}