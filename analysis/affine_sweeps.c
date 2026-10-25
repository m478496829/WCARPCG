#include <complex.h>
#include <stddef.h>

/* One ordered affine weighted row sweep, with fixed row-scaled A and b.
   The homogeneous kernel in row_sweeps.c is the archived spectral kernel. */
int sweep_affine(int nr, int nc, const int *ptr, const int *idx,
                 const double complex *a, const double *invw,
                 const double *invden, double theta, int reverse,
                 const double complex *b, double complex *x) {
    for (int step=0; step<nr; ++step) {
        int r=reverse ? nr-1-step : step;
        double complex residual=b[r];
        for (int p=ptr[r]; p<ptr[r+1]; ++p) {
            if (idx[p]<0 || idx[p]>=nc) return 2;
            residual-=a[p]*x[idx[p]];
        }
        for (int p=ptr[r]; p<ptr[r+1]; ++p)
            x[idx[p]]+=theta*invden[r]*invw[idx[p]]*conj(a[p])*residual;
    }
    return 0;
}
