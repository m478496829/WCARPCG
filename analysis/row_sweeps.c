#include <complex.h>
#include <stdlib.h>
#include <string.h>

/* Apply actual ordered homogeneous row updates to k independent RHS columns.
   All matrices use row-major storage. No Krylov iteration is performed. */
int sweep_rows(int m, int nc, int k, const int *ptr, const int *idx,
               const double complex *a, const double *invw,
               const double *invden, double theta, int reverse,
               double complex *x, double *loss) {
    double complex *t = malloc((size_t)k*sizeof(*t));
    if (!t) return 1;
    if (loss) memset(loss,0,(size_t)k*sizeof(*loss));
    for (int step=0;step<m;step++) {
        int r=reverse ? m-1-step : step;
        memset(t,0,(size_t)k*sizeof(*t));
        for (int p=ptr[r];p<ptr[r+1];p++) {
            int c=idx[p]; if(c<0 || c>=nc){free(t);return 2;}
            const double complex *xr=x+(size_t)c*k;
            double complex ar=a[p];
            for(int j=0;j<k;j++) t[j]+=ar*xr[j];
        }
        if(loss) for(int j=0;j<k;j++)
            loss[j]+=theta*(2.0-theta)*invden[r]*
                     (creal(t[j])*creal(t[j])+cimag(t[j])*cimag(t[j]));
        for (int p=ptr[r];p<ptr[r+1];p++) {
            int c=idx[p];
            double complex *xr=x+(size_t)c*k;
            double complex fac=theta*invden[r]*invw[c]*conj(a[p]);
            for(int j=0;j<k;j++) xr[j]-=fac*t[j];
        }
    }
    free(t); return 0;
}
