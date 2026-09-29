#include "PDE.h"
#include <cmath>
#include <cstdio>
#include <omp.h>

// Compare the tiled, parallel SGS sweep with the original pointwise order.
static bool check(int ny, int nx, int threads)
{
    PDE pde(1, 1, nx, ny);
    Grid rhs(nx, ny), expected(nx, ny), actual(nx, ny), stencil(nx, ny);
    rhs.fill([](int i, int j) { return 0.01*i + 0.03*j; });
    const double hx = 1.0/(nx+1), hy = 1.0/(ny+1);
    const double wx = 1.0/(hx*hx), wy = 1.0/(hy*hy);
    const double invDiag = 1.0/(2.0*wx+2.0*wy);

    for (int j=1; j<=ny; ++j)
        for (int i=1; i<=nx; ++i)
            expected(j,i) = invDiag*(rhs(j,i) + wy*expected(j-1,i) + wx*expected(j,i-1));
    for (int j=ny; j>=1; --j)
        for (int i=nx; i>=1; --i)
            expected(j,i) += invDiag*(wy*expected(j+1,i) + wx*expected(j,i+1));

    omp_set_num_threads(threads);
    pde.GSPreCon(&rhs, &actual);
    double maxError = 0.0;
    for (int j=1; j<=ny; ++j)
        for (int i=1; i<=nx; ++i)
            maxError = std::fmax(maxError, std::fabs(expected(j,i)-actual(j,i)));

    // A second call must not depend on the old interior values of actual.
    pde.GSPreCon(&rhs, &actual);
    for (int j=1; j<=ny; ++j)
        for (int i=1; i<=nx; ++i)
            maxError = std::fmax(maxError, std::fabs(expected(j,i)-actual(j,i)));

    pde.applyStencil(&stencil, &rhs);
    for (int j=1; j<=ny; ++j)
        for (int i=1; i<=nx; ++i) {
            const double reference = (2*wx+2*wy)*rhs(j,i)
                - wy*(rhs(j-1,i)+rhs(j+1,i))
                - wx*(rhs(j,i-1)+rhs(j,i+1));
            maxError = std::fmax(maxError, std::fabs(reference-stencil(j,i)));
        }
    std::printf("%d x %d, %d threads: max absolute error %.3e\n", ny, nx, threads, maxError);
    return maxError < 1e-10;
}

int main()
{
    bool okay = true;
    for (int threads : {1, 4, 8}) {
        okay &= check(257, 129, threads);
        okay &= check(129, 4097, threads);
        okay &= check(4097, 129, threads);
        okay &= check(513, 1024, threads);
    }
    return okay ? 0 : 1;
}
