#include "Grid.h"
#include "PDE.h"
#include "Solver.h"
#include "test_macros.h"
#include <omp.h>

#ifdef LIKWID_PERFMON
    #include <likwid.h>
#endif

double uSineFunc(int i, int j, double h_x, double h_y)
{
    return sin(M_PI*i*h_x)*sin(M_PI*j*h_y);
}

double rhsSineFunc(int i, int j, double h_x, double h_y)
{
    return 2*M_PI*M_PI*sin(M_PI*i*h_x)*sin(M_PI*j*h_y);
}


#define RESIDUAL(res_norm_sq_, res_vec_, b_, x_)\
    laplace.applyStencil(&res_vec_,&x_);\
    axpby(&res_vec_,1.0,&b_,-1.0,&res_vec_);\
    res_norm_sq_ = dotProduct(&res_vec_, &res_vec_);



int main(const int argc, char* const argv[])
{
    if(argc < 3)
    {
        printf("Usage: %s <outer dimension y> <inner dimension x>\n", argv[0]);
        return 0;
    }

#ifdef LIKWID_PERFMON
    LIKWID_MARKER_INIT;
#endif

    int ny = atoi(argv[1]);
    int nx = atoi(argv[2]);
    int numThreads = 1;

#pragma omp parallel
    {
#pragma omp single
        {
            numThreads = omp_get_num_threads();
        }
    }

    printf("Total number of threads active = %d\n", numThreads);

    TESTS_START(4);

    PDE laplace(1,1,nx,ny);
    Grid u_sine(nx,ny);
    Grid rhs_sine(nx,ny);
    laplace.initFunc = uSineFunc;
    laplace.init(&u_sine);
    laplace.initFunc = rhsSineFunc;
    laplace.init(&rhs_sine);
    Grid residual(nx,ny);

    //Now check CG solver with fixed iteration number
    Grid x(nx,ny);
    x.rand();//fill(0);//rand();
    axpby(&residual, 1.0, &u_sine, -1.0, &x);
    double err_start = dotProduct(&residual, &residual);
    double res_start;
    RESIDUAL(res_start, residual, rhs_sine, x);
    int iter_sine_cg = laplace.solve(&x, &rhs_sine, CG, 20);
    printf("CG iterations = %d\n", iter_sine_cg);
    double cg_time = 0;
    GET_TIMER(cg_time, CG);
    double totLUP = nx*ny;
    printf("Performance CG = %f [MLUP/s]\n", static_cast<double>(iter_sine_cg*totLUP*1e-6)/cg_time);
    axpby(&residual, 1.0, &u_sine, -1.0, &x);
    double err_sine_cg = dotProduct(&residual, &residual);
    double res_sine_cg;
    RESIDUAL(res_sine_cg, residual, rhs_sine, x);
    CHECK_LESS_THAN(res_sine_cg, res_start, "Solver::CG - residual check");
    CHECK_LESS_THAN(err_sine_cg, err_start, "Solver::CG - error check");
#ifdef DEBUG
    printf("Initial residual = %.9e, curr residual CG = %.9e\n", sqrt(res_start), sqrt(res_sine_cg));
    printf("Initial error = %.9e, curr error CG = %.9e\n", sqrt(err_start), sqrt(err_sine_cg));
#endif

    //Now check PCG solver with fixed iteration number
    x.rand();//fill(0);//rand();
    int iter_sine_pcg = laplace.solve(&x, &rhs_sine, PCG, 20);
    printf("PCG iterations = %d\n", iter_sine_pcg);
    double pcg_time = 0;
    GET_TIMER(pcg_time, PCG);
    printf("Performance PCG = %f [MLUP/s]\n", static_cast<double>(iter_sine_pcg*totLUP*1e-6)/pcg_time);
    axpby(&residual, 1.0, &u_sine, -1.0, &x);
    double err_sine_pcg = dotProduct(&residual, &residual);
    double res_sine_pcg;
    RESIDUAL(res_sine_pcg, residual, rhs_sine, x);
    CHECK_LESS_THAN(res_sine_pcg, res_start, "Solver::PCG - residual check");
    CHECK_LESS_THAN(err_sine_pcg, err_start, "Solver::PCG - error check");
#ifdef DEBUG
    printf("Initial residual = %.9e, curr residual PCG = %.9e\n", sqrt(res_start), sqrt(res_sine_pcg));
    printf("Initial error = %.9e, curr error PCG = %.9e\n", sqrt(err_start), sqrt(err_sine_pcg));
#endif

    TESTS_END;

    PRINT_TIME_SUMMARY;

#ifdef LIKWID_PERFMON
    LIKWID_MARKER_CLOSE;
#endif


}
