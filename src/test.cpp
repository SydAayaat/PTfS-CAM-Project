#include "Grid.h"
#include "PDE.h"
#include "Solver.h"
#include "test_macros.h"
#include <cmath>

double uSineFunc(int i, int j, double h_x, double h_y)
{
    return sin(M_PI*i*h_x)*sin(M_PI*j*h_y);
}

double rhsSineFunc(int i, int j, double h_x, double h_y)
{
    return 2*M_PI*M_PI*sin(M_PI*i*h_x)*sin(M_PI*j*h_y);
}

double rhsLinearFunc(int i, int j, double h_x, double h_y)
{
    return 2*i*h_x + 2*j*h_y;
}

#define RESIDUAL(res_norm_sq_, res_vec_, b_, x_)\
    laplace.applyStencil(&res_vec_,&x_);\
    axpby(&res_vec_,1.0,&b_,-1.0,&res_vec_);\
    res_norm_sq_ = dotProduct(&res_vec_, &res_vec_);

//for test we set u = sin(πx)*sin(πy); for x,y ∈ [0,1]
//            =>-Δu = 2π²*sin(πx)*sin(πy) = 2π²*u
//To test Solver we solve the inverse problem
int main()
{
    TESTS_START(13);

    int nx = 199, ny = 99; //nx and ny contains only inner points
    PDE laplace(1,1,nx,ny);
    Grid u_sine(nx,ny);
    Grid u_linear(nx,ny);
    Grid rhs_sine(nx,ny);
    Grid rhs_linear(nx,ny);
    laplace.initFunc = uSineFunc;
    laplace.init(&u_sine);
    laplace.initFunc = rhsSineFunc;
    laplace.init(&rhs_sine);
    laplace.initFunc = rhsLinearFunc;
    laplace.init(&rhs_linear);
    Grid residual(nx,ny);
    residual.fill(std::nan("0"));
    Grid zero(nx,ny);
    zero.fill(0);
    Grid one(nx,ny);
    one.fill(1);
    Grid two(nx,ny);
    two.fill(2);
    double err_norm_1 = 0, err_norm_2 = 0;

    //Check dot product operation
    err_norm_1 = dotProduct(&one, &one);
    err_norm_2 = nx*ny;
    CHECK_ERR(err_norm_1-err_norm_2, 1e-12, "dotProduct sum test");
#ifdef DEBUG
    printf("dotProduct = %f,  analytical = %f\n", err_norm_1, err_norm_2);
#endif
    err_norm_1 = dotProduct(&one, &zero);
    CHECK_ERR(err_norm_1, 1e-12, "dotProduct product test");
#ifdef DEBUG
    printf("dotProduct = %f,  analytical = 0\n", err_norm_1);
#endif
    err_norm_1 = dotProduct(&one, &two);
    CHECK_ERR(err_norm_1 - 2 * err_norm_2 , 1e-12, "dotProduct product test"); 
#ifdef DEBUG
    printf("dotProduct = %f,  analytical = %f\n", err_norm_1, 2 * err_norm_2);
#endif


    //Check axpby operations
    axpby(&residual,1.0,&u_sine,-1.0,&u_sine);
    err_norm_1  = dotProduct(&residual, &residual);
    CHECK_ERR(err_norm_1, 1e-12, "axpy");
#ifdef DEBUG
	printf("axpby = %f, analytical = 0\n", err_norm_1);
#endif
    axpby(&residual,1.0,&u_sine,1.0,&u_linear);
    err_norm_1  = dotProduct(&residual, &residual);
    CHECK_ERR(err_norm_1 - 5000, 1e-10, "axpy");
#ifdef DEBUG
	printf("axpby = %f, analytical = 5000\n", err_norm_1);
#endif
    axpby(&residual,1.0,&rhs_sine,-1.0,&u_sine);
    err_norm_1 = dotProduct(&residual, &residual);
#ifdef DEBUG
    printf("L2 Error before applying operator = %.9e\n", err_norm_1);
#endif

    //Check Stencil operation
    Grid rhs_fdm(nx,ny);
    laplace.applyStencil(&rhs_fdm,&u_sine);
    axpby(&residual,1.0,&rhs_sine,-1.0,&rhs_fdm);
    err_norm_2 = dotProduct(&residual, &residual);
#ifdef DEBUG
    printf("L2 Error after applying operator = %.9e\n", err_norm_2);
#endif
    CHECK_ERR(err_norm_2, 1e-2, "PDE::applyStencil");

    // Check Precon operation
    Grid precon(nx, ny);
    laplace.GSPreCon(&one, &precon);
    Grid ref(nx, ny);
    ref.readFile("./include/PRECON.test", false);
    CHECK_GRIDS(precon, ref, 1e-7, "PDE::GSPreCon");

    //Now check CG solver
    Grid x(nx,ny);
    x.rand();
    int niter_cg = 850;
    int iter_sine_cg = laplace.solve(&x, &rhs_sine, CG, niter_cg);
    printf("CG iterations for Sine problem = %d\n", iter_sine_cg);
    axpby(&residual, 1.0, &u_sine, -1.0, &x);
    double err_sine_cg = dotProduct(&residual, &residual);
    double res_sine_cg;
    RESIDUAL(res_sine_cg, residual, rhs_sine, x);
    CHECK_ERR(sqrt(res_sine_cg), 1e-7, "Solver::CG - sine residual check");
    CHECK_ERR(sqrt(err_sine_cg), 1e-2, "Solver::CG - sine error check");
#ifdef DEBUG
    printf("Curr residual CG, sine = %.9e\n", sqrt(res_sine_cg));
    printf("Curr error CG, sine = %.9e\n", sqrt(err_sine_cg));
#endif
    writeGnuplotFile("./results/CG_sine.dat", x, 1, 1, false);

    x.rand();
    int iter_linear_cg = laplace.solve(&x, &rhs_linear, CG, niter_cg);
    printf("CG iterations for Linear problem = %d\n", iter_linear_cg);
    double res_linear_cg;
    RESIDUAL(res_linear_cg, residual, rhs_linear, x);
    CHECK_ERR(sqrt(res_linear_cg), 1e-7, "Solver::CG - linear residual check");
#ifdef DEBUG
    printf("Curr residual CG,  linear = %.9e\n", sqrt(res_linear_cg));
#endif
    writeGnuplotFile("./results/CG_linear.dat", x, 1, 1, false);

    //Now check PCG solver
    x.rand();
    int niter_pcg = 350;
    int iter_sine_pcg = laplace.solve(&x, &rhs_sine, PCG, niter_pcg);
    printf("PCG iterations for Sine problem = %d\n", iter_sine_pcg);
    axpby(&residual, 1.0, &u_sine, -1.0, &x);
    double err_sine_pcg = dotProduct(&residual, &residual);
    double res_sine_pcg;
    RESIDUAL(res_sine_pcg, residual, rhs_sine, x);
    CHECK_ERR(sqrt(res_sine_pcg), 1e-7, "Solver::PCG - sine residual check");
    CHECK_ERR(sqrt(err_sine_pcg), 1e-2, "Solver::PCG - sine error check");
#ifdef DEBUG
    printf("Curr residual PCG, sine = %.9e\n", sqrt(res_sine_pcg));
    printf("Curr error PCG, sine = %.9e\n", sqrt(err_sine_pcg));
#endif
    writeGnuplotFile("./results/PCG_sine.dat", x, 1, 1, false);

    x.rand();
    int iter_linear_pcg = laplace.solve(&x, &rhs_linear, PCG, niter_pcg);
    printf("PCG iterations for Linear problem = %d\n", iter_linear_pcg);
    double res_linear_pcg;
    RESIDUAL(res_linear_pcg, residual, rhs_linear, x);
    CHECK_ERR(sqrt(res_linear_pcg), 1e-7, "Solver::PCG - linear residual check");
#ifdef DEBUG
    printf("Curr residual PCG, linear = %.9e\n", sqrt(res_linear_pcg));
#endif
    writeGnuplotFile("./results/PCG_linear.dat", x, 1, 1, false);

    TESTS_END;
}
