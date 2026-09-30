#include "Solver.h"
#include "Grid.h"
#ifdef LIKWID_PERFMON
#include <likwid.h>
#endif

#define IS_VALID(a)\
	!(std::isnan(a) || std::isinf(a))

SolverClass::SolverClass(PDE *pde_, Grid *x_, Grid *b_):pde(pde_),x(x_),b(b_)
{
}

// One pass updates both vectors and obtains the stopping criterion. This
// replaces two AXPBY traversals and a separate residual dot product.
static double updateSolutionResidual(Grid *x, Grid *r, Grid *p, Grid *v,
                                     double lambda)
{
    double norm = 0.0;
    const int columns = x->columns;
    const int rows = x->rows;
#pragma omp parallel reduction(+:norm) if((rows-2)*(columns-2) >= 32768)
    {
#ifdef LIKWID_PERFMON
    LIKWID_MARKER_START("UPDATE_RESIDUAL");
#endif
#pragma omp for schedule(static)
    for (int j=1; j<rows-1; ++j) {
        const int base = j*columns;
        for (int i=1; i<columns-1; ++i) {
            const int index = base+i;
            x->arrayPtr[index] += lambda*p->arrayPtr[index];
            const double residual = r->arrayPtr[index] - lambda*v->arrayPtr[index];
            r->arrayPtr[index] = residual;
            norm += residual*residual;
        }
    }
#ifdef LIKWID_PERFMON
    LIKWID_MARKER_STOP("UPDATE_RESIDUAL");
#endif
    }
    return norm;
}

int SolverClass::CG(int niter, double tol)
{
    Grid *p = new Grid(pde->numGrids_x(), pde->numGrids_y());
    Grid *v = new Grid(pde->numGrids_x(), pde->numGrids_y());

    int iter = 0;
    double lambda = 0;
    double alpha_0 = 0, alpha_1 = 0;
    //Calculate residual
    //p=A*x
    pde->applyStencil(p,x);
    //p=b-A*x
    axpby(p,1,b,-1,p);
    //calculate alpha_0
    alpha_0 = dotProduct(p,p);

    Grid *r = new Grid(*p);

    START_TIMER(CG);

    while( (iter<niter) && (alpha_0>tol*tol) && (IS_VALID(alpha_0)) )
    {
        pde->applyStencil(v,p);
        lambda =  alpha_0/dotProduct(v,p);
        alpha_1 = updateSolutionResidual(x,r,p,v,lambda);
        //Update p
        axpby(p, 1.0, r, alpha_1/alpha_0, p);
        alpha_0 = alpha_1;
#ifdef DEBUG
        printf("iter = %d, res = %.15e\n", iter, alpha_0);
#endif
        ++iter;
    }

    STOP_TIMER(CG);

    if( !IS_VALID(alpha_0) ){
        printf("\x1B[31mWARNING: NaN/INF detected after iteration %d\x1B[0m\n", iter);
    }

    delete p;
    delete v;
    delete r;

    return (iter);
}

int SolverClass::PCG(int niter, double tol)
{
    Grid* r = new Grid(pde->numGrids_x(), pde->numGrids_y());
    Grid* z = new Grid(pde->numGrids_x(), pde->numGrids_y());
    Grid* v = new Grid(pde->numGrids_x(), pde->numGrids_y());

    int iter = 0;
    double lambda = 0;
    double alpha_0 = 0, alpha_1 = 0;
    double res_norm_sq = 0;
    //Calculate residual
    pde->applyStencil(r,x);
    axpby(r,1,b,-1,r);
    res_norm_sq = dotProduct(r,r);
    pde->GSPreCon(r,z);

    alpha_0 = dotProduct(r,z);
    Grid* p = new Grid(*z);

    START_TIMER(PCG);

    while( (iter<niter) && (res_norm_sq>tol*tol) && (IS_VALID(res_norm_sq)) )
    {
        pde->applyStencil(v,p);
        lambda =  alpha_0/dotProduct(v,p);
        res_norm_sq = updateSolutionResidual(x,r,p,v,lambda);
        //Update z
        pde->GSPreCon(r, z);
        alpha_1 = dotProduct(r,z);
        //Update p
        axpby(p, 1.0, z, alpha_1/alpha_0, p);
        alpha_0 = alpha_1;

#ifdef DEBUG
        printf("iter = %d, res = %.15e\n", iter, res_norm_sq);
#endif
        ++iter;
    }

    STOP_TIMER(PCG);

    if( !IS_VALID(res_norm_sq) ){
        printf("\x1B[31mWARNING: NaN/INF detected after iteration %d\x1B[0m\n", iter);
    }

    delete r;
    delete z;
    delete v;
    delete p;

    return (iter);
}
