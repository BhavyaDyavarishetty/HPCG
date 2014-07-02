

//
// HPCG: High Performance Conjugate Gradient Benchmark
//
// Contact:
// Michael A. Heroux ( maherou@sandia.gov)
// Jack Dongarra     (dongarra@eecs.utk.edu)
// Piotr Luszczek    (luszczek@eecs.utk.edu)
//
// ***************************************************
//@HEADER

/*!
 @file ComputeDotProduct_ref.cpp

 HPCG routine
 */

#ifndef HPCG_NOHPX
#include "parallel_for.hpp"
#endif
#ifndef HPCG_NOMPI
#include <mpi.h>
#include "mytimer.hpp"
#endif
#ifndef HPCG_NOOPENMP
#include <omp.h>
#endif
#include <cassert>
#include "ComputeDotProduct_ref.hpp"
#include<hpx/include/algorithm.hpp>

/*!
  Routine to compute the dot product of two vectors where:

  This is the reference dot-product implementation.  It _CANNOT_ be modified for the
  purposes of this benchmark.

  @param[in] n the number of vector elements (on this processor)
  @param[in] x, y the input vectors
  @param[in] result a pointer to scalar value, on exit will contain result.
  @param[out] time_allreduce the time it took to perform the communication between processes

  @return returns 0 upon success and non-zero otherwise

  @see ComputeDotProduct
*/
int ComputeDotProduct_ref(const local_int_t n, const Vector & x, const Vector & y,
    double & result, double & time_allreduce) {
  assert(x.localLength>=n); // Test vector lengths
  assert(y.localLength>=n);
double arr[n];
double *a= arr;
double *r=arr;
  double local_result = 0.0;
  double * xv = x.values;
  double * yv = y.values;
#ifndef HPCG_NOHPX
  auto plus = [](double& a,const double& b) {
    a += b;
  };
  if(yv == xv) {
    auto f = [xv](local_int_t i) {
      return xv[i]*xv[i];
    };
 local_result = hpx::parallel::reduce(hpx::parallel::par, xv, xv+n, local_result, [] (double &a, double &b)
 {
 return b*b;
 });
/* 
local_result += parallel_sum<decltype(f),local_int_t,decltype(plus)>(
      hpx_nprocs,hpx_chunksize,f,0,n,plus,0);*/

  }
 else {
    auto f = [xv,yv](local_int_t i) {
      return xv[i]*yv[i];
    };
hpx::parallel::transform(hpx::parallel::par, xv, xv+n, xy, r, [](double a, double b){
return a*b;
});
local_result= hpx::parallel::reduce(hpx::parallel::par, r,r+n, local_result, [] (double a, double b){return a+b;});
   /* local_result += parallel_sum<decltype(f),local_int_t,decltype(plus)>(
      hpx_nprocs,hpx_chunksize,f,0,n,plus,0);*/
  }
#else
  if (yv==xv) {
#ifndef HPCG_NOOPENMP
    #pragma omp parallel for reduction (+:local_result)
#endif
    for (local_int_t i=0; i<n; i++) local_result += xv[i]*xv[i];
  } else {
#ifndef HPCG_NOOPENMP
    #pragma omp parallel for reduction (+:local_result)
#endif
    for (local_int_t i=0; i<n; i++) local_result += xv[i]*yv[i];
  }
#endif

#ifndef HPCG_NOMPI
  // Use MPI's reduce function to collect all partial sums
  double t0 = mytimer();
  double global_result = 0.0;
  MPI_Allreduce(&local_result, &global_result, 1, MPI_DOUBLE, MPI_SUM,
      MPI_COMM_WORLD);
  result = global_result;
  time_allreduce += mytimer() - t0;
#else
  result = local_result;
#endif

  return(0);
}
