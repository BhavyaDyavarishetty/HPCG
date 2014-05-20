
//@HEADER
// ***************************************************
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
 @file ComputeWAXPBY_ref.cpp

 HPCG routine
 */

#ifndef HPCG_NOHPX
#include "parallel_for.hpp"
#endif
#include "ComputeWAXPBY_ref.hpp"
#ifndef HPCG_NOOPENMP
#include <omp.h>
#endif
#include <cassert>
/*!
  Routine to compute the update of a vector with the sum of two
  scaled vectors where: w = alpha*x + beta*y

  This is the reference WAXPBY impmentation.  It CANNOT be modified for the
  purposes of this benchmark.

  @param[in] n the number of vector elements (on this processor)
  @param[in] alpha, beta the scalars applied to x and y respectively.
  @param[in] x, y the input vectors
  @param[out] w the output vector.

  @return returns 0 upon success and non-zero otherwise

  @see ComputeWAXPBY
*/
int ComputeWAXPBY_ref(const local_int_t n, const double alpha, const Vector & x,
    const double beta, const Vector & y, Vector & w) {

	assert(x.localLength>=n); // Test vector lengths
	assert(y.localLength>=n);

	const double * const xv = x.values;
	const double * const yv = y.values;
	double * const wv = w.values;
  if (alpha==1.0) {
#ifndef HPCG_NOHPX
    auto r = [](double&,double){ return 0.0; };
    auto f = [xv,yv,&beta,wv](local_int_t i) {
      wv[i] = xv[i] + beta * yv[i];
      return 0.0;
    };
    parallel_sum<decltype(f),local_int_t,decltype(r)>(
      hpx_nprocs,hpx_chunksize,f,0,n,r,0);
#else
#ifndef HPCG_NOOPENMP
    #pragma omp parallel for
#endif
    for (local_int_t i=0; i<n; i++) wv[i] = xv[i] + beta * yv[i];
#endif
  } else if (beta==1.0) {
#ifndef HPCG_NOHPX
    auto r = [](double&,double){ return 0.0; };
    auto f = [xv,yv,&alpha,wv](local_int_t i) {
      wv[i] = alpha * xv[i] + yv[i];
      return 0.0;
    };
    parallel_sum<decltype(f),local_int_t,decltype(r)>(
      hpx_nprocs,hpx_chunksize,f,0,n,r,0);
#else
#ifndef HPCG_NOOPENMP
    #pragma omp parallel for
#endif
    for (local_int_t i=0; i<n; i++) wv[i] = alpha * xv[i] + yv[i];
#endif
  } else  {
#ifndef HPCG_NOHPX
    auto r = [](double&,double){ return 0.0; };
    auto f = [xv,yv,&alpha,&beta,wv](local_int_t i) {
      wv[i] = alpha * xv[i] + beta * yv[i];
      return 0.0;
    };
    parallel_sum<decltype(f),local_int_t,decltype(r)>(
      hpx_nprocs,hpx_chunksize,f,0,n,r,0);
#else
#ifndef HPCG_NOOPENMP
    #pragma omp parallel for
#endif
    for (local_int_t i=0; i<n; i++) wv[i] = alpha * xv[i] + beta * yv[i];
#endif
  }

  return(0);
}
