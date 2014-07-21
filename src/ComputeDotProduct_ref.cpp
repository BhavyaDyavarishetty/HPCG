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
  #include<hpx/hpx_init.hpp>                                                       
  #include<hpx/hpx.hpp>                                                            
  #include<hpx/include/algorithm.hpp>                                              
  #include<hpx/include/numeric.hpp>                                                
  #include<boost/iterator/counting_iterator.hpp>                                   
  #include"parallel_for.hpp"                                                       
  #endif                                                                           
  #ifndef HPCG_NOMPI                                                               
  #include <mpi.h>                                                                 
  #include "mytimer.hpp"                                                           
  #endif                                                                           
  #ifndef HPCG_NOOPENMP                                                            
  #include <omp.h>                                                                 
  #endif                                                                           
  #include<time.h>                                                                 
  #include <cassert>                                                               
  #include "ComputeDotProduct_ref.hpp"                                             
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
                                                                                   
#ifndef HPCG_NOHPX
  using hpx::util::make_zip_iterator;                                              
  using hpx::util::tuple;                                                          
  using hpx::util::make_tuple;                                                     
  using hpx::util::get;                                                            
#endif                                                                                   
                                                                                   
  int ComputeDotProduct_ref(const local_int_t n, const Vector & x, const Vector & y,
      double & result, double & time_allreduce) {                                  
    assert(x.localLength>=n); // Test vector lengths                               
    assert(y.localLength>=n);                                                      
    double local_result = 0.0;                                                     
    double * xv = x.values;                                                        
    double * yv = y.values;                                                        
  #ifndef HPCG_NOHPX                                                                
                                                            
    auto plus = [](double& a,const double& b) {                                    
      a += b;                                                                      
    };                                                                             
                                                                                   
    if(yv == xv) {                                                                 
     local_result =                                                                
          hpx::parallel::transform_reduce(                                                   
              hpx::parallel::par(hpx_chunksize),                                                  
              boost::counting_iterator<size_t>(0),                                 
              boost::counting_iterator<size_t>(n),                                 
              0.0,                                                                 
              std::plus<double>(),                                                 
              [&xv](size_t i){                                                     
                  return xv[i] * xv[i];                                            
              }                                                                    
          );                                                                       
                                                                                   
                                                                                   
    }                                                                              
   else {                                                                          
                                                                                   
      // the result of the execution will be stored in location 0 of the tuple     
      local_result =                                                               
          hpx::parallel::transform_reduce(                                                   
              hpx::parallel::par(hpx_chunksize),                                                  
              boost::counting_iterator<size_t>(0),                                 
              boost::counting_iterator<size_t>(n),                                 
              0.0,                            
              std::plus<double>(),                                                 
             [&xv,&yv](size_t i){                                                 
                 return xv[i] * yv[i];                                            
             }                                                                    
         );                                                                       
                                                                                  
   }                                                                              
                                                                                  
 #else                                                                            
 clock_t t1, t2;                                                                  
  t1=clock();                                                                     
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
 t2=clock();                                                                      
 float diff((float)t2-(float)t1);                                                 
 float seconds= diff/CLOCKS_PER_SEC;                                              
                                                    
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
