#ifndef PARALLEL_FOR_HPP
#define PARALLEL_FOR_HPP
#include <future>
#include <functional>
#include "Geometry.hpp"

const int hpx_nprocs = 4;
const int hpx_chunksize = 1<<12;

template<typename Func,typename Iter,typename Reduce>
auto parallel_sum(
    int procs,
    const int min_chunk_size,
    Func f,
    Iter ilo,Iter ihi,
    Reduce r,
    decltype(f(ilo)) ident) -> decltype(f(ilo))
{
  const int chunk_size = (ihi - ilo)/2;
  typedef decltype(f(ilo)) value_type;
  if(procs > 1 && chunk_size > min_chunk_size) {
    Iter imid = ilo + chunk_size;
    const int p1 = procs/2;
    const int p2 = procs-p1-1;
    std::future<value_type> f1 = std::async(std::launch::deferred,
      parallel_sum<Func,Iter,Reduce>,
      p1,min_chunk_size,
      f,ilo,imid,
      r,ident);
    std::future<value_type> f2 = std::async(std::launch::async,
      parallel_sum<Func,Iter,Reduce>,
      p2,min_chunk_size,
      f,imid,ihi,
      r,ident);
    return f1.get()+f2.get();
  } else {
    value_type result = ident;
    for(Iter i=ilo;i != ihi;++i) {
      r(result, f(i));
    }
    return result;
  }
}
#endif
