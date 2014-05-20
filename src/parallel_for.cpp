#include "parallel_for.hpp"
#include <future>
#include <functional>
#include <atomic>

/*
std::atomic<int> max_threads(0);

template<typename Func,typename Iter,typename Reduce>
auto parallel_sum(
    Func f,
    Iter ilo,Inter ihi,
    Reduc r,
    decltype(f(ilo)) ident,
    const local_int_t min_chunk_size) -> decltype(f(ilo))
{
  const local_int_t imid = (ihi+ilo)>>1;
  if(ilo+min_chunk_size <= imid) {
    int nn = ++max_threads;
    if(nn >= 256) {
      --max_threads;
      return f(ilo,ihi);
    }
    std::future<double> f1 = std::async(std::launch::deferred,parallel_sum,f,ilo,imid,min_chunk_size);
    std::future<double> f2 = std::async(std::launch::async,parallel_sum,f,imid,ihi,min_chunk_size);
    return f1.get()+f2.get();
  } else {
    return f(ilo,ihi);
  }
}
*/
