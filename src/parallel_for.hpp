#ifndef PARALLEL_FOR_HPP
#define PARALLEL_FOR_HPP
#include <functional>
#include "Geometry.hpp"

extern double parallel_sum(
    std::function<double(int,int)> f,
    const local_int_t ilo, const local_int_t ihi,
    const local_int_t min_chunk_size);
#endif
