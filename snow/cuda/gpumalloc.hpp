/**
 * gpumalloc.hpp - Defines the function which allocates data on the GPU
 * which is imported from CUDA.
 *
 * @author Mars Semenova
 */

#ifndef GPUMALLOC_HPP
#define GPUMALLOC_HPP

extern void* mallocGPU(size_t size);

#endif
