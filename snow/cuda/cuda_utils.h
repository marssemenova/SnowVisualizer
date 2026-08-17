/**
 * cuda_utils.h - Contains util functions for GPU code.
 *
 * @author Mars Semenova
 */

#ifndef CUDA_UTILS_H
#define CUDA_UTILS_H

#include "cuda_errors.h"
#include "../../util/CPUTimer.hpp"

/**
 * Get random float in the range [min, max) on the GPU.
 * @param min - Min float generated.
 * @param max - Max float generated, non-inclusive.
 * @return Generated float.
 */
__device__ __forceinline__ float getRandFloatGPU(float min, float max, curandState* localState) {
    return min + (max - min) * curand_uniform(localState);
}

/**
 * Allocate memory via cudaMallocHost.
 * @param size - Size of the memory to allocate.
 * @return A pointer to the allocated data.
 */
extern void* mallocGPU(size_t size) {
    void *ptr;
    cudaMallocHost((void**)&ptr, size);
    return ptr;
}

#endif
