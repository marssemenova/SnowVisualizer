/**
 * gpu.cu - Contains the CUDA code which accelerates the
 * program on the GPU.
 *
 * @author Mars Semenova
 */

#include <vector>
#include <stdio.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>

using namespace std;

#include "../SnowGeneratorData.hpp"

#define GRAVITY 0.2f
#define SNOW_NOISE_Y 0.1f // TODO: ok to def?

// kernel params
// apply grav
#define SNOW_GRAV_BLOCK_SIZE 1024
#define SNOW_GRAV_BATCH_SIZE 8
unsigned h_snowGravNumBlocks;
// update snow
#define SNOW_UPDATE_BLOCK_SIZE 1024
unsigned h_snowUpdateNumBlocks;

// host vars
float *h_verts;
SnowflakeData *h_snowflakeData;
unsigned h_numPolys;
unsigned h_numParticles;

// dev vars
float *d_verts;
float *d_snowflakeDataFlat;
__constant__ __device__ float d_extent[6];
unsigned *d_numParticles;
float *d_snowOffsets;

/**
 * Generate a random number usind curand. Source: https://codingbyexample.com/2020/09/15/curand/.
 * @param seed - Seed.
 * @param kernelInd - Index of thread.
 * @param threadCallCount - Call # in kernel.
 * @return Generated float.
 */
__device__ float getRandom(uint64_t seed, int kernelInd, int threadCallCount) {
    curandState s;
    curand_init(seed + kernelInd + threadCallCount, 0, 0, &s);
    return curand_uniform(&s);
}

/**
 * Get random float in the range [min, max) on the GPU.
 * @param min - Min float generated.
 * @param max - Max float generated, non-inclusive.
 * @param seed - Seed.
 * @param kernelInd - Index of thread.
 * @param threadCallCount - Call # in kernel.
 * @return Generated float.
 */
__device__ float getRandFloatGPU(float min, float max, uint64_t seed, int kernelInd, int threadCallCount) {
    if (min == max) {
        return min;
    }
    return min + (max - min) * (float) getRandom(seed, kernelInd, threadCallCount);
}

/**
 * Kernel which applies gravity to snow particles and stores the offset.
 * @param d_snowflakeDataFlat - Flattened snowflake data. Every 3 elements correspond
 * to a snow particle's coordinates.
 * @param d_snowOffsets - The offsets array into which to write. Every 5 elements correspond to a
 * particle's data, where the first 3 elements are the offset, the 4th element
 * is the number of polygons in the snowflake, and the 5th parameter is the first index of its
 * vertices in the vertices array.
 */
__global__ void snowApplyGrav(float *d_snowflakeDataFlat, unsigned *d_numParticles, float *d_snowOffsets, uint64_t seed) {
    unsigned kernelInd = blockIdx.x*SNOW_GRAV_BLOCK_SIZE*SNOW_GRAV_BATCH_SIZE + threadIdx.x;
    unsigned stride = SNOW_GRAV_BLOCK_SIZE;
    float xOffset, yOffset, zOffset;
    float yOffsetRand;
    for (int x = kernelInd; x < min(kernelInd + SNOW_GRAV_BATCH_SIZE*stride, *d_numParticles); x+=stride) {
        if (d_snowflakeDataFlat[3*x+1] < d_extent[2]) {
            //xOffset = getRandFloatGPU(d_extent[0], d_extent[1], seed, kernelInd, 0) - d_snowflakeDataFlat[3*x];
            //yOffsetRand = getRandFloatGPU(-(SNOW_NOISE_Y*(d_extent[3] - d_extent[2])), SNOW_NOISE_Y*(d_extent[3] - d_extent[2]), seed, kernelInd, 1);
            //zOffset = getRandFloatGPU(d_extent[4], d_extent[5], seed, kernelInd, 2) - d_snowflakeDataFlat[3*x+2];
            xOffset = 0;
            yOffsetRand = 0;
            zOffset = 0;
            yOffset = (d_extent[3] - d_extent[2]) + yOffsetRand;
        } else {
            yOffset = -GRAVITY;
            xOffset = 0;
            zOffset = 0;
        }

        // update x
         //d_snowOffsets[5*x] = xOffset;
         //d_snowflakeDataFlat[3*x] += xOffset;

        // update y
        d_snowOffsets[5*x+1] = yOffset;
        d_snowflakeDataFlat[3*x+1] += yOffset;

        // update z
         //d_snowOffsets[5*x+2] = zOffset;
         //d_snowflakeDataFlat[3*x+2] += zOffset;
    }
}

/**
 * Kernel which updates snow vertices and normals based on previously computed offsets.
 * @param d_verts - Vertice data on the GPU.
 * @param d_snowOffsets - Computed offset. Every 5 elements correspond to a
 * particle's data, where the first 3 elements are the offset, the 4th element
 * is the number of polygons in the snowflake, and the 5th parameter is the first index of its
 * vertices in the vertices array.
 */
__global__ void snowUpdate(float *d_verts, float *d_snowOffsets, unsigned *d_numParticles) {
    unsigned kernelInd = blockIdx.x*SNOW_UPDATE_BLOCK_SIZE + threadIdx.x;
    if (kernelInd < *d_numParticles) {
        unsigned currInd = d_snowOffsets[5*kernelInd + 4];
        for (int x = currInd; x < currInd + d_snowOffsets[5*kernelInd + 3]*9; x+=3) {
            for (int i = 0; i < 3; i++) {
                d_verts[x + i] += d_snowOffsets[5*kernelInd + i];
            }
        }
    }
}

/**
 * Initialize data on the GPU.
 * @param data - SnowGeneratorData object with particle data.
 * @param numParticles - Number of particles.
 * @param extent - Extent of volume in which to generate the particles, where extent[0] is a pair for the x extent,
 * extent[1] is a pair for the y extent, and extent[2] is a pair for the z extent. If numParticles = 1 this
 * parameter has no effect and the snow particle is generated at the origin.
 */
extern void snowInitGPU(SnowGeneratorData data, unsigned numParticles, float extent[3][2]) {
    // save refs
    h_numPolys = data.numPolys;
    h_numParticles = numParticles;
    h_verts = data.verts;
    h_snowGravNumBlocks = ceil(h_numParticles/(SNOW_GRAV_BLOCK_SIZE*SNOW_GRAV_BATCH_SIZE*1.0)); // TODO: mem access block size (256) dictates snow batch size
    h_snowUpdateNumBlocks = ceil(h_numParticles/(SNOW_UPDATE_BLOCK_SIZE*1.0));
    h_snowflakeData = data.snowflakeData;

    // flatten snowflakeData
    float *h_snowflakeDataFlat;
    float *h_snowOffsets;
    cudaMallocHost((void**)&h_snowflakeDataFlat, 3*h_numParticles*sizeof(float));
    cudaMallocHost((void**)&h_snowOffsets, 5*h_numParticles*sizeof(float));
    for (int x = 0; x < h_numParticles; x++) {
        for (int i = 0; i < 3; i++) {
            h_snowflakeDataFlat[3*x+i] = h_snowflakeData[x].pos[i];
        }
        h_snowOffsets[5*x+3] = h_snowflakeData[x].numPolys;
        h_snowOffsets[5*x+4] = h_snowflakeData[x].ind;
    }

    // flatten extent
    float *h_extent;
    cudaMallocHost((void**)&h_extent, 6*sizeof(float));
    for (int x = 0; x < 3; x++) {
        for (int i = 0; i < 2; i++) {
            h_extent[2*x+i] = extent[x][i];
        }
    }

    // copy data to host
    cudaMalloc((void**)&d_verts, h_numPolys*9*sizeof(float));
    cudaMalloc((void**)&d_snowflakeDataFlat, 3*h_numParticles*sizeof(float));
    cudaMalloc((void**)&d_snowOffsets, 5*h_numParticles*sizeof(float));
    cudaMalloc((void**)&d_numParticles, sizeof(unsigned));
    cudaMemcpy(d_verts, h_verts, h_numPolys*9*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_snowflakeDataFlat, h_snowflakeDataFlat, 3*h_numParticles*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_snowOffsets, h_snowOffsets, 5*h_numParticles*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_numParticles, &h_numParticles, sizeof(unsigned), cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(d_extent, h_extent, 6*sizeof(float));

    // free
    cudaFree(h_snowflakeDataFlat);
    cudaFree(h_snowOffsets);
    cudaFree(h_extent);
}

/**
 * Update snow on the GPU via kernel calls.
 */
extern void snowUpdateGPU() {
    // dispatch kernel
    cudaDeviceSynchronize();
    snowApplyGrav<<<h_snowGravNumBlocks,SNOW_GRAV_BLOCK_SIZE>>>(d_snowflakeDataFlat, d_numParticles, d_snowOffsets, time(NULL));
    cudaDeviceSynchronize();
    snowUpdate<<<h_snowUpdateNumBlocks,SNOW_UPDATE_BLOCK_SIZE>>>(d_verts, d_snowOffsets, d_numParticles);
    cudaDeviceSynchronize();

    // fetch work
    cudaMemcpy(h_verts, d_verts, h_numPolys*9*sizeof(float), cudaMemcpyDeviceToHost);
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