/**
 * gpu.cu - Contains the CUDA code which accelerates the
 * program on the GPU.
 *
 * @author Mars Semenova
 */

#include <vector>
#include <stdio.h>
#include <cuda_runtime.h>

using namespace std;

#include "../SnowGeneratorData.hpp"

#define GRAVITY 2.0f

// snow kernel params
#define SNOW_BLOCK_SIZE 1024
#define SNOW_BATCH_SIZE 8

// host vars
float *h_verts;
float *h_normals;
SnowflakeData *h_snowflakeData;
float *h_snowflakeDataFlat;
unsigned h_snowNumBlocks;
unsigned h_numPolys;
unsigned h_numParticles;

// dev vars
float *d_verts;
float *d_normals;
float *d_snowflakeDataFlat;
__constant__ __device__ float d_extent[6];
__constant__ __device__ unsigned d_numParticles;

// TODO: remember abt cudaHostMalloc

__global__ void updateSnow(float *d_verts, float *d_normals, float *d_snowflakeDataFlat) { // TODO: pass params every time or make dev vars?
    unsigned kernelInd = blockIdx.x*SNOW_BLOCK_SIZE + threadIdx.x;
    unsigned alignment = SNOW_BLOCK_SIZE/SNOW_BATCH_SIZE; // TODO: check math
    if (kernelInd >= alignment) { // TODO: is there a way to avoid this if statement
        return;
    }
    unsigned currInd;
    float offset;
    for (int x = kernelInd; x < min(kernelInd + SNOW_BATCH_SIZE, d_numParticles); x+=alignment) {
        currInd = d_snowflakeDataFlat[5*x+4];
        if (d_snowflakeDataFlat[5*x+1] < d_extent[2]) {
            offset = d_extent[3] - d_snowflakeDataFlat[5*x+1];
        } else {
            offset = -GRAVITY;
        }
        for (int i = currInd; i < currInd + d_snowflakeDataFlat[5*x+3]*9; i+=3) {
            d_verts[i + 1] += offset;
        }
        d_snowflakeDataFlat[5*x+1] += offset;
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
extern void initSnowOnGPU(SnowGeneratorData data, unsigned numParticles, float extent[3][2]) {
    // save refs
    h_verts = data.verts;
    h_normals = data.normals;
    h_snowNumBlocks = ceil(numParticles/(SNOW_BLOCK_SIZE*SNOW_BATCH_SIZE*1.0)); // TODO: mem access block size (256) dictates snow batch size
    h_numPolys = data.numPolys;
    h_numParticles = numParticles;
    h_snowflakeData = data.snowflakeData;

    // flatten snowflakeData
    cudaMallocHost((void**)&h_snowflakeDataFlat, 5*h_numParticles*sizeof(float));
    for (int x = 0; x < h_numParticles; x++) {
        for (int i = 0; i < 3; i++) {
            h_snowflakeDataFlat[5*x+i] = h_snowflakeData[x].pos[i];
        }
        h_snowflakeDataFlat[5*x+3] = h_snowflakeData[x].numPolys;
        h_snowflakeDataFlat[5*x+4] = h_snowflakeData[x].ind;
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
    cudaMalloc((void**)&d_normals, h_numPolys*9*sizeof(float));
    cudaMalloc((void**)&d_snowflakeDataFlat, 5*h_numParticles*sizeof(float));
    cudaMemcpy(d_verts, h_verts, h_numPolys*9*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_normals, h_normals, h_numPolys*9*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_snowflakeDataFlat, h_snowflakeDataFlat, 5*h_numParticles*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(d_extent, h_extent, 6*sizeof(float));
    cudaMemcpyToSymbol(d_numParticles, &h_numParticles, sizeof(unsigned));

    // free
    cudaFree(h_snowflakeDataFlat);
    cudaFree(h_extent);
}

/**
 * Update snow on the GPU via kernel calls.
 */
extern void updateSnowOnGPU() {
    // dispatch kernel
    cudaDeviceSynchronize();
    updateSnow<<<h_snowNumBlocks,SNOW_BLOCK_SIZE>>>(d_verts, d_normals, d_snowflakeDataFlat);
    cudaDeviceSynchronize();

    // fetch work
    cudaMemcpy(h_verts, d_verts, h_numPolys*9*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_normals, d_normals, h_numPolys*9*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_snowflakeDataFlat, d_snowflakeDataFlat, 5*h_numParticles*sizeof(float), cudaMemcpyDeviceToHost);

    // update snowflakeData
    for (int x = 0; x < h_numParticles; x++) {
        for (int i = 0; i < 3; i++) {
            h_snowflakeData[x].pos[i] = h_snowflakeDataFlat[5*x+i];
        }
    }
}

/** TODO
 * - split kernels (in vert update can fo 1 particle per thread)
*/