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
#define SNOW_BLOCK_SIZE 256
#define SNOW_BATCH_SIZE 256

__global__ void updateSnow(float *verts, SnowflakeData *snowflakeData, unsigned numParticles, float (*extent)[2]) {
    unsigned kernelInd = (blockIdx.x*SNOW_BLOCK_SIZE + threadIdx.x)*SNOW_BATCH_SIZE;
    if (kernelInd >= numParticles) {
        return;
    }
    SnowflakeData currSnowflakeData;
    unsigned currInd;
    float offset;
    for (int x = kernelInd; x < min(kernelInd + SNOW_BATCH_SIZE, numParticles); x++) {
        currSnowflakeData = snowflakeData[x];
        currInd = currSnowflakeData.ind;
        if (currSnowflakeData.pos[1] < extent[1][0]) {
            offset = extent[1][1] - currSnowflakeData.pos[1];
        } else {
            offset = -GRAVITY;
        }
        for (int i = currInd; i < currInd + currSnowflakeData.numPolys*9; i+=3) {
            verts[i + 1] += offset;
        }
        snowflakeData[x].pos[1] += offset;
    }
}

extern void updateSnowOnGPU(SnowGeneratorData data, unsigned numParticles, float extent[3][2]) { // TODO: refactor bc 10k already looks bad (could be lack of freeing?)
    // allocate dev mem
    float *d_verts;
    cudaMalloc((void**)&d_verts, data.numPolys*9*sizeof(float));
    SnowflakeData *d_snowflakeData;
    cudaMalloc((void**)&d_snowflakeData, numParticles*sizeof(SnowflakeData));
    float (*d_extent)[2];
    cudaMalloc((void**)&d_extent, sizeof(float[3][2]));

    // copy host to dev
    cudaMemcpy(d_verts, data.verts, data.numPolys*9*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_snowflakeData, data.snowflakeData, numParticles*sizeof(SnowflakeData), cudaMemcpyHostToDevice);
    cudaMemcpy(d_extent, extent, sizeof(float[3][2]), cudaMemcpyHostToDevice);

    // dispatch kernel
    int snowNumBlocks = ceil(numParticles/(SNOW_BLOCK_SIZE*SNOW_BATCH_SIZE*1.0));
    updateSnow<<<snowNumBlocks,SNOW_BLOCK_SIZE>>>(d_verts, d_snowflakeData, numParticles, d_extent);
    cudaDeviceSynchronize();

    // fetch work
    int error = cudaMemcpy(data.verts, d_verts, data.numPolys*9*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(data.snowflakeData, d_snowflakeData, numParticles*sizeof(SnowflakeData), cudaMemcpyDeviceToHost);
    cudaFree(d_verts);
    cudaFree(d_snowflakeData);
    cudaFree(d_extent);
}
