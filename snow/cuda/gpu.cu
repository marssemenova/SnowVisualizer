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
int snowNumBlocks;
int numPolys;
int h_numParticles;

// dev vars
float *d_verts;
float *d_normals;
SnowflakeData *d_snowflakeData;
float (*d_extent)[2];

// TODO: remember abt cudaHostMalloc

__global__ void updateSnow(float *verts, float *norms, SnowflakeData *snowflakeData, unsigned numParticles, float (*extent)[2]) { // TODO: pass params every time or make dev vars?
    unsigned kernelInd = (blockIdx.x*SNOW_BLOCK_SIZE + threadIdx.x)*SNOW_BATCH_SIZE;
    if (kernelInd > numParticles) {
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

extern void initSnowOnGPU(SnowGeneratorData data, unsigned numParticles, float extent[3][2]) {
    // save refs
    h_verts = data.verts;
    h_normals = data.normals;
    snowNumBlocks = ceil(numParticles/(SNOW_BLOCK_SIZE*SNOW_BATCH_SIZE*1.0)); // TODO: mem access block size (256) dictates snow batch size
    numPolys = data.numPolys;
    h_numParticles = numParticles;

    // cuda malloc
    cudaMalloc((void**)&d_verts, data.numPolys*9*sizeof(float));
    cudaMalloc((void**)&d_normals, data.numPolys*9*sizeof(float));
    cudaMalloc((void**)&d_snowflakeData, numParticles*sizeof(SnowflakeData));
    cudaMalloc((void**)&d_extent, sizeof(float[3][2]));

    // copy data to host
    cudaMemcpy(d_verts, data.verts, data.numPolys*9*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_normals, data.normals, data.numPolys*9*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_snowflakeData, data.snowflakeData, numParticles*sizeof(SnowflakeData), cudaMemcpyHostToDevice);
    cudaMemcpy(d_extent, extent, sizeof(float[3][2]), cudaMemcpyHostToDevice);
}

extern void updateSnowOnGPU() {
    // dispatch kernel
    updateSnow<<<snowNumBlocks,SNOW_BLOCK_SIZE>>>(d_verts, d_normals, d_snowflakeData, h_numParticles, d_extent);
    cudaDeviceSynchronize();

    // fetch work
    cudaMemcpy(h_verts, d_verts, numPolys*9*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_normals, d_normals, numPolys*9*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_snowflakeData, d_snowflakeData, h_numParticles*sizeof(SnowflakeData), cudaMemcpyDeviceToHost);
}

/** TODO
 * - flatten SnowflakeData
 * - make gpu vars __device__ vars
 * - refactor alignment
 * - check if any if statements avoidable
 * - free mem (global)
 * - __constant__ for extent
 * - split kernels (in vert update can fo 1 particle per thread)
 * -
*/