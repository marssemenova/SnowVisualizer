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

#define GRAVITY 2.0f
#define SNOW_NOISE_Y 0.1f // TODO: ok to def?

#define LBM_Q 19
#define LBM_C 1.0/3.0f
#define LBM_TAU 5.0 // TODO: ????
__constant__ static const int D_LATTICE_VELOCITIES[19][3] = {
    {0,-1,-1},{-1,0,-1},{0,0,-1},{1,0,-1},{0,1,-1},{-1,-1,0},{0,-1,0},{1,-1,0},
    {-1,0,0}, {0,0,0},  {1,0,0}, {-1,1,0},{0,1,0}, {1,1,0},  {0,-1,1},{-1,0,1},
    {0,0,1},  {1,0,1},  {0,1,1}
};

__constant__ static const float D_LATTICE_WEIGHTS[19] = {
    1.0/36.0, 1.0/36.0, 2.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0, 2.0/36.0, 1.0/36.0,
    2.0/36.0, 12.0/36.0,2.0/36.0, 1.0/36.0, 2.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0,
    2.0/36.0, 1.0/36.0, 1.0/36.0
};
static const float H_LATTICE_WEIGHTS[19] = { // TODO: see if way to not have 2
    1.0/36.0, 1.0/36.0, 2.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0, 2.0/36.0, 1.0/36.0,
    2.0/36.0, 12.0/36.0,2.0/36.0, 1.0/36.0, 2.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0,
    2.0/36.0, 1.0/36.0, 1.0/36.0
};
#define Nx 16 // TODO: make adjustable based on extent
#define Ny 16
#define Nz 16
struct Lattice {
    float f0[Nx*Ny*Nz];
    float f1[Nx*Ny*Nz];
    float f2[Nx*Ny*Nz];
    float f3[Nx*Ny*Nz];
    float f4[Nx*Ny*Nz];
    float f5[Nx*Ny*Nz];
    float f6[Nx*Ny*Nz];
    float f7[Nx*Ny*Nz];
    float f8[Nx*Ny*Nz];
    float f9[Nx*Ny*Nz];
    float f10[Nx*Ny*Nz];
    float f11[Nx*Ny*Nz];
    float f12[Nx*Ny*Nz];
    float f13[Nx*Ny*Nz];
    float f14[Nx*Ny*Nz];
    float f15[Nx*Ny*Nz];
    float f16[Nx*Ny*Nz];
    float f17[Nx*Ny*Nz];
    float f18[Nx*Ny*Nz];
};


// kernel params
// LBM
dim3 h_lbmBlockSize; // TODO: make variable
dim3 h_lbmGridSize;
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
float *d_srcGrid;
float *d_tempGrid;
float *d_destGrid;

__global__ void lbmKernel(float *d_srcGrid, float *d_destGrid) {
    // compute the 3D position of the thread
    int x = threadIdx.x;
    int y = blockIdx.x;
    int z = blockIdx.y;
    // compute the corresponding 1D index
    int ind = x + y * Nx + z * Nx*Ny;

    if (true) { // TODO: rem
        return;
    }

    if (!(0 < x && x < Nx && 0 < y && y < Ny && 0 < z && z < Nz)){
        return; // TODO: refactor
    }
    //float value = Lattice.f5[ind]; // TODO: ref, del

    // stream 19 pdfs from adjacent cells to curr cell
    int accessX, accessY, accessZ, accessInd;
    for (int i = 0; i < LBM_Q; i++) { // TODO: is it better to unroll?
        accessX = x-D_LATTICE_VELOCITIES[i][0];
        accessY = y-D_LATTICE_VELOCITIES[i][1];
        accessZ = z-D_LATTICE_VELOCITIES[i][2];
        accessInd = accessX + accessY * Nx + accessZ * Nx*Ny + i;
        d_destGrid[LBM_Q*ind + i] = d_srcGrid[LBM_Q*accessInd + i]; // TODO: might need to swap, rn dest = stream + src = collide
    }

    // apply boundary conds
    // TODO: ?????

    float *currCell = &d_srcGrid[LBM_Q*ind];
    // calc density (rho)
    float density = 0;
    for (int i = 0; i < LBM_Q; i++) {
        density += currCell[i];
    }

    // calc velocity (u)
    float velocity[3];
    for (int i = 0; i < 3; i++) {
        velocity[i] = 0;
        for (int j = 0; j < LBM_Q; j++) {
            velocity[i] += currCell[j]*D_LATTICE_VELOCITIES[j][i];
        }
    }

    // calc the loc equilibrium distro funcs f_qi^eq
    float feq[LBM_Q];
    float t1, t2, t3;
    t3 = velocity[0]*velocity[0] + velocity[1]*velocity[1] + velocity[2]*velocity[2];
    for (int i = 0; i < LBM_Q; i++) {
        t1 = D_LATTICE_VELOCITIES[i][0]*velocity[0] + D_LATTICE_VELOCITIES[i][1]*velocity[1] + D_LATTICE_VELOCITIES[i][2]*velocity[2];
        t2 = t1*t1;
        feq[i] = D_LATTICE_WEIGHTS[i]*density*(1 + (3.0/LBM_C)*t1 + (9.0/(2.0*LBM_C*LBM_C))*t2 - (3.0/(2.0*LBM_C*LBM_C))*t3);
    }

    // calc distro func (f_qi) at new time step + save 19 vals of distro func (f_qi) to curr cell
    for (int i = 0; i < LBM_Q; i++) {
        currCell[i] = currCell[i] - (currCell[i] - feq[i])/LBM_TAU;
    }
}

__global__ void swapGrid(float *d_srcGrid, float *d_destGrid){
    float *swap = d_srcGrid;
    d_srcGrid=d_destGrid;
    d_destGrid=d_srcGrid;
}

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
    h_lbmBlockSize = dim3(Nx, 1, 1);
    h_lbmGridSize = dim3(Ny, Nz, 1);

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

    // init macroscopic quatities (i.e. density (rho) + velocity(u))

    // init distro func (f_qi)

    // init equilibrium func (f_qi^eq)

    // init dest + src grids
    float *h_destGrid, *h_srcGrid;
    unsigned numCells = Nx*Ny*Nz;
    cudaMallocHost((void**)&h_destGrid, LBM_Q*numCells*sizeof(float));
    cudaMallocHost((void**)&h_srcGrid, LBM_Q*numCells*sizeof(float));
    for (int x = 0; x < Nx; x++) {
        for (int y = 0; y < Ny; y++) {
            for (int z = 0;z < Nz;z++) {
                for (int i = 0; i < LBM_Q; i++) {
                    h_destGrid[LBM_Q*(x + y*Nx + z*Nx*Ny) + i] = 0;
                    h_srcGrid[LBM_Q*(x + y*Nx + z*Nx*Ny) + i] = H_LATTICE_WEIGHTS[i];
                }
            }
        }
    }
    cudaMalloc((void**)&d_destGrid, LBM_Q*numCells*sizeof(float));
    cudaMalloc((void**)&d_srcGrid, LBM_Q*numCells*sizeof(float));
    cudaMemcpy(d_destGrid, h_destGrid, LBM_Q*numCells*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_srcGrid, h_srcGrid, LBM_Q*numCells*sizeof(float), cudaMemcpyHostToDevice);
    cudaFree(h_destGrid);
    cudaFree(h_srcGrid);

}

/**
 * Update snow on the GPU via kernel calls.
 */
extern void snowUpdateGPU() {
    // dispatch kernels
    // LBM
    lbmKernel<<<h_lbmBlockSize, h_lbmGridSize>>>(d_srcGrid, d_destGrid);
    cudaDeviceSynchronize();
    swapGrid<<<1,1>>>(d_srcGrid, d_destGrid);
    cudaDeviceSynchronize();

    // apply forces to snow
    snowApplyGrav<<<h_snowGravNumBlocks,SNOW_GRAV_BLOCK_SIZE>>>(d_snowflakeDataFlat, d_numParticles, d_snowOffsets, time(NULL));
    cudaDeviceSynchronize();

    // update snow verts
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