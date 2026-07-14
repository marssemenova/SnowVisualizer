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
#include "cuda_errors.h"

#define GRAVITY 0.2f
#define SNOW_NOISE_Y 0.1f // TODO: ok to def?

#define LBM_Q 19
#define LBM_C 1.0/3.0f // from speed of sounds = 1/sqrt(3)
#define LBM_TAU 1.0 // 3 * viscosity - 1/2
__constant__ static const int D_LATTICE_VELOCITIES[19][3] = { {0,0,0},
    {1,0,0}, {-1,0,0}, {0,1,0}, {0,-1,0}, {0,0,1}, {0,0,1},
    {1,1,0}, {1,-1,0}, {-1,1,0}, {-1,-1,0}, {1,0,1}, {1,0,-1}, {-1,0,1}, {-1,0,-1}, {0,1,1}, {0,1,-1},  {0,-1,1},  {0,-1,-1}};
__constant__ static const float D_LATTICE_WEIGHTS[19] = { 1.0/3.0,
    1.0/18.0, 1.0/18.0, 1.0/18.0, 1.0/18.0, 1.0/18.0, 1.0/18.0,
    1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0,
    1.0/36.0, 1.0/36.0, 1.0/36.0};
static const float H_LATTICE_WEIGHTS[19] = { 1.0/3.0, // TODO: see if way to not have 2
    1.0/18.0, 1.0/18.0, 1.0/18.0, 1.0/18.0, 1.0/18.0, 1.0/18.0,
    1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0};
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
Lattice h_srcLattice; // TODO: del
Lattice h_destLattice; // TODO: del

// dev vars
float *d_verts;
float *d_snowflakeDataFlat;
__constant__ __device__ float d_extent[6];
unsigned *d_numParticles;
float *d_snowOffsets;
curandState *d_globalState;
Lattice *d_srcLattice;
Lattice *d_destLattice;

__global__ void lbmKernel(Lattice *d_srcLattice, Lattice *d_destLattice) {
    // compute the 3D position of the thread
    int x = threadIdx.x;
    int y = blockIdx.x;
    int z = blockIdx.y;
    // compute the corresponding 1D index
    int ind = x + y * Nx + z * Nx*Ny;


    if (!(ind < Nx*Ny*Nz)) { // TODO: ref
        return;
    }
/*
    float f0 = d_srcLattice->f0[ind];
    float f1 = d_srcLattice->f1[ind];
    float f2 = d_srcLattice->f2[ind];
    float f3 = d_srcLattice->f3[ind];
    float f4 = d_srcLattice->f4[ind];
    float f5 = d_srcLattice->f5[ind];
    float f6 = d_srcLattice->f6[ind];
    float f7 = d_srcLattice->f7[ind];
    float f8 = d_srcLattice->f8[ind];
    float f9 = d_srcLattice->f9[ind];
    float f10 = d_srcLattice->f10[ind];
    float f11 = d_srcLattice->f11[ind];
    float f12 = d_srcLattice->f12[ind];
    float f13 = d_srcLattice->f13[ind];
    float f14 = d_srcLattice->f14[ind];
    float f15 = d_srcLattice->f15[ind];
    float f16 = d_srcLattice->f16[ind];
    float f17 = d_srcLattice->f17[ind];
    float f18 = d_srcLattice->f18[ind];
    printf("%f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f\n", f0, f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13, f14, f15, f16, f17, f18);
*/

    // stream 19 pdfs from adjacent cells to curr cell
    int accessX, accessY, accessZ, accessInd; // TODO!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!: i think boundary is based on accessX/Y/Z n if its out of bounds do smth
    accessX = x-D_LATTICE_VELOCITIES[0][0];
    accessY = y-D_LATTICE_VELOCITIES[0][1];
    accessZ = z-D_LATTICE_VELOCITIES[0][2];
    accessInd = accessX + accessY * Nx + accessZ * Nx*Ny;
    if (accessInd < Nx*Ny*Nz) {
        d_destLattice->f0[ind] = d_srcLattice->f0[accessInd]; // TODO: might need to swap, rn dest = stream + src = collide
    }
    accessX = x-D_LATTICE_VELOCITIES[1][0];
    accessY = y-D_LATTICE_VELOCITIES[1][1];
    accessZ = z-D_LATTICE_VELOCITIES[1][2];
    accessInd = accessX + accessY * Nx + accessZ * Nx*Ny;
    if (accessInd < Nx*Ny*Nz) {
        d_destLattice->f1[ind] = d_srcLattice->f1[accessInd];
    }
    accessX = x-D_LATTICE_VELOCITIES[2][0];
    accessY = y-D_LATTICE_VELOCITIES[2][1];
    accessZ = z-D_LATTICE_VELOCITIES[2][2];
    accessInd = accessX + accessY * Nx + accessZ * Nx*Ny;
    if (accessInd < Nx*Ny*Nz) {
        d_destLattice->f2[ind] = d_srcLattice->f2[accessInd];
    }
    accessX = x-D_LATTICE_VELOCITIES[3][0];
    accessY = y-D_LATTICE_VELOCITIES[3][1];
    accessZ = z-D_LATTICE_VELOCITIES[3][2];
    accessInd = accessX + accessY * Nx + accessZ * Nx*Ny;
    if (accessInd < Nx*Ny*Nz) {
        d_destLattice->f3[ind] = d_srcLattice->f3[accessInd];
    }
    accessX = x-D_LATTICE_VELOCITIES[4][0];
    accessY = y-D_LATTICE_VELOCITIES[4][1];
    accessZ = z-D_LATTICE_VELOCITIES[4][2];
    accessInd = accessX + accessY * Nx + accessZ * Nx*Ny;
    if (accessInd < Nx*Ny*Nz) {
        d_destLattice->f4[ind] = d_srcLattice->f4[accessInd];
    }
    accessX = x-D_LATTICE_VELOCITIES[5][0];
    accessY = y-D_LATTICE_VELOCITIES[5][1];
    accessZ = z-D_LATTICE_VELOCITIES[5][2];
    accessInd = accessX + accessY * Nx + accessZ * Nx*Ny;
    if (accessInd < Nx*Ny*Nz) {
        d_destLattice->f5[ind] = d_srcLattice->f5[accessInd];
    }
    accessX = x-D_LATTICE_VELOCITIES[6][0];
    accessY = y-D_LATTICE_VELOCITIES[6][1];
    accessZ = z-D_LATTICE_VELOCITIES[6][2];
    accessInd = accessX + accessY * Nx + accessZ * Nx*Ny;
    if (accessInd < Nx*Ny*Nz) {
        d_destLattice->f6[ind] = d_srcLattice->f6[accessInd];
    }
    accessX = x-D_LATTICE_VELOCITIES[7][0];
    accessY = y-D_LATTICE_VELOCITIES[7][1];
    accessZ = z-D_LATTICE_VELOCITIES[7][2];
    accessInd = accessX + accessY * Nx + accessZ * Nx*Ny;
    if (accessInd < Nx*Ny*Nz) {
        d_destLattice->f7[ind] = d_srcLattice->f7[accessInd];
    }
    accessX = x-D_LATTICE_VELOCITIES[8][0];
    accessY = y-D_LATTICE_VELOCITIES[8][1];
    accessZ = z-D_LATTICE_VELOCITIES[8][2];
    accessInd = accessX + accessY * Nx + accessZ * Nx*Ny;
    if (accessInd < Nx*Ny*Nz) {
        d_destLattice->f8[ind] = d_srcLattice->f8[accessInd];
    }
    accessX = x-D_LATTICE_VELOCITIES[9][0];
    accessY = y-D_LATTICE_VELOCITIES[9][1];
    accessZ = z-D_LATTICE_VELOCITIES[9][2];
    accessInd = accessX + accessY * Nx + accessZ * Nx*Ny;
    if (accessInd < Nx*Ny*Nz) {
        d_destLattice->f9[ind] = d_srcLattice->f9[accessInd];
    }
    accessX = x-D_LATTICE_VELOCITIES[10][0];
    accessY = y-D_LATTICE_VELOCITIES[10][1];
    accessZ = z-D_LATTICE_VELOCITIES[10][2];
    accessInd = accessX + accessY * Nx + accessZ * Nx*Ny;
    if (accessInd < Nx*Ny*Nz) {
        d_destLattice->f10[ind] = d_srcLattice->f10[accessInd];
    }
    accessX = x-D_LATTICE_VELOCITIES[11][0];
    accessY = y-D_LATTICE_VELOCITIES[11][1];
    accessZ = z-D_LATTICE_VELOCITIES[11][2];
    accessInd = accessX + accessY * Nx + accessZ * Nx*Ny;
    if (accessInd < Nx*Ny*Nz) {
        d_destLattice->f11[ind] = d_srcLattice->f11[accessInd];
    }
    accessX = x-D_LATTICE_VELOCITIES[12][0];
    accessY = y-D_LATTICE_VELOCITIES[12][1];
    accessZ = z-D_LATTICE_VELOCITIES[12][2];
    accessInd = accessX + accessY * Nx + accessZ * Nx*Ny;
    if (accessInd < Nx*Ny*Nz) {
        d_destLattice->f12[ind] = d_srcLattice->f12[accessInd];
    }
    accessX = x-D_LATTICE_VELOCITIES[13][0];
    accessY = y-D_LATTICE_VELOCITIES[13][1];
    accessZ = z-D_LATTICE_VELOCITIES[13][2];
    accessInd = accessX + accessY * Nx + accessZ * Nx*Ny;
    if (accessInd < Nx*Ny*Nz) {
        d_destLattice->f13[ind] = d_srcLattice->f13[accessInd];
    }
    accessX = x-D_LATTICE_VELOCITIES[14][0];
    accessY = y-D_LATTICE_VELOCITIES[14][1];
    accessZ = z-D_LATTICE_VELOCITIES[14][2];
    accessInd = accessX + accessY * Nx + accessZ * Nx*Ny;
    if (accessInd < Nx*Ny*Nz) {
        d_destLattice->f14[ind] = d_srcLattice->f14[accessInd];
    }
    accessX = x-D_LATTICE_VELOCITIES[15][0];
    accessY = y-D_LATTICE_VELOCITIES[15][1];
    accessZ = z-D_LATTICE_VELOCITIES[15][2];
    accessInd = accessX + accessY * Nx + accessZ * Nx*Ny;
    if (accessInd < Nx*Ny*Nz) {
        d_destLattice->f15[ind] = d_srcLattice->f15[accessInd];
    }
    accessX = x-D_LATTICE_VELOCITIES[16][0];
    accessY = y-D_LATTICE_VELOCITIES[16][1];
    accessZ = z-D_LATTICE_VELOCITIES[16][2];
    accessInd = accessX + accessY * Nx + accessZ * Nx*Ny;
    if (accessInd < Nx*Ny*Nz) {
        d_destLattice->f16[ind] = d_srcLattice->f16[accessInd];
    }
    accessX = x-D_LATTICE_VELOCITIES[17][0];
    accessY = y-D_LATTICE_VELOCITIES[17][1];
    accessZ = z-D_LATTICE_VELOCITIES[17][2];
    accessInd = accessX + accessY * Nx + accessZ * Nx*Ny;
    if (accessInd < Nx*Ny*Nz) {
        d_destLattice->f17[ind] = d_srcLattice->f17[accessInd];
    }
    accessX = x-D_LATTICE_VELOCITIES[18][0];
    accessY = y-D_LATTICE_VELOCITIES[18][1];
    accessZ = z-D_LATTICE_VELOCITIES[18][2];
    accessInd = accessX + accessY * Nx + accessZ * Nx*Ny;
    if (accessInd < Nx*Ny*Nz) {
        d_destLattice->f18[ind] = d_srcLattice->f18[accessInd];
    }


    // apply boundary conds
    // TODO: ?????

   // float *currCell = &d_srcGrid[LBM_Q*ind];
    // calc density (rho)
    float density = d_srcLattice->f0[ind] + d_srcLattice->f1[ind] + d_srcLattice->f2[ind] + d_srcLattice->f3[ind]
                    + d_srcLattice->f4[ind] + d_srcLattice->f5[ind] + d_srcLattice->f6[ind]
                    + d_srcLattice->f7[ind] + d_srcLattice->f8[ind] + d_srcLattice->f9[ind]
                    + d_srcLattice->f10[ind] + d_srcLattice->f11[ind] + d_srcLattice->f12[ind]
                    + d_srcLattice->f13[ind] + d_srcLattice->f14[ind] + d_srcLattice->f15[ind]
                    + d_srcLattice->f16[ind] + d_srcLattice->f17[ind] + d_srcLattice->f18[ind];

    // calc velocity (u)
    float velocity[3];
    for (int i = 0; i < 3; i++) {
        velocity[i] = d_srcLattice->f0[ind]*D_LATTICE_VELOCITIES[0][i] + d_srcLattice->f1[ind]*D_LATTICE_VELOCITIES[1][i] + d_srcLattice->f2[ind]*D_LATTICE_VELOCITIES[2][i]
                    + d_srcLattice->f3[ind]*D_LATTICE_VELOCITIES[3][i] + d_srcLattice->f4[ind]*D_LATTICE_VELOCITIES[4][i] + d_srcLattice->f5[ind]*D_LATTICE_VELOCITIES[5][i]
                    + d_srcLattice->f6[ind]*D_LATTICE_VELOCITIES[6][i] + d_srcLattice->f7[ind]*D_LATTICE_VELOCITIES[7][i] + d_srcLattice->f8[ind]*D_LATTICE_VELOCITIES[8][i]
                    + d_srcLattice->f9[ind]*D_LATTICE_VELOCITIES[9][i] + d_srcLattice->f10[ind]*D_LATTICE_VELOCITIES[10][i] + d_srcLattice->f11[ind]*D_LATTICE_VELOCITIES[11][i]
                    + d_srcLattice->f12[ind]*D_LATTICE_VELOCITIES[12][i] + d_srcLattice->f13[ind]*D_LATTICE_VELOCITIES[13][i] + d_srcLattice->f14[ind]*D_LATTICE_VELOCITIES[14][i]
                    + d_srcLattice->f15[ind]*D_LATTICE_VELOCITIES[15][i] + d_srcLattice->f16[ind]*D_LATTICE_VELOCITIES[16][i] + d_srcLattice->f5[ind]*D_LATTICE_VELOCITIES[17][i]
                    + d_srcLattice->f18[ind]*D_LATTICE_VELOCITIES[18][i];
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
    d_srcLattice->f0[ind] = d_srcLattice->f0[ind] - (d_srcLattice->f0[ind] - feq[0])/LBM_TAU;
    d_srcLattice->f1[ind] = d_srcLattice->f1[ind] - (d_srcLattice->f1[ind] - feq[1])/LBM_TAU;
    d_srcLattice->f2[ind] = d_srcLattice->f2[ind] - (d_srcLattice->f2[ind] - feq[2])/LBM_TAU;
    d_srcLattice->f3[ind] = d_srcLattice->f3[ind] - (d_srcLattice->f3[ind] - feq[3])/LBM_TAU;
    d_srcLattice->f4[ind] = d_srcLattice->f4[ind] - (d_srcLattice->f4[ind] - feq[4])/LBM_TAU;
    d_srcLattice->f5[ind] = d_srcLattice->f5[ind] - (d_srcLattice->f5[ind] - feq[5])/LBM_TAU;
    d_srcLattice->f6[ind] = d_srcLattice->f6[ind] - (d_srcLattice->f6[ind] - feq[6])/LBM_TAU;
    d_srcLattice->f7[ind] = d_srcLattice->f7[ind] - (d_srcLattice->f7[ind] - feq[7])/LBM_TAU;
    d_srcLattice->f8[ind] = d_srcLattice->f8[ind] - (d_srcLattice->f8[ind] - feq[8])/LBM_TAU;
    d_srcLattice->f9[ind] = d_srcLattice->f9[ind] - (d_srcLattice->f9[ind] - feq[9])/LBM_TAU;
    d_srcLattice->f10[ind] = d_srcLattice->f10[ind] - (d_srcLattice->f10[ind] - feq[10])/LBM_TAU;
    d_srcLattice->f11[ind] = d_srcLattice->f11[ind] - (d_srcLattice->f11[ind] - feq[11])/LBM_TAU;
    d_srcLattice->f12[ind] = d_srcLattice->f12[ind] - (d_srcLattice->f12[ind] - feq[12])/LBM_TAU;
    d_srcLattice->f13[ind] = d_srcLattice->f13[ind] - (d_srcLattice->f13[ind] - feq[13])/LBM_TAU;
    d_srcLattice->f14[ind] = d_srcLattice->f14[ind] - (d_srcLattice->f14[ind] - feq[14])/LBM_TAU;
    d_srcLattice->f15[ind] = d_srcLattice->f15[ind] - (d_srcLattice->f15[ind] - feq[15])/LBM_TAU;
    d_srcLattice->f16[ind] = d_srcLattice->f16[ind] - (d_srcLattice->f16[ind] - feq[16])/LBM_TAU;
    d_srcLattice->f17[ind] = d_srcLattice->f17[ind] - (d_srcLattice->f17[ind] - feq[17])/LBM_TAU;
    d_srcLattice->f18[ind] = d_srcLattice->f18[ind] - (d_srcLattice->f18[ind] - feq[18])/LBM_TAU;
}

__global__ void swapGrid(Lattice *d_srcLattice, Lattice *d_destLattice){
    Lattice *swap = d_srcLattice;
    d_srcLattice = d_destLattice;
    d_destLattice = swap;
}

/**
 * Create and initialize curandState using seed, one for each thread.
 * Stores result in globalState[tid]. // TODO: params
 */
__global__ void setupSnowRandState(curandState* d_globalState, uint64_t seed, unsigned *d_numParticles) {
    int tid = threadIdx.x  + blockDim.x * blockIdx.x;
    if (tid < *d_numParticles) {
        curand_init(seed, tid, 0, &d_globalState[tid]);
    }
}

/**
 * Generate a floating point number in the range (min,max].
 * Note curand_uniform returns numbers in the  range (0.0, 1.0]. // TODO: params
 */
__device__ __forceinline__ float getRandFloatGPU(float min, float max, curandState* localState) {
    return min + (max - min) * curand_uniform(localState);
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
__global__ void snowApplyGrav(float *d_snowflakeDataFlat, unsigned *d_numParticles, float *d_snowOffsets, curandState* d_globalState) {
    unsigned kernelInd = blockIdx.x*SNOW_GRAV_BLOCK_SIZE*SNOW_GRAV_BATCH_SIZE + threadIdx.x;
    unsigned stride = SNOW_GRAV_BLOCK_SIZE;
    float xOffset, yOffset, zOffset;
    float yOffsetRand;
    curandState localState;
    for (int x = kernelInd; x < min(kernelInd + SNOW_GRAV_BATCH_SIZE*stride, *d_numParticles); x+=stride) {
        if (d_snowflakeDataFlat[3*x+1] < d_extent[2]) {
            localState = d_globalState[kernelInd];
            xOffset = getRandFloatGPU(d_extent[0], d_extent[1], &localState) - d_snowflakeDataFlat[3*x];
            yOffsetRand = getRandFloatGPU(-(SNOW_NOISE_Y*(d_extent[3] - d_extent[2])), SNOW_NOISE_Y*(d_extent[3] - d_extent[2]), &localState);
            zOffset = getRandFloatGPU(d_extent[4], d_extent[5], &localState) - d_snowflakeDataFlat[3*x+2];
            yOffset = (d_extent[3] - d_extent[2]) + yOffsetRand;
            d_globalState[kernelInd] = localState;
        } else {
            yOffset = -GRAVITY;
            xOffset = 0;
            zOffset = 0;
        }

        // update x
         d_snowOffsets[5*x] = xOffset;
         d_snowflakeDataFlat[3*x] += xOffset;

        // update y
        d_snowOffsets[5*x+1] = yOffset;
        d_snowflakeDataFlat[3*x+1] += yOffset;

        // update z
         d_snowOffsets[5*x+2] = zOffset;
         d_snowflakeDataFlat[3*x+2] += zOffset;
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
    cudaFreeHost(h_snowflakeDataFlat);
    cudaFreeHost(h_snowOffsets);
    cudaFreeHost(h_extent);

    // init macroscopic quatities (i.e. density (rho) + velocity(u))

    // init distro func (f_qi)

    // init equilibrium func (f_qi^eq)

    // init dest + src grids
    cudaMallocHost((void**)&h_srcLattice, sizeof(Lattice));
    cudaMallocHost((void**)&h_destLattice, sizeof(Lattice));
    unsigned numCells = Nx*Ny*Nz;
    for (int x = 0; x < numCells; x++) {
        h_srcLattice.f0[x] = H_LATTICE_WEIGHTS[0];
        h_srcLattice.f1[x] = H_LATTICE_WEIGHTS[1];
        h_srcLattice.f2[x] = H_LATTICE_WEIGHTS[2];
        h_srcLattice.f3[x] = H_LATTICE_WEIGHTS[3];
        h_srcLattice.f4[x] = H_LATTICE_WEIGHTS[4];
        h_srcLattice.f5[x] = H_LATTICE_WEIGHTS[5];
        h_srcLattice.f6[x] = H_LATTICE_WEIGHTS[6];
        h_srcLattice.f7[x] = H_LATTICE_WEIGHTS[7];
        h_srcLattice.f8[x] = H_LATTICE_WEIGHTS[8];
        h_srcLattice.f9[x] = H_LATTICE_WEIGHTS[9];
        h_srcLattice.f10[x] = H_LATTICE_WEIGHTS[10];
        h_srcLattice.f11[x] = H_LATTICE_WEIGHTS[11];
        h_srcLattice.f12[x] = H_LATTICE_WEIGHTS[12];
        h_srcLattice.f13[x] = H_LATTICE_WEIGHTS[13];
        h_srcLattice.f14[x] = H_LATTICE_WEIGHTS[14];
        h_srcLattice.f15[x] = H_LATTICE_WEIGHTS[15];
        h_srcLattice.f16[x] = H_LATTICE_WEIGHTS[16];
        h_srcLattice.f17[x] = H_LATTICE_WEIGHTS[17];
        h_srcLattice.f18[x] = H_LATTICE_WEIGHTS[18];
    }
    cudaMalloc((void**)&d_srcLattice, sizeof(Lattice));
    cudaMalloc((void**)&d_destLattice, sizeof(Lattice));
    cudaMemcpy(d_srcLattice, &h_srcLattice, sizeof(Lattice), cudaMemcpyHostToDevice);
    cudaMemcpy(d_destLattice, &h_destLattice, sizeof(Lattice), cudaMemcpyHostToDevice);

    // init rand
    cudaMalloc((void**)&d_globalState, h_numParticles*sizeof(curandState));
    setupSnowRandState<<<h_snowGravNumBlocks,SNOW_GRAV_BLOCK_SIZE>>>(d_globalState, time(NULL), d_numParticles);
    cudaDeviceSynchronize();
}

/**
 * Update snow on the GPU via kernel calls.
 */
extern void snowUpdateGPU() {
    // dispatch kernels
    // LBM
    lbmKernel<<<h_lbmBlockSize, h_lbmGridSize>>>(d_srcLattice, d_destLattice);
    cudaDeviceSynchronize();
    swapGrid<<<1,1>>>(d_srcLattice, d_destLattice);
    cudaDeviceSynchronize();

    // apply forces to snow
    snowApplyGrav<<<h_snowGravNumBlocks,SNOW_GRAV_BLOCK_SIZE>>>(d_snowflakeDataFlat, d_numParticles, d_snowOffsets, d_globalState);
    cudaDeviceSynchronize();

    // update snow verts
    snowUpdate<<<h_snowUpdateNumBlocks,SNOW_UPDATE_BLOCK_SIZE>>>(d_verts, d_snowOffsets, d_numParticles);
    cudaDeviceSynchronize();

    // fetch work
    cudaMemcpy(h_verts, d_verts, h_numPolys*9*sizeof(float), cudaMemcpyDeviceToHost);

    // TODO: del
    cudaMemcpy(&h_srcLattice, d_srcLattice, sizeof(Lattice), cudaMemcpyDeviceToHost);
    for (int x = 0; x < 1; x++) {
        float f0 = h_srcLattice.f0[x];
        float f1 = h_srcLattice.f1[x];
        float f2 = h_srcLattice.f2[x];
        float f3 = h_srcLattice.f3[x];
        float f4 = h_srcLattice.f4[x];
        float f5 = h_srcLattice.f5[x];
        float f6 = h_srcLattice.f6[x];
        float f7 = h_srcLattice.f7[x];
        float f8 = h_srcLattice.f8[x];
        float f9 = h_srcLattice.f9[x];
        float f10 = h_srcLattice.f10[x];
        float f11 = h_srcLattice.f11[x];
        float f12 = h_srcLattice.f12[x];
        float f13 = h_srcLattice.f13[x];
        float f14 = h_srcLattice.f14[x];
        float f15 = h_srcLattice.f15[x];
        float f16 = h_srcLattice.f16[x];
        float f17 = h_srcLattice.f17[x];
        float f18 = h_srcLattice.f18[x];
        printf("%f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f\n", f0, f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13, f14, f15, f16, f17, f18);
    }
    printf("\n\n\n");
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