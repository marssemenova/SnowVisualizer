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

#define GRAVITY -0.5f
#define SNOW_NOISE_Y 0.1f

#define LBM_Q 19
#define LBM_C 1.0 // from speed of sounds = 1/sqrt(3)
#define LBM_C_S 1.0/sqrt(3) // from speed of sounds = 1/sqrt(3)
#define LBM_M_MAX 0.1
#define LBM_TAU 0.55 // TODO: rem
__constant__ static const int D_LATTICE_VELOCITIES[19][3] = { {0,0,0},
    {1,0,0}, {-1,0,0}, {0,1,0}, {0,-1,0}, {0,0,1}, {0,0,-1},
    {1,1,0}, {1,-1,0}, {-1,1,0}, {-1,-1,0}, {1,0,1}, {1,0,-1}, {-1,0,1}, {-1,0,-1}, {0,1,1}, {0,1,-1},  {0,-1,1},  {0,-1,-1}};
__constant__ static const float D_LATTICE_WEIGHTS[19] = { 1.0/3.0,
    1.0/18.0, 1.0/18.0, 1.0/18.0, 1.0/18.0, 1.0/18.0, 1.0/18.0,
    1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0,
    1.0/36.0, 1.0/36.0, 1.0/36.0};
struct Lattice {
    float *f0;
    float *f1;
    float *f2;
    float *f3;
    float *f4;
    float *f5;
    float *f6;
    float *f7;
    float *f8;
    float *f9;
    float *f10;
    float *f11;
    float *f12;
    float *f13;
    float *f14;
    float *f15;
    float *f16;
    float *f17;
    float *f18;
};

// kernel params
// LBM
dim3 h_lbmBlockSize;
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
__constant__ __device__ float d_extents[6];
__constant__ __device__ float d_tau;
__constant__ __device__ float d_delta_x_phys;
__constant__ __device__ float d_delta_t_phys;
__constant__ __device__ unsigned d_N;
__constant__ __device__ float d_windVel;
float *d_verts;
float *d_snowflakeDataFlat;
unsigned *d_numParticles;
float *d_snowOffsets;
float *d_velocities;
curandState *d_globalState;
Lattice *d_srcLattice;
Lattice *d_destLattice;

__device__ int getInd(int x, int y, int z) {
    return x + y * d_N + z * d_N*d_N;
}

__device__ int getInd(int pos[3]) {
    return getInd(pos[0], pos[1], pos[2]);
}

// TODO
__device__ float applyBoundaryConds(float accessInd, unsigned dim) {
    if (accessInd < 0) {
        return d_N - 1;
    }
    if (accessInd >= d_N) {
        return 0;
    }

    return accessInd;
}

// TODO
__device__ float feqFunc(float velocity[3], float density, float t3, int f) {
    float t1 = D_LATTICE_VELOCITIES[f][0]*velocity[0] + D_LATTICE_VELOCITIES[f][1]*velocity[1] + D_LATTICE_VELOCITIES[f][2]*velocity[2];
    float t2 = t1*t1;
    return D_LATTICE_WEIGHTS[f]*density*(1 + (3.0/LBM_C)*t1 + (9.0/(2.0*pow(LBM_C,2)))*t2 - (3.0/(2.0*pow(LBM_C,2)))*t3);
}

// TODO
__global__ void lbmKernel(Lattice *d_srcLattice, Lattice *d_destLattice, float* d_velocities) {
    // compute the 3D position of the thread
    int x = threadIdx.x;
    int y = blockIdx.x;
    int z = blockIdx.y;
    // compute the corresponding 1D index
    int ind = getInd(x, y, z);

    if (!(ind < d_N*d_N*d_N)) { // TODO: refactor
        return;
    }

    float* destRefs[] = {d_destLattice->f0, d_destLattice->f1, d_destLattice->f2, d_destLattice->f3, d_destLattice->f4, d_destLattice->f5, d_destLattice->f6, d_destLattice->f7, d_destLattice->f8, d_destLattice->f9, d_destLattice->f10, d_destLattice->f11, d_destLattice->f12, d_destLattice->f13, d_destLattice->f14, d_destLattice->f15, d_destLattice->f16, d_destLattice->f17, d_destLattice->f18};
    float* srcRefs[] = {d_srcLattice->f0, d_srcLattice->f1, d_srcLattice->f2, d_srcLattice->f3, d_srcLattice->f4, d_srcLattice->f5, d_srcLattice->f6, d_srcLattice->f7, d_srcLattice->f8, d_srcLattice->f9, d_srcLattice->f10, d_srcLattice->f11, d_srcLattice->f12, d_srcLattice->f13, d_srcLattice->f14, d_srcLattice->f15, d_srcLattice->f16, d_srcLattice->f17, d_srcLattice->f18};
    // stream 19 pdfs from adjacent cells to curr cell + apply boundary conds
    int accessX, accessY, accessZ, accessInd;
    for (int x = 0; x < LBM_Q; x++) {
        accessX = x-D_LATTICE_VELOCITIES[x][0];
        accessY = y-D_LATTICE_VELOCITIES[x][1];
        accessZ = z-D_LATTICE_VELOCITIES[x][2];
        accessX = applyBoundaryConds(accessX, 0);
        accessY = applyBoundaryConds(accessY, 1);
        accessZ = applyBoundaryConds(accessZ, 2);
        accessInd = getInd(accessX, accessY, accessZ);
        (destRefs[x])[ind] = (srcRefs[x])[accessInd];
    }

    // calc density (rho)
    float density = 0;
    for (int x = 0; x < LBM_Q; x++) {
        density += (destRefs[x])[ind];
    }

    // calc velocity (u)
    float velocity[3];
    for (int x = 0; x < 3; x++) {
        velocity[x] = 0;
        for (int i = 0; i < LBM_Q; i++) {
            velocity[x] += (destRefs[i])[ind]*D_LATTICE_VELOCITIES[i][x];
        }
        velocity[x]/=density;
        d_velocities[3*ind + x] = velocity[x]; // save
    }

    // calc the loc equilibrium distro funcs f_qi^eq
    float feq[LBM_Q];
    float t3 = velocity[0]*velocity[0] + velocity[1]*velocity[1] + velocity[2]*velocity[2];
    for (int x = 0; x < LBM_Q; x++) {
        feq[x] = feqFunc(velocity, density, t3, x);
    }

    // calc distro func (f_qi) at new time step + save 19 vals of distro func (f_qi) to curr cell
    for (int x = 0; x < LBM_Q; x++) {
        (destRefs[x])[ind] = (destRefs[x])[ind] - ((destRefs[x])[ind] - feq[x])/LBM_TAU;
    }
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

// TODO
__device__ void calcVelocity(int pos[3], float* d_velocities, float velocity[3]) {
    int ind = getInd(pos);
    for (int x = 0; x < 3; x++) {
        velocity[x] = d_velocities[3*ind + x];
    }
}

__device__ void clampInterpolation(int xPos[3]) {
    for (int x = 0; x < 3; x++) {
        xPos[x] = xPos[x] < d_N ? xPos[x] : d_N-1;
    }
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
__global__ void snowApplyGrav(float *d_snowflakeDataFlat, unsigned *d_numParticles, float *d_snowOffsets, curandState* d_globalState, float* d_velocities) {
    unsigned kernelInd = blockIdx.x*SNOW_GRAV_BLOCK_SIZE*SNOW_GRAV_BATCH_SIZE + threadIdx.x;
    unsigned stride = SNOW_GRAV_BLOCK_SIZE;

    // apply forces to snow particles
    float xOffset, yOffset, zOffset;
    float yOffsetRand;
    curandState localState;
    bool checkX, checkY, checkZ;
    float x_phys, y_phys, z_phys, x_lat, y_lat, z_lat;
    float x_phys_offset = min(d_extents[0], d_extents[1]), y_phys_offset = min(d_extents[2], d_extents[3]), z_phys_offset = min(d_extents[4], d_extents[5]);
    float dx, dy, dz;
    float u_x0[3], u_x1[3], u_x2[3], u_x3[3], u_x4[3], u_x5[3], u_x6[3], u_x7[3];
    float ux, uy, uz;
    int x_lat_int, y_lat_int, z_lat_int;
    int x0[3], x1[3], x2[3], x3[3], x4[3], x5[3], x6[3], x7[3];
    for (int x = kernelInd; x < min(kernelInd + SNOW_GRAV_BATCH_SIZE*stride, *d_numParticles); x+=stride) {
        // apply wind field to snow particles
        x_phys = d_snowflakeDataFlat[3*x] - x_phys_offset;
        y_phys = d_snowflakeDataFlat[3*x+1] - y_phys_offset;
        z_phys = d_snowflakeDataFlat[3*x+2] - z_phys_offset;
        x_lat = x_phys/d_delta_x_phys;
        y_lat = y_phys/d_delta_x_phys;
        z_lat = z_phys/d_delta_x_phys;
        x_lat_int = (int) x_lat;
        y_lat_int = (int) y_lat;
        z_lat_int = (int) z_lat;
        if (x_lat != x_lat_int || y_lat != y_lat_int || z_lat != z_lat_int) { // interpolation
            x0[0] = x_lat_int, x0[1] = y_lat_int, x0[2] = z_lat_int;
            x1[0] = x_lat_int, x1[1] = y_lat_int, x1[2] = z_lat_int + 1;
            x2[0] = x_lat_int + 1, x2[1] = y_lat_int, x2[2] = z_lat_int;
            x3[0] = x_lat_int, x3[1] = y_lat_int + 1, x3[2] = z_lat_int;
            x4[0] = x_lat_int, x4[1] = y_lat_int + 1, x4[2] = z_lat_int + 1;
            x5[0] = x_lat_int + 1, x5[1] = y_lat_int, x5[2] = z_lat_int + 1;
            x6[0] = x_lat_int + 1, x6[1] = y_lat_int + 1, x6[2] = z_lat_int;
            x7[0] = x_lat_int + 1, x7[1] = y_lat_int + 1, x7[2] = z_lat_int + 1;
            clampInterpolation(x0);
            clampInterpolation(x1);
            clampInterpolation(x2);
            clampInterpolation(x3);
            clampInterpolation(x4);
            clampInterpolation(x5);
            clampInterpolation(x6);
            clampInterpolation(x7);
            dx = x_lat-x0[0];
            dy = y_lat-x0[1];
            dz = z_lat-x0[2];
            calcVelocity(x0, d_velocities, u_x0);
            calcVelocity(x1, d_velocities, u_x1);
            calcVelocity(x2, d_velocities, u_x2);
            calcVelocity(x3, d_velocities, u_x3);
            calcVelocity(x4, d_velocities, u_x4);
            calcVelocity(x5, d_velocities, u_x5);
            calcVelocity(x6, d_velocities, u_x6);
            calcVelocity(x7, d_velocities, u_x7);
            ux = (1-dx)*(1-dy)*(1-dz)*u_x0[0]
                + (1-dx)*(1-dy)*dz*u_x1[0]
                + (1-dx)*dy*(1-dz)*u_x2[0]
                + dx*(1-dy)*(1-dz)*u_x3[0]
                + (1-dx)*dy*dz*u_x4[0]
                + dx*(1-dy)*dz*u_x5[0]
                + dx*dy*(1-dz)*u_x6[0]
                + dx*dy*dz*u_x7[0];
            uy = (1-dx)*(1-dy)*(1-dz)*u_x0[1]
               + (1-dx)*(1-dy)*dz*u_x1[1]
               + (1-dx)*dy*(1-dz)*u_x2[1]
               + dx*(1-dy)*(1-dz)*u_x3[1]
               + (1-dx)*dy*dz*u_x4[1]
               + dx*(1-dy)*dz*u_x5[1]
               + dx*dy*(1-dz)*u_x6[1]
               + dx*dy*dz*u_x7[1];
            uz = (1-dx)*(1-dy)*(1-dz)*u_x0[2]
                + (1-dx)*(1-dy)*dz*u_x1[2]
                + (1-dx)*dy*(1-dz)*u_x2[2]
                + dx*(1-dy)*(1-dz)*u_x3[2]
                + (1-dx)*dy*dz*u_x4[2]
                + dx*(1-dy)*dz*u_x5[2]
                + dx*dy*(1-dz)*u_x6[2]
                + dx*dy*dz*u_x7[2];
        } else {
            x0[0] = x_lat_int, x0[1] = y_lat_int, x0[2] = z_lat_int;
            calcVelocity(x0, d_velocities, u_x0);
            ux = u_x0[0];
            uy = u_x0[1];
            uz = u_x0[2];
        }
        ux = ux*d_delta_x_phys/d_delta_t_phys;
        uy = uy*d_delta_x_phys/d_delta_t_phys;
        uz = uz*d_delta_x_phys/d_delta_t_phys;
        //printf("vel %f %f %f\n\n", ux, uy, uz);

        checkX = d_snowflakeDataFlat[3*x] + ux < d_extents[0] || d_snowflakeDataFlat[3*x] + ux > d_extents[1];
        checkY = d_snowflakeDataFlat[3*x+1] + uy + GRAVITY < d_extents[2] || d_snowflakeDataFlat[3*x+1] + uy + GRAVITY > d_extents[3];
        checkZ = d_snowflakeDataFlat[3*x+2] + uz < d_extents[4] || d_snowflakeDataFlat[3*x+2] + uz > d_extents[5];
        if (checkX || checkY || checkZ) {
            localState = d_globalState[kernelInd];
            xOffset = getRandFloatGPU(d_extents[0], d_extents[1], &localState) - d_snowflakeDataFlat[3*x];
            yOffsetRand = getRandFloatGPU(-(SNOW_NOISE_Y*(d_extents[3] - d_extents[2])), 0, &localState);
            yOffset = max(d_extents[2], d_extents[3]) + yOffsetRand- d_snowflakeDataFlat[3*x+1];
            zOffset = getRandFloatGPU(d_extents[4], d_extents[5], &localState) - d_snowflakeDataFlat[3*x+2];
            d_globalState[kernelInd] = localState;
        } else {
            xOffset =  ux;
            yOffset = uy + GRAVITY;
            zOffset = uz;
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

// TODO
__global__ void initLBMModel(Lattice *d_srcLattice) {
    float velocity[] = {d_windVel, 0, d_windVel};
    float feq[LBM_Q];
    float t3 = velocity[0]*velocity[0] + velocity[1]*velocity[1] + velocity[2]*velocity[2];
    for (int x = 0; x < LBM_Q; x++) {
        feq[x] = feqFunc(velocity, 1.0f, t3, x);
    }
    float* srcRefs[] = {d_srcLattice->f0, d_srcLattice->f1, d_srcLattice->f2, d_srcLattice->f3, d_srcLattice->f4, d_srcLattice->f5, d_srcLattice->f6, d_srcLattice->f7, d_srcLattice->f8, d_srcLattice->f9, d_srcLattice->f10, d_srcLattice->f11, d_srcLattice->f12, d_srcLattice->f13, d_srcLattice->f14, d_srcLattice->f15, d_srcLattice->f16, d_srcLattice->f17, d_srcLattice->f18};
    for (int x = 0; x < d_N*d_N*d_N; x++) { // TODO: turn into kernel = max block size + however many blocks needed
        for (int i = 0; i < LBM_Q; i++) {
            (srcRefs[i])[x] = feq[i];
        }

    }
}

// TODO
bool isLBMModelValid(float h_extents[3][2], float h_windVel, unsigned h_N, float h_temp) {
    float h_viscosity_phys = 0.000012890; // m^2/s from temp (-5) (https://theengineeringmindset.com/properties-of-air-at-atmospheric-pressure/) TODO
    float h_cs_phys =  328.25; // m/s from temp (-5) (https://en.wikipedia.org/wiki/Speed_of_sound)
    float h_delta_x_phys = abs(h_extents[0][1]-h_extents[0][0])/h_N;
    float h_delta_t_phys = (LBM_C_S/h_cs_phys)*h_delta_x_phys;
    float h_tau = 3*(h_viscosity_phys*(h_delta_t_phys/pow(h_delta_x_phys, 2)))+1.0/2;
    printf("%f %f %f\n", h_delta_x_phys, h_delta_t_phys, h_tau);
    float h_windVel_lbm = h_windVel*(h_delta_t_phys/h_delta_x_phys);

    // if valid write to dev
    cudaMemcpyToSymbol(d_tau, &h_tau, sizeof(float));
    cudaMemcpyToSymbol(d_delta_x_phys, &h_delta_x_phys, sizeof(float));
    cudaMemcpyToSymbol(d_delta_t_phys, &h_delta_t_phys, sizeof(float));
    cudaMemcpyToSymbol(d_N, &h_N, sizeof(unsigned));
    cudaMemcpyToSymbol(d_windVel, &h_windVel_lbm, sizeof(float));

    return false; // TODO: rem
}

/**
 * Initialize data on the GPU.
 * @param data - SnowGeneratorData object with particle data.
 * @param numParticles - Number of particles.
 * @param extents - Extents of the volume in which to generate the particles, where extents[0] is a pair for the x extent,
 * extents[1] is a pair for the y extent, and extents[2] is a pair for the z extent. If numParticles = 1 this
 * parameter has no effect and the snow particle is generated at the origin.
 */
extern void snowInitGPU(SnowGeneratorData data, unsigned numParticles, float extents[3][2], float windVel, unsigned latticeRes, float temp) {
    // validate model
    if (!isLBMModelValid(extents, windVel, latticeRes, temp)) { // TODO
        printf("Invalid model parameters\n");
        // exit(-1); TODO
    }

    // save refs
    h_numPolys = data.numPolys;
    h_numParticles = numParticles;
    h_verts = data.verts;
    h_snowGravNumBlocks = ceil(h_numParticles/(SNOW_GRAV_BLOCK_SIZE*SNOW_GRAV_BATCH_SIZE*1.0)); // TODO: mem access block size (256) dictates snow batch size
    h_snowUpdateNumBlocks = ceil(h_numParticles/(SNOW_UPDATE_BLOCK_SIZE*1.0));
    h_snowflakeData = data.snowflakeData;
    h_lbmBlockSize = dim3(latticeRes, 1, 1);
    h_lbmGridSize = dim3(latticeRes, latticeRes, 1);

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

    // flatten extents
    float *h_extents;
    cudaMallocHost((void**)&h_extents, 6*sizeof(float));
    for (int x = 0; x < 3; x++) {
        for (int i = 0; i < 2; i++) {
            h_extents[2*x+i] = extents[x][i];
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
    cudaMemcpyToSymbol(d_extents, h_extents, 6*sizeof(float));

    // free
    cudaFreeHost(h_snowflakeDataFlat);
    cudaFreeHost(h_snowOffsets);
    cudaFreeHost(h_extents);

    // init dest & src grids + velocities
    Lattice h_srcLattice; // TODO: better way to do this?
    float *srcf0, *srcf1, *srcf2, *srcf3, *srcf4, *srcf5, *srcf6, *srcf7, *srcf8, *srcf9, *srcf10, *srcf11, *srcf12, *srcf13, *srcf14, *srcf15, *srcf16, *srcf17, *srcf18;
    cudaMalloc((void**)&d_srcLattice, sizeof(Lattice));
    cudaMalloc((void**)&srcf0, latticeRes*latticeRes*latticeRes*sizeof(float));
    cudaMalloc((void**)&srcf1, latticeRes*latticeRes*latticeRes*sizeof(float));
    cudaMalloc((void**)&srcf2, latticeRes*latticeRes*latticeRes*sizeof(float));
    cudaMalloc((void**)&srcf3, latticeRes*latticeRes*latticeRes*sizeof(float));
    cudaMalloc((void**)&srcf4, latticeRes*latticeRes*latticeRes*sizeof(float));
    cudaMalloc((void**)&srcf5, latticeRes*latticeRes*latticeRes*sizeof(float));
    cudaMalloc((void**)&srcf6, latticeRes*latticeRes*latticeRes*sizeof(float));
    cudaMalloc((void**)&srcf7, latticeRes*latticeRes*latticeRes*sizeof(float));
    cudaMalloc((void**)&srcf8, latticeRes*latticeRes*latticeRes*sizeof(float));
    cudaMalloc((void**)&srcf9, latticeRes*latticeRes*latticeRes*sizeof(float));
    cudaMalloc((void**)&srcf10, latticeRes*latticeRes*latticeRes*sizeof(float));
    cudaMalloc((void**)&srcf11, latticeRes*latticeRes*latticeRes*sizeof(float));
    cudaMalloc((void**)&srcf12, latticeRes*latticeRes*latticeRes*sizeof(float));
    cudaMalloc((void**)&srcf13, latticeRes*latticeRes*latticeRes*sizeof(float));
    cudaMalloc((void**)&srcf14, latticeRes*latticeRes*latticeRes*sizeof(float));
    cudaMalloc((void**)&srcf15, latticeRes*latticeRes*latticeRes*sizeof(float));
    cudaMalloc((void**)&srcf16, latticeRes*latticeRes*latticeRes*sizeof(float));
    cudaMalloc((void**)&srcf17, latticeRes*latticeRes*latticeRes*sizeof(float));
    cudaMalloc((void**)&srcf18, latticeRes*latticeRes*latticeRes*sizeof(float));
    h_srcLattice.f0 = srcf0;
    h_srcLattice.f1 = srcf1;
    h_srcLattice.f2 = srcf2;
    h_srcLattice.f3 = srcf3;
    h_srcLattice.f4 = srcf4;
    h_srcLattice.f5 = srcf5;
    h_srcLattice.f6 = srcf6;
    h_srcLattice.f7 = srcf7;
    h_srcLattice.f8 = srcf8;
    h_srcLattice.f9 = srcf9;
    h_srcLattice.f10 = srcf10;
    h_srcLattice.f11 = srcf11;
    h_srcLattice.f12 = srcf12;
    h_srcLattice.f13 = srcf13;
    h_srcLattice.f14 = srcf14;
    h_srcLattice.f15 = srcf15;
    h_srcLattice.f16 = srcf16;
    h_srcLattice.f17 = srcf17;
    h_srcLattice.f18 = srcf18;
    cudaMemcpy(d_srcLattice, &h_srcLattice, sizeof(Lattice), cudaMemcpyHostToDevice);
    Lattice h_destLattice; // TODO: better way to do this?
    float *destf0, *destf1, *destf2, *destf3, *destf4, *destf5, *destf6, *destf7, *destf8, *destf9, *destf10, *destf11, *destf12, *destf13, *destf14, *destf15, *destf16, *destf17, *destf18;
    cudaMalloc((void**)&d_destLattice, sizeof(Lattice));
    cudaMalloc((void**)&destf0, latticeRes*latticeRes*latticeRes*sizeof(float));
    cudaMalloc((void**)&destf1, latticeRes*latticeRes*latticeRes*sizeof(float));
    cudaMalloc((void**)&destf2, latticeRes*latticeRes*latticeRes*sizeof(float));
    cudaMalloc((void**)&destf3, latticeRes*latticeRes*latticeRes*sizeof(float));
    cudaMalloc((void**)&destf4, latticeRes*latticeRes*latticeRes*sizeof(float));
    cudaMalloc((void**)&destf5, latticeRes*latticeRes*latticeRes*sizeof(float));
    cudaMalloc((void**)&destf6, latticeRes*latticeRes*latticeRes*sizeof(float));
    cudaMalloc((void**)&destf7, latticeRes*latticeRes*latticeRes*sizeof(float));
    cudaMalloc((void**)&destf8, latticeRes*latticeRes*latticeRes*sizeof(float));
    cudaMalloc((void**)&destf9, latticeRes*latticeRes*latticeRes*sizeof(float));
    cudaMalloc((void**)&destf10, latticeRes*latticeRes*latticeRes*sizeof(float));
    cudaMalloc((void**)&destf11, latticeRes*latticeRes*latticeRes*sizeof(float));
    cudaMalloc((void**)&destf12, latticeRes*latticeRes*latticeRes*sizeof(float));
    cudaMalloc((void**)&destf13, latticeRes*latticeRes*latticeRes*sizeof(float));
    cudaMalloc((void**)&destf14, latticeRes*latticeRes*latticeRes*sizeof(float));
    cudaMalloc((void**)&destf15, latticeRes*latticeRes*latticeRes*sizeof(float));
    cudaMalloc((void**)&destf16, latticeRes*latticeRes*latticeRes*sizeof(float));
    cudaMalloc((void**)&destf17, latticeRes*latticeRes*latticeRes*sizeof(float));
    cudaMalloc((void**)&destf18, latticeRes*latticeRes*latticeRes*sizeof(float));
    h_destLattice.f0 = destf0;
    h_destLattice.f1 = destf1;
    h_destLattice.f2 = destf2;
    h_destLattice.f3 = destf3;
    h_destLattice.f4 = destf4;
    h_destLattice.f5 = destf5;
    h_destLattice.f6 = destf6;
    h_destLattice.f7 = destf7;
    h_destLattice.f8 = destf8;
    h_destLattice.f9 = destf9;
    h_destLattice.f10 = destf10;
    h_destLattice.f11 = destf11;
    h_destLattice.f12 = destf12;
    h_destLattice.f13 = destf13;
    h_destLattice.f14 = destf14;
    h_destLattice.f15 = destf15;
    h_destLattice.f16 = destf16;
    h_destLattice.f17 = destf17;
    h_destLattice.f18 = destf18;
    cudaMemcpy(d_destLattice, &h_destLattice, sizeof(Lattice), cudaMemcpyHostToDevice);
    initLBMModel<<<1, 1>>>(d_srcLattice);
    cudaMalloc((void**)&d_velocities, 3*latticeRes*latticeRes*latticeRes*sizeof(float));

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
    lbmKernel<<<h_lbmGridSize, h_lbmBlockSize>>>(d_srcLattice, d_destLattice, d_velocities);
    cudaDeviceSynchronize();

    // apply forces to snow
    snowApplyGrav<<<h_snowGravNumBlocks,SNOW_GRAV_BLOCK_SIZE>>>(d_snowflakeDataFlat, d_numParticles, d_snowOffsets, d_globalState, d_velocities);

    // update snow verts
    snowUpdate<<<h_snowUpdateNumBlocks,SNOW_UPDATE_BLOCK_SIZE>>>(d_verts, d_snowOffsets, d_numParticles);

    // fetch work
    cudaMemcpy(h_verts, d_verts, h_numPolys*9*sizeof(float), cudaMemcpyDeviceToHost);

    // swap grids
    Lattice *swap = d_srcLattice;
    d_srcLattice = d_destLattice;
    d_destLattice = swap;
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