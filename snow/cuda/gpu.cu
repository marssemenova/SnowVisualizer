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

// external code dependencies
#include "../SnowGeneratorData.hpp"
#include "../../util/DevInput.hpp"

// consts
__constant__ __device__ float GRAVITY = -9.81f*10.0f;
__constant__ __device__ float SNOW_NOISE_Y = 0.1f;
#define LBM_Q 19
__constant__ __device__ float LBM_C = 1.0; // from speed of sounds = 1/sqrt(3)
const float LBM_C_S = 1.0/sqrt(3); // from speed of sounds = 1/sqrt(3)
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
// apply forces
const unsigned SNOW_FORCES_BLOCK_SIZE = 1024;
const unsigned SNOW_FORCES_BATCH_SIZE = 8;
unsigned h_snowForcesNumBlocks;
// update snow
const unsigned SNOW_UPDATE_BLOCK_SIZE = 1024;
unsigned h_snowUpdateNumBlocks;

// host vars
float *h_verts;
SnowflakeData *h_snowflakeData;
unsigned h_numPolys;
unsigned h_numParticles;

// dev vars
__constant__ __device__ float d_extents[6];
__constant__ __device__ float d_delta_x_phys;
__constant__ __device__ float d_delta_t_phys;
__constant__ __device__ unsigned d_N;
__constant__ __device__ float d_tau;
__constant__ __device__ float d_windVel;
__constant__ __device__ float d_feqInit[LBM_Q];
float *d_verts;
float *d_snowflakeDataFlat;
unsigned *d_numParticles;
float *d_snowOffsets;
float *d_velocities;
curandState *d_globalState;
Lattice *d_srcLattice;
Lattice *d_destLattice;

#include "lbm_utils.h"

/**
 * Helper function to check whether a lattice point is on the boundary or not.
 * @param accessX - 3D x index in of the lattice point to check.
 * @param accessY - 3D y index in of the lattice point to check.
 * @param accessZ - 3D z index in of the lattice point to check.
 * @return Whether the lattice point is not on the boundary.
 */
__device__ bool applyBoundaryConds(float accessX, float accessY, float accessZ) {
    return (0 <= accessX && accessX < d_N && 0 <= accessY && accessY < d_N && 0 <= accessZ && accessZ < d_N);
}

/**
 * Helper function to calculate the local equilibrium distribution function.
 * @param velocity - A float[3] containing the velocity vector used in the local equilibrium distribution function.
 * @param density
 * @param t3 - Term 3 of the local equilibrium distribution functions. Since it is the same for all
 * local equilibrium distribution functions of a lattice point it is pre-calculated to avoid calculating it 19 times.
 * @return The computed value of the local equilibrium distribution function with the given parameters.
 */
__device__ float feqFunc(float velocity[3], float density, float t3, int f) {
    float t1 = D_LATTICE_VELOCITIES[f][0]*velocity[0] + D_LATTICE_VELOCITIES[f][1]*velocity[1] + D_LATTICE_VELOCITIES[f][2]*velocity[2];
    float t2 = t1*t1;
    return D_LATTICE_WEIGHTS[f]*density*(1 + (3.0/LBM_C)*t1 + (9.0/(2.0*pow(LBM_C,2)))*t2 - (3.0/(2.0*pow(LBM_C,2)))*t3);
}

/**
 * LBM kernel.
 * @param d_srcLattice - Source lattice.
 * @param d_destLattice - Destination lattice.
 * @param d_velocities - Array on the GPU which stores the calculated velocity at each lattice point.
 */
__global__ void lbmKernel(Lattice *d_srcLattice, Lattice *d_destLattice, float* d_velocities) {
    // compute the 3D position of the thread
    int x = threadIdx.x;
    int y = blockIdx.x;
    int z = blockIdx.y;
    // compute the corresponding 1D index
    int ind = getInd(x, y, z);

    if (!(ind < d_N*d_N*d_N)) {
        return;
    }

    // stream 19 pdfs from adjacent cells to curr cell + apply boundary conds
    float* destRefs[] = {d_destLattice->f0, d_destLattice->f1, d_destLattice->f2, d_destLattice->f3, d_destLattice->f4, d_destLattice->f5, d_destLattice->f6, d_destLattice->f7, d_destLattice->f8, d_destLattice->f9, d_destLattice->f10, d_destLattice->f11, d_destLattice->f12, d_destLattice->f13, d_destLattice->f14, d_destLattice->f15, d_destLattice->f16, d_destLattice->f17, d_destLattice->f18};
    float* srcRefs[] = {d_srcLattice->f0, d_srcLattice->f1, d_srcLattice->f2, d_srcLattice->f3, d_srcLattice->f4, d_srcLattice->f5, d_srcLattice->f6, d_srcLattice->f7, d_srcLattice->f8, d_srcLattice->f9, d_srcLattice->f10, d_srcLattice->f11, d_srcLattice->f12, d_srcLattice->f13, d_srcLattice->f14, d_srcLattice->f15, d_srcLattice->f16, d_srcLattice->f17, d_srcLattice->f18};

    if (x == 0) { // inlet for dynamic wind
        for (int i = 0; i < LBM_Q; i++) {
            (destRefs[i])[ind] = (destRefs[i])[ind] - ((destRefs[i])[ind] - d_feqInit[i])/d_tau;
        }
        return;
    }

    bool flag[19];
    int accessX, accessY, accessZ, accessInd;
    for (int i = 0; i < LBM_Q; i++) {
        flag[i] = true;
        //subtract lattice velocities because we are "pulling in"
        accessX = x-D_LATTICE_VELOCITIES[i][0];
        accessY = y-D_LATTICE_VELOCITIES[i][1];
        accessZ = z-D_LATTICE_VELOCITIES[i][2];
        if (applyBoundaryConds(accessX, accessY, accessZ)) {
            accessInd = getInd(accessX, accessY, accessZ);
            (destRefs[i])[ind] = (srcRefs[i])[accessInd];
            flag[i] = false;
        }
    }

    // apply boundary conds
    if (flag[1]) { // f1=f2
        d_destLattice->f1[ind] = d_srcLattice->f2[ind];
    }
    if (flag[2]) { // f2=f1
        d_destLattice->f2[ind] = d_srcLattice->f1[ind];
    }
    if (flag[3]) { // f3=f4
        d_destLattice->f3[ind] = d_srcLattice->f4[ind];
    }
    if (flag[4]) { // f4=f3
        d_destLattice->f4[ind] = d_srcLattice->f3[ind];
    }
    if (flag[5]) { // f5=f6
        d_destLattice->f5[ind] = d_srcLattice->f6[ind];
    }
    if (flag[6]) { // f6=f5
        d_destLattice->f6[ind] = d_srcLattice->f5[ind];
    }
    if (flag[7]) { // f7=f10
        d_destLattice->f7[ind] = d_srcLattice->f10[ind];
    }
    if (flag[8]) { // f8=f9
        d_destLattice->f8[ind] = d_srcLattice->f9[ind];
    }
    if (flag[9]) { // f9=f8
        d_destLattice->f9[ind] = d_srcLattice->f8[ind];
    }
    if (flag[10]) { // f10=f7
        d_destLattice->f10[ind] = d_srcLattice->f7[ind];
    }
    if (flag[11]) { // f11=f14
        d_destLattice->f11[ind] = d_srcLattice->f14[ind];
    }
    if (flag[12]) { // f12=f13
        d_destLattice->f12[ind] = d_srcLattice->f13[ind];
    }
    if (flag[13]) { // f13=f12
        d_destLattice->f13[ind] = d_srcLattice->f12[ind];
    }
    if (flag[14]) { // f14=f11
        d_destLattice->f14[ind] = d_srcLattice->f11[ind];
    }
    if (flag[15]) { // f15=f18
        d_destLattice->f15[ind] = d_srcLattice->f18[ind];
    }
    if (flag[16]) { // f16=f17
        d_destLattice->f16[ind] = d_srcLattice->f17[ind];
    }
    if (flag[17]) { // f17=f16
        d_destLattice->f17[ind] = d_srcLattice->f16[ind];
    }
    if (flag[18]) { // f18=f15
        d_destLattice->f18[ind] = d_srcLattice->f15[ind];
    }

    // calc density (rho)
    float density = 0;
    for (int i = 0; i < LBM_Q; i++) {
        density += (destRefs[i])[ind];
    }

    // calc velocity (u)
    float velocity[3];
    for (int i = 0; i < 3; i++) {
        velocity[i] = 0;
        for (int j = 0; j < LBM_Q; j++) {
            velocity[i] += (destRefs[j])[ind]*D_LATTICE_VELOCITIES[j][i];
        }
        velocity[i]/=density;
        d_velocities[3*ind + i] = velocity[i]; // save
    }

    // calc the loc equilibrium distro funcs f_qi^eq
    float feq[LBM_Q];
    float t3 = velocity[0]*velocity[0] + velocity[1]*velocity[1] + velocity[2]*velocity[2];
    for (int i = 0; i < LBM_Q; i++) {
        feq[i] = feqFunc(velocity, density, t3, i);
    }

    // calc distro func (f_qi) at new time step + save 19 vals of distro func (f_qi) to curr cell
    for (int i = 0; i < LBM_Q; i++) {
        (destRefs[i])[ind] = (destRefs[i])[ind] - ((destRefs[i])[ind] - feq[i])/d_tau;
    }
}

/**
 * Create and initialize curandState using seed, one for each thread.
 * Stores result in globalState.
 * @param g_globalState - Allocated array on the GPU to store the global states.
 * @param seed - Seed used for random generation.
 * @param d_numParticles - Number of snow particles.
 */
__global__ void setupSnowRandState(curandState* d_globalState, uint64_t seed, unsigned *d_numParticles) {
    unsigned kernelInd = blockIdx.x*SNOW_FORCES_BLOCK_SIZE*SNOW_FORCES_BATCH_SIZE + threadIdx.x;
    unsigned stride = SNOW_FORCES_BLOCK_SIZE;
    for (int x = kernelInd; x < min(kernelInd + SNOW_FORCES_BATCH_SIZE*stride, *d_numParticles); x+=stride) {
        if (x < *d_numParticles) {
            curand_init(seed, x, 0, &d_globalState[x]);
        }
    }
}

/**
 * Helper function to calculate the velocity at a lattice point.
 * @param pos - A float[3] containing the lattice point's 3D coordinates in the form {x, y, z}.
 * @param d_velocities - Array on the GPU which stores the calculated velocity at each lattice point.
 * @param velocity - A float[3] into which to store the computed velocity.
 */
__device__ void calcVelocity(int pos[3], float* d_velocities, float velocity[3]) {
    int ind = getInd(pos);
    for (int x = 0; x < 3; x++) {
        velocity[x] = d_velocities[3*ind + x];
    }
}

/**
 * Helper function to clamp the xi used in the linear interpolation of snow particles to lattice points
 * within the boundaries of the lattice.
 * @param xPos - A float[3] containing the xi's 3D coordinates in lattice space in the form {x, y, z}.
 */
__device__ void clampInterpolation(int xPos[3]) {
    for (int x = 0; x < 3; x++) {
        xPos[x] = xPos[x] < d_N ? xPos[x] : d_N-1;
    }
}

/**
 * Kernel which applies forces, gravity and wind, to snow particles and stores the offset.
 * @param d_snowflakeDataFlat - Flattened snowflake data. Every 3 elements correspond
 * to a snow particle's coordinates.
 * @param d_numParticles - Number of snow particles.
 * @param d_snowOffsets - The offsets array into which to write. Every 5 elements correspond to a
 * particle's data, where the first 3 elements are the offset, the 4th element
 * is the number of polygons in the snowflake, and the 5th parameter is the first index of its
 * vertices in the vertices array.
 * @param d_globalState - Global state array used for random generation on the GPU.
 * @param d_velocities - Array on the GPU which stores the calculated velocity at each lattice point.
 */
__global__ void snowApplyForces(float *d_snowflakeDataFlat, unsigned *d_numParticles, float *d_snowOffsets, curandState* d_globalState, float* d_velocities) {
    unsigned kernelInd = blockIdx.x*SNOW_FORCES_BLOCK_SIZE*SNOW_FORCES_BATCH_SIZE + threadIdx.x;
    unsigned stride = SNOW_FORCES_BLOCK_SIZE;

    // apply forces to snow particles
    curandState localState;
    float xOffset, yOffset, zOffset;
    float yOffsetRand;
    bool checkX, checkY, checkZ;
    float x_phys, y_phys, z_phys, x_lat, y_lat, z_lat;
    float x_phys_offset = d_extents[0], y_phys_offset = d_extents[2], z_phys_offset = d_extents[4];
    float dx, dy, dz;
    float u_x0[3], u_x1[3], u_x2[3], u_x3[3], u_x4[3], u_x5[3], u_x6[3], u_x7[3];
    float ux, uy, uz;
    int x_lat_int, y_lat_int, z_lat_int;
    int x0[3], x1[3], x2[3], x3[3], x4[3], x5[3], x6[3], x7[3];
    for (int x = kernelInd; x < min(kernelInd + SNOW_FORCES_BATCH_SIZE*stride, *d_numParticles); x+=stride) {
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

        checkX = d_snowflakeDataFlat[3*x] + ux <= d_extents[0] || d_snowflakeDataFlat[3*x] + ux >= d_extents[1];
        checkY = d_snowflakeDataFlat[3*x+1] + uy + GRAVITY*d_delta_t_phys <= d_extents[2] || d_snowflakeDataFlat[3*x+1] + uy + GRAVITY*d_delta_t_phys >= d_extents[3];
        checkZ = d_snowflakeDataFlat[3*x+2] + uz <= d_extents[4] || d_snowflakeDataFlat[3*x+2] + uz >= d_extents[5];

        if (checkY) {
            localState = d_globalState[x];
            xOffset = getRandFloatGPU(d_extents[0], d_extents[1], &localState) - d_snowflakeDataFlat[3*x];
            yOffsetRand = getRandFloatGPU(-(SNOW_NOISE_Y*(d_extents[3] - d_extents[2])), 0, &localState);
            yOffset = d_extents[3] + yOffsetRand- d_snowflakeDataFlat[3*x+1];
            zOffset = getRandFloatGPU(d_extents[4], d_extents[5], &localState) - d_snowflakeDataFlat[3*x+2];
            d_globalState[x] = localState;
        } else {
            xOffset = checkX ? -ux : ux;
            yOffset = uy + GRAVITY*d_delta_t_phys;
            zOffset = checkZ ? -uz : uz;
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
 * @param d_numParticles - Number of snow particles.
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
 * Calculate and store the initial local equilibrium distribution functions.
 * @param feqInitCpy - Array temporarily allocated on the devices used to copy to d_feqInit, the constant
 * field on the device.
 */
__global__ void feqInitKernel(float *feqInitCpy) {
    float velocity[] = {d_windVel, 0, d_windVel};
    float feq[LBM_Q];
    float t3 = velocity[0]*velocity[0] + velocity[1]*velocity[1] + velocity[2]*velocity[2];
    for (int x = 0; x < LBM_Q; x++) {
        feq[x] = feqFunc(velocity, 1.0f, t3, x);
        feqInitCpy[x] = feq[x];
    }
}

/**
 * Initialize the model.
 * @param d_srcLattice - Source lattice allocated on the device.
 */
__global__ void initLBMModel(Lattice *d_srcLattice) {
    // compute the 3D position of the thread
    int x = threadIdx.x;
    int y = blockIdx.x;
    int z = blockIdx.y;
    // compute the corresponding 1D index
    int ind = getInd(x, y, z);

    if (!(ind < d_N*d_N*d_N)) {
        return;
    }

    float* srcRefs[] = {d_srcLattice->f0, d_srcLattice->f1, d_srcLattice->f2, d_srcLattice->f3, d_srcLattice->f4, d_srcLattice->f5, d_srcLattice->f6, d_srcLattice->f7, d_srcLattice->f8, d_srcLattice->f9, d_srcLattice->f10, d_srcLattice->f11, d_srcLattice->f12, d_srcLattice->f13, d_srcLattice->f14, d_srcLattice->f15, d_srcLattice->f16, d_srcLattice->f17, d_srcLattice->f18};
    for (int i = 0; i < LBM_Q; i++) {
        (srcRefs[i])[ind] = d_feqInit[i];
    }
}

/**
 * Initialize data on the GPU.
 * @param data - SnowGeneratorData object with particle data.
 * @param numParticles - Number of particles.
 * @param extents - Extents of the volume in which to generate the wind field, where extents[0] is a pair for the x extent,
 * extents[1] is a pair for the y extent, and extents[2] is a pair for the z extent.
 * @param windVel - The macroscopic flow velocity of the wind in km/h.
 * @param latticeRes - Lattice resolution.
 * @param temp - Temperature of the simulation.
 */
extern void snowInitGPU(SnowGeneratorData data, unsigned numParticles, float extents[3][2], float windVel, unsigned latticeRes, float temp) {
    // format extents
    float tempExt;
    for (int x = 0; x < 3; x++) {
        if (extents[x][0] > extents[x][1]) {
            tempExt = extents[x][1];
            extents[x][1] = extents[x][0];
            extents[x][0] = tempExt;
        }
    }

    // set model params
    setLBMModelParams(extents, windVel, latticeRes, temp);

    // save refs
    h_numPolys = data.numPolys;
    h_numParticles = numParticles;
    h_verts = data.verts;
    h_snowForcesNumBlocks = ceil(h_numParticles/(SNOW_FORCES_BLOCK_SIZE*SNOW_FORCES_BATCH_SIZE*1.0));
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
    float *d_feqInitCpy;
    cudaMalloc((void**)&d_feqInitCpy, LBM_Q*sizeof(float));
    feqInitKernel<<<1, 1>>>(d_feqInitCpy);
    cudaMemcpyToSymbol(d_feqInit, d_feqInitCpy, LBM_Q*sizeof(float));
    initLBMModel<<<h_lbmGridSize, h_lbmBlockSize>>>(d_srcLattice);
    cudaMalloc((void**)&d_velocities, 3*latticeRes*latticeRes*latticeRes*sizeof(float));

    // free
    cudaFreeHost(h_snowflakeDataFlat);
    cudaFreeHost(h_snowOffsets);
    cudaFreeHost(h_extents);
    cudaFree(d_feqInitCpy);

    // init rand
    cudaMalloc((void**)&d_globalState, h_numParticles*sizeof(curandState));
    setupSnowRandState<<<h_snowForcesNumBlocks,SNOW_FORCES_BLOCK_SIZE>>>(d_globalState, time(NULL), d_numParticles);
    cudaDeviceSynchronize();
}

/**
 * Update snow on the GPU via kernel calls.
 */
extern void snowUpdateGPU() {
    // dispatch kernels
    // LBM
    time_point lbmStart;
    if (PROFILING) {
        frameCount++;
        lbmStart = startTimer();
    }
    lbmKernel<<<h_lbmGridSize, h_lbmBlockSize>>>(d_srcLattice, d_destLattice, d_velocities);
    if (PROFILING) {
        cudaDeviceSynchronize();
        time_point lbmEnd = stopTimer();
        float lbmTime = elapsedTime(lbmStart, lbmEnd);
        lbmTimeTot += lbmTime;
    }

    // apply forces to snow
    time_point forcesStart;
    if (PROFILING) {
        forcesStart = startTimer();
    }
    snowApplyForces<<<h_snowForcesNumBlocks,SNOW_FORCES_BLOCK_SIZE>>>(d_snowflakeDataFlat, d_numParticles, d_snowOffsets, d_globalState, d_velocities);
    if (PROFILING) {
        cudaDeviceSynchronize();
        time_point forcesEnd = stopTimer();
        float forcesTime = elapsedTime(forcesStart, forcesEnd);
        forcesTimeTot += forcesTime;
    }

    // update snow verts
    time_point updateStart;
    if (PROFILING) {
        updateStart = startTimer();
    }
    snowUpdate<<<h_snowUpdateNumBlocks,SNOW_UPDATE_BLOCK_SIZE>>>(d_verts, d_snowOffsets, d_numParticles);
    if (PROFILING) {
        cudaDeviceSynchronize();
        time_point updateEnd = stopTimer();
        float updateTime = elapsedTime(updateStart, updateEnd);
        updateTimeTot += updateTime;
    }

    // fetch work
    time_point cpyStart;
    if (PROFILING) {
        cpyStart = startTimer();
    }
    cudaMemcpy(h_verts, d_verts, h_numPolys*9*sizeof(float), cudaMemcpyDeviceToHost);
    if (PROFILING) {
        cudaDeviceSynchronize();
        time_point cpyEnd = stopTimer();
        cpyTime = elapsedTime(cpyStart, cpyEnd);
        cpyTimeTot += cpyTime;
        if (frameCount >= NUM_FRAMES) {
            printf("LBM kernel time over %d frames: %fms\n", NUM_FRAMES, lbmTimeTot/NUM_FRAMES);
            lbmTimeTot = 0;
            printf("Snow particle update kernel time over %d frames: %fms\n", NUM_FRAMES, forcesTimeTot/NUM_FRAMES);
            forcesTimeTot = 0;
            printf("Vertices update kernel time over %d frames: %fms\n", NUM_FRAMES, updateTimeTot/NUM_FRAMES);
            updateTimeTot = 0;
            printf("GPU to CPU copy time over %d frames: %fms\n", NUM_FRAMES, cpyTimeTot/NUM_FRAMES);
            cpyTimeTot = 0;
            frameCount = 0;
        }
    }

    // swap grids
    Lattice *swap = d_srcLattice;
    d_srcLattice = d_destLattice;
    d_destLattice = swap;
}