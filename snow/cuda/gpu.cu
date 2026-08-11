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
#define N 5 // TODO: make adjustable
struct Lattice {
    float f0[N*N*N];
    float f1[N*N*N];
    float f2[N*N*N];
    float f3[N*N*N];
    float f4[N*N*N];
    float f5[N*N*N];
    float f6[N*N*N];
    float f7[N*N*N];
    float f8[N*N*N];
    float f9[N*N*N];
    float f10[N*N*N];
    float f11[N*N*N];
    float f12[N*N*N];
    float f13[N*N*N];
    float f14[N*N*N];
    float f15[N*N*N];
    float f16[N*N*N];
    float f17[N*N*N];
    float f18[N*N*N];
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
Lattice h_srcLattice; // TODO: del
Lattice h_destLattice; // TODO: del

// dev vars
__constant__ __device__ float d_extents[6];
__constant__ __device__ float d_tau;
__constant__ __device__ float d_delta_x_phys;
__constant__ __device__ float d_delta_t_phys;
__constant__ __device__ unsigned d_N;
__constant__ __device__ float d_windVel;
float *d_verts;
float *d_snowflakeDataFlat;
unsigned *d_numParticles; // TODO: make just a param
float *d_snowOffsets;
float *d_velocities;
curandState *d_globalState;
Lattice *d_srcLattice;
Lattice *d_destLattice;

// TODO
__device__ bool applyBoundaryConds(float accessX, float accessY, float accessZ, int f) {
    if (f == 1) {
        return accessX > 0;
    }
    if (f == 2) {
        return accessX < N-1;
    }
    if (f == 3) {
        return accessY > 0;
    }
    if (f == 4) {
        return accessY < N-1;
    }
    if (f == 5) {
        return accessZ > 0;
    }
    if (f == 6) {
        return accessZ < N-1;
    }
    if (f == 7) {
        return accessX > 0 && accessY > 0;
    }
    if (f == 8) {
        return accessX > 0 && accessY < N-1;
    }
    if (f == 9) {
        return accessX < N-1 && accessY > 0;
    }
    if (f == 10) {
        return accessX < N-1 && accessY < N-1;
    }
    if (f == 11) {
        return accessX > 0 && accessZ > 0;
    }
    if (f == 12) {
        return accessX > 0 && accessZ < N-1;
    }
    if (f == 13) {
        return accessX < N-1 && accessZ > 0;
    }
    if (f == 14) {
        return accessX < N-1 && accessZ < N-1;
    }
    if (f == 15) {
        return accessY > 0 && accessZ > 0;
    }
    if (f == 16) {
        return accessY > 0 && accessZ < N-1;
    }
    if (f == 17) {
        return accessY < N-1 && accessZ > 0;
    }
    if (f == 18) {
        return accessY < N-1 && accessZ < N-1;
    }
}

// TODO
__global__ void lbmKernel(Lattice *d_srcLattice, Lattice *d_destLattice, float* d_velocities) {
    // compute the 3D position of the thread
    int x = threadIdx.x;
    int y = blockIdx.x;
    int z = blockIdx.y;

    // compute the corresponding 1D index
    int ind = x + y * N + z * N*N;
    if (!(ind < N*N*N)) {
        return;
    }

    // stream 19 pdfs from adjacent cells to curr cell + apply boundary conds
    int accessX, accessY, accessZ, accessInd;
    bool flag[19] = {true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true};
    d_destLattice->f0[ind] = d_srcLattice->f0[ind];
    accessX = x-D_LATTICE_VELOCITIES[1][0];
    accessY = y-D_LATTICE_VELOCITIES[1][1];
    accessZ = z-D_LATTICE_VELOCITIES[1][2];
    if (applyBoundaryConds(accessX, accessY, accessZ, 1)) {
        accessInd = accessX + accessY * N + accessZ * N*N;
        d_destLattice->f1[ind] = d_srcLattice->f1[accessInd];
        flag[1] = false;
    }
    accessX = x-D_LATTICE_VELOCITIES[2][0];
    accessY = y-D_LATTICE_VELOCITIES[2][1];
    accessZ = z-D_LATTICE_VELOCITIES[2][2];
    if (applyBoundaryConds(accessX, accessY, accessZ, 2)) {
        accessInd = accessX + accessY * N + accessZ * N*N;
        d_destLattice->f2[ind] = d_srcLattice->f2[accessInd];
        flag[2] = false;
    }
    accessX = x-D_LATTICE_VELOCITIES[3][0];
    accessY = y-D_LATTICE_VELOCITIES[3][1];
    accessZ = z-D_LATTICE_VELOCITIES[3][2];
    if (applyBoundaryConds(accessX, accessY, accessZ, 3)) {
        accessInd = accessX + accessY * N + accessZ * N*N;
        d_destLattice->f3[ind] = d_srcLattice->f3[accessInd];
        flag[3] = false;
    }
    accessX = x-D_LATTICE_VELOCITIES[4][0];
    accessY = y-D_LATTICE_VELOCITIES[4][1];
    accessZ = z-D_LATTICE_VELOCITIES[4][2];
    if (applyBoundaryConds(accessX, accessY, accessZ, 4)) {
        accessInd = accessX + accessY * N + accessZ * N*N;
        d_destLattice->f4[ind] = d_srcLattice->f4[accessInd];
        flag[4] = false;
    }
    accessX = x-D_LATTICE_VELOCITIES[5][0];
    accessY = y-D_LATTICE_VELOCITIES[5][1];
    accessZ = z-D_LATTICE_VELOCITIES[5][2];
    if (applyBoundaryConds(accessX, accessY, accessZ, 5)) {
        accessInd = accessX + accessY * N + accessZ * N*N;
        d_destLattice->f5[ind] = d_srcLattice->f5[accessInd];
        flag[5] = false;
    }
    accessX = x-D_LATTICE_VELOCITIES[6][0];
    accessY = y-D_LATTICE_VELOCITIES[6][1];
    accessZ = z-D_LATTICE_VELOCITIES[6][2];
    if (applyBoundaryConds(accessX, accessY, accessZ, 6)) {
        accessInd = accessX + accessY * N + accessZ * N*N;
        d_destLattice->f6[ind] = d_srcLattice->f6[accessInd];
        flag[6] = false;
    }
    accessX = x-D_LATTICE_VELOCITIES[7][0];
    accessY = y-D_LATTICE_VELOCITIES[7][1];
    accessZ = z-D_LATTICE_VELOCITIES[7][2];
    if (applyBoundaryConds(accessX, accessY, accessZ, 7)) {
        accessInd = accessX + accessY * N + accessZ * N*N;
        d_destLattice->f7[ind] = d_srcLattice->f7[accessInd];
        flag[7] = false;
    }
    accessX = x-D_LATTICE_VELOCITIES[8][0];
    accessY = y-D_LATTICE_VELOCITIES[8][1];
    accessZ = z-D_LATTICE_VELOCITIES[8][2];
    if (applyBoundaryConds(accessX, accessY, accessZ, 8)) {
        accessInd = accessX + accessY * N + accessZ * N*N;
        d_destLattice->f8[ind] = d_srcLattice->f8[accessInd];
        flag[8] = false;
    }
    accessX = x-D_LATTICE_VELOCITIES[9][0];
    accessY = y-D_LATTICE_VELOCITIES[9][1];
    accessZ = z-D_LATTICE_VELOCITIES[9][2];
    if (applyBoundaryConds(accessX, accessY, accessZ, 9)) {
        accessInd = accessX + accessY * N + accessZ * N*N;
        d_destLattice->f9[ind] = d_srcLattice->f9[accessInd];
        flag[9] = false;
    }
    accessX = x-D_LATTICE_VELOCITIES[10][0];
    accessY = y-D_LATTICE_VELOCITIES[10][1];
    accessZ = z-D_LATTICE_VELOCITIES[10][2];
    if (applyBoundaryConds(accessX, accessY, accessZ, 10)) {
        accessInd = accessX + accessY * N + accessZ * N*N;
        d_destLattice->f10[ind] = d_srcLattice->f10[accessInd];
        flag[10] = false;
    }
    accessX = x-D_LATTICE_VELOCITIES[11][0];
    accessY = y-D_LATTICE_VELOCITIES[11][1];
    accessZ = z-D_LATTICE_VELOCITIES[11][2];
    if (applyBoundaryConds(accessX, accessY, accessZ, 11)) {
        accessInd = accessX + accessY * N + accessZ * N*N;
        d_destLattice->f11[ind] = d_srcLattice->f11[accessInd];
        flag[11] = false;
    }
    accessX = x-D_LATTICE_VELOCITIES[12][0];
    accessY = y-D_LATTICE_VELOCITIES[12][1];
    accessZ = z-D_LATTICE_VELOCITIES[12][2];
    if (applyBoundaryConds(accessX, accessY, accessZ, 12)) {
        accessInd = accessX + accessY * N + accessZ * N*N;
        d_destLattice->f12[ind] = d_srcLattice->f12[accessInd];
        flag[12] = false;
    }
    accessX = x-D_LATTICE_VELOCITIES[13][0];
    accessY = y-D_LATTICE_VELOCITIES[13][1];
    accessZ = z-D_LATTICE_VELOCITIES[13][2];
    if (applyBoundaryConds(accessX, accessY, accessZ, 13)) {
        accessInd = accessX + accessY * N + accessZ * N*N;
        d_destLattice->f13[ind] = d_srcLattice->f13[accessInd];
        flag[13] = false;
    }
    accessX = x-D_LATTICE_VELOCITIES[14][0];
    accessY = y-D_LATTICE_VELOCITIES[14][1];
    accessZ = z-D_LATTICE_VELOCITIES[14][2];
    if (applyBoundaryConds(accessX, accessY, accessZ, 14)) {
        accessInd = accessX + accessY * N + accessZ * N*N;
        d_destLattice->f14[ind] = d_srcLattice->f14[accessInd];
        flag[14] = false;
    }
    accessX = x-D_LATTICE_VELOCITIES[15][0];
    accessY = y-D_LATTICE_VELOCITIES[15][1];
    accessZ = z-D_LATTICE_VELOCITIES[15][2];
    if (applyBoundaryConds(accessX, accessY, accessZ, 15)) {
        accessInd = accessX + accessY * N + accessZ * N*N;
        d_destLattice->f15[ind] = d_srcLattice->f15[accessInd];
        flag[15] = false;
    }
    accessX = x-D_LATTICE_VELOCITIES[16][0];
    accessY = y-D_LATTICE_VELOCITIES[16][1];
    accessZ = z-D_LATTICE_VELOCITIES[16][2];
    if (applyBoundaryConds(accessX, accessY, accessZ, 16)) {
        accessInd = accessX + accessY * N + accessZ * N*N;
        d_destLattice->f16[ind] = d_srcLattice->f16[accessInd];
        flag[16] = false;
    }
    accessX = x-D_LATTICE_VELOCITIES[17][0];
    accessY = y-D_LATTICE_VELOCITIES[17][1];
    accessZ = z-D_LATTICE_VELOCITIES[17][2];
    if (applyBoundaryConds(accessX, accessY, accessZ, 17)) {
        accessInd = accessX + accessY * N + accessZ * N*N;
        d_destLattice->f17[ind] = d_srcLattice->f17[accessInd];
        flag[17] = false;
    }
    accessX = x-D_LATTICE_VELOCITIES[18][0];
    accessY = y-D_LATTICE_VELOCITIES[18][1];
    accessZ = z-D_LATTICE_VELOCITIES[18][2];
    if (applyBoundaryConds(accessX, accessY, accessZ, 18)) {
        accessInd = accessX + accessY * N + accessZ * N*N;
        d_destLattice->f18[ind] = d_srcLattice->f18[accessInd];
        flag[18] = false;
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
    float density = d_destLattice->f0[ind] + d_destLattice->f1[ind] + d_destLattice->f2[ind] + d_destLattice->f3[ind]
                    + d_destLattice->f4[ind] + d_destLattice->f5[ind] + d_destLattice->f6[ind]
                    + d_destLattice->f7[ind] + d_destLattice->f8[ind] + d_destLattice->f9[ind]
                    + d_destLattice->f10[ind] + d_destLattice->f11[ind] + d_destLattice->f12[ind]
                    + d_destLattice->f13[ind] + d_destLattice->f14[ind] + d_destLattice->f15[ind]
                    + d_destLattice->f16[ind] + d_destLattice->f17[ind] + d_destLattice->f18[ind];

    // calc velocity (u)
    float velocity[3];
    for (int i = 0; i < 3; i++) {
        velocity[i] = d_destLattice->f0[ind]*D_LATTICE_VELOCITIES[0][i] + d_destLattice->f1[ind]*D_LATTICE_VELOCITIES[1][i] + d_destLattice->f2[ind]*D_LATTICE_VELOCITIES[2][i]
                    + d_destLattice->f3[ind]*D_LATTICE_VELOCITIES[3][i] + d_destLattice->f4[ind]*D_LATTICE_VELOCITIES[4][i] + d_destLattice->f5[ind]*D_LATTICE_VELOCITIES[5][i]
                    + d_destLattice->f6[ind]*D_LATTICE_VELOCITIES[6][i] + d_destLattice->f7[ind]*D_LATTICE_VELOCITIES[7][i] + d_destLattice->f8[ind]*D_LATTICE_VELOCITIES[8][i]
                    + d_destLattice->f9[ind]*D_LATTICE_VELOCITIES[9][i] + d_destLattice->f10[ind]*D_LATTICE_VELOCITIES[10][i] + d_destLattice->f11[ind]*D_LATTICE_VELOCITIES[11][i]
                    + d_destLattice->f12[ind]*D_LATTICE_VELOCITIES[12][i] + d_destLattice->f13[ind]*D_LATTICE_VELOCITIES[13][i] + d_destLattice->f14[ind]*D_LATTICE_VELOCITIES[14][i]
                    + d_destLattice->f15[ind]*D_LATTICE_VELOCITIES[15][i] + d_destLattice->f16[ind]*D_LATTICE_VELOCITIES[16][i] + d_destLattice->f17[ind]*D_LATTICE_VELOCITIES[17][i]
                    + d_destLattice->f18[ind]*D_LATTICE_VELOCITIES[18][i];
        velocity[i]/=density;
        d_velocities[3*ind + i] = velocity[i]; // save
    }

    // calc the loc equilibrium distro funcs f_qi^eq
    float feq[LBM_Q];
    float t1, t2, t3;
    t3 = velocity[0]*velocity[0] + velocity[1]*velocity[1] + velocity[2]*velocity[2];
    for (int i = 0; i < LBM_Q; i++) {
        t1 = D_LATTICE_VELOCITIES[i][0]*velocity[0] + D_LATTICE_VELOCITIES[i][1]*velocity[1] + D_LATTICE_VELOCITIES[i][2]*velocity[2];
        t2 = t1*t1;
        feq[i] = D_LATTICE_WEIGHTS[i]*density*(1 + (3.0/LBM_C)*t1 + (9.0/(2.0*pow(LBM_C,2)))*t2 - (3.0/(2.0*pow(LBM_C,2)))*t3);
    }

    // calc distro func (f_qi) at new time step + save 19 vals of distro func (f_qi) to curr cell
    d_destLattice->f0[ind] = d_destLattice->f0[ind] - (d_destLattice->f0[ind] - feq[0])/LBM_TAU; // TODO: make loc arr of 19 els + load ptrs so its coalesced reads/writes
    d_destLattice->f1[ind] = d_destLattice->f1[ind] - (d_destLattice->f1[ind] - feq[1])/LBM_TAU;
    d_destLattice->f2[ind] = d_destLattice->f2[ind] - (d_destLattice->f2[ind] - feq[2])/LBM_TAU;
    d_destLattice->f3[ind] = d_destLattice->f3[ind] - (d_destLattice->f3[ind] - feq[3])/LBM_TAU;
    d_destLattice->f4[ind] = d_destLattice->f4[ind] - (d_destLattice->f4[ind] - feq[4])/LBM_TAU;
    d_destLattice->f5[ind] = d_destLattice->f5[ind] - (d_destLattice->f5[ind] - feq[5])/LBM_TAU;
    d_destLattice->f6[ind] = d_destLattice->f6[ind] - (d_destLattice->f6[ind] - feq[6])/LBM_TAU;
    d_destLattice->f7[ind] = d_destLattice->f7[ind] - (d_destLattice->f7[ind] - feq[7])/LBM_TAU;
    d_destLattice->f8[ind] = d_destLattice->f8[ind] - (d_destLattice->f8[ind] - feq[8])/LBM_TAU;
    d_destLattice->f9[ind] = d_destLattice->f9[ind] - (d_destLattice->f9[ind] - feq[9])/LBM_TAU;
    d_destLattice->f10[ind] = d_destLattice->f10[ind] - (d_destLattice->f10[ind] - feq[10])/LBM_TAU;
    d_destLattice->f11[ind] = d_destLattice->f11[ind] - (d_destLattice->f11[ind] - feq[11])/LBM_TAU;
    d_destLattice->f12[ind] = d_destLattice->f12[ind] - (d_destLattice->f12[ind] - feq[12])/LBM_TAU;
    d_destLattice->f13[ind] = d_destLattice->f13[ind] - (d_destLattice->f13[ind] - feq[13])/LBM_TAU;
    d_destLattice->f14[ind] = d_destLattice->f14[ind] - (d_destLattice->f14[ind] - feq[14])/LBM_TAU;
    d_destLattice->f15[ind] = d_destLattice->f15[ind] - (d_destLattice->f15[ind] - feq[15])/LBM_TAU;
    d_destLattice->f16[ind] = d_destLattice->f16[ind] - (d_destLattice->f16[ind] - feq[16])/LBM_TAU;
    d_destLattice->f17[ind] = d_destLattice->f17[ind] - (d_destLattice->f17[ind] - feq[17])/LBM_TAU;
    d_destLattice->f18[ind] = d_destLattice->f18[ind] - (d_destLattice->f18[ind] - feq[18])/LBM_TAU;
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
__global__ void snowApplyGrav(float *d_snowflakeDataFlat, unsigned *d_numParticles, float *d_snowOffsets, curandState* d_globalState, float* d_velocities) {
    unsigned kernelInd = blockIdx.x*SNOW_GRAV_BLOCK_SIZE*SNOW_GRAV_BATCH_SIZE + threadIdx.x;
    unsigned stride = SNOW_GRAV_BLOCK_SIZE;

    // apply forces to snowflakes
    float xOffset, yOffset, zOffset;
    float yOffsetRand;
    curandState localState;
    float x_phys, y_phys, z_phys, x_lat, y_lat, z_lat;
    float x_phys_offset = min(d_extents[0], d_extents[1]), y_phys_offset = min(d_extents[2], d_extents[3]), z_phys_offset = min(d_extents[4], d_extents[5]);
    float dx = 0, dy = 0, dz = 0;
    float u_x0[3], u_x1[3], u_x2[3], u_x3[3], u_x4[3], u_x5[3], u_x6[3], u_x7[3];
    float ux, uy, uz;
    int x_lat_int, y_lat_int, z_lat_int;
    int x0[3], x1[3], x2[3], x3[3], x4[3], x5[3], x6[3], x7[3];
    int ind;
    for (int x = kernelInd; x < min(kernelInd + SNOW_GRAV_BATCH_SIZE*stride, *d_numParticles); x+=stride) {
        // translate x y z into a lattice point
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
            for (int i = 0; i < 3; i++) {
                x0[i] = x0[i] < N ? x0[i] : N-1; // TODO: good strat?
                x1[i] = x1[i] < N ? x1[i] : N-1;
                x2[i] = x2[i] < N ? x2[i] : N-1;
                x3[i] = x3[i] < N ? x3[i] : N-1;
                x4[i] = x4[i] < N ? x4[i] : N-1;
                x5[i] = x5[i] < N ? x5[i] : N-1;
                x6[i] = x6[i] < N ? x6[i] : N-1;
                x7[i] = x7[i] < N ? x7[i] : N-1;
            }
            dx = x_lat-x0[0];
            dy = y_lat-x0[1];
            dz = z_lat-x0[2];
            ind = x0[0] + x0[1] * N + x0[2] * N*N;
            for (int i = 0; i < 3; i++) {
                u_x0[i] = d_velocities[3*ind + i];
            }
            ind = x1[0] + x1[1] * N + x1[2] * N*N;
            for (int i = 0; i < 3; i++) {
                u_x1[i] = d_velocities[3*ind + i];
            }
            ind = x2[0] + x2[1] * N + x2[2] * N*N;
            for (int i = 0; i < 3; i++) {
                u_x2[i] = d_velocities[3*ind + i];
            }
            ind = x3[0] + x3[1] * N + x3[2] * N*N;
            for (int i = 0; i < 3; i++) {
                u_x3[i] = d_velocities[3*ind + i];
            }
            ind = x4[0] + x4[1] * N + x4[2] * N*N;
            for (int i = 0; i < 3; i++) {
                u_x4[i] = d_velocities[3*ind + i];
            }
            ind = x5[0] + x5[1] * N + x5[2] * N*N;
            for (int i = 0; i < 3; i++) {
                u_x5[i] = d_velocities[3*ind + i];
            }
            ind = x6[0] + x6[1] * N + x6[2] * N*N;
            for (int i = 0; i < 3; i++) {
                u_x6[i] = d_velocities[3*ind + i];
            }
            ind = x7[0] + x7[1] * N + x7[2] * N*N;
            for (int i = 0; i < 3; i++) {
                u_x7[i] = d_velocities[3*ind + i];
            }
            ux = (1-dx)*(1-dy)*(1-dz)*u_x0[0]
                + (1-dx)*(1-dy)*dz*u_x1[0]
                + (1-dx)*dy*(1-dz)*u_x2[0]
                + dx*(1-dy)*(1-dz)*u_x3[0]
                + (1-dx)*dy*dz*u_x4[0]
                + dx*(1-dy)*dz*u_x5[0]
                + dx*dy*(1-dz)*u_x6[0]
                + dx*dy*dz*u_x7[0];
            ux*=d_delta_x_phys;
            uy = (1-dx)*(1-dy)*(1-dz)*u_x0[1]
                                       + (1-dx)*(1-dy)*dz*u_x1[1]
                                       + (1-dx)*dy*(1-dz)*u_x2[1]
                                       + dx*(1-dy)*(1-dz)*u_x3[1]
                                       + (1-dx)*dy*dz*u_x4[1]
                                       + dx*(1-dy)*dz*u_x5[1]
                                       + dx*dy*(1-dz)*u_x6[1]
                                       + dx*dy*dz*u_x7[1];
            uy*=d_delta_x_phys;
            uz = (1-dx)*(1-dy)*(1-dz)*u_x0[2]
                            + (1-dx)*(1-dy)*dz*u_x1[2]
                            + (1-dx)*dy*(1-dz)*u_x2[2]
                            + dx*(1-dy)*(1-dz)*u_x3[2]
                            + (1-dx)*dy*dz*u_x4[2]
                            + dx*(1-dy)*dz*u_x5[2]
                            + dx*dy*(1-dz)*u_x6[2]
                            + dx*dy*dz*u_x7[2];
            uz*=d_delta_x_phys;
            printf("vel %f %f %f\n\n", ux, uy, uz);
        } else {
            ind = x_lat_int + y_lat_int * N + z_lat_int * N*N;
            ux = d_velocities[3*ind]*d_delta_x_phys;
            uy = d_velocities[3*ind + 1]*d_delta_x_phys;
            uz = d_velocities[3*ind + 2]*d_delta_x_phys;
        }

        // update x
        d_snowOffsets[5*x] = ux;
        d_snowflakeDataFlat[3*x] += ux;

        // update y
        d_snowOffsets[5*x+1] = uy;
        d_snowflakeDataFlat[3*x+1] += uy;

        // update z
        d_snowOffsets[5*x+2] = uz;
        d_snowflakeDataFlat[3*x+2] += uz;

        if (d_snowflakeDataFlat[3*x+1] < d_extents[2]) {
            localState = d_globalState[kernelInd];
            xOffset = getRandFloatGPU(d_extents[0], d_extents[1], &localState) - d_snowflakeDataFlat[3*x];
            yOffsetRand = getRandFloatGPU(-(SNOW_NOISE_Y*(d_extents[3] - d_extents[2])), SNOW_NOISE_Y*(d_extents[3] - d_extents[2]), &localState);
            zOffset = getRandFloatGPU(d_extents[4], d_extents[5], &localState) - d_snowflakeDataFlat[3*x+2];
            yOffset = (d_extents[3] - d_extents[2]) + yOffsetRand;
            d_globalState[kernelInd] = localState;
        } else {
            yOffset = -GRAVITY;
            xOffset = 0;
            zOffset = 0;
        }

        // update x
         d_snowOffsets[5*x] += xOffset;
         d_snowflakeDataFlat[3*x] += xOffset;

        // update y
        d_snowOffsets[5*x+1] += yOffset;
        d_snowflakeDataFlat[3*x+1] += yOffset;

        // update z
         d_snowOffsets[5*x+2] += zOffset;
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
    float feq[LBM_Q];
    float t1, t2, t3;
    t3 = 2*d_windVel*d_windVel;
    for (int i = 0; i < LBM_Q; i++) {
        t1 = D_LATTICE_VELOCITIES[i][0]*d_windVel + D_LATTICE_VELOCITIES[i][1]*0 + D_LATTICE_VELOCITIES[i][2]*d_windVel;
        t2 = t1*t1;
        feq[i] = D_LATTICE_WEIGHTS[i]*(1 + (3.0/LBM_C)*t1 + (9.0/(2.0*pow(LBM_C,2)))*t2 - (3.0/(2.0*pow(LBM_C,2)))*t3);
    }
    for (int x = 0; x < N*N*N; x++) { // TODO: turn into kernel = max block size + however many blocks needed
        int ind = x;
        d_srcLattice->f0[ind] = feq[0];
        d_srcLattice->f1[ind] = feq[1];
        d_srcLattice->f2[ind] = feq[2];
        d_srcLattice->f3[ind] = feq[3];
        d_srcLattice->f4[ind] = feq[4];
        d_srcLattice->f5[ind] = feq[5];
        d_srcLattice->f6[ind] = feq[6];
        d_srcLattice->f7[ind] = feq[7];
        d_srcLattice->f8[ind] = feq[8];
        d_srcLattice->f9[ind] = feq[9];
        d_srcLattice->f10[ind] = feq[10];
        d_srcLattice->f11[ind] = feq[11];
        d_srcLattice->f12[ind] = feq[12];
        d_srcLattice->f13[ind] = feq[13];
        d_srcLattice->f14[ind] = feq[14];
        d_srcLattice->f15[ind] = feq[15];
        d_srcLattice->f16[ind] = feq[16];
        d_srcLattice->f17[ind] = feq[17];
        d_srcLattice->f18[ind] = feq[18];
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
    h_lbmBlockSize = dim3(N, 1, 1);
    h_lbmGridSize = dim3(N, N, 1);

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
    cudaMallocHost((void**)&h_srcLattice, sizeof(Lattice)); // TODO: del
    cudaMallocHost((void**)&h_destLattice, sizeof(Lattice)); // TODO: del
    cudaMalloc((void**)&d_srcLattice, sizeof(Lattice));
    cudaMalloc((void**)&d_destLattice, sizeof(Lattice));
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

    // TODO: del
    cudaMemcpy(&h_destLattice, d_destLattice, sizeof(Lattice), cudaMemcpyDeviceToHost);
    int H_LATTICE_VELOCITIES[19][3] = { {0,0,0},
    {1,0,0}, {-1,0,0}, {0,1,0}, {0,-1,0}, {0,0,1}, {0,0,-1},
    {1,1,0}, {1,-1,0}, {-1,1,0}, {-1,-1,0}, {1,0,1}, {1,0,-1}, {-1,0,1}, {-1,0,-1}, {0,1,1}, {0,1,-1},  {0,-1,1},  {0,-1,-1}};;
    for (int x = 0; x < N*N*N; x++) {
        float density = h_destLattice.f0[x] + h_destLattice.f1[x] + h_destLattice.f2[x] + h_destLattice.f3[x]
                    + h_destLattice.f4[x] + h_destLattice.f5[x] + h_destLattice.f6[x]
                    + h_destLattice.f7[x] + h_destLattice.f8[x] + h_destLattice.f9[x]
                    + h_destLattice.f10[x] + h_destLattice.f11[x] + h_destLattice.f12[x]
                    + h_destLattice.f13[x] + h_destLattice.f14[x] + h_destLattice.f15[x]
                    + h_destLattice.f16[x] + h_destLattice.f17[x] + h_destLattice.f18[x];
        float velocity[3];
        for (int i = 0; i < 3; i++) {
            velocity[i] = h_destLattice.f0[x]*H_LATTICE_VELOCITIES[0][i] + h_destLattice.f1[x]*H_LATTICE_VELOCITIES[1][i] + h_destLattice.f2[x]*H_LATTICE_VELOCITIES[2][i]
                        + h_destLattice.f3[x]*H_LATTICE_VELOCITIES[3][i] + h_destLattice.f4[x]*H_LATTICE_VELOCITIES[4][i] + h_destLattice.f5[x]*H_LATTICE_VELOCITIES[5][i]
                        + h_destLattice.f6[x]*H_LATTICE_VELOCITIES[6][i] + h_destLattice.f7[x]*H_LATTICE_VELOCITIES[7][i] + h_destLattice.f8[x]*H_LATTICE_VELOCITIES[8][i]
                        + h_destLattice.f9[x]*H_LATTICE_VELOCITIES[9][i] + h_destLattice.f10[x]*H_LATTICE_VELOCITIES[10][i] + h_destLattice.f11[x]*H_LATTICE_VELOCITIES[11][i]
                        + h_destLattice.f12[x]*H_LATTICE_VELOCITIES[12][i] + h_destLattice.f13[x]*H_LATTICE_VELOCITIES[13][i] + h_destLattice.f14[x]*H_LATTICE_VELOCITIES[14][i]
                        + h_destLattice.f15[x]*H_LATTICE_VELOCITIES[15][i] + h_destLattice.f16[x]*H_LATTICE_VELOCITIES[16][i] + h_destLattice.f17[x]*H_LATTICE_VELOCITIES[17][i]
                        + h_destLattice.f18[x]*H_LATTICE_VELOCITIES[18][i];
            velocity[i]/=density;
        }
        printf("(%f,%f,%f) ", velocity[0], velocity[1], velocity[2]);
    }
    printf("\n\n\n\n");

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