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
#define LBM_TAU 0.55 // 3 * viscosity - 1/2 // TODO: rem
#define LBM_M_MAX 0.1
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
#define Nx 5 // TODO: make adjustable
#define Ny 5
#define Nz 5
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
__constant__ __device__ float d_extent[6];
__constant__ __device__ float d_tau;
float *d_verts;
float *d_snowflakeDataFlat;
unsigned *d_numParticles;
float *d_snowOffsets;
curandState *d_globalState;
Lattice *d_srcLattice;
Lattice *d_destLattice;

// TODO
__device__ float applyBoundaryConds(float accessInd, unsigned dim) {
    if (dim == 0) {
        if (accessInd < 0) {
            return Nx - 1;
        }
        if (accessInd >= Nx) {
            return 0;
        }
    }
    if (dim == 1) {
        if (accessInd < 0) {
            return Ny - 1;
        }
        if (accessInd >= Ny) {
            return 0;
        }
    }
    if (dim == 2) {
        if (accessInd < 0) {
            return Nz - 1;
        }
        if (accessInd >= Nz) {
            return 0;
        }
    }

    return accessInd;
}

// TODO
__global__ void lbmKernel(Lattice *d_srcLattice, Lattice *d_destLattice) {
    // compute the 3D position of the thread
    int x = threadIdx.x;
    int y = blockIdx.x;
    int z = blockIdx.y;
    // compute the corresponding 1D index
    int ind = x + y * Nx + z * Nx*Ny;


    if (!(ind < Nx*Ny*Nz)) { // TODO: refactor
        return;
    }

    float f0 = d_destLattice->f0[ind];
    float f1 = d_destLattice->f1[ind];
    float f2 = d_destLattice->f2[ind];
    float f3 = d_destLattice->f3[ind];
    float f4 = d_destLattice->f4[ind];
    float f5 = d_destLattice->f5[ind];
    float f6 = d_destLattice->f6[ind];
    float f7 = d_destLattice->f7[ind];
    float f8 = d_destLattice->f8[ind];
    float f9 = d_destLattice->f9[ind];
    float f10 = d_destLattice->f10[ind];
    float f11 = d_destLattice->f11[ind];
    float f12 = d_destLattice->f12[ind];
    float f13 = d_destLattice->f13[ind];
    float f14 = d_destLattice->f14[ind];
    float f15 = d_destLattice->f15[ind];
    float f16 = d_destLattice->f16[ind];
    float f17 = d_destLattice->f17[ind];
    float f18 = d_destLattice->f18[ind];
    if (ind == 5) {
        //printf("+ %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f\n", f0, f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13, f14, f15, f16, f17, f18);
    }


    // stream 19 pdfs from adjacent cells to curr cell + apply boundary conds
    int accessX, accessY, accessZ, accessInd;
    accessX = x-D_LATTICE_VELOCITIES[0][0];
    accessY = y-D_LATTICE_VELOCITIES[0][1];
    accessZ = z-D_LATTICE_VELOCITIES[0][2];
    accessX = applyBoundaryConds(accessX, 0);
    accessY = applyBoundaryConds(accessY, 1);
    accessZ = applyBoundaryConds(accessZ, 2);
    accessInd = accessX + accessY * Nx + accessZ * Nx*Ny;
    d_destLattice->f0[ind] = d_srcLattice->f0[accessInd];
    accessX = x-D_LATTICE_VELOCITIES[1][0];
    accessY = y-D_LATTICE_VELOCITIES[1][1];
    accessZ = z-D_LATTICE_VELOCITIES[1][2];
    accessX = applyBoundaryConds(accessX, 0);
    accessY = applyBoundaryConds(accessY, 1);
    accessZ = applyBoundaryConds(accessZ, 2);
    accessInd = accessX + accessY * Nx + accessZ * Nx*Ny;
    int test = accessInd; // TODO: rem
    d_destLattice->f1[ind] = d_srcLattice->f1[accessInd];
    accessX = x-D_LATTICE_VELOCITIES[2][0];
    accessY = y-D_LATTICE_VELOCITIES[2][1];
    accessZ = z-D_LATTICE_VELOCITIES[2][2];
    accessX = applyBoundaryConds(accessX, 0);
    accessY = applyBoundaryConds(accessY, 1);
    accessZ = applyBoundaryConds(accessZ, 2);
    accessInd = accessX + accessY * Nx + accessZ * Nx*Ny;
    d_destLattice->f2[ind] = d_srcLattice->f2[accessInd];
    accessX = x-D_LATTICE_VELOCITIES[3][0];
    accessY = y-D_LATTICE_VELOCITIES[3][1];
    accessZ = z-D_LATTICE_VELOCITIES[3][2];
    accessX = applyBoundaryConds(accessX, 0);
    accessY = applyBoundaryConds(accessY, 1);
    accessZ = applyBoundaryConds(accessZ, 2);
    accessInd = accessX + accessY * Nx + accessZ * Nx*Ny;
    d_destLattice->f3[ind] = d_srcLattice->f3[accessInd];
    accessX = x-D_LATTICE_VELOCITIES[4][0];
    accessY = y-D_LATTICE_VELOCITIES[4][1];
    accessZ = z-D_LATTICE_VELOCITIES[4][2];
    accessX = applyBoundaryConds(accessX, 0);
    accessY = applyBoundaryConds(accessY, 1);
    accessZ = applyBoundaryConds(accessZ, 2);
    accessInd = accessX + accessY * Nx + accessZ * Nx*Ny;
    d_destLattice->f4[ind] = d_srcLattice->f4[accessInd];
    accessX = x-D_LATTICE_VELOCITIES[5][0];
    accessY = y-D_LATTICE_VELOCITIES[5][1];
    accessZ = z-D_LATTICE_VELOCITIES[5][2];
    accessX = applyBoundaryConds(accessX, 0);
    accessY = applyBoundaryConds(accessY, 1);
    accessZ = applyBoundaryConds(accessZ, 2);
    accessInd = accessX + accessY * Nx + accessZ * Nx*Ny;
    d_destLattice->f5[ind] = d_srcLattice->f5[accessInd];
    accessX = x-D_LATTICE_VELOCITIES[6][0];
    accessY = y-D_LATTICE_VELOCITIES[6][1];
    accessZ = z-D_LATTICE_VELOCITIES[6][2];
    accessX = applyBoundaryConds(accessX, 0);
    accessY = applyBoundaryConds(accessY, 1);
    accessZ = applyBoundaryConds(accessZ, 2);
    accessInd = accessX + accessY * Nx + accessZ * Nx*Ny;
    d_destLattice->f6[ind] = d_srcLattice->f6[accessInd];
    accessX = x-D_LATTICE_VELOCITIES[7][0];
    accessY = y-D_LATTICE_VELOCITIES[7][1];
    accessZ = z-D_LATTICE_VELOCITIES[7][2];
    accessX = applyBoundaryConds(accessX, 0);
    accessY = applyBoundaryConds(accessY, 1);
    accessZ = applyBoundaryConds(accessZ, 2);
    accessInd = accessX + accessY * Nx + accessZ * Nx*Ny;
    d_destLattice->f7[ind] = d_srcLattice->f7[accessInd];
    accessX = x-D_LATTICE_VELOCITIES[8][0];
    accessY = y-D_LATTICE_VELOCITIES[8][1];
    accessZ = z-D_LATTICE_VELOCITIES[8][2];
    accessX = applyBoundaryConds(accessX, 0);
    accessY = applyBoundaryConds(accessY, 1);
    accessZ = applyBoundaryConds(accessZ, 2);
    accessInd = accessX + accessY * Nx + accessZ * Nx*Ny;
    d_destLattice->f8[ind] = d_srcLattice->f8[accessInd];
    accessX = x-D_LATTICE_VELOCITIES[9][0];
    accessY = y-D_LATTICE_VELOCITIES[9][1];
    accessZ = z-D_LATTICE_VELOCITIES[9][2];
    accessX = applyBoundaryConds(accessX, 0);
    accessY = applyBoundaryConds(accessY, 1);
    accessZ = applyBoundaryConds(accessZ, 2);
    accessInd = accessX + accessY * Nx + accessZ * Nx*Ny;
    d_destLattice->f9[ind] = d_srcLattice->f9[accessInd];
    accessX = x-D_LATTICE_VELOCITIES[10][0];
    accessY = y-D_LATTICE_VELOCITIES[10][1];
    accessZ = z-D_LATTICE_VELOCITIES[10][2];
    accessX = applyBoundaryConds(accessX, 0);
    accessY = applyBoundaryConds(accessY, 1);
    accessZ = applyBoundaryConds(accessZ, 2);
    accessInd = accessX + accessY * Nx + accessZ * Nx*Ny;
    d_destLattice->f10[ind] = d_srcLattice->f10[accessInd];
    accessX = x-D_LATTICE_VELOCITIES[11][0];
    accessY = y-D_LATTICE_VELOCITIES[11][1];
    accessZ = z-D_LATTICE_VELOCITIES[11][2];
    accessX = applyBoundaryConds(accessX, 0);
    accessY = applyBoundaryConds(accessY, 1);
    accessZ = applyBoundaryConds(accessZ, 2);
    accessInd = accessX + accessY * Nx + accessZ * Nx*Ny;
    d_destLattice->f11[ind] = d_srcLattice->f11[accessInd];
    accessX = x-D_LATTICE_VELOCITIES[12][0];
    accessY = y-D_LATTICE_VELOCITIES[12][1];
    accessZ = z-D_LATTICE_VELOCITIES[12][2];
    accessX = applyBoundaryConds(accessX, 0);
    accessY = applyBoundaryConds(accessY, 1);
    accessZ = applyBoundaryConds(accessZ, 2);
    accessInd = accessX + accessY * Nx + accessZ * Nx*Ny;
    d_destLattice->f12[ind] = d_srcLattice->f12[accessInd];
    accessX = x-D_LATTICE_VELOCITIES[13][0];
    accessY = y-D_LATTICE_VELOCITIES[13][1];
    accessZ = z-D_LATTICE_VELOCITIES[13][2];
    accessX = applyBoundaryConds(accessX, 0);
    accessY = applyBoundaryConds(accessY, 1);
    accessZ = applyBoundaryConds(accessZ, 2);
    accessInd = accessX + accessY * Nx + accessZ * Nx*Ny;
    d_destLattice->f13[ind] = d_srcLattice->f13[accessInd];
    accessX = x-D_LATTICE_VELOCITIES[14][0];
    accessY = y-D_LATTICE_VELOCITIES[14][1];
    accessZ = z-D_LATTICE_VELOCITIES[14][2];
    accessX = applyBoundaryConds(accessX, 0);
    accessY = applyBoundaryConds(accessY, 1);
    accessZ = applyBoundaryConds(accessZ, 2);
    accessInd = accessX + accessY * Nx + accessZ * Nx*Ny;
    d_destLattice->f14[ind] = d_srcLattice->f14[accessInd];
    accessX = x-D_LATTICE_VELOCITIES[15][0];
    accessY = y-D_LATTICE_VELOCITIES[15][1];
    accessZ = z-D_LATTICE_VELOCITIES[15][2];
    accessX = applyBoundaryConds(accessX, 0);
    accessY = applyBoundaryConds(accessY, 1);
    accessZ = applyBoundaryConds(accessZ, 2);
    accessInd = accessX + accessY * Nx + accessZ * Nx*Ny;
    d_destLattice->f15[ind] = d_srcLattice->f15[accessInd];
    accessX = x-D_LATTICE_VELOCITIES[16][0];
    accessY = y-D_LATTICE_VELOCITIES[16][1];
    accessZ = z-D_LATTICE_VELOCITIES[16][2];
    accessX = applyBoundaryConds(accessX, 0);
    accessY = applyBoundaryConds(accessY, 1);
    accessZ = applyBoundaryConds(accessZ, 2);
    accessInd = accessX + accessY * Nx + accessZ * Nx*Ny;
    d_destLattice->f16[ind] = d_srcLattice->f16[accessInd];
    accessX = x-D_LATTICE_VELOCITIES[17][0];
    accessY = y-D_LATTICE_VELOCITIES[17][1];
    accessZ = z-D_LATTICE_VELOCITIES[17][2];
    accessX = applyBoundaryConds(accessX, 0);
    accessY = applyBoundaryConds(accessY, 1);
    accessZ = applyBoundaryConds(accessZ, 2);
    accessInd = accessX + accessY * Nx + accessZ * Nx*Ny;
    d_destLattice->f17[ind] = d_srcLattice->f17[accessInd];
    accessX = x-D_LATTICE_VELOCITIES[18][0];
    accessY = y-D_LATTICE_VELOCITIES[18][1];
    accessZ = z-D_LATTICE_VELOCITIES[18][2];
    accessX = applyBoundaryConds(accessX, 0);
    accessY = applyBoundaryConds(accessY, 1);
    accessZ = applyBoundaryConds(accessZ, 2);
    accessInd = accessX + accessY * Nx + accessZ * Nx*Ny;
    d_destLattice->f18[ind] = d_srcLattice->f18[accessInd];
    __syncthreads();
    if (ind ==120){
        printf("0\n");
        printf("test %f\n", d_srcLattice->f1[test]);
        for (int x = 0; x < 25; x+=1) {
            for (int y = 0; y < 5; y++) {
                f1 = d_destLattice->f1[5*x + y];
                printf("%f ", f1);
            }
            printf("\n");
        }
        printf("\n\n\n\n");
    }
    // calc density (rho)
    float density = d_destLattice->f0[ind] + d_destLattice->f1[ind] + d_destLattice->f2[ind] + d_destLattice->f3[ind]
                    + d_destLattice->f4[ind] + d_destLattice->f5[ind] + d_destLattice->f6[ind]
                    + d_destLattice->f7[ind] + d_destLattice->f8[ind] + d_destLattice->f9[ind]
                    + d_destLattice->f10[ind] + d_destLattice->f11[ind] + d_destLattice->f12[ind]
                    + d_destLattice->f13[ind] + d_destLattice->f14[ind] + d_destLattice->f15[ind]
                    + d_destLattice->f16[ind] + d_destLattice->f17[ind] + d_destLattice->f18[ind]; // TODO: should be preinitialized?

    // calc velocity (u)
    float velocity[3]; // TODO: should be preinitialized?
    for (int i = 0; i < 3; i++) {
        velocity[i] = d_destLattice->f0[ind]*D_LATTICE_VELOCITIES[0][i] + d_destLattice->f1[ind]*D_LATTICE_VELOCITIES[1][i] + d_destLattice->f2[ind]*D_LATTICE_VELOCITIES[2][i]
                    + d_destLattice->f3[ind]*D_LATTICE_VELOCITIES[3][i] + d_destLattice->f4[ind]*D_LATTICE_VELOCITIES[4][i] + d_destLattice->f5[ind]*D_LATTICE_VELOCITIES[5][i]
                    + d_destLattice->f6[ind]*D_LATTICE_VELOCITIES[6][i] + d_destLattice->f7[ind]*D_LATTICE_VELOCITIES[7][i] + d_destLattice->f8[ind]*D_LATTICE_VELOCITIES[8][i]
                    + d_destLattice->f9[ind]*D_LATTICE_VELOCITIES[9][i] + d_destLattice->f10[ind]*D_LATTICE_VELOCITIES[10][i] + d_destLattice->f11[ind]*D_LATTICE_VELOCITIES[11][i]
                    + d_destLattice->f12[ind]*D_LATTICE_VELOCITIES[12][i] + d_destLattice->f13[ind]*D_LATTICE_VELOCITIES[13][i] + d_destLattice->f14[ind]*D_LATTICE_VELOCITIES[14][i]
                    + d_destLattice->f15[ind]*D_LATTICE_VELOCITIES[15][i] + d_destLattice->f16[ind]*D_LATTICE_VELOCITIES[16][i] + d_destLattice->f5[ind]*D_LATTICE_VELOCITIES[17][i]
                    + d_destLattice->f18[ind]*D_LATTICE_VELOCITIES[18][i];
        velocity[i]/=density;
    }

    // calc the loc equilibrium distro funcs f_qi^eq
    float feq[LBM_Q]; // TODO: should be preinitialized?
    float t1, t2, t3; // TODO: should be preinitialized?
    t3 = velocity[0]*velocity[0] + velocity[1]*velocity[1] + velocity[2]*velocity[2];
    for (int i = 0; i < LBM_Q; i++) {
        t1 = D_LATTICE_VELOCITIES[i][0]*velocity[0] + D_LATTICE_VELOCITIES[i][1]*velocity[1] + D_LATTICE_VELOCITIES[i][2]*velocity[2];
        t2 = t1*t1;
        feq[i] = D_LATTICE_WEIGHTS[i]*density*(1 + (3.0/LBM_C)*t1 + (9.0/(2.0*pow(LBM_C,2)))*t2 - (3.0/(2.0*pow(LBM_C,2)))*t3);
    }

    f0 = d_destLattice->f0[ind];
     f1 = d_destLattice->f1[ind];
     f2 = d_destLattice->f2[ind];
     f3 = d_destLattice->f3[ind];
     f4 = d_destLattice->f4[ind];
     f5 = d_destLattice->f5[ind];
     f6 = d_destLattice->f6[ind];
     f7 = d_destLattice->f7[ind];
     f8 = d_destLattice->f8[ind];
     f9 = d_destLattice->f9[ind];
     f10 = d_destLattice->f10[ind];
     f11 = d_destLattice->f11[ind];
     f12 = d_destLattice->f12[ind];
     f13 = d_destLattice->f13[ind];
     f14 = d_destLattice->f14[ind];
     f15 = d_destLattice->f15[ind];
     f16 = d_destLattice->f16[ind];
     f17 = d_destLattice->f17[ind];
     f18 = d_destLattice->f18[ind];
    if (ind == 120) {
        printf("ruh new %f old %f minus %f feq %f\n\n\n", d_destLattice->f1[ind] - (d_destLattice->f1[ind] - feq[1])/LBM_TAU, d_destLattice->f1[ind], (d_destLattice->f1[ind] - feq[1])/LBM_TAU, feq[1]); // -0.02487272727
        //printf("%f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f\n\n", f0, f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13, f14, f15, f16, f17, f18);
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
     f0 = d_destLattice->f0[ind];
     f1 = d_destLattice->f1[ind];
     f2 = d_destLattice->f2[ind];
     f3 = d_destLattice->f3[ind];
     f4 = d_destLattice->f4[ind];
     f5 = d_destLattice->f5[ind];
     f6 = d_destLattice->f6[ind];
     f7 = d_destLattice->f7[ind];
     f8 = d_destLattice->f8[ind];
     f9 = d_destLattice->f9[ind];
     f10 = d_destLattice->f10[ind];
     f11 = d_destLattice->f11[ind];
     f12 = d_destLattice->f12[ind];
     f13 = d_destLattice->f13[ind];
     f14 = d_destLattice->f14[ind];
     f15 = d_destLattice->f15[ind];
     f16 = d_destLattice->f16[ind];
     f17 = d_destLattice->f17[ind];
     f18 = d_destLattice->f18[ind];
      if (ind == 120) {
         // printf(">>>>>>>>>>>>>>>>%d %f\n\n", ind, (d_destLattice->f0[ind] - (d_destLattice->f0[ind] - feq[0])/LBM_TAU));
         // printf("> %d %f\n\n", ind, f0);
         //printf("> %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f\n\n", f0, f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13, f14, f15, f16, f17, f18);
        }

    if (ind == 120) {
        printf("1\n");
        for (int x = 0; x < 25; x+=1) {
            for (int y = 0; y < 5; y++) {
                f1 = d_destLattice->f1[5*x + y];
                printf("%f ", f1);
            }
            printf("\n");
        }
        printf("\n\n\n\n");
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

    // apply forces to snowflakes
    float xOffset, yOffset, zOffset;
    float yOffsetRand;
    curandState localState;
    float x_phys, y_phys, z_phys, x_lat, y_lat, z_lat;
    float dx, dy, dz;
    float u, u_x0, u_x1, u_x2, u_x3, u_x4, u_x5, u_x6, u_x7;
    int x_lat_int, y_lat_int, z_lat_int;
    int x0[3], x1[3], x2[3], x3[3], x4[3], x5[3], x6[3], x7[3];
    for (int x = kernelInd; x < min(kernelInd + SNOW_GRAV_BATCH_SIZE*stride, *d_numParticles); x+=stride) {
        // translate x y z into a lattice point
        x_phys = d_snowflakeDataFlat[3*x];
        y_phys = d_snowflakeDataFlat[3*x+1];
        z_phys = d_snowflakeDataFlat[3*x+2];
        x_lat = x_phys; // TODO
        y_lat = y_phys; // TODO
        z_lat = z_phys; // TODO
        x_lat_int = (int) x_lat;
        y_lat_int = (int) y_lat;
        z_lat_int = (int) z_lat;
        if (x_lat != x_lat_int || y_lat != y_lat_int || z_lat != z_lat_int) { // interpolation
            x_lat_int = (int) x_lat;
            y_lat_int = (int) y_lat;
            z_lat_int = (int) z_lat;
            x0[0] = x_lat, x0[1] = y_lat, x0[2] = z_lat; // TODO: closest or furthest?
            x1[0] = x_lat, x1[1] = y_lat, x1[2] = z_lat; // TODO
            x2[0] = x_lat, x2[1] = y_lat, x2[2] = z_lat; // TODO
            x3[0] = x_lat, x3[1] = y_lat, x3[2] = z_lat; // TODO
            x4[0] = x_lat, x4[1] = y_lat, x4[2] = z_lat; // TODO
            x5[0] = x_lat, x5[1] = y_lat, x5[2] = z_lat; // TODO
            x6[0] = x_lat, x6[1] = y_lat, x6[2] = z_lat; // TODO
            x7[0] = x_lat, x7[1] = y_lat, x7[2] = z_lat; // TODO
            dx = x_lat-x0[0];
            dy = y_lat-x0[1];
            dz = z_lat-x0[2];
            u_x0 = 0; // TODO
            u_x1 = 0; // TODO
            u_x2 = 0; // TODO
            u_x3 = 0; // TODO
            u_x4 = 0; // TODO
            u_x5 = 0; // TODO
            u_x6 = 0; // TODO
            u_x7 = 0; // TODO
            u = (1-dx)*(1-dy)*(1-dz)*u_x0
                + (1-dx)*(1-dy)*dz*u_x1
                + (1-dx)*dy*(1-dz)*u_x2
                + dx*(1-dy)*(1-dz)*u_x3
                + (1-dx)*dy*dz*u_x4
                + dx*(1-dy)*dz*u_x5
                + dx*dy*(1-dz)*u_x6
                + dx*dy*dz*u_x7;
        }
        // TODO

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

// TODO
bool isModelValid(float h_extent[3][2], float h_windVel, float h_latticeRes, float h_temp) {
    float h_viscosity_phys = 0.000012890; // m^2/s from temp (-5) (https://theengineeringmindset.com/properties-of-air-at-atmospheric-pressure/)
    float h_cs_phys =  328.25; // m/s from temp (-5) (https://en.wikipedia.org/wiki/Speed_of_sound)
    float h_delta_x_phys = abs(h_extent[0][1]-h_extent[0][0])/h_latticeRes;
    float h_delta_t_phys = (LBM_C_S/h_cs_phys)*h_delta_x_phys;
    float h_tau = 3*(h_viscosity_phys*(h_delta_t_phys/pow(h_delta_x_phys, 2)))+1.0/2;
    printf("%f %f %f\n", h_delta_x_phys, h_delta_t_phys, h_tau);
    return false;
}

/**
 * Initialize data on the GPU.
 * @param data - SnowGeneratorData object with particle data.
 * @param numParticles - Number of particles.
 * @param extent - Extent of volume in which to generate the particles, where extent[0] is a pair for the x extent,
 * extent[1] is a pair for the y extent, and extent[2] is a pair for the z extent. If numParticles = 1 this
 * parameter has no effect and the snow particle is generated at the origin.
 */
extern void snowInitGPU(SnowGeneratorData data, unsigned numParticles, float extent[3][2], float windVel, unsigned latticeRes, float temp) {
    // validate model
    if (!isModelValid(extent, windVel, latticeRes, temp)) { // TODO
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
    // TODO: free lattices

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
    lbmKernel<<<h_lbmGridSize, h_lbmBlockSize>>>(d_srcLattice, d_destLattice);
    cudaDeviceSynchronize();

    // swap grids
    Lattice *swap = d_srcLattice;
    d_srcLattice = d_destLattice;
    d_destLattice = swap;

    // apply forces to snow
    snowApplyGrav<<<h_snowGravNumBlocks,SNOW_GRAV_BLOCK_SIZE>>>(d_snowflakeDataFlat, d_numParticles, d_snowOffsets, d_globalState);

    // update snow verts
    snowUpdate<<<h_snowUpdateNumBlocks,SNOW_UPDATE_BLOCK_SIZE>>>(d_verts, d_snowOffsets, d_numParticles);

    // fetch work
    cudaMemcpy(h_verts, d_verts, h_numPolys*9*sizeof(float), cudaMemcpyDeviceToHost);

    // TODO: del
    cudaMemcpy(&h_destLattice, d_destLattice, sizeof(Lattice), cudaMemcpyDeviceToHost);
    for (int x = 0; x < 25; x+=1) {
        for (int y = 0; y < 5; y++) {
            float f1 = h_destLattice.f1[5*x + y];
           // printf("%f ", f1);
        }
        //printf("\n");
    }
    //printf("\n\n\n\n");
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