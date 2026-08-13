/**
* lbm_utils.h - Contains util functions for the LBM implementation.
 *
 * @author Mars Semenova
 */

#ifndef LBM_UTILS_H
#define LBM_UTILS_H

#include "../../util/Unix_Timer.h"

timer_id globalGPUTimer;

__global__ void printLattice(Lattice *d_lattice) {
    float* refs[] = {d_lattice->f0, d_lattice->f1, d_lattice->f2, d_lattice->f3, d_lattice->f4, d_lattice->f5, d_lattice->f6, d_lattice->f7, d_lattice->f8, d_lattice->f9, d_lattice->f10, d_lattice->f11, d_lattice->f12, d_lattice->f13, d_lattice->f14, d_lattice->f15, d_lattice->f16, d_lattice->f17, d_lattice->f18};
    for (int x = 0; x < d_N*d_N*d_N; x++) {
        printf("%d\n", x);
        for (int i = 0; i < LBM_Q; i++) {
            printf("%f ", (refs[i])[x]);
        }
        printf("\n");
    }
}

__global__ void printVelocities(float *d_velocities) {
    for (int x = 0; x < d_N*d_N*d_N; x++) {
        printf("%d: (%f %f %f), ", x, d_velocities[3*x], d_velocities[3*x+1], d_velocities[3*x+2]);
    }
    printf("\n");
}

__device__ int getInd(int x, int y, int z) {
    return x + y * d_N + z * d_N*d_N;
}

__device__ int getInd(int pos[3]) {
    return getInd(pos[0], pos[1], pos[2]);
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

#endif
