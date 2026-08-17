/**
* gpu.hpp - Defines functions that are imported from CUDA.
 *
 * @author Mars Semenova
 */

#ifndef GPU_HPP
#define GPU_HPP

const float DEFAULT_WIND_VELOCITY = 20; // TODO: update when physics fixed
const unsigned DEFAULT_LATTICE_RES = 20; // TODO: update when physics fixed

extern void snowUpdateGPU();
extern void snowInitGPU(SnowGeneratorData data, unsigned numParticles, float extents[3][2], float windVel, unsigned latticeRes, float temp);

#endif
