/**
* gpu.hpp - Defines functions that are imported from CUDA.
 *
 * @author Mars Semenova
 */

#ifndef GPU_HPP
#define GPU_HPP

extern void snowUpdateGPU();
extern void snowInitGPU(SnowGeneratorData data, unsigned numParticles, float extents[3][2], float windVel, unsigned latticeRes, float temp);

#endif
