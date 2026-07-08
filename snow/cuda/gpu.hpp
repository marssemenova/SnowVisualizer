/**
* gpu.hpp - Defines functions that are imported from CUDA.
 *
 * @author Mars Semenova
 */

#ifndef GPU_HPP
#define GPU_HPP

extern void updateSnowOnGPU(SnowGeneratorData data, unsigned numParticles, float extent[3][2]);

#endif
