/**
* gpu.hpp - Defines functions that are imported from CUDA.
 *
 * @author Mars Semenova
 */

#ifndef GPU_HPP
#define GPU_HPP

extern void updateSnowOnGPU(float *verts, unsigned numParticles, SnowGenerator snowGenerator, SnowGeneratorData data, float extent[3][2]);

#endif
