/**
 * gpu.hpp - Contains the CUDA code which accelerates the
 * program on the GPU..
 *
 * @author Mars Semenova
 */

#ifndef GPU_H
#define GPU_H

// #include "cuda.h"

void updateSnowOnGPU(GLfloat *verts, GLint numVerts) {
    for (int x = 0; x < numVerts; x++) {
        verts[3*x+1] -= 2.0f;
    }
}
#endif
