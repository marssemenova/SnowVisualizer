/**
 * gpu.cu - Contains the CUDA code which accelerates the
 * program on the GPU.
 *
 * @author Mars Semenova
 */

#include <stdio.h>

extern void test() {
    printf("hi");
}

extern void updateSnowOnGPU(float *verts, int numVerts) {
    for (int x = 0; x < numVerts; x++) {
        verts[3*x+1] -= 2.0f;
    }
}
