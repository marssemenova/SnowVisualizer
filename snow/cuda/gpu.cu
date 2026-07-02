/**
 * gpu.cu - Contains the CUDA code which accelerates the
 * program on the GPU.
 *
 * @author Mars Semenova
 */

#include <vector>
#include <stdio.h>
#include <cuda_runtime.h>

using namespace std;

#include "../SnowGeneratorData.hpp"

extern void updateSnowOnGPU(float *verts, unsigned numParticles, SnowGeneratorData data, float extent[3][2]) {
    SnowflakeData snowflakeData;
    unsigned currInd;
    float offset;
    for (int x = 0; x < numParticles; x++) {
        snowflakeData = data.snowflakeData[x];
        currInd = snowflakeData.ind;
        if (snowflakeData.pos[1] < extent[1][0]) {
            offset = extent[1][1] - snowflakeData.pos[1];
        } else {
            offset = -2.0f; // TODO: make var
        }
        for (int i = currInd; i < currInd + snowflakeData.numPolys*9; i+=3) {
            verts[i + 1] += offset;
        }
        data.snowflakeData[x].pos[1] += offset;
    }
}
