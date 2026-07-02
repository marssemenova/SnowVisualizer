/**
 * gpu.cu - Contains the CUDA code which accelerates the
 * program on the GPU.
 *
 * @author Mars Semenova
 */

#include <vector>
#include <stdio.h>
#include <cuda_runtime.h>

#define GLM_ENABLE_EXPERIMENTAL
#include "../../include/glm/glm.hpp"
#include "../../include/glm/gtx/string_cast.hpp"

using namespace glm;
using namespace std;

#include "../SnowGenerator.hpp"

extern void updateSnowOnGPU(float *verts, unsigned numParticles, SnowGenerator snowGenerator, SnowGeneratorData data, float extent[3][2]) {
    vector<int> newSnowInds;
    SnowflakeData snowflakeData;
    unsigned currInd, numNewSnow = 0;
    for (int x = 0; x < numParticles; x++) {
        snowflakeData = data.snowflakeData[x];
        if (snowflakeData.pos[1] < extent[1][0]) {
            newSnowInds.push_back(x);
            numNewSnow++;
        } else {
            currInd = snowflakeData.ind;
            for (int i = currInd; i < currInd + snowflakeData.numPolys*9; i+=3) {
                verts[i + 1] -= 2.0f; // TODO: make var
            }
            data.snowflakeData[x].pos[1] -= 2.0f; // TODO: make var
        }
    }
    if (numNewSnow > 0) {
        float topExtent[3][2];
        memcpy(topExtent, extent, 6*sizeof(float));
        topExtent[1][0] = topExtent[1][1];
        SnowGeneratorData newSnowData = snowGenerator.generateSnowExperimental(numNewSnow, topExtent); // TODO: for varied snowflake numPolys would have to change implementation
        int currVert = 0;
        for (int x = 0; x < newSnowInds.size(); x++) {
            snowflakeData = data.snowflakeData[newSnowInds[x]];
            memcpy(data.snowflakeData[newSnowInds[x]].pos, newSnowData.snowflakeData[x].pos, 3*sizeof(float));
            data.snowflakeData[newSnowInds[x]].pos[1] = newSnowData.snowflakeData[x].pos[1];
            data.snowflakeData[newSnowInds[x]].pos[2] = newSnowData.snowflakeData[x].pos[2];
            currInd = snowflakeData.ind;
            for (int i = 0; i < snowflakeData.numPolys*9; i++) {
                verts[currInd + i] = newSnowData.verts[currVert + i];
            }
            currVert += snowflakeData.numPolys*9;
        }
    }
}
