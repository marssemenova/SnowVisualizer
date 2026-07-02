/**
* SnowGeneratorData.hpp - Contains the data structure used to return
 * generated snow data.
 *
 * @author Mars Semenova
 */

#ifndef SNOWGENERATORDATA_HPP
#define SNOWGENERATORDATA_HPP

struct SnowflakeData {
    float pos[3];
    unsigned numPolys;
    unsigned ind;
};

struct SnowGeneratorData {
    unsigned numPolys;
    float* verts;
    float* normals;
    float* colours;
    SnowflakeData* snowflakeData;
};

#endif
