/**
 * SnowGeneratorData.hpp - Contains the data structure used to return
 * generated snow data.
 *
 * @author Mars Semenova
 * @date April 5, 2026
 */

#ifndef SNOWGENERATORDATA_HPP
#define SNOWGENERATORDATA_HPP

struct SnowGeneratorData {
    GLuint numPolys;
    GLfloat* verts;
    GLfloat* normals;
    GLfloat* colours;
};

#endif
