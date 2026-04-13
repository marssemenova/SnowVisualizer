/**
 * SnowBuilderData.hpp - Contains the data structure used to return
 * generated snow data.
 *
 * @author Mars Semenova
 * @date April 5, 2026
 */

#ifndef SNOWBUILDERDATA_HPP
#define SNOWBUILDERDATA_HPP

struct SnowBuilderData {
    GLuint numPolys;
    GLfloat* verts;
    GLfloat* normals;
    GLfloat* colours;
};

#endif
