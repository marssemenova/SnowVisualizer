/**
 * Util.hpp - Contains util functions.
 *
 * @author Mars Semenova
 * @date April 5, 2026
 */

#ifndef UTIL_HPP
#define UTIL_HPP

/**
 * Get random float in the range [min, max).
 * @param min - Min float generated.
 * @param max - Max float generated, non-inclusive.
 * @return Generated float.
 */
GLfloat getRandFloat(GLfloat min, GLfloat max) {
    if (min == max) {
        return min;
    }
    return min + (max - min) * (GLfloat) rand() / (GLfloat) RAND_MAX;
}

/**
 * Get random integer in the range [min, max].
 * @param min - Min integer generated.
 * @param max - Max integer generated.
 * @return Generated integer.
 */
GLint getRandInt(GLfloat min, GLfloat max) {
    if (min == max) {
        return min;
    }
    return min + (rand() % (int)(max+1-min));
}

/**
 * Calculate normals. Normals are calculated for each vertex independently
 * but in the future it is worth looking into calculating normals for the entire
 * triangle.
 * @param trig - Pointer to the vertices of the triangle for which to calculate normals.
 * @param norm - Pointer the array where to write the calculated normals.
 */
void calcNormal(GLfloat* trig, GLfloat* norm) {
    for (int x = 0; x < 9; x+=3) {
        glm::vec3 v(trig[x], trig[x+1], trig[x+2]);
        v = glm::normalize(v);
        norm[x] = v[0];
        norm[x+1] = v[1];
        norm[x+2] = v[2];
    }
}

/**
 * Return a random point inside of a triangle.
 * @param trig - Pointer to the vertices of the triangle in which to find the random point.
 * @param pt - Pointer the array where to write the point.
 */
void findPointInTriangle(GLfloat* trig, GLfloat pt[3]) {
    GLfloat x1 = trig[0], y1 = trig[1], z1 = trig[2], x2=trig[3], y2=trig[4], z2=trig[5], x3=trig[6], y3=trig[7], z3=trig[8];
    GLfloat v1x=x1-x2, v1y=y1-y2, v1z=z1-z2;
    GLfloat v2x=x3-x2, v2y=y3-y2, v2z=z3-z2;
    GLfloat s = getRandFloat(0, 1), t = getRandFloat(0, 1);
    if (s + t <= 1) {
        pt[0] = s*v1x + t*v2x;
        pt[1] = s*v1y + t*v2y;
        pt[2] = s*v1z + t*v2z;
    } else {
        pt[0] = (1-s)*v1x + (1-t)*v2x;
        pt[1] = (1-s)*v1y + (1-t)*v2y;
        pt[2] = (1-s)*v1z + (1-t)*v2z;
    }
    pt[0] += x2;
    pt[1] += y2;
    pt[2] += z2;
}

/**
 * Convert degreed to radians.
 * @param deg - Value in degrees to convert.
 * @return Degrees in radians.
 */
GLfloat deg2rad(GLfloat deg) {
    return deg * (_PI/180);
}


#endif
