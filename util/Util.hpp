/**
 * Util.hpp - Contains util functions.
 *
 * @author Mars Semenova
 * @date April 5, 2026
 */
#ifndef UTIL_HPP
#define UTIL_HPP

GLfloat getRandFloat(GLfloat min, GLfloat max) {
    return min + (max - min) * (GLfloat) rand() / (GLfloat) RAND_MAX;
}

void calcNormal(GLfloat* trig, GLfloat* norm) {
    glm::vec3 p1, p2, p3;
    p1.x = trig[0];
    p1.y = trig[1];
    p1.z = trig[2];
    p2.x = trig[3];
    p2.y = trig[4];
    p2.z = trig[5];
    p3.x = trig[6];
    p3.y = trig[7];
    p3.z = trig[8];

    glm::vec3 v1 = p2-p1;
    glm::vec3 v2 = p3-p2;

    v1 = glm::cross(v1,v2);
    v1 = glm::normalize(v1);
    for (int x = 0; x < 9; x+=3) {
        norm[x] = v1[0];
        norm[x+1] = v1[1];
        norm[x+2] = v1[2];
    }
}

#endif
