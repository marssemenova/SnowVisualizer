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

GLint getRandInt(GLfloat min, GLfloat max) {
    return min + (rand() % (int)(max+1-min));
}

void calcNormal(GLfloat* trig, GLfloat* norm) { // TODO: get right 1
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
        glm::vec3 v(trig[x], trig[x+1], trig[x+2]);
        v = glm::normalize(v);
        norm[x] = v[0];
        norm[x+1] = v[1];
        norm[x+2] = v[2];
        //norm[x] = v1[0];
        //norm[x+1] = v1[1];
        //norm[x+2] = v1[2];
    }
}

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

GLfloat deg2rad(GLfloat deg) {
    return deg * (_PI/180);
}


#endif
