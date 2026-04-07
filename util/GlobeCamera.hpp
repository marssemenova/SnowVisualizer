/**
 * GlobeCamera.hpp - Contains helper methods to control the camera. Modified version of a standard
 * template by Dr. Brandt.
 *
 * @author Dr. Alexander Brandt, Mars Semenova
 */
#ifndef _CAM_CONTROLS_H_
#define _CAM_CONTROLS_H_

glm::vec3 cameraControlsGlobe(glm::mat4& V, glm::vec3 eye, GLFWwindow* window) {
    glm::vec3 targ = {0.0f, 0.0f, 0.0f};
    glm::vec3 up = {0.0f, 1.0f, 0.0f};
    static float radiusFromOrigin = glm::length(eye);

    static glm::vec3 position = eye;
    static GLfloat phi = 1*3.14159f - acos(eye.y/radiusFromOrigin);

    // theta based on +/- x and z
    float thetaVal = 0;
    if (eye.x < 0 && eye.z >= 0) {
        thetaVal = acos(eye.x/(radiusFromOrigin*sin(phi)));
    }
    if (eye.x < 0 && eye.z < 0) {
        thetaVal = 1*3.14159f - acos(eye.x/(radiusFromOrigin*sin(phi)));
    }
    if (eye.x >= 0 && eye.z >= 0) {
        thetaVal = 1*3.14159f + acos(eye.x/(radiusFromOrigin*sin(phi)));
    }
    if (eye.x >= 0 && eye.z < 0) {
        thetaVal = -acos(eye.x/(radiusFromOrigin*sin(phi)));
    }
    static GLfloat theta = thetaVal;

    static double mouseDownX;
    static double mouseDownY;
    static bool firstPress = true;
    static double lastTime = glfwGetTime();

    double dx = 0.0, dy = 0.0;
    int state = glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT);
    if (state == GLFW_PRESS)
    {
        if (firstPress) {
            glfwGetCursorPos(window, &mouseDownX, &mouseDownY);
            firstPress = false;
            dx = 0.0;
            dy = 0.0;
        } else {
            double xpos, ypos;
            glfwGetCursorPos(window, &xpos, &ypos);

            dx = xpos - mouseDownX;
            dy = ypos - mouseDownY;

            dx *= -1; dy *= -1; //for globe movement

            mouseDownX = xpos;
            mouseDownY = ypos;
        }
    }
    if (state == GLFW_RELEASE) {
        firstPress = true;
    }

    theta += 0.002f * dx;
    phi   += -0.002f * dy;
    if (theta > 2*_PI) {
        theta -= 2*_PI;
    }
    if (phi >= _PI) {
        phi = 0.9999999 * _PI;
    }
    if (phi <= 0) {
        phi = 0.0000001;
    }

    glm::vec3 direction(
        sin(phi) * sin(theta),
        cos(phi),
        sin(phi) * cos(theta)
    );
    glm::normalize(direction);

    float speed = 0.25f * radiusFromOrigin; //move faster further away
    double currentTime = glfwGetTime();
    float deltaTime = (currentTime - lastTime);
    lastTime = currentTime;

    // Move forward
    if (glfwGetKey( window, GLFW_KEY_UP ) == GLFW_PRESS) {
        radiusFromOrigin -= deltaTime * speed;
    }
    // Move backward
    if (glfwGetKey( window, GLFW_KEY_DOWN ) == GLFW_PRESS) {
        radiusFromOrigin += deltaTime * speed;
    }

    position = -1.0f * direction * radiusFromOrigin;
    V = glm::lookAt(position, targ, up);

    return position;
}

#endif
