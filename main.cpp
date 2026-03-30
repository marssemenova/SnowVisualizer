// include standard headers
#include <stdio.h>
#include <stdlib.h>
#include <cmath>

#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <iostream>

// include GLM
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtx/string_cast.hpp>
#include <glm/gtc/matrix_transform.hpp>

using namespace glm;

// settings
const unsigned int SCR_WIDTH = 800;
const unsigned int SCR_HEIGHT = 600;

#include "util/shader.hpp"
#include "util/GlobeCamera.hpp"
#include "sample/LoadBMP.hpp"
#include "sample/Plane.hpp"
#include "sample/Sphere.hpp"

void framebuffer_size_callback(GLFWwindow* window, int width, int height);
void processInput(GLFWwindow *window);

int main()
{
    // glfw: initialize and configure
    glfwInit();
    glfwWindowHint(GLFW_SAMPLES, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

#ifdef __APPLE__
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE); // uncomment this statement to fix compilation on OS X
#endif

    // glfw window creation
    GLFWwindow* window = glfwCreateWindow(SCR_WIDTH, SCR_HEIGHT, "Variety Visualizer", NULL, NULL);
    if (window == NULL) {
        std::cout << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return -1;
    }
    glfwMakeContextCurrent(window);
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);

    // glad: load all OpenGL function pointers
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
        std::cout << "Failed to initialize GLAD" << std::endl;
        return -1;
    }

    // define vars
    float screenW = SCR_WIDTH;
    float screenH = SCR_HEIGHT;
    GLenum err;

    // ensure we can capture the escape key being pressed below
    glfwSetInputMode(window, GLFW_STICKY_KEYS, GL_TRUE);

    // dark blue background
    glClearColor(0.2f, 0.2f, 0.3f, 0.0f);

    glm::mat4 Projection = glm::perspective(glm::radians(45.0f), screenW/screenH, 0.001f, 1000.0f);
    glm::vec3 eye = {0.0f, 3.0f, 5.0f};
    glm::vec3 up = {0.0f, 1.0f, 0.0f};
    glm::vec3 center = {0.0f, 0.0f, 0.0f};

    glm::mat4 V = glm::lookAt(eye, center, up);
    cameraControlsGlobe(V, eye, window);

    glm::mat4 V2 = glm::lookAt(eye, center, up);

    glm::mat4 M1(1.0f);
    glm::mat4 M2(1.0f);
    glm::mat4 M3(1.0f);
    glm::mat4 MFloor(1.0f);
    M1 = glm::translate(M2, glm::vec3(-3.0f, 0.0f, 0.0f));
    M3 = glm::translate(M2, glm::vec3(3.0f, 0.0f, 0.0f));
    MFloor = glm::translate(MFloor, glm::vec3(0.f, -1.1f, 0.f));

    Sphere sphere;
    sphere.setUpAxis(2);
    Sphere sphere2;
    Sphere sphere3;
    sphere2.setUpAxis(2);
    sphere3.setUpAxis(2);

    Plane floor(10.0f, "wood.bmp");

    glm::vec3 lightpos(5.0f, 5.0f, 5.0f);

    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LESS);

    glm::vec4 color1(0.f, 0.8f, 0.8f, 1.0f);
    glm::vec4 color2(0.f, 0.8f, 0.8f, 1.0f);
    glm::vec4 color3(0.f, 0.8f, 0.8f, 1.0f);
    float alpha1 = 2;
    float alpha2 = 16;
    float alpha3 = 64;

    do {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); // clear the screen

        sphere.draw(lightpos, M1, V, Projection, color1, alpha1);
        sphere2.draw(lightpos, M2, V, Projection, color2, alpha2);
        sphere3.draw(lightpos, M3, V, Projection, color3, alpha3);

        floor.draw(lightpos, MFloor, V, Projection, color3, alpha1);

        cameraControlsGlobe(V, eye, window);

        // process/log the error
        while ((err = glGetError()) != GL_NO_ERROR) {
            fprintf(stderr, "GLEnum error after draw: %d\n", err);
        }

        // swap buffers
        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    // check if the ESC key was pressed or the window was closed
    while (glfwGetKey(window, GLFW_KEY_ESCAPE ) != GLFW_PRESS && glfwWindowShouldClose(window) == 0);

    // render loop
    while (!glfwWindowShouldClose(window)) {
        // input
        processInput(window);

        // render
        glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);

        // glfw: swap buffers and poll IO events (keys pressed/released, mouse moved etc.)
        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    // glfw: terminate, clearing all previously allocated GLFW resources.
    glfwTerminate();
    return 0;
}

// process all input: query GLFW whether relevant keys are pressed/released this frame and react accordingly
void processInput(GLFWwindow *window) {
    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
        glfwSetWindowShouldClose(window, true);
}

// glfw: whenever the window size changed (by OS or user resize) this callback function executes
void framebuffer_size_callback(GLFWwindow* window, int width, int height) {
    glViewport(0, 0, width, height); // make sure the viewport matches the new window dimensions (note that width and height will be significantly larger than specified on retina displays)
}