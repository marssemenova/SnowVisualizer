/**
 * main.cpp - Runner file for the application. Structure largely follows a standard
 * template by Dr. Brandt.
 *
 * @author Mars Semenova, Dr. Alexander Brandt
 */

#include <thread>

#include "util/Constants.hpp"

// OpenGL includes
#include "util/ImportGL.hpp"

#include "snow/SnowRenderer.hpp"
#include "util/Input.hpp"
#include "util/CPUTimer.hpp"

void framebuffer_size_callback(GLFWwindow* window, int width, int height);
void processInput(GLFWwindow *window);

int main() {
    srand(time(0));
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
    GLFWwindow* window = glfwCreateWindow(SCR_WIDTH, SCR_HEIGHT, "Snow Visualizer", NULL, NULL);
    if (window == NULL) {
        std::cout << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return -1;
    }
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);

    // glad: load all OpenGL function pointers
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
        std::cout << "Failed to initialize GLAD" << std::endl;
        return -1;
    }

    // def vars
    float screenW = SCR_WIDTH;
    float screenH = SCR_HEIGHT;
    GLenum err;
    windVel = (windVel / 3.6f); // km/h > m/s

    // ensure we can capture the escape key being pressed below
    glfwSetInputMode(window, GLFW_STICKY_KEYS, GL_TRUE);

    // dark blue background
    glClearColor(0.0f, 0.0f, 0.0f, 0.0f);

    // set up rendering vars
    glm::mat4 Projection = glm::perspective(glm::radians(45.0f), screenW/screenH, 0.001f, zFarClip);
    glm::vec3 eye;
    if (whichCam == GLOBE_CAM) {
        eye= {0.0f, 50.0f, 100.0f};
    } else {
        eye = {0.0f, 0.0f, 400.0f};
    }
    glm::vec3 up = {0.0f, 1.0f, 0.0f};
    glm::vec3 center = {0.0f, 0.0f, 0.0f};

    glm::mat4 V = glm::lookAt(eye, center, up);
    if (whichCam == GLOBE_CAM) {
        cameraControlsGlobe(V, eye, window);
    } else {
        cameraControlsFirstPerson(V, eye, window);
    }
    glm::mat4 MSnow(1.0f);
    glm::vec3 lightpos(0.0f, 10.0f, -10.0f);

    // setup snow gen obj
    GLfloat extents[3][2] = {{minX, maxX}, {minY, maxY}, {minZ, maxZ}};
    SnowRenderer snowGen(numParticles, extents, temp, EXPERIMENTAL_ALG, windVel, latticeRes);

    GLfloat axlength = 25.0f;
    glm::vec3 origin(0.0f, 0.0f, 0.0f);
    glm::vec3 ax_ext(axlength, axlength, axlength);
    glm::mat4 m_axes(1.0f);
    Axes ax(origin, ax_ext);

    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LESS);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    do {

        time_point updateStart = startTimer();
        snowGen.updateSnow(); //includes simulation + vertex copy to OGL
        time_point updateEnd = stopTimer();
        float updateTime = elapsedTime(updateStart, updateEnd);
        printf("update time: %f\n", updateTime);

        time_point renderStart = startTimer();
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); // clear the screen

        ax.draw(m_axes, V, Projection);
        snowGen.draw(lightpos, MSnow, V, Projection);

        if (whichCam == GLOBE_CAM) {
            cameraControlsGlobe(V, eye, window);
        } else {
            cameraControlsFirstPerson(V, eye, window);
        }

        // process/log the error
        while ((err = glGetError()) != GL_NO_ERROR) {
            fprintf(stderr, "GLEnum error after draw: %d\n", err);
        }

        // swap buffers
        glfwSwapBuffers(window);
        glfwPollEvents();
        time_point renderEnd = stopTimer();

        float renderTime = elapsedTime(renderStart, renderEnd);
        printf("render time: %f\n", renderTime);
        // std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    // check if the ESC key was pressed or the window was closed
    while (glfwGetKey(window, GLFW_KEY_ESCAPE ) != GLFW_PRESS && glfwWindowShouldClose(window) == 0);


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