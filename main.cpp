/**
 * Main.cpp - Runner file for the application. Structure largely follows a standard
 * template by Dr. Brandt.
 *
 * @author Mars Semenova, Dr. Alexander Brandt
 * @date March 30, 2026
 */
#include "util/Constants.hpp"
#include "sample/LoadBMP.hpp"

#include "util/GlobeCamera.hpp"
#include "snow/SnowGenerator.hpp"

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
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);

    // glad: load all OpenGL function pointers
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
        std::cout << "Failed to initialize GLAD" << std::endl;
        return -1;
    }

    // temp input vars, TODO: del n get thru CLI
    GLuint numParticles = 1;
    GLfloat minX = 0.0, maxX = 5.0, minY = 0.0, maxY = 5.0, minZ = 0.0, maxZ = 5.0, scale = 1.0f;
    GLfloat temp = -5.0;
    bool isWet = true;

    // def vars
    float screenW = SCR_WIDTH;
    float screenH = SCR_HEIGHT;
    GLenum err;
    Axes axes(glm::vec3(0,  0, 0), glm::vec3(5.0f, 5.0f, 5.0f));

    // ensure we can capture the escape key being pressed below
    glfwSetInputMode(window, GLFW_STICKY_KEYS, GL_TRUE);

    // dark blue background
    glClearColor(0.1f, 0.1f, 0.3f, 0.0f);

    // set up rendering vars
    glm::mat4 Projection = glm::perspective(glm::radians(45.0f), screenW/screenH, 0.001f, 1000.0f);
    glm::vec3 eye = {0.0f, 3.0f, 5.0f};
    glm::vec3 up = {0.0f, 1.0f, 0.0f};
    glm::vec3 center = {0.0f, 0.0f, 0.0f};

    glm::mat4 V = glm::lookAt(eye, center, up);
    cameraControlsGlobe(V, eye, window);
    glm::mat4 MSnow(1.0f);
    MSnow = glm::scale(MSnow, glm::vec3(scale, scale, scale));
    glm::mat4 MAxes(1.0f);
    glm::vec3 lightpos(5.0f, 5.0f, 5.0f);

    Sphere sphere;
    sphere.setUpAxis(2);
    sphere.setRadius((0.008540*0.5)/2);
    glm::vec4 color1(0.f, 0.8f, 0.8f, 0.1f);
    glm::mat4 M1(1.0f);
    M1 = glm::scale(M1, glm::vec3(50.0f, 50.0f, 50.0f));
    float alpha1 = 2;

    // setup snow gen obj
    GLfloat extent[3][2] = {{minX/scale, maxX/scale}, {minY/scale, maxY/scale}, {minZ/scale, maxZ/scale}};
    SnowGenerator snowGen(numParticles, extent, temp, isWet);

    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LESS);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    do {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); // clear the screen

        //axes.draw(MAxes, V, Projection);
        //sphere.draw(lightpos, M1, V, Projection, color1, alpha1);
        snowGen.draw(lightpos, MSnow, V, Projection);

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