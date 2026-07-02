/**
* Constants.hpp - Contains includes, definitions, and constants for the application.
 *
 * @author Mars Semenova
 */

#ifndef ImportGL_HPP
#define ImportGL_HPP

// OpenGL includes
#include <glad/glad.h>
#include <GLFW/glfw3.h>

// include GLM
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/glm.hpp>
#include <glm/gtx/string_cast.hpp>

using namespace glm;

// settings
const unsigned int SCR_WIDTH = 1280;
const unsigned int SCR_HEIGHT = 720;

#include "Shader.hpp"
#include "Axes.hpp"
#include "Sphere.hpp"
#include "CameraControls.hpp"

#endif