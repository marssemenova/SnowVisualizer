/**
* ImportGL.hpp - Contains includes, definitions, and constants for the OpenGL part of the application.
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
const float  SCR_WIDTH = 1280.0f;
const float SCR_HEIGHT = 720.0f;

#include "Shader.hpp"
#include "Axes.hpp"
#include "Sphere.hpp"
#include "CameraControls.hpp"

#endif