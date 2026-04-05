/**
 * Constants.hpp - Contains includes, definitions, and constants for the application.
 *
 * @author Mars Semenova
 * @date March 30, 2026
 */
#ifndef CONSTANTS_HPP
#define CONSTANTS_HPP

// include libs
#include <stdio.h>
#include <stdlib.h>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <algorithm>
#include <sstream>

// OpenGL includes
#include <glad/glad.h>
#include <GLFW/glfw3.h>

// include GLM
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/glm.hpp>
#include <glm/gtx/string_cast.hpp>

using namespace glm;

// settings
const unsigned int SCR_WIDTH = 800;
const unsigned int SCR_HEIGHT = 600;

// consts
static const double _PI = 2.0*asin(1);

#include "Util.hpp"
#include "Shader.hpp"
#include "Axes.hpp"
#include "Sphere.hpp"

#endif