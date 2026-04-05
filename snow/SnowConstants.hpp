/**
 * SnowConstants.hpp - Contains includes, definitions, and constants for the
 * snow generation.
 *
 * @author Mars Semenova
 * @date March 30, 2026
 */
#ifndef SNOWCONSTANTS_HPP
#define SNOWCONSTANTS_HPP

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

// def vals
const GLuint DEFAULT_SNOW_COUNT = 100;
const GLfloat DEFAULT_TEMP = -15.0; // C
const GLfloat DEFAULT_EXTENT[3][2] = {{-100, 100}, {-100, 100}, {-100, 100}}; // x range, y range, z range

// consts
const GLfloat DRY_HUMIDITY_CONST = 0.17; // km/m^2
const GLfloat WET_HUMIDITY_CONST = 0.724; // km/m^2
const GLfloat GRAV = 9.81; // used w density
const GLfloat EPS = 80.0; // val from Moeslund

#endif
