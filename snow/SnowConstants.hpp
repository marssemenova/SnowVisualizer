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
const GLfloat SNOW_STATE_THRESH = -1.0; // above = wet, below = dry
const GLfloat DIAMETER_THRESH = -0.061;
const GLfloat DRY_HUMIDITY_CONST = 0.17; // km/m^2
const GLfloat WET_HUMIDITY_CONST = 0.724; // km/m^2
const GLfloat GRAV = 9.81; // used w density
const GLfloat EPS = deg2rad(80.0); // val from Moeslund
const GLfloat SHININESS_COEFF = 1.0;

// alg types enum
const GLuint MOESLUND_ALG = 1;
const GLuint EXPERIMENTAL_ALG = 2;

// experimentally determined values for the experimental alg
const GLfloat EPS_THETA_MOESLUND = deg2rad(80.0);
const GLfloat EPS_PHI_MOESLUND  = deg2rad(80.0);
const GLfloat OPACITY_COEFF_MOESLUND = 1.0/50; // approx
const GLfloat EPS_OPACITY_MOESLUND  = 0.0; // percentage
const GLuint NUM_LAYERS_WET_MOESLUND  = 6;
const GLuint NUM_LAYERS_DRY_MOESLUND  = 6;
const GLfloat EPS_THETA_WET = deg2rad(60.0);
const GLfloat EPS_PHI_WET = deg2rad(30.0);
const GLfloat EPS_THETA_DRY = deg2rad(70.0);
const GLfloat EPS_PHI_DRY = deg2rad(35.0);
const GLfloat OPACITY_COEFF = 1.0/50;
const GLfloat EPS_OPACITY = 0.25; // percentage
const GLuint NUM_LAYERS_WET = 5;
const GLuint NUM_LAYERS_DRY = 6;

#endif
