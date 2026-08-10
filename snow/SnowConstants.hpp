/**
* SnowConstants.hpp - Contains includes, definitions, and constants for the
 * snow generation.
 *
 * @author Mars Semenova
 */

#ifndef SNOW_CONSTANTS_HPP
#define SNOW_CONSTANTS_HPP

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

#include "../util/Util.hpp"

// def vals
const unsigned DEFAULT_SNOW_COUNT = 100;
const float DEFAULT_TEMP = -15.0; // C
const float DEFAULT_EXTENTS[3][2] = {{-100, 100}, {-100, 100}, {-100, 100}}; // x range, y range, z range

// consts
const float SNOW_STATE_THRESH = -1.0; // above = wet, below = dry
const float DIAMETER_THRESH = -0.061;
const float DRY_HUMIDITY_CONST = 0.17; // km/m^2
const float WET_HUMIDITY_CONST = 0.724; // km/m^2
const float GRAV = 9.81; // used w density
const float EPS = deg2rad(80.0); // val from Moeslund
const float SHININESS_COEFF = 1.0;

// alg types enum
const unsigned MOESLUND_ALG = 1;
const unsigned EXPERIMENTAL_ALG = 2;

// experimentally determined values for the experimental alg
const float EPS_THETA_MOESLUND = deg2rad(80.0);
const float EPS_PHI_MOESLUND  = deg2rad(80.0);
const float OPACITY_COEFF_MOESLUND = 1.0/50.0; // approx
const float EPS_OPACITY_MOESLUND  = 0.0; // percentage
const unsigned NUM_LAYERS_WET_MOESLUND  = 6;
const unsigned NUM_LAYERS_DRY_MOESLUND  = 6;
const float EPS_THETA_WET = deg2rad(60.0);
const float EPS_PHI_WET = deg2rad(30.0);
const float EPS_THETA_DRY = deg2rad(70.0);
const float EPS_PHI_DRY = deg2rad(35.0);
const float OPACITY_COEFF = 1.0/50.0;
const float EPS_OPACITY = 0.25; // percentage
const unsigned NUM_LAYERS_WET = 5;
const unsigned NUM_LAYERS_DRY = 6;

#endif
