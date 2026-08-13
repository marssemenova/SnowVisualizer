/**
 * Input.hpp - Contains inputs for the program. Replaces input through CLI for
 * development purposes.
 *
 * @author Mars Semenova
 */

#ifndef INPUT_HPP
#define INPUT_HPP

// temp input vars (TODO: get from CLI args)
// snow
unsigned numParticles = 1000;
unsigned whichCam = FIRST_PERSON_CAM;
float minX = -100.0, maxX = 100.0, minY = -100.0, maxY = 100.0, minZ = -100.0, maxZ = 100.0;
float temp = -5.0;

// wind
unsigned latticeRes = 20;
float windVel = 5; // km/h

#endif
