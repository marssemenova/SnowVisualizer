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
unsigned numParticles = 10000;
unsigned whichCam = GLOBE_CAM;
// unsigned whichCam = FIRST_PERSON_CAM;
float zFarClip = 1000;
float simExtents = 100.0;
float minX = -simExtents, maxX = simExtents;
float minY = -simExtents, maxY = simExtents;
float minZ = -simExtents, maxZ = simExtents;
float temp = -5.0;

// wind
unsigned latticeRes = 20;
float windVel = 20; // km/h

#endif
