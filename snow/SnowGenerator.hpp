/**
 * SnowGenerator.hpp - Contains the class which implements the generation of
 * a single instance of snow.
 *
 * @author Mars Semenova
 */

#ifndef SNOWGENERATOR_HPP
#define SNOWGENERATOR_HPP

#include "SnowConstants.hpp"
#include "SnowGeneratorData.hpp"
#include "cuda/gpumalloc.hpp"

class SnowGenerator {
private:
	// params
	float diameter; // calced in cm based on temp, fluctuates by 50%, in cm
	float density; // calced based on temp, affects amt of layers + amt of polys + alpha
	bool isWet;

	// Moeslund implementation functions
	unsigned getNumLayersMoeslund() {
		return 6; // guessed
	}

	unsigned getNumPolysPerLayerMoeslund() {
		return isWet ? 40 : 10; // for Zou says only needs to be 1/4 ot the other
	}

	float getOpacityMoeslund() {
		return 0.5; // const guessed from imgs in paper
	}

	// improvement(?) functions
	unsigned getNumLayersExperimental() {
		return isWet ? NUM_LAYERS_WET : NUM_LAYERS_DRY;
	}
	unsigned getNumPolysPerLayerExperimental() {
		return isWet ? 40 : 10; // TODO: for Zou says only needs to be 1/4 ot the other, need to experiment
	}

	float getOpacityExperimental(float opacityCoeff) {
		return opacityCoeff * density;
	}

	/**
	 * Helper function which generates snow using the specified algorithm
	 * @param numParticles - Number of particles to generate.
	 * @param extents - Extents of volume in which to generate the particles, where extents[0] is a pair for the x extent,
	 * extents[1] is a pair for the y extent, and extents[2] is a pair for the z extent. If numParticles = 1 this
	 * parameter has no effect and the snow particle is generated at the origin.
	 * @param alg - Which algorithm to use for the snow particle generation. Either the Moeslund algorithm (1) or the
	 * experimental algorithm (2). Constants for these are defined in SnowConstants.hpp (MOESLUND_ALG and EXPERIMENTAL_ALG, respectively).
	 * @return A SnowGeneratorData object with the generated data for the snowflake(s).
	 */
	SnowGeneratorData generateSnow(unsigned numParticles, const float extents[3][2], unsigned alg) {
		SnowGeneratorData data;
		float xPos, yPos, zPos;
		if (numParticles == 1) {
			if (alg == MOESLUND_ALG) {
				data = generateSnowOnceMoeslund();
			} else {
				data = generateSnowOnceExperimental();
			}
			xPos = getRandFloat(extents[0][0], extents[0][1]);
			yPos = getRandFloat(extents[1][0], extents[1][1]);
			zPos = getRandFloat(extents[2][0], extents[2][1]);
			data.snowflakeData[0].pos[0] = xPos;
			data.snowflakeData[0].pos[1] = yPos;
			data.snowflakeData[0].pos[2] = zPos;
		}

		SnowGeneratorData* dataReturned = (SnowGeneratorData*) malloc(numParticles*sizeof(SnowGeneratorData));
		data.snowflakeData = (SnowflakeData*) mallocGPU(numParticles*sizeof(SnowflakeData));
		unsigned numEntries = 0;
		for (int x = 0; x < numParticles; x++) {
			xPos = getRandFloat(extents[0][0], extents[0][1]);
			yPos = getRandFloat(extents[1][0], extents[1][1]);
			zPos = getRandFloat(extents[2][0], extents[2][1]);
			data.snowflakeData[x].pos[0] = xPos;
			data.snowflakeData[x].pos[1] = yPos;
			data.snowflakeData[x].pos[2] = zPos;
			if (alg == MOESLUND_ALG) {
				dataReturned[x] = generateSnowOnceMoeslund();
			} else {
				dataReturned[x] = generateSnowOnceExperimental();
			}
			data.snowflakeData[x].ind = numEntries*9;
			numEntries += dataReturned[x].numPolys;
			data.snowflakeData[x].numPolys = dataReturned[x].numPolys;

			// update pos
			for (int i = 0; i < dataReturned[x].numPolys*9; i+=3) {
				dataReturned[x].verts[i] += xPos;
				dataReturned[x].verts[i+1] += yPos;
				dataReturned[x].verts[i+2] += zPos;
			}
		}

		data.numPolys = numEntries;
		data.verts = (float*) mallocGPU(numEntries*9*sizeof(float));
		data.normals = (float*) malloc(numEntries*9*sizeof(float));
		data.colours = (float*) malloc(numEntries*12*sizeof(float));
		unsigned offset = 0;
		for (int x = 0; x < numParticles; x++) {
			for (int i = 0; i < dataReturned[x].numPolys*9; i++) {
				data.verts[i + offset] = dataReturned[x].verts[i];
				data.normals[i + offset] = dataReturned[x].normals[i];
			}
			offset += dataReturned[x].numPolys*9;
		}
		offset = 0;
		for (int x = 0; x < numParticles; x++) {
			for (int i = 0; i < dataReturned[x].numPolys*12; i++) {
				data.colours[i + offset] = dataReturned[x].colours[i];
			}
			offset += dataReturned[x].numPolys*12;
		}

		// free
		free(dataReturned);

		return data;
	}

public:
	/**
	 * Constructor for SnowGenerator.
	 * @param temp - Temperature of the simulation.
	 */
	SnowGenerator(float temp) {
		if (temp <= DIAMETER_THRESH) {
			diameter = 0.015*pow(abs(temp), -0.35);
		} else {
			diameter = 0.04;
		}
		isWet = temp >= SNOW_STATE_THRESH;
		density = isWet ? WET_HUMIDITY_CONST/diameter : DRY_HUMIDITY_CONST/diameter; // TODO: if using probly also need to convert to cm
		diameter *= 100.0; // to cm
	}

	/**
	 * Default constructor for SnowGenerator.
	 */
	SnowGenerator() : SnowGenerator(DEFAULT_TEMP) {};

	/**
	 * Default function to generate a single snow particle. Uses the Experimental algorithm.
	 * @return A SnowGeneratorData object with the generated data for the snowflake.
	 */
	SnowGeneratorData generateSnowOnce() { return generateSnowOnceExperimental(); }

	/**
	 * Generates a single snow particle using the Moeslund algorithm.
	 * @return A SnowGeneratorData object with the generated data for the snowflake.
	 */
	SnowGeneratorData generateSnowOnceMoeslund() {
		// set vars
		unsigned numPolys;
		float d, rho, currRho, theta, phi, newTheta, newPhi;
		unsigned numPolysPerLayer = getNumPolysPerLayerMoeslund(), numLayers = getNumLayersMoeslund();
		numPolys = numPolysPerLayer * numLayers;
		d = getRandFloat(0.5, 1.5)*diameter;
		float layerH = (d/2.0f)/numLayers;

		// set up data obj
		SnowGeneratorData data;
		data.numPolys = numPolys;
		data.verts = (float*) mallocGPU(numPolys*9*sizeof(float));
		data.normals = (float*) malloc(numPolys*9*sizeof(float));
		data.colours = (float*) malloc(numPolys*12*sizeof(float));
		data.snowflakeData = (SnowflakeData*) mallocGPU(sizeof(SnowflakeData));

		// gen first layer verts n norms
		currRho = layerH;
		for (int x = 0; x < numPolysPerLayer*9; x+=9) {
			// gen vars
			rho = getRandFloat(currRho-layerH, currRho+layerH);
			if (rho == 0.0f) {
				rho = 0.000001f;
			}
			theta = deg2rad(getRandFloat(0.0, 360.0));
			phi = deg2rad(getRandFloat(0.0, 180.0));

			data.verts[x] = rho*cos(theta)*sin(phi);
			data.verts[x+1] = rho*sin(theta)*sin(phi);
			data.verts[x+2] = rho*cos(phi);

			for (int i = 3; i < 9; i+=3) {
				rho = getRandFloat(currRho-layerH, currRho+layerH);
				if (rho == 0.0f) {
					rho = 0.000001f;
				}
				newTheta = getRandFloat(theta-EPS, theta+EPS);
				newPhi = getRandFloat(phi-EPS, phi+EPS);

				data.verts[x+i] = rho*cos(newTheta)*sin(newPhi);
				data.verts[x+i+1] = rho*sin(newTheta)*sin(newPhi);
				data.verts[x+i+2] = rho*cos(newPhi);
			}
			calcNormal(&(data.verts[x]), &(data.normals[x]));
		}

		// gen verts n norms
		unsigned refTrig, refPt, refPtInd;
		float newRho, refTheta, refPhi, refX, refY, refZ;
		float pt[3];
		for (int x = numPolysPerLayer*9; x < numPolys*9; x+=numPolysPerLayer*9) {
			// updates
			currRho += layerH;

			// gen vars
			for (int y = 0; y < numPolysPerLayer*9; y+=9) {
				refTrig = getRandInt(0, numPolysPerLayer-1);
				refPtInd = getRandInt(0, 3);
				refPt = x - numPolysPerLayer*9 + refTrig*9 + refPtInd*3;

				findPointInTriangle(&(data.verts[x - numPolysPerLayer*9 + refTrig*9]), pt);
				newRho = sqrt(pow(pt[0], 2) + pow(pt[1], 2) + pow(pt[2], 2));
				refX = data.verts[refPt], refY = data.verts[refPt + 1], refZ = data.verts[refPt+ 2];
				if (refX == 0) {
					refTheta = refY > 0 ? _PI/2.0f : -_PI/2.0f;
				} else {
					refTheta = atan(refY/refX);
					if (refX < 0) {
						if (refY >= 0) {
							refTheta += _PI;
						} else {
							refTheta -= _PI;
						}
					}
				}
				if (refZ == 0 && sqrt(pow(refX, 2) + pow(refY, 2)) != 0) {
					refPhi = _PI/2.0f;
				} else {
					refPhi = atan(sqrt(pow(refX, 2) + pow(refY, 2))/refZ);
					if (refZ < 0) {
						refPhi += _PI;
					}
				}

				data.verts[x+y] = newRho*cos(refTheta)*sin(refPhi);
				data.verts[x+y+1] = newRho*sin(refTheta)*sin(refPhi);
				data.verts[x+y+2] = newRho*cos(refPhi);

				for (int i = 3; i < 9; i+=3) {
					rho = getRandFloat(currRho-layerH, currRho+layerH);
					newTheta = getRandFloat(refTheta-EPS, refTheta+EPS);
					newPhi = getRandFloat(refPhi-EPS, refPhi+EPS);

					data.verts[x+y+i] = rho*cos(newTheta)*sin(newPhi);
					data.verts[x+y+i+1] = rho*sin(newTheta)*sin(newPhi);
					data.verts[x+y+i+2] = rho*cos(newPhi);
				}
				calcNormal(&(data.verts[x+y]), &(data.normals[x+y]));
			}
		}

		// set colours
		for (int x = 0; x < numPolys*12; x+=12) {
			for (int y = 0; y < 12;	y+= 4) {
				for (int i = 0; i < 3; i++) {
					data.colours[x + y + i] = 1.0;
				}
				data.colours[x + y + 3] = getOpacityMoeslund();
			}
		}

		return data;
	}

	/**
	 * Generates a single snow particle using the experimental algorithm with configurable params.
	 * Used in experimentation.
	 * @param epsTheta - Allowed angular change in theta.
	 * @param epsPhi - Allowed angular change in phi.
	 * @param opacityCoeff - Opacity coefficient used with the density to determine opacity of triangles.
	 * @param epsOpacity - Allowed change (percentage) in the opacity of
	 * @param numLayersInp - Number of layers of the snow particle.
	 * @return A SnowGeneratorData object with the generated data for the snowflake.
	 */
	SnowGeneratorData generateSnowOnceExperimental(float epsTheta, float epsPhi, float opacityCoeff, float epsOpacity) {
		// set vars
		unsigned numPolys;
		float d, rho, currRho, theta, phi, newTheta, newPhi;
		unsigned numPolysPerLayer = getNumPolysPerLayerExperimental(), numLayers = getNumLayersExperimental();
		numPolys = numPolysPerLayer * numLayers;
		d = getRandFloat(0.5, 1.5)*diameter;
		float layerH = (d/2.0f)/numLayers;

		// set up data obj
		SnowGeneratorData data;
		data.numPolys = numPolys;
		data.verts = (float*) mallocGPU(numPolys*9*sizeof(float));
		data.normals = (float*) malloc(numPolys*9*sizeof(float));
		data.colours = (float*) malloc(numPolys*12*sizeof(float));
		data.snowflakeData = (SnowflakeData*) mallocGPU(sizeof(SnowflakeData));

		// gen first layer verts n norms
		currRho = layerH;
		for (int x = 0; x < numPolysPerLayer*9; x+=9) {
			// gen vars
			rho = getRandFloat(currRho-layerH, currRho+layerH);
			if (rho == 0.0f) {
				rho = 0.000001f;
			}
			theta = deg2rad(getRandFloat(0.0, 360.0));
			phi = deg2rad(getRandFloat(0.0, 180.0));

			data.verts[x] = rho*cos(theta)*sin(phi);
			data.verts[x+1] = rho*sin(theta)*sin(phi);
			data.verts[x+2] = rho*cos(phi);

			for (int i = 3; i < 9; i+=3) {
				rho = getRandFloat(currRho-layerH, currRho+layerH);
				if (rho == 0.0f) {
					rho = 0.000001f;
				}
				newTheta = getRandFloat(theta-epsTheta, theta+epsTheta);
				newPhi = getRandFloat(phi-epsPhi, phi+epsPhi);

				data.verts[x+i] = rho*cos(newTheta)*sin(newPhi);
				data.verts[x+i+1] = rho*sin(newTheta)*sin(newPhi);
				data.verts[x+i+2] = rho*cos(newPhi);
			}
			calcNormal(&(data.verts[x]), &(data.normals[x]));
		}

		// gen verts n norms
		unsigned refTrig, refPt, refPtInd;
		float newRho, refTheta, refPhi, refX, refY, refZ;
		float pt[3];
		for (int x = numPolysPerLayer*9; x < numPolys*9; x+=numPolysPerLayer*9) {
			// updates
			currRho += layerH;

			// gen vars
			for (int y = 0; y < numPolysPerLayer*9; y+=9) {
				refTrig = getRandInt(0, numPolysPerLayer-1);
				refPtInd = getRandInt(0, 3);
				refPt = x - numPolysPerLayer*9 + refTrig*9 + refPtInd*3;

				findPointInTriangle(&(data.verts[x - numPolysPerLayer*9 + refTrig*9]), pt);
				newRho = sqrt(pow(pt[0], 2) + pow(pt[1], 2) + pow(pt[2], 2));
				refX = data.verts[refPt], refY = data.verts[refPt + 1], refZ = data.verts[refPt+ 2];
				if (refX == 0) {
					refTheta = refY > 0 ? _PI/2.0f : -_PI/2.0f;
				} else {
					refTheta = atan(refY/refX);
					if (refX < 0) {
						if (refY >= 0) {
							refTheta += _PI;
						} else {
							refTheta -= _PI;
						}
					}
				}
				if (refZ == 0 && sqrt(pow(refX, 2) + pow(refY, 2)) != 0) {
					refPhi = _PI/2.0f;
				} else {
					refPhi = atan(sqrt(pow(refX, 2) + pow(refY, 2))/refZ);
					if (refZ < 0) {
						refPhi += _PI;
					}
				}

				data.verts[x+y] = newRho*cos(refTheta)*sin(refPhi);
				data.verts[x+y+1] = newRho*sin(refTheta)*sin(refPhi);
				data.verts[x+y+2] = newRho*cos(refPhi);

				for (int i = 3; i < 9; i+=3) {
					rho = getRandFloat(currRho-layerH, currRho+layerH);
					newTheta = getRandFloat(refTheta-epsTheta, refTheta+epsTheta);
					newPhi = getRandFloat(refPhi-epsPhi, refPhi+epsPhi);

					data.verts[x+y+i] = rho*cos(newTheta)*sin(newPhi);
					data.verts[x+y+i+1] = rho*sin(newTheta)*sin(newPhi);
					data.verts[x+y+i+2] = rho*cos(newPhi);
				}
				calcNormal(&(data.verts[x+y]), &(data.normals[x+y]));
			}
		}

		// set colours
		float opacity;
		for (int x = 0; x < numPolys*12; x+=12) {
			opacity = getOpacityExperimental(opacityCoeff);
			opacity = getRandFloat(opacity - opacity*epsOpacity, opacity + opacity*epsOpacity);
			for (int y = 0; y < 12;	y+= 4) {
				for (int i = 0; i < 3; i++) {
					data.colours[x + y + i] = 1.0;
				}
				data.colours[x + y + 3] = opacity;
			}
		}

		return data;
	}

	/**
	 * Generates a single snow particle using the experimental algorithm with experimentally determined params.
	 * @return A SnowGeneratorData object with the generated data for the snowflake.
	 */
	SnowGeneratorData generateSnowOnceExperimental() {
		return generateSnowOnceExperimental(isWet ? EPS_THETA_WET : EPS_THETA_DRY, isWet ? EPS_PHI_WET : EPS_PHI_DRY, OPACITY_COEFF, EPS_OPACITY);
	}

	/**
	 * Generates snow using the default algorithm (Experimental).
	 * @param numParticles - Number of particles to generate.
	 * @param extents - Extents of volume in which to generate the particles, where extents[0] is a pair for the x extent,
	 * extents[1] is a pair for the y extent, and extents[2] is a pair for the z extent. If numParticles = 1 this
	 * parameter has no effect and the snow particle is generated at the origin.
	 * @return A SnowGeneratorData object with the generated data for the snowflake(s).
	 */
	SnowGeneratorData generateSnow(unsigned numParticles, const float extents[3][2]) {
		return generateSnowExperimental(numParticles, extents);
	}

	/**
	 * Generates snow using the Moeslund algorithm.
	 * @param numParticles - Number of particles to generate.
	 * @param extents - Extents of volume in which to generate the particles, where extents[0] is a pair for the x extent,
	 * extents[1] is a pair for the y extent, and extents[2] is a pair for the z extent. If numParticles = 1 this
	 * parameter has no effect and the snow particle is generated at the origin.
	 * @return A SnowGeneratorData object with the generated data for the snowflake(s).
	 */
	SnowGeneratorData generateSnowMoeslund(unsigned numParticles, const float extents[3][2]) {
		return generateSnow(numParticles, extents, MOESLUND_ALG);
	}

	/**
	 * Generates snow using the experimental algorithm.
	 * @param numParticles - Number of particles to generate.
	 * @param extent - Extent of volume in which to generate the particles, where extent[0] is a pair for the x extent,
	 * extent[1] is a pair for the y extent, and extent[2] is a pair for the z extent. If numParticles = 1 this
	 * parameter has no effect and the snow particle is generated at the origin.
	 * @return A SnowGeneratorData object with the generated data for the snowflake(s).
	 */
	SnowGeneratorData generateSnowExperimental(unsigned numParticles, const float extent[3][2]) {
		return generateSnow(numParticles, extent, EXPERIMENTAL_ALG);
	}
};

#endif