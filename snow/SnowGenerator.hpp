/**
 * SnowGenerator.hpp - Contains the class which implements the generation of
 * a single instance of snow.
 *
 * @author Mars Semenova
 * @date March 30, 2026
 */

#ifndef SNOWGENERATOR_HPP
#define SNOWGENERATOR_HPP

#include "SnowConstants.hpp"
#include "SnowGeneratorData.hpp"
#include "../util/Util.hpp"

class SnowGenerator {
private:
	// params
	GLfloat diameter; // calced in cm based on temp, fluctuates by 50%, in cm
	GLfloat density; // calced based on temp, affects amt of layers + amt of polys + alpha
	bool isWet;
	GLfloat alpha;

	// Moeslund implementation functions
	GLuint getNumLayersMoeslund() {
		return 6; // TODO: guessed
	}

	GLuint getNumPolysPerLayerMoeslund() {
		return isWet ? 40 : 10; // TODO: for Zou says only needs to be 1/4 ot the other
	}

	GLfloat getAlphaMoeslund() {
		return 0.5; // TODO: const guessed from imgs in paper
	}

	// improvement(?) functions

	/**
	 * Helper function which generates snow using the specified algorithm
	 * @param numParticles - Number of particles to generate.
	 * @param extent - Extent of volume in which to generate the particles, where extent[0] is a pair for the x extent,
	 * extent[1] is a pair for the y extent, and extent[2] is a pair for the z extent. If numParticles = 1 this
	 * parameter has no effect and the snow particle is generated at the origin.
	 * @param alg - Which algorithm to use for the snow particle generation. Either the Moeslund algorithm (1) or the
	 * experimental algorithm (2). Constants for these are defined in SnowConstants.hpp (MOESLUND_ALG and EXPERIMENTAL_ALG, respectively).
	 * @return A SnowGeneratorData object with the generated data for the snowflake(s).
	 */
	SnowGeneratorData generateSnow(GLuint numParticles, const GLfloat extent[3][2], GLuint alg) {
		if (numParticles == 1) {
			if (alg == EXPERIMENTAL_ALG) {
				return generateSnowOnceExperimental();
			}
			return generateSnowOnceMoeslund();
		}
		SnowGeneratorData data;
		SnowGeneratorData dataReturned[numParticles];
		GLuint numEntries = 0;
		GLfloat xPos, yPos, zPos;
		for (int x = 0; x < numParticles; x++) {
			xPos = getRandFloat(extent[0][0], extent[0][1]);
			yPos = getRandFloat(extent[1][0], extent[1][1]);
			zPos = getRandFloat(extent[2][0], extent[2][1]);
			if (alg == EXPERIMENTAL_ALG) {
				dataReturned[x] = generateSnowOnceExperimental();
			} else {
				dataReturned[x] = generateSnowOnceMoeslund();
			}
			numEntries += dataReturned[x].numPolys;

			// update pos
			for (int i = 0; i < dataReturned[x].numPolys*9; i+=3) {
				dataReturned[x].verts[i] += xPos;
				dataReturned[x].verts[i+1] += yPos;
				dataReturned[x].verts[i+2] += zPos;
			}
		}

		data.numPolys = numEntries;
		data.verts = (GLfloat*) malloc(numEntries*9*sizeof(GLfloat));
		data.normals = (GLfloat*) malloc(numEntries*9*sizeof(GLfloat));
		data.colours = (GLfloat*) malloc(numEntries*12*sizeof(GLfloat));
		data.alpha = dataReturned[0].alpha;
		GLuint offset = 0;
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

		return data;
	}

public:
	/**
	 * Constructor for SnowGenerator.
	 * @param temp - Temperature of the simulation.
	 */
	SnowGenerator(GLfloat temp) {
		if (temp <= DIAMETER_THRESH) {
			diameter = 0.015*pow(abs(temp), -0.35);
		} else {
			diameter = 0.04;
		}
		isWet = temp >= SNOW_STATE_THRESH;
		alpha = isWet ? 3.0 : 2.0; // TODO: experiment to get good values
		density = isWet ? WET_HUMIDITY_CONST/diameter : DRY_HUMIDITY_CONST/diameter;
		diameter *= 100.0; // to cm
	}

	/**
	 * Default constructor for SnowGenerator.
	 */
	SnowGenerator() : SnowGenerator(DEFAULT_TEMP) {};

	/**
	 * Default function to generate a single snow particle. Uses the Moeslund algorithm.
	 * @return A SnowGeneratorData object with the generated data for the snowflake.
	 */
	SnowGeneratorData generateSnowOnce() { return generateSnowOnceMoeslund(); }

	/**
	 * Generates a single snow particle using the Moeslund algorithm.
	 * @return A SnowGeneratorData object with the generated data for the snowflake.
	 */
	SnowGeneratorData generateSnowOnceMoeslund() {
		// set vars
		GLuint numPolys;
		GLfloat d, rho, currRho, theta, phi, newTheta, newPhi;
		GLuint numPolysPerLayer = getNumPolysPerLayerMoeslund(), numLayers = getNumLayersMoeslund();
		numPolys = numPolysPerLayer * numLayers;
		d = getRandFloat(0.5, 1.5)*diameter;
		GLfloat layerH = (d/2.0f)/numLayers;

		// set up data obj
		SnowGeneratorData data;
		data.numPolys = numPolys;
		data.verts = (GLfloat*) malloc(numPolys*9*sizeof(GLfloat));
		data.normals = (GLfloat*) malloc(numPolys*9*sizeof(GLfloat));
		data.colours = (GLfloat*) malloc(numPolys*12*sizeof(GLfloat));
		data.alpha = alpha;

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
		GLuint refTrig, refPt, refPtInd;
		GLfloat newRho, refTheta, refPhi, refX, refY, refZ;
		GLfloat pt[3];
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
		for (int x = 0; x < numPolys*12; x+=4) {
			for (int i = 0; i < 3; i++) {
				data.colours[x + i] = 1.0;
			}
			data.colours[x + 3] = getAlphaMoeslund();
		}

		return data;
	}

	/**
	 * Generates a single snow particle using the experimental algorithm.
	 * @return A SnowGeneratorData object with the generated data for the snowflake.
	 */
	SnowGeneratorData generateSnowOnceExperimental() {

	}

	/**
	 * Generates snow using the default algorithm (Moeslund).
	 * @param numParticles - Number of particles to generate.
	 * @param extent - Extent of volume in which to generate the particles, where extent[0] is a pair for the x extent,
	 * extent[1] is a pair for the y extent, and extent[2] is a pair for the z extent. If numParticles = 1 this
	 * parameter has no effect and the snow particle is generated at the origin.
	 * @return A SnowGeneratorData object with the generated data for the snowflake(s).
	 */
	SnowGeneratorData generateSnow(GLuint numParticles, const GLfloat extent[3][2]) {
		return generateSnowMoeslund(numParticles, extent);
	}

	/**
	 * Generates snow using the Moeslund algorithm.
	 * @param numParticles - Number of particles to generate.
	 * @param extent - Extent of volume in which to generate the particles, where extent[0] is a pair for the x extent,
	 * extent[1] is a pair for the y extent, and extent[2] is a pair for the z extent. If numParticles = 1 this
	 * parameter has no effect and the snow particle is generated at the origin.
	 * @return A SnowGeneratorData object with the generated data for the snowflake(s).
	 */
	SnowGeneratorData generateSnowMoeslund(GLuint numParticles, const GLfloat extent[3][2]) {
		return generateSnow(numParticles, extent, MOESLUND_ALG);
	}

	/**
	 * Generates snow using the experimental algorithm.
	 * @param numParticles - Number of particles to generate.
	 * @param extent - Extent of volume in which to generate the particles, where extent[0] is a pair for the x extent,
	 * extent[1] is a pair for the y extent, and extent[2] is a pair for the z extent. If numParticles = 1 this
	 * parameter has no effect and the snow particle is generated at the origin.
	 * @return A SnowGeneratorData object with the generated data for the snowflake(s).
	 */
	SnowGeneratorData generateSnowExperimental(GLuint numParticles, const GLfloat extent[3][2]) {
		return generateSnow(numParticles, extent, EXPERIMENTAL_ALG);
	}
};

#endif