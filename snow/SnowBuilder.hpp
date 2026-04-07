/**
 * SnowBuilder.hpp - Contains the class which implements the generation of
 * a single instance of snow.
 *
 * @author Mars Semenova
 * @date March 30, 2026
 */
#ifndef SNOWBUILDER_HPP
#define SNOWBUILDER_HPP

#include "SnowConstants.hpp"
#include "SnowBuilderData.hpp"
#include "../util/Util.hpp"

class SnowBuilder {
private:
	// params
	GLfloat diameter; // fluctuates by 50%, in cm
	GLfloat density; // affects amt of layers + amt of polys + alpha
	bool isWet;

	// fully random implementation functions
	GLuint getNumLayersRand() {
		return getRandInt(3, 6);
	}

	GLuint getNumPolysRand() {
		return getRandInt(10, 40)*getNumLayersRand();
	}

	GLfloat getAlphaRand() {
		return getRandFloat(0.0, 1.0);
	}

	// Moeslund implementation
	GLuint getNumLayersMoeslund() {
		return 6; // TODO: ???
	}

	GLuint getNumPolysPerLayerMoeslund() {
		return isWet ? 40 : 10;
	}

	GLfloat getAlphaMoeslund() {
		return 0.5; // const guessed from imgs in paper
	}

	// Zou implementation

	// improvement?

	SnowBuilderData generateSnow(GLuint n, const GLfloat extent[3][2], GLuint alg) {
		SnowBuilderData data;
		SnowBuilderData dataReturned[n];
		GLuint numEntries = 0;
		GLfloat xPos, yPos, zPos;
		for (int x = 0; x < n; x++) {
			xPos = getRandFloat(extent[0][0], extent[0][1]);
			yPos = getRandFloat(extent[1][0], extent[1][1]);
			zPos = getRandFloat(extent[2][0], extent[2][1]);
			if (alg == RAND_ALG) {
				dataReturned[x] = generateSnowOnceRand();
			}
			if (alg == MOESLUND_ALG) {
				dataReturned[x] = generateSnowOnceMoeslund();
			}
			if (alg == ZOU_ALG) {
				dataReturned[x] = generateSnowOnceZou();
			}
			if (alg == EXPERIMENTAL_ALG) {
				dataReturned[x] = generateSnowOnceExperimental();
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
		GLuint offset = 0;
		for (int x = 0; x < n; x++) {
			for (int i = 0; i < dataReturned[x].numPolys*9; i++) {
				data.verts[i + offset] = dataReturned[x].verts[i];
				data.normals[i + offset] = dataReturned[x].normals[i];
			}

			offset += dataReturned[x].numPolys*9;
		}
		offset = 0;
		for (int x = 0; x < n; x++) {
			for (int i = 0; i < dataReturned[x].numPolys*12; i++) {
				data.colours[i + offset] = dataReturned[x].colours[i];
			}

			offset += dataReturned[x].numPolys*12;
		}

		return data;
	}

public:
	SnowBuilder(GLfloat diameter, GLfloat density, bool isWet) : diameter(diameter), density(density), isWet(isWet) {
	}
	SnowBuilder() : SnowBuilder(0.015*pow(abs(DEFAULT_TEMP), -0.35), WET_HUMIDITY_CONST/(0.015*pow(abs(DEFAULT_TEMP), -0.35)), true) {};

	SnowBuilderData generateSnowOnce() { return generateSnowOnceMoeslund(); }

	SnowBuilderData generateSnowOnceRand() {
		// set vars
		GLuint numPolys;
		GLfloat d, rho, theta, phi, newTheta, newPhi;
		numPolys = getNumPolysRand();

		// set up data obj
		SnowBuilderData data;
		data.numPolys = numPolys;
		data.verts = (GLfloat*) malloc(numPolys*9*sizeof(GLfloat));
		data.normals = (GLfloat*) malloc(numPolys*9*sizeof(GLfloat));
		data.colours = (GLfloat*) malloc(numPolys*12*sizeof(GLfloat));

		// gen verts n norms
		for (int x = 0; x < numPolys*9; x+=9) {
			// gen vars
			d = getRandFloat(0.5, 1.5)*diameter;
			rho = d/2.0f;
			theta = deg2rad(getRandFloat(0.0, 360.0));
			phi = deg2rad(getRandFloat(-90.0, 90.0));

			data.verts[x] = rho*cos(theta)*sin(phi);
			data.verts[x+1] = rho*sin(theta)*sin(phi);
			data.verts[x+2] = rho*cos(phi);

			for (int i = 3; i < 9; i+=3) {
				newTheta = getRandFloat(theta-EPS, theta+EPS);
				newPhi = getRandFloat(phi-EPS, phi+EPS);

				data.verts[x+i] = rho*cos(newTheta)*sin(newPhi);
				data.verts[x+i+1] = rho*sin(newTheta)*sin(newPhi);
				data.verts[x+i+2] = rho*cos(newPhi);
			}
			calcNormal(&(data.verts[x]), &(data.normals[x]));
		}

		// set colours
		for (int x = 0; x < numPolys*12; x+=4) {
			for (int i = 0; i < 3; i++) {
				data.colours[x + i] = 1.0;
			}
			data.colours[x + 3] = getAlphaRand();
		}

		return data;
	}

	SnowBuilderData generateSnowOnceMoeslund() {
		// set vars
		GLuint numPolys;
		GLfloat d, rho, currRho, theta, phi, newTheta, newPhi;
		GLuint numPolysPerLayer = getNumPolysPerLayerMoeslund(), numLayers = getNumLayersMoeslund();
		numPolys = numPolysPerLayer * numLayers;
		d = getRandFloat(0.5, 1.5)*diameter;
		GLfloat layerH = (d/2.0f)/numLayers;

		// set up data obj
		SnowBuilderData data;
		data.numPolys = numPolys;
		data.verts = (GLfloat*) malloc(numPolys*9*sizeof(GLfloat));
		data.normals = (GLfloat*) malloc(numPolys*9*sizeof(GLfloat));
		data.colours = (GLfloat*) malloc(numPolys*12*sizeof(GLfloat));

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

	SnowBuilderData generateSnowOnceZou() { // TODO
		SnowBuilderData data;
		SnowBuilderData moeslund = generateSnowOnceMoeslund();

		// init linked list

		//

		return moeslund;
	}

	SnowBuilderData generateSnowOnceExperimental() { // TODO

	}

	SnowBuilderData generateSnow(GLuint n, const GLfloat extent[3][2]) {
		return generateSnowMoeslund(n, extent);
	}

	SnowBuilderData generateSnowRand(GLuint n, const GLfloat extent[3][2]) {
		return generateSnow(n, extent, RAND_ALG);
	}

	SnowBuilderData generateSnowMoeslund(GLuint n, const GLfloat extent[3][2]) {
		return generateSnow(n, extent, MOESLUND_ALG);
	}

	SnowBuilderData generateSnowZou(GLuint n, const GLfloat extent[3][2]) {
		return generateSnow(n, extent, ZOU_ALG);
	}

	SnowBuilderData generateSnowExperimental(GLuint n, const GLfloat extent[3][2]) {
		return generateSnow(n, extent, EXPERIMENTAL_ALG);
	}
};

#endif