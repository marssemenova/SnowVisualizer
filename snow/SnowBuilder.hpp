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
	GLfloat diameter; // fluctuates by 50%
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
		return isWet ? 6 : 3; // TODO: ???
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
				printf("%f %f\n", data.colours[i + offset], dataReturned[x].colours[i]);
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
		GLfloat d, phi, theta, rho, newPhi, newTheta;
		numPolys = getNumPolysRand();

		// set up data obj
		SnowBuilderData data;
		numPolys = 40;
		data.numPolys = numPolys;
		data.verts = (GLfloat*) malloc(numPolys*9*sizeof(GLfloat));
		data.normals = (GLfloat*) malloc(numPolys*9*sizeof(GLfloat));
		data.colours = (GLfloat*) malloc(numPolys*12*sizeof(GLfloat));

		// gen verts n norms
		for (int x = 0; x < numPolys*9; x+=9) {
			// gen vars
			d = getRandFloat(0.5, 1.5)*diameter;
			rho = d/2.0f;
			phi = getRandFloat(-90.0, 90.0);
			theta = getRandFloat(0.0, 360.0);

			data.verts[x] = rho*cos(theta)*sin(phi);
			data.verts[x+1] = rho*sin(theta)*sin(phi);
			data.verts[x+2] = rho*cos(theta);

			for (int i = 3; i < 9; i+=3) {
				newPhi = getRandFloat(phi-EPS, phi+EPS);
				newTheta = getRandFloat(theta-EPS, theta+EPS);
				data.verts[x+i] = rho*cos(newTheta)*sin(newPhi);
				data.verts[x+i+1] = rho*sin(newTheta)*sin(newPhi);
				data.verts[x+i+2] = rho*cos(newTheta);
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
		GLfloat d, currD, phi, theta, rho, newPhi, newTheta;
		GLuint numPolysPerLayer = getNumPolysPerLayerMoeslund();
		numPolys = numPolysPerLayer * getNumLayersMoeslund();
		d = getRandFloat(0.5, 1.5)*diameter;
		GLfloat layerH = d/(numPolysPerLayer+1);

		// set up data obj
		SnowBuilderData data;
		numPolys = 40;
		data.numPolys = numPolys;
		data.verts = (GLfloat*) malloc(numPolys*9*sizeof(GLfloat));
		data.normals = (GLfloat*) malloc(numPolys*9*sizeof(GLfloat));
		data.colours = (GLfloat*) malloc(numPolys*12*sizeof(GLfloat));

		// gen first layer verts n norms
		currD = layerH;
		for (int x = 0; x < numPolysPerLayer*9; x+=9) {
			// gen vars
			rho = currD/2.0f;
			phi = getRandFloat(-90.0, 90.0);
			theta = getRandFloat(0.0, 360.0);

			data.verts[x] = rho*cos(theta)*sin(phi);
			data.verts[x+1] = rho*sin(theta)*sin(phi);
			data.verts[x+2] = rho*cos(theta);

			for (int i = 3; i < 9; i+=3) {
				newPhi = getRandFloat(phi-EPS, phi+EPS);
				newTheta = getRandFloat(theta-EPS, theta+EPS);
				data.verts[x+i] = rho*cos(newTheta)*sin(newPhi);
				data.verts[x+i+1] = rho*sin(newTheta)*sin(newPhi);
				data.verts[x+i+2] = rho*cos(newTheta);
			}
			calcNormal(&(data.verts[x]), &(data.normals[x]));
		}

		// gen verts n norms
		for (int x = 0; x < numPolys*9; x+=9) {
			// gen vars
			rho = d/2.0f;
			phi = getRandFloat(-90.0, 90.0);
			theta = getRandFloat(0.0, 360.0);

			data.verts[x] = rho*cos(theta)*sin(phi);
			data.verts[x+1] = rho*sin(theta)*sin(phi);
			data.verts[x+2] = rho*cos(theta);

			for (int i = 3; i < 9; i+=3) {
				newPhi = getRandFloat(phi-EPS, phi+EPS);
				newTheta = getRandFloat(theta-EPS, theta+EPS);
				data.verts[x+i] = rho*cos(newTheta)*sin(newPhi);
				data.verts[x+i+1] = rho*sin(newTheta)*sin(newPhi);
				data.verts[x+i+2] = rho*cos(newTheta);
			}
			calcNormal(&(data.verts[x]), &(data.normals[x]));
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