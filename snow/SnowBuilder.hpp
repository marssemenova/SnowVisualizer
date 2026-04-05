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
	GLfloat extent[3][2]; // x range, y range, z range
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
	GLuint getNumPolysPerLayerMoeslund() {
		return isWet ? 40 : 10;
	}

	// Zou implementation


	// improvement?


	SnowBuilderData generateSnowOnceRand() {
		// set vars
		GLuint numPolys;
		GLfloat xPos, yPos, zPos, d, phi, theta, rho, newPhi, newTheta;
		xPos = getRandFloat(extent[0][0], extent[0][1]);
		yPos = getRandFloat(extent[1][0], extent[1][1]);
		zPos = getRandFloat(extent[2][0], extent[2][1]);
		numPolys = getNumPolysRand();
		printf("pos: (%f, %f, %f), num polys: %d, d: %f\n", xPos, yPos, zPos, numPolys, diameter);

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
			printf("d: %f %f\n", d, d/diameter);
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
			printf("verts\n");
			for (int i = x; i < x+9; i+=3) {
				printf("(%f, %f, %f) ", data.verts[i], data.verts[i+1], data.verts[i+2]);
			}
			printf("\nnorms\n");
			for (int i = x; i < x+9; i+=3) {
				printf("(%f, %f, %f) ", data.normals[i], data.normals[i+1], data.normals[i+2]);
			}
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

public:
	SnowBuilder(GLfloat diameter, GLfloat density, const GLfloat extentInp[3][2], bool isWet) : diameter(diameter), density(density), isWet(isWet) {
		memcpy(extent, extentInp, 6*sizeof(GLfloat));
	}
	SnowBuilder() : SnowBuilder(0.015*pow(abs(DEFAULT_TEMP), -0.35), WET_HUMIDITY_CONST/(0.015*pow(abs(DEFAULT_TEMP), -0.35)), DEFAULT_EXTENT, true) {};

	SnowBuilderData generateSnowOnce() {
		return generateSnowOnceRand();
	}

	SnowBuilderData generateSnow(GLuint n) {
		SnowBuilderData data;
		SnowBuilderData dataReturned[n];
		GLuint numEntries = 0;
		for (int x = 0; x < n; x++) {
			dataReturned[x] = generateSnowOnce();
			numEntries += dataReturned[x].numPolys;
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
				data.colours[i + offset] = dataReturned[x].colours[i]; // TODO: refactor
				printf("%f %f\n", data.colours[i + offset], dataReturned[x].colours[i]);
			}

			offset += dataReturned[x].numPolys*12;
		}

		return data;
	}
};

#endif