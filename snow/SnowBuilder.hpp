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

class SnowBuilder {
private:
	// params
	GLfloat diameter; // fluctuates by 50%
	GLfloat density; // affects amt of polys + alpha
	GLfloat extent[2][2]; // x range, y range

public:
	SnowBuilder(GLfloat diameter, GLfloat density, const GLfloat extentInp[2][2]) : diameter(diameter), density(density) {
		memcpy(extent, extentInp, 4*sizeof(GLfloat));
	}
	SnowBuilder() : SnowBuilder(0.015*pow(abs(DEFAULT_TEMP), -0.35), WET_HUMIDITY_CONST/(0.015*pow(abs(DEFAULT_TEMP), -0.35)), DEFAULT_EXTENT) {};

	void generateSnowOnce(GLfloat vertsRef[], GLfloat normalsRef[], GLfloat coloursRef[], GLuint offset) { // TODO: make priv
		// del
		glm::vec3 origin = glm::vec3(0.0f, 0.0f, 0.0f); // TODO: del
		glm::vec3 extents = glm::vec3(5.0f, 5.0f, 5.0f); // TODO: del
		static const GLfloat tempVerts[] = {
			origin.x, origin.y, origin.z,
			origin.x + extents.x, origin.y, origin.z,
			origin.x, origin.y, origin.z,
			origin.x, origin.y + extents.y, origin.z,
			origin.x, origin.y, origin.z,
			origin.x, origin.y, origin.z + extents.z,
		};
		static const GLfloat tempColours[] = {
			1.0f, 0.0f, 0.0f,
			1.0f, 0.0f, 0.0f,
			0.0f, 1.0f, 0.0f,
			0.0f, 1.0f, 0.0f,
			0.0f, 0.0f, 1.0f,
			0.0f, 0.0f, 1.0f,
		};

		// gen verts
		for (int x = offset; x < offset + 18; x++) {
			vertsRef[x] = tempVerts[x];
		}

		// set norms
		for (int x = offset; x < offset + 18; x++) {
			normalsRef[x] = 0;
		}

		// set colours
		for (int x = offset; x < offset + 18; x++) {
			coloursRef[x] = tempColours[x];
		}
	}

	void generateSnowOnce(GLfloat vertsRef[], GLfloat normalsRef[], GLfloat coloursRef[]) {
		generateSnow(vertsRef, normalsRef, coloursRef, 1);
	}

	void generateSnow(GLfloat vertsRef[], GLfloat normalsRef[], GLfloat coloursRef[], GLuint n) {
		for (int x = 0; x < n; x++) {
			generateSnowOnce(vertsRef, normalsRef, coloursRef, 9*x);
		}
	}
};

#endif