/**
 * SnowGenerator.hpp - Contains the class which implements the snow generation logic.
 *
 * @author Mars Semenova
 * @date March 30, 2026
 */
#ifndef SNOWGENERATOR_HPP
#define SNOWGENERATOR_HPP

#include "SnowConstants.hpp"
#include "SnowBuilder.hpp"

class SnowGenerator {
private:
	// params
	GLuint numParticles;
	GLfloat temp; // C
	GLfloat extent[3][2]; // x range, y range, z range
	GLfloat density; // calced
	GLfloat diameter; // calced
	SnowBuilder snowBuilder;
	SnowBuilderData data;

	// render params
	GLuint programID;

	GLuint MVPID;
	GLuint MID;
	GLuint VID;
	GLuint LightPosID;
	GLuint alphaID;

	GLuint vertexArrayID;
	GLuint vertBuffer;
	GLuint normalBuffer;
	GLuint colorBuffer;

public:
	SnowGenerator(GLuint numParticles, const GLfloat extentInp[3][2], GLfloat temp, bool isWet) : numParticles(numParticles), temp(temp) {
		// set vars
		memcpy(extent, extentInp, 6*sizeof(GLfloat));
		if (temp <= -0.061) {
			diameter = 0.015*pow(abs(temp), -0.35);
		} else {
			diameter = 0.04;
		}
		density = isWet ? WET_HUMIDITY_CONST/diameter : DRY_HUMIDITY_CONST/diameter;

		// create builder
		snowBuilder = SnowBuilder(diameter, density, extent, isWet);

		// load shaders
		programID = LoadShaders( "PhongVertexShader.vertexshader", "PhongFragmentShader.fragmentshader");
		MVPID = glGetUniformLocation(programID, "MVP");
		MID = glGetUniformLocation(programID, "M");
		VID = glGetUniformLocation(programID, "V");
		LightPosID = glGetUniformLocation(programID, "LightPosition_worldspace");
		alphaID = glGetUniformLocation(programID, "alpha");
		glUseProgram(programID);

		setupVAO();
	}

	SnowGenerator() : SnowGenerator(DEFAULT_SNOW_COUNT, DEFAULT_EXTENT, DEFAULT_TEMP, true) {};

	void setupVAO() {
		glGenVertexArrays(1, &vertexArrayID);
		glBindVertexArray(vertexArrayID);

		// gen snow
		data = snowBuilder.generateSnowOnce();

		// vertices
		glGenBuffers(1, &vertBuffer);
		glBindBuffer(GL_ARRAY_BUFFER, vertBuffer);
		glBufferData(GL_ARRAY_BUFFER, sizeof(GLfloat) * data.numPolys * 9, data.verts, GL_STATIC_DRAW);
		glEnableVertexAttribArray(0);
		glVertexAttribPointer(
			0,                                // attribute. No particular reason for 1, but must match the layout in the shader.
			3,                                // size
			GL_FLOAT,                         // type
			GL_FALSE,                         // normalized?
			0,                                // stride
			(void*) 0                        // array buffer offset
		);

		// normals
		glGenBuffers(1, &normalBuffer);
		glBindBuffer(GL_ARRAY_BUFFER, normalBuffer);
		glBufferData(GL_ARRAY_BUFFER, sizeof(GLfloat) * data.numPolys * 9, data.normals, GL_STATIC_DRAW);
		glEnableVertexAttribArray(1);
		glVertexAttribPointer(
			1,                                // attribute. No particular reason for 1, but must match the layout in the shader.
			3,                                // size
			GL_FLOAT,                         // type
			GL_TRUE,                         // normalized?
			0,                                // stride
			(void*) 0                        // array buffer offset
		);

		// colours
		glGenBuffers(1, &colorBuffer);
		glBindBuffer(GL_ARRAY_BUFFER, colorBuffer);
		glBufferData(GL_ARRAY_BUFFER, sizeof(GLfloat) * data.numPolys * 12, data.colours, GL_STATIC_DRAW);
		glEnableVertexAttribArray(2);
		glVertexAttribPointer(
			2,                                // attribute. No particular reason for 1, but must match the layout in the shader.
			4,                                // size
			GL_FLOAT,                         // type
			GL_FALSE,                         // normalized?
			0,                                // stride
			(void*) 0                        // array buffer offset
		);

		glBindBuffer(GL_ARRAY_BUFFER, 0);
		glBindVertexArray(0);
	}

	void draw(glm::vec3 lightPos, glm::mat4 M, glm::mat4 V, glm::mat4 P) {
		glm::mat4 MVP = P*V*M;
		glBindVertexArray(vertexArrayID);
		glUseProgram(programID);

		glUniformMatrix4fv(MVPID, 1, GL_FALSE, &MVP[0][0]);
		glUniformMatrix4fv(MID, 1, GL_FALSE, &M[0][0]);
		glUniformMatrix4fv(VID, 1, GL_FALSE, &V[0][0]);
		glUniform3f(LightPosID, lightPos.x, lightPos.y, lightPos.z);
		glUniform1f(alphaID, 2); // TODO

		glPointSize(3.0f);
		glDrawArrays(GL_TRIANGLES, 0, data.numPolys*3);
		glPointSize(1.0f);

		glUseProgram(0);
		glBindVertexArray(0);
	}
};

#endif
