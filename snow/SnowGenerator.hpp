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
	GLfloat alpha;
	SnowBuilder snowBuilder;
	SnowBuilderData data;
	GLuint whichAlg;

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
	/**
	 * Constructor for SnowGenerator.
	 * @param numParticles - Number of snow particles to generate.
	 * @param extentInp - Extent of volume in which to generate the particles, where extentInp[0] is a pair for the x extent,
	 * extentInp[1] is a pair for the y extent, and extentInp[2] is a pair for the z extent. If numParticles = 1 this
	 * parameter has no effect and the snow particle is generated at the origin.
	 * @param temp - Temperature of the simulation.
	 * @param whichAlg - Which algorithm to use for the snow particle generation. Either the Moeslund algorithm (1) or the
	 * experimental algorithm (2). Constants for these are defined in SnowConstants.hpp (MOESLUND_ALG and EXPERIMENTAL_ALG, respectively).
	 */
	SnowGenerator(GLuint numParticles, const GLfloat extentInp[3][2], GLfloat temp, GLuint whichAlg) : numParticles(numParticles), temp(temp), whichAlg(whichAlg) {
		// set vars
		memcpy(extent, extentInp, 6*sizeof(GLfloat));
		bool isWet = temp >= SNOW_STATE_THRESH;
		alpha = isWet ? 3.0 : 2.0; // TODO: experiment to get good values

		// create builder
		snowBuilder = SnowBuilder(temp);

		// load shaders
		programID = LoadShaders( "shaders/PhongVertexShader.vertexshader", "shaders/PhongFragmentShader.fragmentshader");
		MVPID = glGetUniformLocation(programID, "MVP");
		MID = glGetUniformLocation(programID, "M");
		VID = glGetUniformLocation(programID, "V");
		LightPosID = glGetUniformLocation(programID, "LightPosition_worldspace");
		alphaID = glGetUniformLocation(programID, "alpha");
		glUseProgram(programID);

		setupVAO();
	}

	/**
	 * Default constructor for SnowGenerator.
	 */
	SnowGenerator() : SnowGenerator(DEFAULT_SNOW_COUNT, DEFAULT_EXTENT, DEFAULT_TEMP, MOESLUND_ALG) {};

	/**
	 * Setup VAOs.
	 */
	void setupVAO() {
		glGenVertexArrays(1, &vertexArrayID);
		glBindVertexArray(vertexArrayID);

		// gen snow
		if (whichAlg == EXPERIMENTAL_ALG) {
			data = snowBuilder.generateSnowExperimental(numParticles, extent);
		} else {
			data = snowBuilder.generateSnowMoeslund(numParticles, extent);
		}

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

	/**
	 * Draw function.
	 * @param lightPos - Position of the light source.
	 * @param M - Model (transformation) matrix.
	 * @param V - View matrix.
	 * @param P - Projection matrix.
	 */
	void draw(glm::vec3 lightPos, glm::mat4 M, glm::mat4 V, glm::mat4 P) {
		glm::mat4 MVP = P*V*M;
		glBindVertexArray(vertexArrayID);
		glUseProgram(programID);

		glUniformMatrix4fv(MVPID, 1, GL_FALSE, &MVP[0][0]);
		glUniformMatrix4fv(MID, 1, GL_FALSE, &M[0][0]);
		glUniformMatrix4fv(VID, 1, GL_FALSE, &V[0][0]);
		glUniform3f(LightPosID, lightPos.x, lightPos.y, lightPos.z);
		glUniform1f(alphaID, alpha);

		glPointSize(3.0f);
		glDrawArrays(GL_TRIANGLES, 0, data.numPolys*3);
		glPointSize(1.0f);

		glUseProgram(0);
		glBindVertexArray(0);
	}
};

#endif
