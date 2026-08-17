/**
 * SnowRenderer.hpp - Contains the class which renders the snow generation.
 *
 * @author Mars Semenova
 */

#ifndef SNOWRENDERER_HPP
#define SNOWRENDERER_HPP

#include "../util/ImportGL.hpp"

#include "SnowGenerator.hpp"
#include "cuda/gpu.hpp"

class SnowRenderer {
private:
	// params
	unsigned numParticles;
	float temp; // C
	float extents[3][2]; // x range, y range, z range
	SnowGenerator snowGenerator;
	SnowGeneratorData data;
	unsigned whichAlg;
	float windVel;
	unsigned latticeRes;

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
	 * Constructor for SnowRenderer.
	 * @param numParticles - Number of snow particles to generate.
	 * @param extentsInp - Extents of volume in which to generate the particles, where extentsInp[0] is a pair for the x extent,
	 * extentsInp[1] is a pair for the y extent, and extentsInp[2] is a pair for the z extent. If numParticles = 1 this
	 * parameter has no effect and the snow particle is generated at the origin.
	 * @param temp - Temperature of the simulation.
	 * @param whichAlg - Which algorithm to use for the snow particle generation. Either the Moeslund algorithm (1) or the
	 * experimental algorithm (2). Constants for these are defined in SnowConstants.hpp (MOESLUND_ALG and EXPERIMENTAL_ALG, respectively).
	 * @param windVel - Macroscopic wind velocity of the wind field.
	 * @param latticeRes - Lattice resolution used in the LBM for the wind field.
	 */
	SnowRenderer(unsigned numParticles, const float extentsInp[3][2], float temp, unsigned whichAlg, float windVelInp, unsigned latticeRes) : numParticles(numParticles), temp(temp), whichAlg(whichAlg), latticeRes(latticeRes) {
		// convert wind vel from km to m
		windVel = (windVelInp / 3.6f); // km/h > m/s

		// copy + clamp extents
		memcpy(extents, extentsInp, 6*sizeof(float));
		float xExt = abs(extents[0][1]-extents[0][0]);
		float yExt = abs(extents[1][0]-extents[1][1]);
		float zExt = abs(extents[2][0]-extents[2][1]);
		if (xExt != yExt || xExt != zExt) {
			float minExtent = std::min(std::min(xExt, yExt), zExt);
			if (xExt != minExtent) {
				if (extents[0][0] < extents[0][1]) {
					extents[0][1] = extents[0][0] + minExtent;
				} else {
					extents[0][0] = extents[0][1] + minExtent;
				}
			}
			if (yExt != minExtent) {
				if (extents[1][0] < extents[1][1]) {
					extents[1][1] = extents[1][0] + minExtent;
				} else {
					extents[1][0] = extents[1][1] + minExtent;
				}
			}
			if (zExt != minExtent) {
				if (extents[2][0] < extents[2][1]) {
					extents[2][1] = extents[2][0] + minExtent;
				} else {
					extents[2][0] = extents[2][1] + minExtent;
				}
			}
		}

		// create generator
		snowGenerator = SnowGenerator(temp);

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
	 * Constructor for SnowRenderer with the default algorithm (EXPERIMENTAL_ALG).
	 * @param numParticles - Number of snow particles to generate.
	 * @param extentsInp - Extents of volume in which to generate the particles, where extentsInp[0] is a pair for the x extent,
	 * extentsInp[1] is a pair for the y extent, and extentsInp[2] is a pair for the z extent. If numParticles = 1 this
	 * parameter has no effect and the snow particle is generated at the origin.
	 * @param temp - Temperature of the simulation.
	 */
	SnowRenderer(unsigned numParticles, const float extentsInp[3][2], float temp) : SnowRenderer(numParticles, extentsInp, temp, EXPERIMENTAL_ALG, DEFAULT_WIND_VELOCITY, DEFAULT_LATTICE_RES) {};

	/**
	 * Default constructor for SnowRenderer.
	 */
	SnowRenderer() : SnowRenderer(DEFAULT_SNOW_COUNT, DEFAULT_EXTENTS, DEFAULT_TEMP) {};

	/**
	 * Setup VAOs.
	 */
	void setupVAO() {
		glGenVertexArrays(1, &vertexArrayID);
		glBindVertexArray(vertexArrayID);

		// gen snow
		if (whichAlg == EXPERIMENTAL_ALG) {
			data = snowGenerator.generateSnowExperimental(numParticles, extents);
		} else {
			data = snowGenerator.generateSnowMoeslund(numParticles, extents);
		}
		snowInitGPU(data, numParticles, extents, windVel, latticeRes, temp);

		// vertices
		glGenBuffers(1, &vertBuffer);
		glBindBuffer(GL_ARRAY_BUFFER, vertBuffer);
		glBufferData(GL_ARRAY_BUFFER, sizeof(float) * data.numPolys * 9, data.verts, GL_DYNAMIC_DRAW);
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
		glBufferData(GL_ARRAY_BUFFER, sizeof(float) * data.numPolys * 9, data.normals, GL_STATIC_DRAW);
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
		glBufferData(GL_ARRAY_BUFFER, sizeof(float) * data.numPolys * 12, data.colours, GL_STATIC_DRAW);
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
	 * Update snow positions.
	 */
	void updateSnow() {
		snowUpdateGPU();
		glBindVertexArray(vertexArrayID);
		glBindBuffer(GL_ARRAY_BUFFER, vertBuffer);
		glBufferData(GL_ARRAY_BUFFER, sizeof(float) * data.numPolys * 9, data.verts, GL_DYNAMIC_DRAW);
	}

	/**
	 * Update snow positions on the GPU. Split up variant for timing purposes.
	 */
	void updateSnowGPUTimed() {
		snowUpdateGPU();
	}

	/**
	 * Update snow positions on the GPU. Split up variant for timing purposes.
	 */
	void updateSnowTimed() {
		glBindVertexArray(vertexArrayID);
		glBindBuffer(GL_ARRAY_BUFFER, vertBuffer);
		glBufferData(GL_ARRAY_BUFFER, sizeof(float) * data.numPolys * 9, data.verts, GL_DYNAMIC_DRAW);
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
		glUniform1f(alphaID, SHININESS_COEFF);

		glPointSize(3.0f);
		glDrawArrays(GL_TRIANGLES, 0, data.numPolys*3);
		glPointSize(1.0f);

		glUseProgram(0);
		glBindVertexArray(0);
	}
};

#endif
