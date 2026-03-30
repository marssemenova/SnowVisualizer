/**
 * Axes.hpp - Renders axes. Used for debugging.
 *
 * @author Mars Semenova
 * @date March 30, 2026
 */
class Axes {
	glm::vec3 origin;
	glm::vec3 extents;

	GLuint programID;

	GLuint MVPID;

	GLuint vertexArrayID;
	GLuint vertBuffer;
	GLuint colorBuffer;

	void setupVAO() {
		glGenVertexArrays(1, &vertexArrayID);
		glBindVertexArray(vertexArrayID);

		// vertices
		static const GLfloat verts[] = {
			origin.x, origin.y, origin.z,
			origin.x + extents.x, origin.y, origin.z,
			origin.x, origin.y, origin.z,
			origin.x, origin.y + extents.y, origin.z,
			origin.x, origin.y, origin.z,
			origin.x, origin.y, origin.z + extents.z,
		};

		glGenBuffers(1, &vertBuffer);
		glBindBuffer(GL_ARRAY_BUFFER, vertBuffer);
		glBufferData(GL_ARRAY_BUFFER, sizeof(verts), verts, GL_STATIC_DRAW);
		glEnableVertexAttribArray(0);
		glVertexAttribPointer(
			0,                                // attribute. No particular reason for 1, but must match the layout in the shader.
			3,                                // size
			GL_FLOAT,                         // type
			GL_FALSE,                         // normalized?
			0,                                // stride
			(void*) 0                        // array buffer offset
		);

		// colours
		static const GLfloat colours[] = {
			1.0f, 0.0f, 0.0f,
			1.0f, 0.0f, 0.0f,
			0.0f, 1.0f, 0.0f,
			0.0f, 1.0f, 0.0f,
			0.0f, 0.0f, 1.0f,
			0.0f, 0.0f, 1.0f,
		};

		glGenBuffers(1, &colorBuffer);
		glBindBuffer(GL_ARRAY_BUFFER, colorBuffer);
		glBufferData(GL_ARRAY_BUFFER, sizeof(colours), colours, GL_STATIC_DRAW);
		glEnableVertexAttribArray(1);
		glVertexAttribPointer(
			1,                                // attribute. No particular reason for 1, but must match the layout in the shader.
			3,                                // size
			GL_FLOAT,                         // type
			GL_FALSE,                         // normalized?
			0,                                // stride
			(void*) 0                        // array buffer offset
		);

		glBindBuffer(GL_ARRAY_BUFFER, 0);
		glBindVertexArray(0);
	}

public:
	Axes(glm::vec3 orig, glm::vec3 ex) : origin(orig), extents(ex) {
		// load shaders
		programID = LoadShaders( "ColorVertexShader.vertexshader", "ColorFragmentShader.fragmentshader");
		MVPID = glGetUniformLocation(programID, "MVP");
		glUseProgram(programID);

		setupVAO();
	}

	Axes() : Axes(glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(1.0f, 1.0f, 1.0f)) {}

	void draw(glm::mat4 M, glm::mat4 V, glm::mat4 P) {
		glm::mat4 MVP = P*V*M;
		glBindVertexArray(vertexArrayID);
		glUseProgram(programID);

		glUniformMatrix4fv(MVPID, 1, GL_FALSE, &MVP[0][0]);

		glLineWidth(5.0f);
		glDrawArrays(GL_LINES, 0, 6);
		glLineWidth(1.0f);

		glUseProgram(0);
		glBindVertexArray(0);
	}
};
