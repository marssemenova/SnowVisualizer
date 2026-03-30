
class Plane {

public:
	enum PLANE_WHICH {
		x,
		y,
		z
	};

private:
	PLANE_WHICH plane = PLANE_WHICH::y;

	glm::vec4 color = glm::vec4(0.9f, 0.9f, 0.9f, 0.1f);

	GLfloat size;
	GLfloat offset = 0;
	GLuint texID = 0;

	GLuint programID;

    GLuint MVPID;
    GLuint MID;
    GLuint VID;
    GLuint LightPosID;
    GLuint colorID;
    GLuint alphaID;

    GLuint vertexArrayID;
    GLuint vertBuffer;
    GLuint normalBuffer;
    GLuint uvBuffer;
    GLuint elementBufferID;



public:

	Plane(GLfloat sz, std::string textureFile) : size(sz) {


		unsigned char* data;
		unsigned int width, height;
		loadBMP(textureFile.c_str(), &data, &width, &height);
		fprintf(stderr, "w: %d, h: %d\n", width, height);

		glGenTextures(1, &texID);
		glBindTexture(GL_TEXTURE_2D, texID);
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_BGR, GL_UNSIGNED_BYTE, data);
    	glGenerateMipmap(GL_TEXTURE_2D);
	    glBindTexture(GL_TEXTURE_2D, 0);

        programID = LoadShaders( "PhongVertexShader.vertexshader", "PhongFragmentShader.fragmentshader" );
//        programID = LoadShaders( "TransformVertexShader.vertexshader", "ColorFragmentShader.fragmentshader" );

        // Get a handle for our "MVP" uniform
        MVPID = glGetUniformLocation(programID, "MVP");
        MID = glGetUniformLocation(programID, "M");
        VID = glGetUniformLocation(programID, "V");
        LightPosID = glGetUniformLocation(programID, "LightPosition_worldspace");
        colorID = glGetUniformLocation(programID, "modelcolor");
        alphaID = glGetUniformLocation(programID, "alpha");

        glUseProgram(programID);
        glm::mat4 M(1.0f);
        glUniformMatrix4fv(MID, 1, GL_FALSE, &M[0][0]); //model matrix always identity.

        setupVAO();
	}

	void setupVAO() {

		float verts[] = {
			-size, offset, -size,
			size, offset, -size,
			size, offset, size,
			-size, offset, size
		};

		float uv[] = {
			0.0f, 0.0f, 0.0f, 1.0f, 1.0f, 1.0f, 1.0f, 0.0f
		};

		float normals[] = {
			0, 1, 0,
			0, 1, 0,
			0, 1, 0,
			0, 1, 0,
		};

	    glGenVertexArrays(1, &vertexArrayID);
	    glBindVertexArray(vertexArrayID);

	    glGenBuffers(1, &vertBuffer);
	    glBindBuffer(GL_ARRAY_BUFFER, vertBuffer);
	    glBufferData(GL_ARRAY_BUFFER, sizeof(GL_FLOAT)*12, verts, GL_STATIC_DRAW);
		glEnableVertexAttribArray(0);
		glVertexAttribPointer(
			0,                                // attribute. No particular reason for 1, but must match the layout in the shader.
			3,                                // size
			GL_FLOAT,                         // type
			GL_FALSE,                         // normalized?
			0,                                // stride
			(void*) 0                        // array buffer offset
		);

		glGenBuffers(1, &normalBuffer);
	    glBindBuffer(GL_ARRAY_BUFFER, normalBuffer);
	    glBufferData(GL_ARRAY_BUFFER, sizeof(GL_FLOAT)*12, normals, GL_STATIC_DRAW);

		glEnableVertexAttribArray(1);
		glVertexAttribPointer(
			1,                                // attribute. No particular reason for 1, but must match the layout in the shader.
			3,                                // size
			GL_FLOAT,                         // type
			GL_TRUE,                         // normalized?
			0,                                // stride
			(void*) 0                          // array buffer offset
		);

		glGenBuffers(1, &uvBuffer);
	    glBindBuffer(GL_ARRAY_BUFFER, uvBuffer);
	    glBufferData(GL_ARRAY_BUFFER, sizeof(GL_FLOAT)*8, uv, GL_STATIC_DRAW);

		// glTexCoordPointer(2, GL_FLOAT, 0, this->texcoordArray);
		glEnableVertexAttribArray(2);
		glVertexAttribPointer(
			2,                                // attribute. No particular reason for 1, but must match the layout in the shader.
			2,                                // size
			GL_FLOAT,                         // type
			GL_TRUE,                         // normalized?
			0,                                // stride
			(void*) 0                         // array buffer offset
		);

		glBindBuffer(GL_ARRAY_BUFFER, 0);


		std::vector<unsigned int> indices = {0, 1, 2, 0, 2, 3};
	    glGenBuffers(1, &elementBufferID);
	    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, elementBufferID);
	    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(GL_UNSIGNED_INT)*indices.size(), indices.data(), GL_STATIC_DRAW);
	   
		glBindVertexArray(0);

	    // fprintf(stderr, "GL Error after setup: %d\n", glGetError());
	}



	Plane(GLfloat sz, PLANE_WHICH which) : size(sz), plane(which) {}
	Plane(GLfloat sz, PLANE_WHICH which, float set) : size(sz), plane(which), offset(set) {}


	void draw(glm::vec3 lightPos, glm::mat4 M, glm::mat4 V, glm::mat4 P, glm::vec4 color, float alpha) {

	    glm::mat4 MVP = P*V*M;
	    glUseProgram(programID);

	    glUniformMatrix4fv(MVPID, 1, GL_FALSE, &MVP[0][0]);
	    glUniformMatrix4fv(VID, 1, GL_FALSE, &V[0][0]);
	    glUniform3f(LightPosID, lightPos.x, lightPos.y, lightPos.z);
	    glUniform1f(alphaID, alpha);
		glBindTexture(GL_TEXTURE_2D, texID);

		glBindVertexArray(vertexArrayID);

		glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, (void*) 0);
		// glDrawArrays(GL_QUADS, 0, 4);

		glBindTexture(GL_TEXTURE_2D, 0);
		glBindVertexArray(0);


		glUseProgram(0);
	}


};
