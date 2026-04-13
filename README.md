# Snow Visualizer
###### CSCI4118 project Winter 2026

This project implements an algorithm for the generation of 
visually and physically realistic clumps of snow in OpenGL by using T. B. Moeslund et al.'s [1]
snow generation model as a reference and running experiments to 
determine optimal modifications to it. The results are also compared to C. Zou et al.'s [2]
work which too builds off of T. B. Moeslund et al.'s [1] model.

## Usage

### 1. Download code

### 2. Install and configure dependencies

- OpenGL 4.0 or newer
- C++11 et al. (i.e. make sure you have everything installed for compiling and running a C++11 application)
- `cmake` and `ninja` (used to compile the application using `CMakeLists.txt`)

__*Note__: OpenGL, especially on Windows, is notoriously horrible to configure. I also set up my 
OpenGL development environment a while ago so I am not 100% sure these are the only dependencies. 
For reference, I am using CLion's bundled MinGW toolchain and I have had trouble with 
using Cygwin and Visual Studio. Debugging info is also hard to provide as the console closes 
when the window crashes and due to time constraints I did not have time to develop a sophisticated 
system to help with installation. 

### 3. Compile the code into an executable

Use `cmake` and the provided `CMakeLists.txt` to compile the code into an executable. Make sure the
generated `/include/glfw/` folder and the shaders folder (`/cmake-build-debug/shaders/`) are in the
same directory as the executable. 

### 4. Run the executable
If arguments are provided, snowflake(s) will be generated as specified. Otherwise, the instance
used in experimentation is rendered. 


## Implementation

The API is implemented in `/snow/SnowBuilder.hpp`, and `/snow/SnowGenerator.hpp` shows an example of using the API. Relevant constants are 
defined in `/snow/SnowConstants.hpp`. The API works by instantiating a SnowBuilder object with a given temperature which then enables 
a user to call an assortment of methods to generate snow by specifying the number of particles to generate and the extent of 
the volume within which snow is generated. The algorithm used to generate the snow is chosen by calling the method for the
desired algorithm. The simple Phong shaders used for the snow are `/cmake-build-debug/shaders/PhongVertexShader.vertexshader`
and `/cmake-build-debug/shaders/PhongFragmentShader.fragmentshader`. The runner is `main.cpp`.

## Documents

Any documents related to this project are available in `/docs/`.

## Citations
[1]	T. B. Moeslund et al. “Modeling Falling and Accumulating Snow”. In: Second
International Conference on Vision, Video and Graphics, VVG 2005. Second
International Conference on Vision, Video and Graphics. European Association for
Computer Graphics, 2005, pp. 61–68. ISBN: 3905673576. URL:
https://www.researchgate.net/publication/267400785_Modeling_Falling_and_Accumulating_Snow.

[2]	C. Zou et al. “Algorithm for generating snow based on GPU”. In: ICIMCS ’10:
Proceedings of the Second International Conference on Internet Multimedia Computing
and Service. Association for Computing Machinery, 2010, pp. 199–202. ISBN:
9781450304603. DOI: 10.1145/1937728.1937775.

## TODOs

- Parse input args + if empty use experiment
- get right normal
- const comments
- enable extend + if extent equal gen at that pos.
- test api


