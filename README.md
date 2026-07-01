# Snow Visualizer

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

### 3. Link CUDA
```shell
nvcc -c gpu.cu -o gpu.o
nvcc -O3 gpu.o -o dlink.o -gencode arch=compute_86,code=sm_86 -dlink
```

### 4. Compile the code into an executable

Use `cmake` and the provided `CMakeLists.txt` to compile the code into an executable. Make sure the
generated `/include/glfw/` folder and the shaders folder (`/cmake-build-debug/shaders/`) are in the
same directory as the executable.

### 5. Run the executable
Running the execulable will generate snow based on preset parameters. Due to time constraints the input of arguments through
the CLI was not implemented so you will have to change parameters in the runner (`main.cpp`) before
compiling.

#### Camera

There are 2 available cameras: globe (mouse drag around the origin, up/down arrow to zoom in/out) and
first person (WASD and LSHIFT/LCTRL). The default is globe but you can change this in the runner
(`main.cpp`) by setting `whichCam` to `FIRST_PERSON_CAM` instead of `GLOBE_CAM`. Ideally the functionality
of these would be merged but due to time and the complexity of trigonometry this is not implemented.

## Implementation

The API is implemented in `/snow/SnowGenerator.hpp`, and `/snow/SnowRenderer.hpp` shows an example of using the API. Relevant constants are
defined in `/snow/SnowConstants.hpp`. The API works by instantiating a `SnowGenerator` object with a given temperature which then enables
a user to call an assortment of methods to generate snow by specifying the number of particles to generate and the extent of
the volume within which snow is generated. The algorithm used to generate the snow is chosen by calling the method for the
desired algorithm. The simple Phong shaders used for the snow are `/cmake-build-debug/shaders/PhongVertexShader.vertexshader`
and `/cmake-build-debug/shaders/PhongFragmentShader.fragmentshader`. The runner is `main.cpp`.

## Documents

Any documents related to this project are available in `/docs/`.

### Videos

Demos of the application can be found in `/docs/videos/`.

## TODOs

- Parse input args + if empty use experiment
- Combine cams in `CameraControls.hpp`
- More rigorous test framework
- Better error-trapping
- Improve shader
- Prevent generated triangles from intersecting
- Optimization + parallelization (setup for GPU)
- Vary # of polys per layer
- Vary layerH but ensure that the gaps or lack thereof are not unnatural
