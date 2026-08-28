# Snow Visualizer

Simulations of natural phenomena in the physical world, such as snowfall, are important 
in many fields, including computer graphics, gaming, and various engineering disciplines. 
This research seeks to implement physically accurate falling snow in real-time while 
maintaining high visual fidelity. This involves generating individual snow particles, 
accurately modelling wind via fluid dynamics, solving for particle motion, and 
rendering the simulation. To mimic the behaviour of snow in the physical world, 
the snow particles are generated as clumps of snowflakes arranged in concentric 
circles. The wind field is modelled using the Lattice Boltzmann Method, a method in 
computational fluid dynamics that models Boltzmann particle dynamics on a lattice, 
which avoids attempting to solve the Navier-Stokes equations directly. 
The computations of the physics, specifically the Lattice Boltzmann Method and 
the application of forces to snow particles, are accelerated through GPU computation 
using CUDA. Various optimizations are applied to improve coalesced memory access and, 
thus, running time. The three-dimensional simulation space is simultaneously rendered in 
real-time using OpenGL.

## Usage

### Setup

#### 1. Download code

#### 2. Install dependencies

- OpenGL 4.0 or newer
- C++11 et al. (i.e. make sure you have everything installed for compiling and running a C++11 application)
- `cmake` and `ninja` (used to compile the application using `CMakeLists.txt`)
- CUDA, and `nvcc` if you intend to recompile the CUDA code (developed with version 13.1)
  - This also means you unfortunately need to have access to an NVIDIA GPU  

__*Note__: OpenGL, especially on Windows, is notoriously horrible to configure, even more so when also integrating CUDA. I 
set up my OpenGL development environment a while ago so I am not 100% sure these are the only dependencies.
For reference, I am developing in CLion using Microsoft's Visual Studio's toolchain. Debugging info is also hard to provide 
as the console closes when the window crashes and due to time constraints I did not have time to develop 
a sophisticated system to help with installation.

#### 3. Compile the code into an executable

Use `cmake` and the provided `CMakeLists.txt` to compile the code into an executable. Make sure the
generated `/include/glfw/` folder and the shaders folder (`/cmake-build-debug/shaders/`) are in the
same directory as the executable.

#### 4. Run the executable
Running the execulable will generate a snow simulation based on the default parameters. 

### Configuring the Simulation

To configure the simulation, you can edit the parameters in `/util/Input.hpp` and recompile. At a later
date this will be refactored to be done through the CLI. 

There is no model validation at this point and the generated simulation will not match 
the conditions the parameters would generate in the physical world due to the lack of an 
implemented subgrid model. The default values are set arbitrarily to values that yield a simulation
which is plausible, determined visually, and which stabilizes.

#### Camera

There are 2 available cameras: first person (WASD and LSHIFT/LCTRL) and 
globe (mouse drag around the origin, up/down arrow to zoom in/out). The default is first person but you can 
change this in the input file by setting `whichCam` to `GLOBE_CAM` instead of `FIRST_PERSON_CAM`. Ideally 
the functionality of these would be merged but due to time and the complexity of 
trigonometry this is not implemented.

### Development

Development of this project is always welcome! The most important thing to note is that if anything
referenced by `/snow/cuda/gpu.cu` (even recursively) is changed, the CUDA code will need to be recompiled. 
To do this, run the shell script `/snow/cuda/compile.sh` to generate the updated files needed to link the CUDA 
code to the rest of the code. You will also need to recompile using `CMakeLists.txt` as usual.

There are also some additional development-related input parameters defined in `/util/DevInput.hpp`, 
such as profiling configurations.

#### Testing

No comprehensive testing has been implemented at this point but most certainly should be for regression testing.

## Implementation

The runner is `main.cpp`. In `/snow/SnowRenderer.hpp`, initial snow particle geometry is generated and each frame the 
snow is updated on the GPU by `/snow/cuda/gpu.cu`. The new vertices are then copied back to the CPU and a new frame is rendered. 
For more details, see the accompanying thesis `MSemenova_Honours` in `/docs/`.

### Snow Generator
The API is implemented in `/snow/SnowGenerator.hpp`, and `/snow/SnowRenderer.hpp` shows an example of using the API. Relevant constants are
defined in `/snow/SnowConstants.hpp`. The API works by instantiating a `SnowGenerator` object with a given temperature which then enables
a user to call an assortment of methods to generate snow by specifying the number of particles to generate and the extents of
the volume within which snow is generated. The algorithm used to generate the snow is chosen by calling the method for the
desired algorithm. The simple Phong shaders used for the snow are `/cmake-build-debug/shaders/PhongVertexShader.vertexshader`
and `/cmake-build-debug/shaders/PhongFragmentShader.fragmentshader`. 

## Documents

Any documents related to this project are available in `/docs/`. The accompanying thesis to 
this project is `MSemenova_Honours`.

### Videos

Demos of the application can be found in `/docs/videos/`.

## TODOs

- Implement input through CLI
- Combine cams in `CameraControls.hpp`
- Better error-trapping
- Implement lookup tables for speed of sound and viscosity from temperature
- Vsync
- Update default values + implement model validation when subgrid model integrated
- Make wind field more variable
- Make sim more flexible (e.g. don't constrain N_x = N_y = N_z)
- Better force application calculation
- Make the snow look better
- Consider loop unrolling + thread divergence
- Testing suite
- Run on proper GPU

### Snow Generator
- Prevent generated triangles from intersecting
- Vary # of polys per layer
- Vary layerH but ensure that the gaps or lack thereof are not unnatural
