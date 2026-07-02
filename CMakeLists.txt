cmake_minimum_required(VERSION 3.25)
project(SnowVisualizer)

find_package(OpenGL REQUIRED)
find_package(CUDAToolkit REQUIRED)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:DynamicRelease>:DynamicRelease>")

# include GLFW
add_subdirectory(include/glfw)

# include GLM
include_directories(include)

# include GLAD
include_directories(include/glad/include)

set(SOURCE_FILES main.cpp include/glad/src/glad.c)

add_executable(${CMAKE_PROJECT_NAME} ${SOURCE_FILES})

# linking GLFW + OGL
target_link_libraries(${CMAKE_PROJECT_NAME} glfw ${GLFW_LIBRARIES} ${OPENGL_LIBRARY})

# CUDA
target_link_libraries(${PROJECT_NAME} CUDA::cudart)
target_link_libraries(SnowVisualizer ${CMAKE_CURRENT_SOURCE_DIR}/snow/cuda/dlink.o)
target_link_libraries(SnowVisualizer ${CMAKE_CURRENT_SOURCE_DIR}/snow/cuda/gpu.o)