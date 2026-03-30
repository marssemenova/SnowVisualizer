cmake_minimum_required(VERSION 3.3)
project(SnowVisualizer)

find_package(OpenGL REQUIRED)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

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
