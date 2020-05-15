cmake_minimum_required(VERSION 3.15)
project(graphics_hello)

set(CMAKE_CXX_STANDARD 20)

add_subdirectory(imgui)

find_package(SDL2 REQUIRED)
find_package(GLEW REQUIRED)

message(STATUS ${SDL2_INCLUDE_DIR})

add_executable(graphics_hello main.cpp)
target_link_libraries(
    graphics_hello
    SDL2::SDL2
    GLEW::GLEW
    GL
    imgui_sdl_opengl3
)
# TODO: static glew or shared glew_s?