cmake_minimum_required(VERSION 3.15)
project(graphics_hello CUDA)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CUDA_STANDARD 14)

#add_subdirectory(imgui)

find_package(SDL2 REQUIRED)
find_package(GLEW REQUIRED)

message(STATUS ${SDL2_INCLUDE_DIR})

add_executable(graphics_hello main.cu)
target_link_libraries(
    graphics_hello
    SDL2::SDL2
    GLEW::GLEW
    GL
)
target_include_directories(graphics_hello PUBLIC "/opt/cuda/include")  # TODO: temporary fix for code insight
target_compile_options(graphics_hello PUBLIC "--expt-relaxed-constexpr")
# TODO: static glew or shared glew_s?
