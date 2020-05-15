add_library(
    imgui
    imgui/imgui.cpp
    imgui/imgui_demo.cpp
    imgui/imgui_draw.cpp
    imgui/imgui_widgets.cpp
)
target_include_directories(imgui PUBLIC imgui)

find_package(SDL2 REQUIRED)
find_package(GLEW REQUIRED)

add_library(
    imgui_sdl_opengl3
    imgui/examples/imgui_impl_opengl3.cpp
    imgui/examples/imgui_impl_sdl.cpp
)
target_link_libraries(
    imgui_sdl_opengl3
    PUBLIC
    imgui
    GLEW::glew
    SDL2::SDL2
    GL
)

add_executable(
    imgui_sdl_opengl3_example
    imgui/examples/example_sdl_opengl3/main.cpp
    /usr/include/GL/glew.h
)
target_link_libraries(imgui_sdl_opengl3_example PUBLIC imgui_sdl_opengl3)
target_include_directories(imgui_sdl_opengl3_example PUBLIC imgui/examples /usr/include/GL/)