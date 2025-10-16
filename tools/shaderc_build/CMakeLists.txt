cmake_minimum_required(VERSION 3.24)
project(shaderc_build)

# Placeholder phony target; actual shader build is configured in cmake/Shaders.cmake
add_custom_target(shader_tools_info
    COMMAND ${CMAKE_COMMAND} -E echo "Shader tools available. Use cmake Shaders.cmake to build."
)
