cmake_minimum_required(VERSION 3.24)
project(voxel_planet LANGUAGES CXX)

# Default build type
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING "Build type" FORCE)
endif()

# C++ standard and policies
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Options (per docs/02-build-system-and-dependencies.md)
option(VOXEL_BUILD_TESTS        "Build tests"                              ON)
option(VOXEL_ENABLE_VALIDATION  "Enable Vulkan validation (Debug default)" ON)
option(VOXEL_ENABLE_TRACY       "Enable Tracy profiler integration"        OFF)
option(VOXEL_ENABLE_SANITIZERS  "Enable Address/UB sanitizers (Debug)"     ON)
option(VOXEL_ENABLE_LTO         "Enable LTO in Release"                    ON)
option(VOXEL_SHADER_DEBUG       "Emit debug info in SPIR-V (Debug)"        ON)
option(VOXEL_USE_FETCHCONTENT   "Use FetchContent for extern deps"         ON)
option(VOXEL_GLFW_WAYLAND       "Use Wayland backend for GLFW (Linux)"     OFF)
option(VOXEL_ENABLE_WINDOW      "Enable window+swapchain (GLFW)"           ON)
option(BUILD_TOOLS              "Build auxiliary tools"                    ON)

# Feature flags bridged to shaders
set(VOXEL_BRICK_SIZE 8 CACHE STRING "Brick dimension (voxels)")
option(VOXEL_USE_TSDF           "Enable TSDF tiles"                         OFF)
option(VOXEL_MATERIAL_4BIT      "Use 4-bit material indices by default"     ON)
option(VOXEL_ENABLE_DENOISER    "Enable denoiser passes"                    ON)

# Compiler definitions
if(WIN32)
  add_definitions(-DNOMINMAX)
endif()
add_definitions(-DGLM_FORCE_RADIANS -DGLM_FORCE_DEPTH_ZERO_TO_ONE)

# Helpers
list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")
include(Warnings)
include(Sanitizers)
include(Shaders)

# Dependencies (extern/) — create interface targets as wrappers
add_library(ext_glm INTERFACE)
target_include_directories(ext_glm INTERFACE ${CMAKE_SOURCE_DIR}/extern/glm)
add_library(ext_spdlog INTERFACE)
target_include_directories(ext_spdlog INTERFACE ${CMAKE_SOURCE_DIR}/extern/spdlog/include ${CMAKE_SOURCE_DIR}/extern/spdlog)
add_library(ext_tracy INTERFACE)
target_include_directories(ext_tracy INTERFACE ${CMAKE_SOURCE_DIR}/extern/tracy)
add_library(ext_stb INTERFACE)
target_include_directories(ext_stb INTERFACE ${CMAKE_SOURCE_DIR}/extern/stb)
add_library(ext_xxhash INTERFACE)
target_include_directories(ext_xxhash INTERFACE ${CMAKE_SOURCE_DIR}/extern/xxhash)
add_library(ext_volk INTERFACE)
target_include_directories(ext_volk INTERFACE ${CMAKE_SOURCE_DIR}/extern/volk)
add_library(ext_vma INTERFACE)
target_include_directories(ext_vma INTERFACE ${CMAKE_SOURCE_DIR}/extern/VMA/include ${CMAKE_SOURCE_DIR}/extern/VMA)
add_library(ext_glfw INTERFACE)
target_include_directories(ext_glfw INTERFACE ${CMAKE_SOURCE_DIR}/extern/glfw/include)

# Always include dependency helper; it will auto-fetch when extern/* is missing.
include(FetchContent)
include(FetchDeps)

# LTO
include(CheckIPOSupported)
check_ipo_supported(RESULT ipo_supported OUTPUT ipo_msg)
if(ipo_supported AND VOXEL_ENABLE_LTO AND CMAKE_BUILD_TYPE STREQUAL "Release")
  set(CMAKE_INTERPROCEDURAL_OPTIMIZATION ON)
endif()

# Shaders: offline compile to build/shaders/*.spv (gracefully noop if glslc missing)
voxel_compile_shaders(
  SOURCE_DIR "${CMAKE_SOURCE_DIR}/shaders"
  OUT_VAR VOXEL_SPV_FILES
  DEBUG_INFO ${VOXEL_SHADER_DEBUG}
)

# Subdirectories
add_subdirectory(src)
if(BUILD_TOOLS)
  add_subdirectory(tools/shaderc_build)
  add_subdirectory(tools/pack_assets)
endif()
if(VOXEL_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()
