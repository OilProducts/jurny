cmake_minimum_required(VERSION 3.24)
if(POLICY CMP0135)
  cmake_policy(SET CMP0135 NEW) # Updated default for external downloads/extract timestamps
endif()
project(voxel_planet VERSION 0.1.0 LANGUAGES CXX)

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
option(VOXEL_ENABLE_SANITIZERS  "Enable Address/UB sanitizers (Debug)"     OFF)
option(VOXEL_ENABLE_LTO         "Enable LTO in Release"                    ON)
option(VOXEL_SHADER_DEBUG       "Emit debug info in SPIR-V (Debug)"        ON)
option(VOXEL_USE_FETCHCONTENT   "Use FetchContent for extern deps"         ON)
option(VOXEL_GLFW_WAYLAND       "Use Wayland backend for GLFW (Linux)"     OFF)
option(VOXEL_ENABLE_WINDOW      "Enable window+swapchain (GLFW)"           ON)
option(BUILD_TOOLS              "Build auxiliary tools"                    ON)

# Feature flags bridged to shaders
set(VOXEL_BRICK_SIZE 8 CACHE STRING "Brick dimension (voxels)")
option(VOXEL_USE_TSDF           "Enable TSDF tiles"                         ON)
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

## Dependency wrapper interface targets (linked/populated in FetchDeps.cmake)
add_library(ext_glm INTERFACE)
add_library(ext_spdlog INTERFACE)
add_library(ext_tracy INTERFACE)
add_library(ext_stb INTERFACE)
add_library(ext_xxhash INTERFACE)
add_library(ext_volk INTERFACE)
add_library(ext_vma INTERFACE)
add_library(ext_glfw INTERFACE)

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
  add_subdirectory(tools/debug_dump)
endif()

set(VOXEL_ASSET_DATA_DIR "${CMAKE_SOURCE_DIR}/data")
file(GLOB_RECURSE VOXEL_ASSET_INPUTS CONFIGURE_DEPENDS "${VOXEL_ASSET_DATA_DIR}/*")
set(VOXEL_ASSET_FILES)
foreach(item ${VOXEL_ASSET_INPUTS})
  if(NOT IS_DIRECTORY "${item}")
    list(APPEND VOXEL_ASSET_FILES "${item}")
  endif()
endforeach()
set(VOXEL_ASSET_OUTPUT_DIR "${CMAKE_BINARY_DIR}/assets")
set(VOXEL_ASSET_PAK "${VOXEL_ASSET_OUTPUT_DIR}/assets.pak")
set(VOXEL_ASSET_MANIFEST "${VOXEL_ASSET_OUTPUT_DIR}/assets_manifest.json")
set(VOXEL_ASSET_INDEX "${VOXEL_ASSET_OUTPUT_DIR}/assets_index.txt")

if(BUILD_TOOLS)
  add_custom_command(
    OUTPUT "${VOXEL_ASSET_PAK}" "${VOXEL_ASSET_MANIFEST}" "${VOXEL_ASSET_INDEX}"
    COMMAND ${CMAKE_COMMAND} -E make_directory "${VOXEL_ASSET_OUTPUT_DIR}"
    COMMAND $<TARGET_FILE:pack_assets>
            --data-dir "${VOXEL_ASSET_DATA_DIR}"
            --out-dir "${VOXEL_ASSET_OUTPUT_DIR}"
            --pak "${VOXEL_ASSET_PAK}"
            --manifest "${VOXEL_ASSET_MANIFEST}"
            --index "${VOXEL_ASSET_INDEX}"
    DEPENDS ${VOXEL_ASSET_FILES} pack_assets
    COMMENT "Packing assets (materials, envmaps, data)"
    VERBATIM
  )
  add_custom_target(assets_packed ALL DEPENDS "${VOXEL_ASSET_PAK}" "${VOXEL_ASSET_MANIFEST}" "${VOXEL_ASSET_INDEX}")
  add_dependencies(assets_packed pack_assets)
else()
  add_custom_target(assets_packed
    COMMAND ${CMAKE_COMMAND} -E echo "Asset packing disabled (BUILD_TOOLS=OFF)"
  )
endif()

if(TARGET voxel_app)
  add_dependencies(voxel_app shaders_spv assets_packed)
endif()

if(VOXEL_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()
