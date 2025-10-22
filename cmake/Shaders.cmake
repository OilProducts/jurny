# Compile GLSL compute shaders (*.comp) to SPIR-V using glslc if available.
# Usage:
#   voxel_compile_shaders(SOURCE_DIR path OUT_VAR out_list [DEFINES ...] [DEBUG_INFO ON|OFF])

function(voxel_compile_shaders)
  set(options)
  set(oneValueArgs SOURCE_DIR OUT_VAR DEBUG_INFO)
  set(multiValueArgs DEFINES)
  cmake_parse_arguments(VOXEL_SHADERS "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

  if (NOT VOXEL_SHADERS_SOURCE_DIR)
    message(FATAL_ERROR "voxel_compile_shaders: SOURCE_DIR is required")
  endif()
  if (NOT VOXEL_SHADERS_OUT_VAR)
    message(FATAL_ERROR "voxel_compile_shaders: OUT_VAR is required")
  endif()

  file(GLOB_RECURSE GLSL_SOURCES CONFIGURE_DEPENDS
    "${VOXEL_SHADERS_SOURCE_DIR}/*.comp"
  )
  # Also track includes so modifying shared headers triggers rebuilds.
  file(GLOB_RECURSE GLSL_INCLUDES CONFIGURE_DEPENDS
    "${VOXEL_SHADERS_SOURCE_DIR}/*.glsl"
  )

  find_program(GLSLC_EXECUTABLE NAMES glslc HINTS ENV VULKAN_SDK PATH_SUFFIXES Bin bin)
  if (NOT GLSLC_EXECUTABLE)
    message(WARNING "glslc not found; shaders will not be compiled. Install the Vulkan SDK.")
    add_custom_target(shaders_spv
      COMMAND ${CMAKE_COMMAND} -E echo "glslc not found; skipping shader compilation"
    )
    set(${VOXEL_SHADERS_OUT_VAR} "" PARENT_SCOPE)
    return()
  endif()

  find_package(Python3 COMPONENTS Interpreter REQUIRED)
  set(SHADER_SCRIPT "${CMAKE_SOURCE_DIR}/tools/shaderc_build/compile_shaders.py")
  if (NOT EXISTS "${SHADER_SCRIPT}")
    message(FATAL_ERROR "Shader wrapper script not found at ${SHADER_SCRIPT}")
  endif()

  set(SHADER_OUTPUT_DIR "${CMAKE_BINARY_DIR}/shaders")
  set(SHADER_ASSET_DIR "${CMAKE_BINARY_DIR}/assets/shaders")
  set(SHADER_DEP_DIR   "${SHADER_OUTPUT_DIR}/deps")
  set(SHADER_MANIFEST  "${SHADER_OUTPUT_DIR}/manifest.json")

  if (GLSL_SOURCES)
    set(OUT_SVPS)
    set(OUT_ASSET_SVPS)
    foreach(src ${GLSL_SOURCES})
      file(RELATIVE_PATH rel "${VOXEL_SHADERS_SOURCE_DIR}" "${src}")
      list(APPEND OUT_SVPS       "${SHADER_OUTPUT_DIR}/${rel}.spv")
      list(APPEND OUT_ASSET_SVPS "${SHADER_ASSET_DIR}/${rel}.spv")
    endforeach()

    set(python_cmd ${Python3_EXECUTABLE} "${SHADER_SCRIPT}"
      --source-dir "${VOXEL_SHADERS_SOURCE_DIR}"
      --output-dir "${SHADER_OUTPUT_DIR}"
      --manifest "${SHADER_MANIFEST}"
      --glslc "${GLSLC_EXECUTABLE}"
      --depdir "${SHADER_DEP_DIR}"
      --copy-dir "${SHADER_ASSET_DIR}"
      --include "${VOXEL_SHADERS_SOURCE_DIR}"
      --include "${VOXEL_SHADERS_SOURCE_DIR}/include"
      --define "VOXEL_BRICK_SIZE=${VOXEL_BRICK_SIZE}"
    )
    if (VOXEL_SHADERS_DEBUG_INFO)
      list(APPEND python_cmd --debug)
    endif()
    if (VOXEL_USE_TSDF)
      list(APPEND python_cmd --define "VOXEL_USE_TSDF=1")
    endif()
    if (VOXEL_MATERIAL_4BIT)
      list(APPEND python_cmd --define "VOXEL_MATERIAL_4BIT=1")
    endif()
    if (VOXEL_ENABLE_DENOISER)
      list(APPEND python_cmd --define "VOXEL_ENABLE_DENOISER=1")
    endif()
    foreach(src ${GLSL_SOURCES})
      list(APPEND python_cmd --source "${src}")
    endforeach()

    set(SHADER_STAMP "${SHADER_OUTPUT_DIR}/shader_compile.stamp")

    add_custom_command(
      OUTPUT "${SHADER_STAMP}"
      BYPRODUCTS ${OUT_SVPS} ${OUT_ASSET_SVPS} "${SHADER_MANIFEST}"
      COMMAND ${CMAKE_COMMAND} -E make_directory "${SHADER_OUTPUT_DIR}"
      COMMAND ${CMAKE_COMMAND} -E make_directory "${SHADER_ASSET_DIR}"
      COMMAND ${CMAKE_COMMAND} -E make_directory "${SHADER_DEP_DIR}"
      COMMAND ${python_cmd}
      COMMAND ${CMAKE_COMMAND} -E touch "${SHADER_STAMP}"
      DEPENDS ${GLSL_SOURCES} ${GLSL_INCLUDES} "${SHADER_SCRIPT}"
      COMMENT "Compiling GLSL shaders via compile_shaders.py"
      VERBATIM
    )
    add_custom_target(shaders_spv ALL DEPENDS "${SHADER_STAMP}")
    set(${VOXEL_SHADERS_OUT_VAR} ${OUT_SVPS} PARENT_SCOPE)
  else()
    add_custom_target(shaders_spv)
    set(${VOXEL_SHADERS_OUT_VAR} "" PARENT_SCOPE)
  endif()
endfunction()
