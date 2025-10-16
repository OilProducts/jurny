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

  find_program(GLSLC_EXECUTABLE NAMES glslc HINTS ENV VULKAN_SDK PATH_SUFFIXES Bin bin)

  set(OUT_SVPS)
  if (GLSLC_EXECUTABLE)
    foreach(src ${GLSL_SOURCES})
      file(RELATIVE_PATH rel "${VOXEL_SHADERS_SOURCE_DIR}" "${src}")
      set(out "${CMAKE_BINARY_DIR}/shaders/${rel}.spv")
      get_filename_component(out_dir "${out}" DIRECTORY)
      file(MAKE_DIRECTORY "${out_dir}")

      set(args -c -o "${out}" "${src}" --target-env=vulkan1.3 -O)
      if (VOXEL_SHADERS_DEBUG_INFO)
        list(APPEND args -g)
      endif()
      list(APPEND args -I "${VOXEL_SHADERS_SOURCE_DIR}" -I "${VOXEL_SHADERS_SOURCE_DIR}/include")
      # Add defines; use generator expressions for toggles to avoid bare -D when false
      list(APPEND args -DVOXEL_BRICK_SIZE=${VOXEL_BRICK_SIZE})
      if (VOXEL_USE_TSDF)
        list(APPEND args -DVOXEL_USE_TSDF=1)
      endif()
      if (VOXEL_MATERIAL_4BIT)
        list(APPEND args -DVOXEL_MATERIAL_4BIT=1)
      endif()
      if (VOXEL_ENABLE_DENOISER)
        list(APPEND args -DVOXEL_ENABLE_DENOISER=1)
      endif()

      add_custom_command(
        OUTPUT "${out}"
        COMMAND ${GLSLC_EXECUTABLE} ${args}
        DEPENDS "${src}"
        COMMENT "[glslc] ${rel}"
        VERBATIM
      )
      list(APPEND OUT_SVPS "${out}")
    endforeach()
    add_custom_target(shaders_spv ALL DEPENDS ${OUT_SVPS})
  else()
    message(WARNING "glslc not found; shaders will not be compiled. Install Vulkan SDK.")
    add_custom_target(shaders_spv
      COMMAND ${CMAKE_COMMAND} -E echo "glslc not found; skipping shader compilation"
    )
  endif()

  set(${VOXEL_SHADERS_OUT_VAR} ${OUT_SVPS} PARENT_SCOPE)
endfunction()
