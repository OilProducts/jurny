## Prefer vendored sources under extern/* when present; otherwise fetch.
## This keeps a modern FetchContent pipeline while supporting local overrides.

## glm
if(EXISTS "${CMAKE_SOURCE_DIR}/extern/glm/CMakeLists.txt" OR EXISTS "${CMAKE_SOURCE_DIR}/extern/glm/glm/glm.hpp")
  set(FETCHCONTENT_SOURCE_DIR_glm "${CMAKE_SOURCE_DIR}/extern/glm")
endif()
FetchContent_Declare(glm
  GIT_REPOSITORY https://github.com/g-truc/glm.git
  GIT_TAG        1.0.1
  GIT_SHALLOW    TRUE
  GIT_PROGRESS   TRUE)
FetchContent_MakeAvailable(glm)
if(TARGET glm::glm)
  target_link_libraries(ext_glm INTERFACE glm::glm)
else()
  target_include_directories(ext_glm INTERFACE "${glm_SOURCE_DIR}")
endif()

## volk
if(EXISTS "${CMAKE_SOURCE_DIR}/extern/volk/CMakeLists.txt" OR EXISTS "${CMAKE_SOURCE_DIR}/extern/volk/volk.h")
  set(FETCHCONTENT_SOURCE_DIR_volk "${CMAKE_SOURCE_DIR}/extern/volk")
endif()
FetchContent_Declare(volk
  GIT_REPOSITORY https://github.com/zeux/volk.git
  GIT_TAG        1.4.304
  GIT_SHALLOW    TRUE
  GIT_PROGRESS   TRUE)
FetchContent_MakeAvailable(volk)
if(TARGET volk)
  target_link_libraries(ext_volk INTERFACE volk)
else()
  target_include_directories(ext_volk INTERFACE "${volk_SOURCE_DIR}")
endif()

## spdlog
if(EXISTS "${CMAKE_SOURCE_DIR}/extern/spdlog/CMakeLists.txt" OR EXISTS "${CMAKE_SOURCE_DIR}/extern/spdlog/include/spdlog/spdlog.h")
  set(FETCHCONTENT_SOURCE_DIR_spdlog "${CMAKE_SOURCE_DIR}/extern/spdlog")
endif()
FetchContent_Declare(spdlog
  GIT_REPOSITORY https://github.com/gabime/spdlog.git
  GIT_TAG        v1.16.0
  GIT_SHALLOW    TRUE
  GIT_PROGRESS   TRUE)
FetchContent_MakeAvailable(spdlog)
if(TARGET spdlog::spdlog_header_only)
  target_link_libraries(ext_spdlog INTERFACE spdlog::spdlog_header_only)
else()
  target_link_libraries(ext_spdlog INTERFACE spdlog::spdlog)
endif()

## tracy (headers)
if(EXISTS "${CMAKE_SOURCE_DIR}/extern/tracy/CMakeLists.txt" OR EXISTS "${CMAKE_SOURCE_DIR}/extern/tracy/public/tracy/Tracy.hpp")
  set(FETCHCONTENT_SOURCE_DIR_tracy "${CMAKE_SOURCE_DIR}/extern/tracy")
endif()
FetchContent_Declare(tracy
  GIT_REPOSITORY https://github.com/wolfpld/tracy.git
  GIT_TAG        v0.12.2
  GIT_SHALLOW    TRUE
  GIT_PROGRESS   TRUE)
FetchContent_MakeAvailable(tracy)
if(tracy_SOURCE_DIR)
  target_include_directories(ext_tracy INTERFACE "${tracy_SOURCE_DIR}/public")
endif()

## stb (headers)
if(EXISTS "${CMAKE_SOURCE_DIR}/extern/stb/CMakeLists.txt" OR EXISTS "${CMAKE_SOURCE_DIR}/extern/stb/stb_image.h")
  set(FETCHCONTENT_SOURCE_DIR_stb "${CMAKE_SOURCE_DIR}/extern/stb")
endif()
## stb has no official releases; pin to a known-good commit used by distros for reproducibility.
FetchContent_Declare(stb
  GIT_REPOSITORY https://github.com/nothings/stb.git
  # Pin to a real commit on master: Create SECURITY.md (2025-05-12)
  # https://github.com/nothings/stb/commit/802cd454f25469d3123e678af41364153c132c2a
  GIT_TAG        802cd454f25469d3123e678af41364153c132c2a
  GIT_SHALLOW    TRUE
  GIT_PROGRESS   TRUE)
FetchContent_MakeAvailable(stb)
if(stb_SOURCE_DIR)
  target_include_directories(ext_stb INTERFACE "${stb_SOURCE_DIR}")
endif()

## xxhash (headers + optional C library)
if(EXISTS "${CMAKE_SOURCE_DIR}/extern/xxhash/CMakeLists.txt" OR EXISTS "${CMAKE_SOURCE_DIR}/extern/xxhash/xxhash.h")
  set(FETCHCONTENT_SOURCE_DIR_xxhash "${CMAKE_SOURCE_DIR}/extern/xxhash")
endif()
FetchContent_Declare(xxhash
  GIT_REPOSITORY https://github.com/Cyan4973/xxHash.git
  GIT_TAG        v0.8.3
  GIT_SHALLOW    TRUE
  GIT_PROGRESS   TRUE)
FetchContent_MakeAvailable(xxhash)
if(xxhash_SOURCE_DIR)
  target_include_directories(ext_xxhash INTERFACE "${xxhash_SOURCE_DIR}")
endif()

## VMA (headers)
if(EXISTS "${CMAKE_SOURCE_DIR}/extern/VMA/CMakeLists.txt" OR EXISTS "${CMAKE_SOURCE_DIR}/extern/VMA/include/vk_mem_alloc.h")
  set(FETCHCONTENT_SOURCE_DIR_vma "${CMAKE_SOURCE_DIR}/extern/VMA")
endif()
FetchContent_Declare(vma
  GIT_REPOSITORY https://github.com/GPUOpen-LibrariesAndSDKs/VulkanMemoryAllocator.git
  GIT_TAG        v3.3.0
  GIT_SHALLOW    TRUE
  GIT_PROGRESS   TRUE)
FetchContent_MakeAvailable(vma)
if(vma_SOURCE_DIR)
  target_include_directories(ext_vma INTERFACE "${vma_SOURCE_DIR}/include" "${vma_SOURCE_DIR}")
endif()

## GLFW
if(EXISTS "${CMAKE_SOURCE_DIR}/extern/glfw/CMakeLists.txt" OR EXISTS "${CMAKE_SOURCE_DIR}/extern/glfw/include/GLFW/glfw3.h")
  set(FETCHCONTENT_SOURCE_DIR_glfw "${CMAKE_SOURCE_DIR}/extern/glfw")
endif()
# Clean up legacy cache var that conflicts with GLFW >= 3.4
unset(GLFW_USE_WAYLAND CACHE)
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
# GLFW 3.4+: select backend via GLFW_BUILD_WAYLAND / GLFW_BUILD_X11
if(VOXEL_GLFW_WAYLAND)
  set(GLFW_BUILD_WAYLAND ON  CACHE BOOL "" FORCE)
  set(GLFW_BUILD_X11     OFF CACHE BOOL "" FORCE)
else()
  set(GLFW_BUILD_WAYLAND OFF CACHE BOOL "" FORCE)
  set(GLFW_BUILD_X11     ON  CACHE BOOL "" FORCE)
endif()
FetchContent_Declare(glfw
  GIT_REPOSITORY https://github.com/glfw/glfw.git
  GIT_TAG        3.4
  GIT_SHALLOW    TRUE
  GIT_PROGRESS   TRUE)
FetchContent_MakeAvailable(glfw)
if(TARGET glfw)
  target_link_libraries(ext_glfw INTERFACE glfw)
elseif(TARGET glfw3)
  target_link_libraries(ext_glfw INTERFACE glfw3)
endif()
