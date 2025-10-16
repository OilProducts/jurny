# Engine modules as static libs, plus app executable.

file(GLOB_RECURSE CORE_SRC CONFIGURE_DEPENDS core/*.cpp)
add_library(voxel_core STATIC ${CORE_SRC})
target_include_directories(voxel_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(voxel_core PUBLIC ext_spdlog ext_tracy)
target_compile_definitions(voxel_core PUBLIC
  $<$<BOOL:${VOXEL_ENABLE_VALIDATION}>:VOXEL_ENABLE_VALIDATION=1>
  $<$<BOOL:${VOXEL_ENABLE_TRACY}>:VOXEL_ENABLE_TRACY=1>
)
voxel_set_warnings(voxel_core)
voxel_enable_sanitizers(voxel_core)

set(PLATFORM_SRC
  platform/VulkanContext.cpp
  platform/Window.cpp
  platform/Swapchain.cpp)
if(NOT VOXEL_ENABLE_WINDOW)
  list(REMOVE_ITEM PLATFORM_SRC platform/Window.cpp platform/Swapchain.cpp)
endif()
add_library(voxel_platform STATIC ${PLATFORM_SRC})
target_include_directories(voxel_platform PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(voxel_platform PUBLIC voxel_core ext_volk ext_vma $<$<BOOL:${VOXEL_ENABLE_WINDOW}>:ext_glfw>)
voxel_set_warnings(voxel_platform)
voxel_enable_sanitizers(voxel_platform)
if(NOT MSVC)
  target_compile_options(voxel_platform PRIVATE -Wno-missing-field-initializers)
endif()

file(GLOB_RECURSE MATH_SRC CONFIGURE_DEPENDS math/*.cpp)
add_library(voxel_math STATIC ${MATH_SRC})
target_include_directories(voxel_math PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(voxel_math PUBLIC ext_glm)
voxel_set_warnings(voxel_math)
voxel_enable_sanitizers(voxel_math)

file(GLOB_RECURSE WORLD_SRC CONFIGURE_DEPENDS world/*.cpp)
add_library(voxel_world STATIC ${WORLD_SRC})
target_include_directories(voxel_world PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(voxel_world PUBLIC voxel_core voxel_math ext_xxhash)
voxel_set_warnings(voxel_world)
voxel_enable_sanitizers(voxel_world)

file(GLOB_RECURSE RENDER_SRC CONFIGURE_DEPENDS render/*.cpp)
add_library(voxel_render STATIC ${RENDER_SRC})
target_include_directories(voxel_render PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(voxel_render PUBLIC voxel_core voxel_math voxel_world)
target_compile_definitions(voxel_render PUBLIC
  VOXEL_BRICK_SIZE=${VOXEL_BRICK_SIZE}
  $<$<BOOL:${VOXEL_USE_TSDF}>:VOXEL_USE_TSDF=1>
  $<$<BOOL:${VOXEL_MATERIAL_4BIT}>:VOXEL_MATERIAL_4BIT=1>
  $<$<BOOL:${VOXEL_ENABLE_DENOISER}>:VOXEL_ENABLE_DENOISER=1>
)
voxel_set_warnings(voxel_render)
voxel_enable_sanitizers(voxel_render)

file(GLOB_RECURSE SIM_SRC CONFIGURE_DEPENDS sim/*.cpp)
add_library(voxel_sim STATIC ${SIM_SRC})
target_include_directories(voxel_sim PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(voxel_sim PUBLIC voxel_world voxel_math)
voxel_set_warnings(voxel_sim)
voxel_enable_sanitizers(voxel_sim)

file(GLOB_RECURSE TOOLS_SRC CONFIGURE_DEPENDS tools/*.cpp)
add_library(voxel_tools STATIC ${TOOLS_SRC})
target_include_directories(voxel_tools PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(voxel_tools PUBLIC voxel_core)
voxel_set_warnings(voxel_tools)
voxel_enable_sanitizers(voxel_tools)

# App executable
file(GLOB APP_SRC CONFIGURE_DEPENDS app/*.cpp main.cpp)
add_executable(voxel_app ${APP_SRC})
target_include_directories(voxel_app PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(voxel_app PRIVATE voxel_platform voxel_render voxel_tools voxel_sim)
voxel_set_warnings(voxel_app)
voxel_enable_sanitizers(voxel_app)
add_dependencies(voxel_app shaders_spv)
target_compile_definitions(voxel_app PRIVATE VOXEL_ENABLE_WINDOW=1)

# Copy shaders to runtime assets after build (if any exist)
add_custom_command(TARGET voxel_app POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E make_directory $<TARGET_FILE_DIR:voxel_app>/assets/shaders
  COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_BINARY_DIR}/shaders $<TARGET_FILE_DIR:voxel_app>/assets/shaders
  COMMAND ${CMAKE_COMMAND} -E make_directory $<TARGET_FILE_DIR:voxel_app>/assets/data
  COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_SOURCE_DIR}/data $<TARGET_FILE_DIR:voxel_app>/assets/data
  VERBATIM)
