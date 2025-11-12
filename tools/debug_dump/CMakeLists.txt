add_executable(dump_brick dump_brick.cpp)

set_target_properties(dump_brick PROPERTIES OUTPUT_NAME dump_brick)

target_include_directories(dump_brick PRIVATE ${CMAKE_SOURCE_DIR}/src)

target_link_libraries(dump_brick PRIVATE
  voxel_world
  voxel_math
  voxel_core)

voxel_set_warnings(dump_brick)
voxel_enable_sanitizers(dump_brick)

target_compile_definitions(dump_brick PRIVATE
  VOXEL_BRICK_SIZE=${VOXEL_BRICK_SIZE})
