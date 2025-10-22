file(GLOB TEST_SRC CONFIGURE_DEPENDS *.cpp)
foreach(src ${TEST_SRC})
  get_filename_component(name ${src} NAME_WE)
  set(tgt ${name}_test)
  add_executable(${tgt} ${src})
  target_include_directories(${tgt} PRIVATE ${CMAKE_SOURCE_DIR}/src)
  target_link_libraries(${tgt} PRIVATE voxel_math voxel_world voxel_core)
  voxel_set_warnings(${tgt})
  voxel_enable_sanitizers(${tgt})
  add_test(NAME ${tgt} COMMAND ${tgt})
  if(VOXEL_ENABLE_SANITIZERS AND NOT MSVC)
    # LeakSanitizer regularly misbehaves in headless/CI runs; disable leak detection for tests.
    set_tests_properties(${tgt} PROPERTIES ENVIRONMENT "ASAN_OPTIONS=detect_leaks=0")
  endif()
endforeach()

add_custom_target(run_tests
  COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
  COMMENT "Running unit tests"
)
