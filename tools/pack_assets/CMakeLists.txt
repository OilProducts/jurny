cmake_minimum_required(VERSION 3.24)
project(pack_assets)

add_executable(pack_assets pack_assets.cpp)
target_compile_features(pack_assets PRIVATE cxx_std_20)

