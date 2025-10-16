# Voxel Planet Tech Demo (C++/Vulkan)

This repository scaffolds a GPU-first voxel renderer and spherical-world tech demo.

Current status
- Initial skeleton structure for code, shaders, tools, and tests.
- CMake options and helper modules wired up (warnings, sanitizers, shader build).
- See `docs/01-repository-layout.md` and `docs/02-build-system-and-dependencies.md` for intent.

Build
- Prereqs: CMake >= 3.24, C++20 compiler, Vulkan SDK (for `glslc`).
- Configure: `cmake -S . -B build -G "Ninja" -DCMAKE_BUILD_TYPE=RelWithDebInfo`
- Build: `cmake --build build -j`
- Notes:
  - Shaders compile to `build/shaders/*.spv` (also copied next to the app under `assets/shaders`).
  - Populate `extern/` with submodules (glm, spdlog, volk, VMA, tracy, stb, xxhash) or point CMake to your local installs. Placeholders are present but headers/libs are not vendored.

Controls (planned)
- Free-fly camera (WASD + mouse).
- Surface walk aligned to local ENU frame.

Key ideas
- World is axis-aligned bricks in Cartesian space; spherical behavior comes from analytic shell clamp and planet signed field F(p).
- Shaders are placeholders for now.

Useful CMake options (ON/OFF)
- `VOXEL_BUILD_TESTS` (default ON)
- `VOXEL_ENABLE_VALIDATION` (default ON)
- `VOXEL_ENABLE_TRACY` (default OFF)
- `VOXEL_ENABLE_SANITIZERS` (default ON in Debug)
- `VOXEL_ENABLE_LTO` (default ON in Release)
- `VOXEL_SHADER_DEBUG` (default ON in Debug)
- Feature flags bridged to shaders: `VOXEL_BRICK_SIZE` (default 8), `VOXEL_USE_TSDF`, `VOXEL_MATERIAL_4BIT` (default ON), `VOXEL_ENABLE_DENOISER` (default ON)
