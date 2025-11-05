# Voxel Planet Tech Demo (C++/Vulkan)

This repository scaffolds a GPU-first voxel renderer and spherical-world tech demo.

Current status
- Initial skeleton structure for code, shaders, tools, and tests.
- CMake options and helper modules wired up (warnings, sanitizers, shader build).
- See `docs/01-repository-layout.md` and `docs/02-build-system-and-dependencies.md` for intent.

Build
- Prereqs: CMake >= 3.24, C++20 compiler, Vulkan headers + shader compiler (`sudo apt install libvulkan-dev glslc` or install the Vulkan SDK), and X11 development headers so GLFW can configure (Ubuntu/Debian: `sudo apt install libxrandr-dev libxinerama-dev libxcursor-dev libxi-dev`).
- Configure: `cmake -S . -B build -G "Ninja" -DCMAKE_BUILD_TYPE=RelWithDebInfo`
- Build: `cmake --build build -j`
- Notes:
- Shaders compile to `build/shaders/*.spv` (also copied next to the app under `assets/shaders`).
- `build/assets/assets.pak` contains packed data from the `data/` folder; see “Toolchain” below.
- `build/assets/assets_index.txt` (tab-separated) mirrors the manifest for quick runtime lookup.
- Populate `extern/` with submodules (glm, spdlog, volk, VMA, tracy, stb, xxhash) or point CMake to your local installs. Placeholders are present but headers/libs are not vendored.

Tests
- After configuring the build directory, run `cmake --build build --target run_tests` (wraps `ctest --output-on-failure`).
- Address/UB sanitizers are enabled in Debug builds by default; LeakSanitizer is disabled for tests to keep CI stable.

Controls (planned)
- Free-fly camera (WASD + mouse).
- Surface walk aligned to local ENU frame.

Key ideas
- World is axis-aligned bricks in Cartesian space; spherical behavior comes from analytic shell clamp and planet signed field F(p).
- Shaders are placeholders for now.

Toolchain
- `glslc` (from the Vulkan SDK) is required at configure time. CMake locates it via `$GLSLC`, `$VULKAN_SDK`, or the system PATH.
- Shaders compile through `tools/shaderc_build/compile_shaders.py` (Python 3). The script writes SPIR-V into `build/shaders`, mirrors the layout under `build/assets/shaders`, and emits `build/shaders/manifest.json`.
- All runtime data in `data/` is packed by `tools/pack_assets` into `build/assets/assets.pak`, alongside `build/assets/assets_manifest.json` and `build/assets/assets_index.txt`. The index is a simple tab-separated table (`path`, `offset`, `size`, `hash`) used by the runtime loader.
- The app uses `core::AssetRegistry` to memory-map `assets.pak`, parse the index, and load files such as `materials.json` at startup.
- Rebuild shaders/assets manually with `cmake --build build --target shaders_spv assets_packed`.

Useful CMake options (ON/OFF)
- `VOXEL_BUILD_TESTS` (default ON)
- `VOXEL_ENABLE_VALIDATION` (default ON)
- `VOXEL_ENABLE_TRACY` (default OFF)
- `VOXEL_ENABLE_SANITIZERS` (default ON in Debug)
- `VOXEL_ENABLE_LTO` (default ON in Release)
- `VOXEL_SHADER_DEBUG` (default ON in Debug)
- Feature flags bridged to shaders: `VOXEL_BRICK_SIZE` (default 8), `VOXEL_USE_TSDF`, `VOXEL_MATERIAL_4BIT` (default ON), `VOXEL_ENABLE_DENOISER` (default ON)

Logging
- Uses `spdlog` with a color console sink by default.
- Configure at runtime:
  - Env: `VOXEL_LOG_LEVEL=trace|debug|info|warn|error|critical|off`, `VOXEL_LOG_FILE=/path/to/voxel.log`.
  - CLI: `--log-level <level>`, `--log-file <path>`, `--no-color`.
- Vulkan/GLFW validation/errors are routed through the logger.
