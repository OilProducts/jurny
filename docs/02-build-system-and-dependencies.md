Part 2 — Build System & Dependencies (C++20 + Vulkan)

Goals
- Fast, reproducible builds on Windows and Linux.
- Deterministic offline shader compilation to SPIR‑V with hot‑reload.
- Clear dependency boundaries (vendored or FetchContent) with version pins.
- Strong validation/profiling in Debug; small, optimized binaries in Release.
- Build options to toggle experimental features without code churn.

Outcomes
- A top‑level CMake project that builds: app, engine libs, shaders, tools, and tests.
- A shader build pipeline that mirrors runtime descriptor layouts.
- Vulkan device bootstrap with required features/extensions enabled.
- CI config that compiles on at least Windows (MSVC) and Linux (Clang/GCC).

Repository Layout (relevant to builds)
- `CMakeLists.txt` — top‑level; config, options, external deps, subdirs.
- `cmake/` — CMake helpers (toolchain, functions for shaders, warnings, sanitizers).
- `extern/` — third‑party deps (submodules or FetchContent fallbacks).
- `tools/` — build helpers (shader compiler wrappers, asset packer).
- `shaders/` — GLSL (`*.comp`, includes under `shaders/include/`).
- `src/` — engine/app code; subdirs become CMake targets.
- `tests/` — unit tests.

Toolchain & Prerequisites
- Compilers
  - Linux: Clang 15+ or GCC 12+.
  - Windows: MSVC (VS 2022, v143 toolset) or Clang‑CL 15+.
- Build tools: CMake 3.24+, Ninja.
- Vulkan SDK 1.3+ installed (sets headers, loader, `glslc`).
- Optional: Python 3.9+ for small build utilities (e.g., asset packer).

Third‑Party Dependencies (extern/)
- `volk` — Vulkan function loader (static init, single translation unit include).
- `VMA` — Vulkan Memory Allocator (GPU memory management).
- `glm` — math library (CPU math; see compile defs below).
- `spdlog` — logging.
- `tracy` — CPU/GPU profiler (zones + Vulkan context).
- `stb` — images (stb_image, stb_image_write) for tools/tests.
- `xxhash` — fast hashing (e.g., brick keys, cache IDs).
- Optional (later): `meshoptimizer`, `glfw`/`SDL` (if not already present via platform layer), `imgui` for debug UI.

Dependency Strategy
- Prefer git submodules in `extern/` for deterministic versions.
- Provide CMake `FetchContent` fallbacks for contributors building without submodules.
- Wrap each dep in a thin CMake target with a unified include interface; do not leak global compile options.

Global Build Options (CMake)
- `VOXEL_BUILD_TESTS` (ON/OFF) — enable unit tests.
- `VOXEL_ENABLE_VALIDATION` (ON/OFF, default ON for Debug) — Vulkan validation layers.
- `VOXEL_ENABLE_TRACY` (ON/OFF) — profiler integration.
- `VOXEL_ENABLE_SANITIZERS` (ON/OFF, on Linux/Clang in Debug) — ASan/UBSan/TSan.
- `VOXEL_ENABLE_LTO` (ON/OFF, default ON for Release) — link‑time optimization.
- `VOXEL_SHADER_DEBUG` (ON/OFF, default ON for Debug) — `-g` in SPIR‑V.
- `VOXEL_USE_FETCHCONTENT` (ON/OFF) — bypass submodules.

Compiler & Warnings
- C++ standard: C++20, no compiler extensions.
- Common defines:
  - `NOMINMAX` on Windows.
  - `GLM_FORCE_RADIANS`, `GLM_FORCE_DEPTH_ZERO_TO_ONE` (Vulkan NDC), `GLM_ENABLE_EXPERIMENTAL` where needed.
- Warnings presets:
  - MSVC: `/W4 /permissive- /Zc:preprocessor`
  - Clang/GCC: `-Wall -Wextra -Wshadow -Wconversion -Wpedantic` (tune conversions).
- Sanitizers (when enabled): `-fsanitize=address,undefined` (and `thread` for tests only).
- LTO: `-flto=thin` (Clang) or `/GL` (MSVC). Ensure all static libs are built with LTO in Release.

Configurations
- `Debug` — Validation layers, Tracy, shader debug info, no LTO, sanitizers ON where supported.
- `RelWithDebInfo` — Validation togglable, partial optimizations, debug info for profiling.
- `Release` — Validation OFF, LTO ON, assertions minimal.

Vulkan Integration
- Loader: initialize `volk` once in platform bootstrap (`platform/VulkanContext.cpp`).
- Target API: Vulkan 1.3 baseline to reduce extension toggles.
- Required features/extensions (enable via feature chain):
  - `VK_KHR_timeline_semaphore` (core in 1.3) — timeline sync.
  - `VK_EXT_descriptor_indexing` (core bits in 1.2) — bindless‑style SSBO arrays.
  - `bufferDeviceAddress` — optional; nice for advanced alloc strategies.
  - `shaderInt64` — if using 64‑bit hashing on GPU (optional; can pack to u32 otherwise).
- Debug & validation:
  - Enable layers in Debug; set message filters to error+warning; crash (assert) on error.
  - Use debug utils for object names and markers; Tracy’s Vulkan context for GPU zones.

GPU Memory (VMA)
- Create one `VmaAllocator` with device/instance/dispatch from `volk`.
- Heaps:
  - `DeviceLocal` — big SSBOs (brick pools, queues), images (framebuffers).
  - `HostVisible` — staging/upload rings, readback.
- Patterns:
  - Use persistently mapped staging buffers, sub‑allocations for uploads.
  - Prefer `VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT` for all runtime‑critical resources.
  - For storage images, choose `VK_FORMAT_R16G16B16A16_SFLOAT` for accum/denoise buffers.

Shader Toolchain (GLSL → SPIR‑V)
- Source layout
  - `shaders/*.comp` — compute kernels: `generate_rays`, `traverse_bricks`, `shade`, `denoise_atrous`, `composite`, `updates_apply`, `sdf_update`.
  - `shaders/include/` — shared headers: `common.glsl`, bindings, structs, hash lookup helpers.
- Compiler
  - Use `glslc` (from the Vulkan SDK) or `shaderc` via a CMake custom command.
  - Target: `--target-env=vulkan1.3`.
  - Debug: `-g` (if `VOXEL_SHADER_DEBUG=ON`), `-O` tuned per config (`-O` Debug, `-O`/`-Os` Release test for performance vs size).
  - `-I shaders/include` for includes; define `-D` switches for feature flags shared with C++.
- Outputs
  - Emit `.spv` into `build/shaders/<name>.spv` (mirrors source tree).
  - Optionally emit a small `.meta` (JSON) containing binding set/layout hashes for hot reload sanity.
- Hot reload
  - Watch `.spv` mtimes; on change, destroy/recreate affected pipelines and descriptor sets safely (defer to next frame).

Descriptor Layout (agree between C++ and GLSL)
- Set 0 — Globals
  - `binding 0`: `GlobalsUBO` (UBO, 256B aligned)
  - `binding 1`: RNG/blue‑noise (SSBO or sampled texture)
- Set 1 — World
  - Brick headers (SSBO)
  - Occupancy bitset (SSBO)
  - Material index stream (SSBO)
  - Palette buffer (SSBO)
  - Hash tables (SSBOs)
  - Materials table (UBO/SSBO)
  - Macro masks / LOD (SSBO)
  - TSDF pool (SSBO) [optional]
- Set 2 — Queues (SSBOs with ring headers)
  - Primary rays, hits, misses, secondary rays
- Set 3 — Frame images (storage images)
  - Accum, albedo, normal, variance/moments, velocity, output

Globals UBO Contract (C++)
- `src/math/Spherical.h` and `shaders/include/common.glsl` define an identical struct.
- Contains: camera matrices (curr/prev), origin rebase delta, frame index, planet radii (`Rin`, `Rout`, `Rsea`), voxel/brick size, image dims, flags.
- Ensure `sizeof(GlobalsUBO) % 256 == 0` for std140 alignment.

Source Code Targets (CMake)
- `voxel_core` — core utilities (jobs, frame graph, descriptors, pipelines, upload helpers).
- `voxel_platform` — window + swapchain + Vulkan context bootstrap.
- `voxel_math` — spherical math, RNG, camera.
- `voxel_world` — brick store, region cache, streaming, materials, worldgen.
- `voxel_render` — raytracer, denoiser, tonemap, GPU buffers.
- `voxel_tools` — overlays, capture.
- `voxel_app` — executable; ties everything together.
- `voxel_tests` — unit tests.

Top‑Level CMake (outline)
- Minimum version and policies; set C++ standard, position independent code for static libs.
- Options listed above; default config (e.g., RelWithDebInfo if none specified).
- Add `extern/` (subdirectories or FetchContent); create interface targets per dep.
- Add `cmake/Warnings.cmake`, `cmake/Sanitizers.cmake`, `cmake/Shaders.cmake` and include them.
- Add subdirectories: `src`, `shaders`, `tools`, `tests` (guarded by options).

Shader & Asset Toolchain
- `voxel_compile_shaders()` invokes the Python helper `tools/shaderc_build/compile_shaders.py`:
  - Scans `shaders/` for `*.comp` sources.
  - Calls `glslc` with the project feature macros; outputs live in `build/shaders/<rel>.spv`.
  - Copies the same layout into `build/assets/shaders/` and writes `build/shaders/manifest.json` (hashes, sizes, compile times).
  - Requires Python 3 and a Vulkan SDK providing `glslc` (resolved via `$GLSLC`, `$VULKAN_SDK`, or PATH). Outputs are tracked via a single `.stamp` target to keep Ninja happy.
- `tools/pack_assets` merges the entire `data/` tree into `build/assets/assets.pak` with a JSON manifest `build/assets/assets_manifest.json` (offsets, sizes, XXHash64 checksums). The `assets_packed` custom target runs automatically and `voxel_app` depends on it.

Install & Runtime Layout
- `bin/` — executables and shared libs.
- `assets/` — generated build artifacts (SPIR‑V, packed data blobs, manifests).
- `config/` — runtime config (optional).

Vulkan Validation & Debugging
- In Debug builds:
  - Enable validation layers; set message severity filter to warning+error.
  - Install a debug message callback that logs via `spdlog` and triggers a debugbreak on error.
  - Name all Vulkan objects with `vkSetDebugUtilsObjectNameEXT`.
- Tracy GPU profiling:
  - Use `TracyVkContext` with a timestamp period; wrap each compute pass with zones.

Static Analysis (optional but recommended)
- Clang‑Tidy: provide a `clang-tidy` preset and a `CMAKE_CXX_CLANG_TIDY` hook for CI runs.
- Cppcheck: optional target.

Continuous Integration (CI) Skeleton
- GitHub Actions workflow (`.github/workflows/ci.yml`) builds and tests every push/PR on Ubuntu:
  - Installs Ninja, Vulkan headers, and `shaderc` (for `glslc`).
  - Configures with tests + sanitizers enabled.
  - Builds via Ninja and executes `run_tests` (ctest with leak detection disabled).
- Windows job *(future work)* — use VS2022 image with Vulkan SDK installed; mirror the Linux steps once dependencies are scripted.

-Environment & Developer Experience
- Scripts
  - `tools/dev_watch.py` — file watcher that rebuilds shaders and restarts the app on change.
  - `tools/shaderc_build/compile_shaders.py` — helper around `glslc`; accepts `--source`, `--output-dir`, `--copy-dir`, `--manifest`, `--define`, etc. The CMake function `voxel_compile_shaders()` invokes it automatically.
  - `tools/pack_assets` — packs the entire `data/` directory into `assets.pak` plus `assets_manifest.json`. Supports `--data-dir`, `--out-dir`, `--pak`, `--manifest`, and `--verbose`.
- Local run
  - Set `VK_LAYER_PATH` and `VK_INSTANCE_LAYERS=VK_LAYER_KHRONOS_validation` in Debug if SDK doesn’t auto‑register.
  - Ensure `assets/` folder is next to the executable or configure a search path.

Platform Notes
- Linux
  - Prefer Clang for consistent diagnostics and sanitizer coverage.
  - Ensure udev permissions allow GPU access in CI containers.
- Windows
  - Use `/Z7` or `/Zi` PDBs in RelWithDebInfo for profiling.
  - Set `DXVK_LOG_LEVEL=none` if using Vulkan layers that might spam.

Feature Flags Bridging (C++ ↔ GLSL)
- Define compile options in CMake and mirror them as shader `-D` macros to keep CPU/GPU codepaths in sync:
  - `VOXEL_USE_TSDF` — enable TSDF tiles.
  - `VOXEL_MATERIAL_4BIT` — use 4‑bit per‑voxel indices (fallback to 8‑bit when flagged).
  - `VOXEL_BRICK_SIZE=8` — specialization constant in shaders; C++ compile‑time constant for layout code.
  - `VOXEL_ENABLE_DENOISER` — include denoise passes.

Linkage & Binaries
- Build as static libs per module (`voxel_*`) linked into `voxel_app` to keep link times reasonable and provide clear boundaries.
- For Release and RelWithDebInfo, enable ThinLTO where supported; verify link order for whole‑archive needs (rare here).

Testing Strategy
- Header‑only test framework (e.g., doctest) or gtest if you prefer.
- Unit tests for math (sphere shell intersection, ENU basis), brick addressing, and small DDA sequences.
- GPU tests are smoke‑tests that run a frame with a fixed seed and assert counters (e.g., rays/hits > 0, queues not overflowing).

Security & Reproducibility
- Treat shader binaries as build artifacts (not checked in); rebuild on demand.
- Pin third‑party commits in submodules; mirror to FetchContent by tag+SHA.
- Avoid downloading tools during configure; document SDK install explicitly.

Milestones for Part 2
- M2.1 — Bring up CMake skeleton, warnings, options, and extern deps.
- M2.2 — Add shader pipeline target; compile a no‑op compute shader to SPIR‑V.
- M2.3 — Implement `VulkanContext` bootstrap (instance/device/queues), validation, and `volk` init.
- M2.4 — Integrate VMA and a basic `GpuAllocator` with upload staging.
- M2.5 — Wire Descriptor sets scaffolding to match shader bindings and create placeholder pipelines.
- M2.6 — CI builds on Linux/Windows; tests run; artifacts package shaders alongside the binary.

Appendix A — CMake Fragments (Reference)
- Top‑level options and presets

  set(CMAKE_CXX_STANDARD 20)
  option(VOXEL_ENABLE_VALIDATION "Enable Vulkan validation" ON)
  option(VOXEL_ENABLE_TRACY "Enable Tracy profiling" ON)
  option(VOXEL_ENABLE_SANITIZERS "Sanitizers in Debug" ON)
  option(VOXEL_ENABLE_LTO "Enable LTO in Release" ON)
  option(VOXEL_BUILD_TESTS "Build tests" ON)

- Warnings (Clang/GCC)

  function(voxel_set_warnings target)
    if (CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
      target_compile_options(${target} PRIVATE -Wall -Wextra -Wpedantic -Wshadow)
    elseif (MSVC)
      target_compile_options(${target} PRIVATE /W4 /permissive-)
    endif()
  endfunction()

- Shaders (per‑file rule)

  function(compile_glsl name)
    set(src ${ARGV1})
    set(out ${CMAKE_BINARY_DIR}/shaders/${name}.spv)
    add_custom_command(
      OUTPUT ${out}
      COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_BINARY_DIR}/shaders
      COMMAND ${GLSLC_PROGRAM} -c --target-env=vulkan1.3
              $<$<CONFIG:Debug>:-g> -O
              -I ${CMAKE_SOURCE_DIR}/shaders/include
              -o ${out} ${src}
      DEPENDS ${src}
      COMMENT "Compiling GLSL ${name}.comp → SPV"
      VERBATIM)
    set_property(DIRECTORY APPEND PROPERTY ADDITIONAL_MAKE_CLEAN_FILES ${out})
    set(${name}_SPV ${out} PARENT_SCOPE)
  endfunction()

Appendix B — Vulkan Feature Enablement (Pseudo‑code)
- Query features and chain enablements:

  VkPhysicalDeviceDescriptorIndexingFeatures indexing{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_INDEXING_FEATURES };
  VkPhysicalDeviceTimelineSemaphoreFeatures timeline{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TIMELINE_SEMAPHORE_FEATURES };
  VkPhysicalDeviceBufferDeviceAddressFeatures bda{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_FEATURES };
  indexing.pNext = &timeline; timeline.pNext = &bda; // chain

  VkDeviceCreateInfo dci{ VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO };
  dci.pNext = &indexing;
  dci.pEnabledFeatures = &core10Features; // core
  vkCreateDevice(..., &dci, ...);

Appendix C — Shared Compile Definitions
- C++ targets:
  - `GLM_FORCE_RADIANS`
  - `GLM_FORCE_DEPTH_ZERO_TO_ONE`
  - `VOXEL_ENABLE_TRACY` (when ON)
  - `VOXEL_DEBUG_VALIDATION` (Debug only)
- GLSL compile `-D`:
  - `VOXEL_BRICK_SIZE=8`
  - `VOXEL_USE_TSDF=1` (toggle)
  - `VOXEL_SHADER_DEBUG=1` in Debug
