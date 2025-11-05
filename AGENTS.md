# Repository Guidelines

## Project Structure & Module Organization
- **`src/`** — C++20 source grouped by feature: `app/` (entry, input), `platform/` (windowing, Vulkan device/swapchain), `core/` (frame graph, uploads, jobs), `math/` (spherical helpers), `world/` (brick store, streaming), `render/` (ray tracer, denoiser, overlays), `sim/` (future fluids/collision).
- **`shaders/`** — GLSL compute kernels (`generate_rays.comp`, `traverse_bricks.comp`, etc.) compiled to SPIR-V via CMake.
- **`data/`** — Runtime assets packed into `assets.pak` (`materials.json`, env maps, noise tables).
- **`tests/`** — Lightweight C++ executables invoked through CTest; expand here for unit coverage.
- **`docs/`** — Design notes and architecture walk-throughs; read `03_architecture_overview.md` before touching core systems.

## Build, Test, and Development Command
- `cmake -S . -B build -G "Ninja" -DCMAKE_BUILD_TYPE=RelWithDebInfo` — configure toolchain, locate `glslc`, stage shader steps.
- `cmake --build build -j` — build the app, shaders, packed assets, and tests.
- `cmake --build build --target shaders_spv` — recompile GLSL only (helpful while iterating on `.comp` files).
- `cmake --build build --target assets_packed` — regenerate `assets.pak` after editing `data/`.
- `cmake --build build --target run_tests` — execute unit tests via CTest; fails fast on assertion.
- `build/bin/voxel_planet` — launch the tech demo; expects `build/assets/` alongside the executable.

## Coding Style & Naming Conventions
- Follow existing formatting: 4-space indentation, brace on same line, no trailing whitespace. Use `clang-format` with the repo’s `.clang-format` when available; otherwise mirror current files.
- Classes/structs use `PascalCase`, functions and methods `camelCase`, constants `kPascalCase` (`kTimestampCount`), GL/Vulkan handles `snake_case`.
- Prefer explicit namespaces (`app::`, `world::`), RAII wrappers for Vulkan objects, and `glm` math types.

## Tooling & Automation
- A repo-provided pre-commit hook auto-runs `clang-format` on staged C/C++ files. Enable it once per clone:
  1. `git config core.hooksPath tools/git-hooks`
  2. Ensure `clang-format` is on your PATH (`brew install clang-format`, `apt install clang-format`, etc.).
- The hook reformats and re-stages files; review the diff after it runs so commits contain only intentional changes.

## Testing Guidelines
- Add focused tests under `tests/` (one executable per feature). Use simple `main()` or introduce a test framework if needed; register the binary in `tests/CMakeLists.txt`.
- Name test sources `<Feature>Tests.cpp` and keep them deterministic (seeded noise, fixed cameras).
- Run `cmake --build build --target run_tests` locally before commits; ensure new tests fail without your change and pass after.

## Commit & Pull Request Guidelines
- Commits follow the existing imperative, capitalized style (`Add surface walk camera mode`, `Fix descriptor refresh after streaming`). Keep messages under ~72 characters and wrap longer explanations in the body.
- Each PR should: describe the feature/fix, link related issues, outline validation (build/tests run), and attach screenshots or GPU timings if visuals/perf change.
- Group related changes; avoid mixing refactors with functional edits unless necessary. Update docs/tests alongside code.

## Architecture & Agent Tips
- Respect the wavefront pipeline: edits must update CPU `BrickStore`, GPU SSBOs, and descriptor bindings consistently.
- When touching streaming, test with `enableStreaming` toggled on in `App::run`; watch log output for residency counts.
- Instruments (`Tracy`, GPU timestamps) are wired in—capture traces when performance shifts more than ±10%.

## Agent Notes
- Investigated cliffy terrain; replaced lattice hash with PCG-based mix in `src/math/Spherical.cpp` and mirrored in `shaders/spherical.glsl`. Continuity test in `tests/MathTests.cpp` now checks local stability. Large-scale continents should be back once you rebuild.
