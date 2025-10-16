Part 1 — Repository Layout (Detailed Outline)

Purpose
- Define a clean, engine-ready directory structure that scales from a tech demo to a game engine.
- Make spherical-world math and GPU-first rendering explicit in where code and data live.
- Enable fast iteration: clear ownership, stable include paths, simple build rules, hot-reload for shaders.

Top-Level Tree (annotated)

voxel-planet/
├─ CMakeLists.txt                # Root build; adds extern/, src/, shaders/, tools/
├─ README.md                     # Quickstart, build/run, controls
├─ LICENSE                       # Project license
├─ .clang-format                 # (Optional) house style
├─ .editorconfig                 # Tabs/spaces, EOLs, encodings
├─ .gitattributes                # Normalize line endings; mark binaries (e.g., .spv)
├─ .gitignore                    # Build outputs, cache, captures
│
├─ extern/                       # Third-party libs (as submodules or FetchContent)
│  ├─ volk/                      # Vulkan loader
│  ├─ VMA/                       # Vulkan Memory Allocator
│  ├─ glm/                       # Math (vec/mat/quaternion); use doubles on CPU where noted
│  ├─ spdlog/                    # Logging
│  ├─ tracy/                     # CPU+GPU profiler
│  ├─ stb/                       # stb_image, stb_image_write
│  └─ xxhash/                    # Fast hashing for keys, materials, assets
│
├─ tools/                        # Host-side tooling (no engine runtime deps)
│  ├─ shaderc_build/             # Offline GLSL→SPIR-V compile scripts/wrapper
│  │  ├─ CMakeLists.txt
│  │  ├─ compile_shaders.py      # Scans shaders/, emits build/shaders/*.spv
│  │  └─ shader_pch.glsl         # Common #includes injected during compile
│  ├─ pack_assets/               # Packs JSON/materials/envmaps → binary blobs
│  │  ├─ CMakeLists.txt
│  │  └─ pack_assets.cpp
│  └─ telemetry/                 # Post-process Tracy captures / frame stats
│
├─ data/                         # Runtime-readable assets (small; hashed & packed)
│  ├─ materials.json             # PBR entries (IDs→params)
│  ├─ envmaps/                   # Sky cubemaps/HDRIs for early testing
│  └─ noise_tables.bin           # Prebaked blue-noise/LDS sequences (optional)
│
├─ shaders/                      # GLSL compute (compiled to SPIR-V at build time)
│  ├─ common.glsl                # Types, descriptor bindings, RNG, packing helpers
│  ├─ spherical.glsl             # F(p), ∇F, sphere-shell clamp; matches math/Spherical.h
│  ├─ generate_rays.comp         # K0
│  ├─ traverse_bricks.comp       # K1 (primary/secondary traversal via queues)
│  ├─ shade.comp                 # K2 (materials, NEE, env)
│  ├─ denoise_atrous.comp        # SVGF/A-trous
│  ├─ composite.comp             # Tonemap + overlays to swapchain
│  ├─ updates_apply.comp         # Batched voxel edit apply on GPU
│  └─ sdf_update.comp            # Optional TSDF (jump-flood / fast-sweep)
│
├─ src/                          # Engine/tech-demo source (C++20)
│  ├─ app/
│  │  ├─ App.h / App.cpp         # Owns main loop, frame graph orchestration
│  │  ├─ Input.h / Input.cpp     # Input devices, camera modes (fly/walk)
│  │  └─ UI.h / UI.cpp           # ImGui (optional) overlays & controls
│  ├─ platform/
│  │  ├─ Window.h / Window.cpp   # GLFW/SDL wrapper; DPI, resize, input hookup
│  │  ├─ VulkanContext.h/.cpp    # Instance/device/queues/features; validation toggles
│  │  └─ Swapchain.h/.cpp        # Swapchain, present images, format/extent management
│  ├─ core/
│  │  ├─ FrameGraph.h/.cpp       # Lightweight pass scheduling; resources & barriers
│  │  ├─ Descriptors.h/.cpp      # Set layouts, pools, bindless indexing
│  │  ├─ Pipelines.h/.cpp        # Compute pipeline cache, specialization constants, hot-reload
│  │  ├─ GpuAllocator.h/.cpp     # VMA wrapper (buffers/images, suballocation helpers)
│  │  ├─ Upload.h/.cpp           # Staging rings, timeline semaphores, batch uploads
│  │  ├─ Jobs.h/.cpp             # Thread pool for streaming & worldgen
│  │  ├─ GpuTimers.h/.cpp        # Timestamp queries + averages for passes
│  │  └─ Debug.h/.cpp            # Validation hooks, crash logs, asserts
│  ├─ math/
│  │  ├─ Spherical.h/.cpp        # Planet math: F(p), ∇F, ENU, shell/AABB overlap, gravity
│  │  ├─ Camera.h/.cpp           # Jittered projection, prev/curr matrices, origin rebase
│  │  └─ RNG.h/.cpp              # PCG32 per-pixel seeds; blue-noise helpers
│  ├─ world/
│  │  ├─ BrickFormats.h          # Packed GPU layouts, flags, palette formats
│  │  ├─ BrickStore.h/.cpp       # CPU+GPU brick pools; edits; device offsets; residency
│  │  ├─ RegionCache.h/.cpp      # 64^3 brick regions; LRU/clock eviction; stats
│  │  ├─ Materials.h/.cpp        # Global PBR table; palette validation; GPU upload
│  │  ├─ WorldGen.h/.cpp         # Deterministic H(û), caves C(p), materials rules
│  │  ├─ MacroMask.h/.cpp        # Empty/full/mixed masks for skipping
│  │  └─ Streaming.h/.cpp        # Radius+angle policy; AABB vs shell; scoring; jobs
│  ├─ render/
│  │  ├─ Raytracer.h/.cpp        # Wavefront orchestration; queue management; pass binds
│  │  ├─ GpuBuffers.h/.cpp       # Ray/Hit/Miss/Secondary queues; frame images; history
│  │  ├─ Denoiser.h/.cpp         # Temporal reprojection, A-trous; motion vectors
│  │  ├─ Tonemap.h/.cpp          # Filmic/ACES, exposure, UI toggles
│  │  └─ Overlays.h/.cpp         # Debug visualizations: heatmaps, normals, shell cull
│  ├─ sim/                       # (Optional early) scoped simulations
│  │  ├─ Fluids.h/.cpp           # Active AABB MAC grid; brick-channel allocation
│  │  └─ Collide.h/.cpp          # TSDF/analytic SDF collisions; jump-flood updates
│  └─ tools/
│     ├─ Capture.h/.cpp          # Golden frame capture/replay; RNG seeds; repro buffers
│     └─ Inspect.h/.cpp          # Brick/region inspectors; palette/material browsers
│
├─ tests/                        # Unit tests (gtest or similar)
│  ├─ MathTests.cpp              # Shell intersection, gradients, ENU, gravity vectors
│  ├─ BrickTests.cpp             # Micro-DDA sequences, palette overflow fallback
│  ├─ HashTests.cpp              # packKey/lookup; probe lengths; collision stats
│  └─ StreamingTests.cpp         # AABB vs shell; region scoring invariants
│
└─ docs/                         # Design docs (this file and peers)
   ├─ 01-repository-layout.md    # You are here
   ├─ 02-descriptor-layouts.md   # Bindings, sets, push constants (authoritative)
   ├─ 03-frame-graph.md          # Pass DAG, resources, barriers, timings
   └─ 04-brick-data-format.md    # Offsets, packing, alignment, memory budgets

Conventions & Ownership
- One module per directory, clear APIs in headers, minimal cross-deps.
- CPU world math uses double precision (glm::dvec); GPU uses float (vec) with per-frame origin rebase.
- All GPU-visible binary formats live in world/BrickFormats.h and docs/04-brick-data-format.md and must remain in sync with shaders/common.glsl.
- Shaders compiled offline to build/shaders/*.spv; runtime hot-reload on timestamp change.
- Keep tests close to code semantics; no heavy mocks. Math/algorithms are unit tested; GPU results validated via golden-frame captures.

Module Details (what goes where and why)

extern/
- Consume via git submodules or CMake FetchContent; pin versions.
- volk: static loader; no global loader dependence.
- VMA: allocator for buffers/images; wrap in core/GpuAllocator.
- tracy: CPU zones; GPU timestamp groups per pass. Build flags gated by CMake options.

tools/
- shaderc_build/: entry points to compile all shaders to SPIR-V; supports dependency scanning (#include graph) and macro toggles for debug vs release.
- pack_assets/: read materials.json/envmaps and pack into a single binary blob with a JSON index (content-hash keys). Output lives in build/assets/*.bin.
- telemetry/: scripts to summarize Tracy captures; output tables/plots in build/telemetry/.

data/
- Checked-in small, text-based assets for early bring-up (materials.json). Larger assets should be packed; original sources may live outside the repo.
- envmaps/ contains small LDR/HDR samples to validate sky/IBL paths. Replace with an atmospheric sky later.

shaders/
- common.glsl: shared types (Ray, Hit, QueueHeader), descriptor set numbers/bindings (authoritative mirror of docs/02-descriptor-layouts.md), RNG, bit ops for 4-bit material indices.
- spherical.glsl: F(p) (crust field), gradF(), IntersectSphereShell(), AABB-vs-shell helpers for GPU usage; math consistency with math/Spherical.h.
- generate_rays.comp: creates primary rays (origin-rebased camera), TAA jitter, per-pixel RNG seeds → RayQueueIn.
- traverse_bricks.comp: sphere-shell clamp → brick DDA → micro-DDA in 8×8×8 → isosurface refinement (F=0) → normal from ∇F/TSDF → Hit/Miss queues. Persistent threads + chunked queue ops.
- shade.comp: PBR shading (albedo, roughness, metalness, emission, absorption), env, shadow rays (optional). Spawns limited secondaries.
- denoise_atrous.comp: SVGF/A-trous using albedo/normal/velocity and variance; temporal reprojection with motion vectors aware of origin rebase.
- composite.comp: tonemap to swapchain; overlay debug.
- updates_apply.comp: merge per-frame voxel edits into brick SSBOs (occupancy/material indices/palettes).
- sdf_update.comp: optional TSDF tile refresh via jump-flood / fast-sweep for edited bricks.

src/app/
- App: sets up VulkanContext, Swapchain, FrameGraph; owns the per-frame OriginRebase (world origin = camera world pos) and uploads GlobalsUBO.
- Input: keyboard/mouse/gamepad, camera modes (free-fly and surface-walk: walk aligns to ENU at ground hit point).
- UI: toggles (exposure, spp, bounces), overlays (shell culling ring, residency heatmap), profiler readouts.

src/platform/
- Window: GLFW/SDL wrapper; vsync toggle; DPI-awareness; surface creation.
- VulkanContext: instance/device selection (features for descriptor indexing, timeline semaphores); command pools/buffers.
- Swapchain: format/extent management, image views; acquire/present with fences.

src/core/
- FrameGraph: tiny struct-of-passes with declared read/write resources; emits vkCmdDispatch + barriers in order; timeline semaphore values per stage.
- Descriptors: Set 0 (Globals UBO + blue-noise), Set 1 (world SSBOs/palettes/materials), Set 2 (queues), Set 3 (frame images). Optional bindless indexing.
- Pipelines: create/own VkPipelines (compute); hot-reload if SPIR-V mtime changes; specialization constants for brick size, feature toggles.
- GpuAllocator: VMA-backed; helpers for SSBOs with 16B alignment; buffer aliasing for 4-bit index streams.
- Upload: persistent staging buffers; batched buffer-to-buffer copies; brick uploads grouped by region.
- Jobs: thread pool for streaming/worldgen; queues are MPMC; simple work-stealing.
- GpuTimers: per-pass timestamps; averages/percentiles for overlays.
- Debug: asserts (palette ≤16, queue capacities), validation layer messages → spdlog.

src/math/
- Spherical: CPU reference implementations for analytic F(p) (crust), gradF by central differences, IntersectSphereShell (inner/outer radii), AABB vs shell overlap; ENU basis; gravity g(p) = -g0 * normalize(p) (or inverse-square). This mirrors shaders/spherical.glsl.
- Camera: stores prev/curr view/proj, jitter, and origin-rebase (double world origin on CPU; float local on GPU). Provides motion vectors consistent across rebases.
- RNG: PCG32 for per-pixel seeds; blue-noise sampling helpers for stratified spp.

src/world/
- BrickFormats: packed GPU layouts, headers (bx/by/bz, offsets, flags), occupancy bitset encoding (8×u64), 4-bit/8-bit material index streams, palette encoding, TSDF tile format (optional, 16-bit truncation radius & scale).
- BrickStore: CPU-side brick creation, edit batching, offset packing, residency state; mirrors GPU buffers (headers/occupancy/indices/palettes/TSDF). UploadBatch() encodes contiguous regions for fewer vkCmdCopy.
- RegionCache: buckets bricks into 64^3 regions for streaming and eviction; AABB for regions; maintains LRU/clock and scores.
- Materials: loads materials.json; assigns stable IDs; builds GPU table/SSBO; palette validation and overflow fallback (auto switch to 8-bit for a brick).
- WorldGen: deterministic procedural rules (height on unit sphere, caves in 3D) for first-touch brick synthesis; material assignment by altitude/slope/latitude.
- MacroMask: empty/full/mixed masks per coarse macro-cell to enable traversal skipping.
- Streaming: radius band [Rin,Rout] + angular window policy around camera; brick/region scoring (angle, sea-level proximity, sim-active, age); schedules jobs and uploads.

src/render/
- GpuBuffers: allocates queue SSBOs (Ray/Hit/Miss/Secondary) and per-frame images (accumColor, normal, albedo, variance, velocity, output); ring sizes sized to resolution*spp with head/tail atomics.
- Raytracer: dispatch order (K0→K1→K2→K1→K2), persistent threads, global barriers between passes, queue overflow handling (drop secondaries first); uploads GlobalsUBO each frame.
- Denoiser: temporal reprojection using prev matrices + originDelta; variance estimation; A-trous iterations with albedo/normal weighting; history clamping.
- Tonemap: ACES/filmic; exposure from UI or auto.
- Overlays: per-pixel overlays (normals, curvature, material IDs), macro-mask visualization, residency heatmap, sphere-shell culling stats.

src/sim/ (optional early)
- Fluids: allocate MAC grid channels only inside active AABBs; pressure projection (PCG or GS) with solid walls from occupancy; write liquid channel for bricks; release when inactive.
- Collide: coarse analytic SDF for crust; per-region TSDF for edits/destruction; jump-flood updates for dirty regions; rigid-body sleeping returns clusters to static bricks.

src/tools/
- Capture: frame capture/replay (camera pose, RNG seed, toggles, minimal world deltas) for reproducible tests.
- Inspect: debug UIs for bricks/regions/materials; live palette edits; queue depths; timer histograms.

tests/
- MathTests: numeric checks for Spherical (edge cases near poles; shell clamp intervals), gradient finite-diff sanity, AABB vs shell.
- BrickTests: micro-DDA path on synthetic masks; palette overflow fallback to 8-bit; TSDF halo correctness across brick borders.
- HashTests: key packing, probe lengths under chosen load factor; collision distribution.
- StreamingTests: overlapsShell invariants; region scoring monotonicity.

docs/
- 01-repository-layout.md: detailed rationale and responsibilities per directory.
- 02-descriptor-layouts.md: authoritative binding numbers, set layouts, push constants; kept in sync with shaders/common.glsl and core/Descriptors.
- 03-frame-graph.md: pass graph order, resources, barriers, timeline semaphore values.
- 04-brick-data-format.md: exact byte layouts; occupancy packing; 4-bit/8-bit index streams; TSDF tile format; alignment; budgets.

Build System (CMake) Guidelines
- Root CMakeLists.txt adds extern/, shaders/ (SPIR-V build), tools/ (optional), src/ (engine), tests/.
- Options: VOXEL_BUILD_TOOLS, VOXEL_ENABLE_TRACY, VOXEL_ENABLE_VALIDATION, VOXEL_USE_BINDLESS.
- Shader compile target regenerates when GLSL changes; emits build/shaders/*.spv; add_custom_command with file lists from tools/shaderc_build/.
- Enforce C++20; warnings-as-errors in CI; disable exceptions in engine (ok in tools/tests if desired).

Descriptor Sets (where they live)
- Document bindings in docs/02-descriptor-layouts.md and mirror in shaders/common.glsl and core/Descriptors.
- Set 0: Globals UBO + RNG/blue-noise. Set 1: World SSBOs (headers, occupancy, indices, palettes, materials, macro masks, TSDF). Set 2: Queues SSBOs. Set 3: Frame images.

Data & Asset Pipeline
- Source-of-truth for materials: data/materials.json → packed by tools/pack_assets/ into build/assets/materials.bin; engine maps material IDs to indices.
- envmaps/ are small and optional; replace with atmospheric sky model later.
- Large assets must not be checked in raw; pack or fetch at build time.

Naming & Coding Style (short)
- Filenames: PascalCase for classes (FooBar.h/.cpp); snake_case.glsl for shaders.
- Namespaces: app, platform, core, math, world, render, sim, tools.
- Header-only where it buys clarity (small math utilities); otherwise .h/.cpp pairs.
- Avoid global singletons; pass contexts explicitly (VulkanContext&, Descriptors&, Pipelines&).

Spherical-World Hooks in Layout (why this structure fits the math)
- math/Spherical.* holds CPU reference for F(p), ∇F, ENU, gravity, and shell/AABB overlap; shaders/spherical.glsl mirrors these for GPU use. Keeping them adjacent but separate enforces consistency and testability.
- render/Raytracer uses Set 0 Globals (Rin, Rout, Rsea, voxel/brick size, originDelta, prev/curr matrices) to ensure traversal is clamped to the spherical shell and motion vectors remain correct across origin rebates.
- world/Streaming couples camera geocentric direction and radius to prioritize regions; it depends on math/Spherical for AABB-vs-shell tests.
- world/BrickFormats centralizes GPU-visible packing (occupancy, palette indices, TSDF) so both traversal and TSDF updates agree on memory layout.

Milestone Artifacts in Repo (deliverables per step)
- M0: Minimal src/app, src/platform, src/core (Vulkan bring-up), shaders/generate_rays.comp + shade.comp (sky only), docs/02-descriptor-layouts.md drafted.
- M1: Add render/Raytracer, GpuBuffers, denoise/composite; temporal accumulation.
- M2: Add world/BrickStore, BrickFormats, RegionCache; shaders/traverse_bricks.comp; simple static bricks.
- M3: math/Spherical + shaders/spherical.glsl; isosurface refinement; normals from ∇F; materials + env.
- M4: world/Streaming + MacroMask; fly around the globe with smooth streaming.
- M5: Editing + optional TSDF tiles; tools/Inspect + overlays.

CI & Quality Gates (where to put configs)
- .github/workflows/ (if using GitHub): build matrix (Win/Linux), run tests, archive artifacts; or ci/ with scripts for your provider.
- tests/ run headless math/format/unit tests; golden-frame GPU tests captured via tools/Capture when feasible.

Getting Started Checklist (repo bootstrap)
- Clone extern/ deps (submodules or FetchContent). Configure CMake options.
- Build tools/shaderc_build to compile shaders → build/shaders/.
- Build voxel-planet; run. Expect a sky-only image (M0). Enable validation layers and Tracy in Debug.

