Architecture Overview (Key Classes)

Purpose

- Define the engine’s core modules, responsibilities, and interactions.
- Specify key classes, primary data structures, public APIs, lifetimes, threading, and ownership.
- Emphasize spherical‑world math (floating origin, shell clamp, ENU frames) and GPU‑first rendering.

Guiding Principles

- Separation of concerns: platform, core, math, world, render, sim, tools.
- GPU‑first dataflow: prepare SSBOs/descriptors once; reuse in a wavefront pipeline.
- One logical world, multiple internal layouts (render/physics/audio fit‑for‑purpose).
- Origin rebasing each frame (double on CPU, float on GPU) for precision.
- Brick‑paged sparse world; allocate dynamic channels only for active bricks.
- Deterministic systems; instrumentation everywhere (Tracy, counters, overlays).

Module Graph (High Level)

- app → platform, render, world, tools
- render ↔ world (SSBOs/queues) and ↔ math (spherical functions)
- world ↔ core (jobs, uploads, frame graph) and ↔ math (field F, ENU)
- sim ↔ world (dynamic bricks, TSDF) ↔ render (visual channels)
- tools read from render/world counters; do not mutate core state

Nomenclature & Conventions

- World space (CPU double): geocentric Cartesian. Render space (GPU float): camera‑relative.
- Brick coordinates (bx,by,bz) are int32 in world space; brick size S = B×voxelSize (B=8 by default).
- All Vulkan resources live behind RAII wrappers; device‑local memory via VMA; descriptor indexing centralised.
- Public APIs are exception‑free; return Result/Status or assert in Debug.

App Layer

- Class App
  - Purpose: Own the main loop, input handling, camera, high‑level orchestration.
  - Responsibilities:
    - Initialize subsystems (platform, core, render, world, tools).
    - Drive per‑frame: input → camera → origin rebase → streaming → uploads → frame graph submit → present.
    - Toggle modes (free‑fly/walk), debug overlays, capture.
  - Key API:
    - init(args), run(), shutdown().
    - onResize(w,h), onKey, onMouse.
  - Data:
    - Camera state (curr/prev view+proj), jitter, exposure.
    - OriginRebase (worldOriginPrev, worldOriginCurr, originDeltaPrevToCurr).
  - Threading: Main thread only; delegates background jobs to core::Jobs.

- Class Input
  - Purpose: Abstract window events and map to actions (move, look, edit brush).
  - API: update(dt), query methods (isDown, axis), captureMouse(bool).

Platform / Vulkan

- Class Window
  - Purpose: OS window + event pump.
  - API: create, poll, getFramebufferSize, setTitle, requestClose.

- Class VulkanContext
  - Purpose: Create instance/device/queues; expose capabilities & loaders (Volk).
  - Responsibilities: Validation layers in Debug, feature toggles, descriptor indexing.
  - API: init(Window&), device(), graphicsQueue(), computeQueue(), allocator(), createCommandPool().

- Class Swapchain
  - Purpose: Manage presentable images and sync.
  - API: init(VulkanContext&, Window&), acquireNextImage(), present(), recreateOnResize().
  - Data: images, imageViews, format, extent, per‑frame semaphores/fences.

- Class GpuAllocator
  - Purpose: Thin wrapper over VMA for buffers/images with convenience helpers.
  - API: createBuffer(desc), createImage(desc), map/unmap, destroy.

- Class UploadContext
  - Purpose: Staging/upload ring buffers + timeline semaphore handoff.
  - API: stageBuffer(data,size,dstBuffer,offset), stageImage(...), submit(), getSemaphore().
  - Threading: Called from main and streaming jobs; serializes GPU submits.

Core

- Class FrameGraph
  - Purpose: Lightweight compute pass graph, resource lifetimes, and submits.
  - Responsibilities: Record passes (generate, traverse, shade, denoise, composite) with their read/write sets and insert barriers.
  - API: beginFrame(), addComputePass(name, binds, dispatch, barriers), submit(queue, wait, signal), endFrame().

- Class Descriptors
  - Purpose: Global descriptor set/layout management; bindless where available.
  - API: createSetLayouts(), allocSet(setId), update(setId,binding,resource), getSet(setId).
  - Sets (suggested):
    - Set0 Globals (UBO, RNG textures/SSBO), Set1 World (SSBOs), Set2 Queues (SSBOs), Set3 Frame Images.

- Class Pipelines
  - Purpose: Create/cache compute pipelines; hot‑reload on SPIR‑V timestamp changes.
  - API: getCompute(name, specialization), reloadChanged().

- Class Jobs
  - Purpose: Thread pool for streaming/worldgen/IO.
  - API: schedule(fn), parallel_for(range,fn), setThreadName.

- Class Debug (Tracy/validation helpers)
  - Purpose: Zone markers, GPU timestamp queries, counters registry.

Math

- Namespace spherical (header Spherical.h)
  - Purpose: All planet math and spherical helpers used across systems.
  - Invariants: Planet center at origin; R (radius), T (crust thickness), sea level Rsea.
  - Key functions:
    - IntersectSphere(o,d,R) → (t0,t1,hit): clamp rays to spherical bounds.
    - IntersectSphereShell(o,d,Rin,Rout) → (tEnter,tExit,hit): restrict traversal to content shell.
    - SampleCrust(p, planet, noise, seed) → { field, height } shared with renderer/WorldGen.
    - F_crust(p, planet, noise, seed) → float: signed field (neg = solid).
    - gradF(p, planet, noise, seed, eps) → vec3: finite‑difference gradient for smooth normals/collision.
    - ApplyDomainWarp(dir, noise, seed) → vec3: shared warp used for continents/moisture.
    - ENU(p) → (east,north,up): local tangent frame at p (radial up).
    - gravity(p, g0) → vec3: radial gravity; clamp magnitude near surface.
  - Types:
    - PlanetParams { double R, T, sea, Hmax; }
    - NoiseParams { continent/detail/warp/cave/moisture frequencies, amplitudes, octaves }.
    - OriginRebase { dvec3 worldOriginPrev, worldOriginCurr; vec3 originDeltaPrevToCurr }.
  - Precision: CPU doubles for accumulation; GPU floats in origin‑relative frame.

- Class RNG (PCG32)
  - Purpose: Per‑pixel RNG for path tracer; deterministic across frames with seed = pixel + frameIdx hash.
  - API: nextUint(), nextFloat(), shuffleSeed().

World / Voxels

- Struct BrickHeader (GPU‑mirrored)
  - Fields: bx,by,bz (int32); occOffset; matIdxOffset; paletteOffset; flags; paletteCount; tsdfOffset.

- Class BrickStore
  - Purpose: Own all brick GPU buffers and metadata; encode/decode brick payloads; apply edits.
  - Responsibilities:
    - Maintain SSBOs: headers, occupancy (8×u64), material index stream (4‑bit or 8‑bit), palettes, optional TSDF pool.
    - Provide CPU‑side brick allocation and defragmentation (indices stable within a frame).
    - Enforce palette rules (≤16 entries → 4‑bit, else flip to 8‑bit); track flags.
  - API:
    - getOrCreate(bc) → BrickHandle; free(handle).
    - encodeBrickPayload(handle, cpuBrickData), uploadBatch(handles, UploadContext&).
    - applyEdits(EditBatch&), allocateTSDF(handle), freeTSDF(handle).
    - deviceBuffers() → struct with VkBuffers & sizes for descriptor updates.
  - Threading: Brick mutation on main/stream threads; GPU uploads serialized via UploadContext.

- Class BrickHash
  - Purpose: GPU‑usable linear‑probe hash (keys/vals SSBOs) for (bx,by,bz) → brickIndex.
  - API: buildFrom(BrickStore), deviceKeys(), deviceVals(), capacity(). Rebuilt when brick set changes.

- Class RegionCache
  - Purpose: Group bricks into regions (e.g., 64³) for streaming, scoring, eviction.
  - API: ensureRegion(regionCoord), forEachRegionInBand(visitor), evictByScore(limit).

- Class Streaming
  - Purpose: Decide which regions/bricks are resident given camera and budgets.
  - Responsibilities:
    - Spherical policy: radius band [Rin,Rout], angular window θ_max; priority score (angle, altitude to sea, recency, simActive).
    - Enumerate candidate regions; cull by AABB vs shell; schedule worldgen/IO jobs; trigger uploads.
  - API: update(cameraWorld, OriginRebase, budgets, PlanetParams, BrickStore&, RegionCache&, UploadContext&).
  - Data: LRU clocks, scores, residence sets (RAM/VRAM), pending uploads.

- Class WorldGen
  - Purpose: Deterministic procedural generation for first‑touch bricks/regions.
  - API: bakeRegion(regionCoord, PlanetParams, Out: occupancy/materials/macro masks).
  - Notes: Use noise on unit sphere for heights; 3D noise in world space for caves; output material IDs.

- Class Materials
  - Purpose: Global PBR table; small immutable blobs keyed by content hash.
  - API: getIndex(name), uploadToGPU().

- Class Edits
  - Purpose: Record user ops as (brick, localMaskΔ, materialΔ) and batch to GPU kernel `updates_apply`.
  - API: queueBrush(op), commit(UploadContext&, BrickStore&).

Render

- Class Raytracer
  - Purpose: Wavefront path tracer and frame graph driver.
  - Responsibilities:
    - Allocate and bind descriptor sets (Globals, World, Queues, Frame images).
    - Launch compute passes: generate → traverse (primary) → shade → traverse (secondary) → shade → denoise → composite.
    - Manage accumulation buffers; motion vectors using prev/curr matrices and origin delta.
  - API:
    - init(VulkanContext&, Descriptors&, Pipelines&, GpuAllocator&).
    - resize(w,h), setPlanetParams(PlanetParams), setMaxBounces(n).
    - render(params, BrickStore&, BrickHash&, VkSemaphore wait, VkSemaphore signal).
    - getDebugViews() → images/SSBOs for overlays.
  - Data: Ray/Hit/Secondary queues (SSBOs), frame images (accum, normal, albedo, variance, velocity, output).

- Class GpuBuffers
  - Purpose: Encapsulate creation/resize of all per‑frame images and SSBOs used by render.
  - API: ensureSize(w,h), descriptorsFor(Set2/Set3), clearAccumulation().

- Class Denoiser
  - Purpose: Temporal reprojection + A‑trous spatial filter.
  - API: reprojection(pass resources), atrous(pass resources, iterations), resetHistory(rect).
  - Inputs: color, normal (from refined F/TSDF), albedo, motion vectors, variance.

- Class Tonemap
  - Purpose: Map HDR to LDR for presentation (ACES/filmic) and draw overlays.
  - API: compositeToSwapchain(outputImage, overlays, exposure).

Simulation (optional early)

- Class Fluids
  - Purpose: Active‑region MAC grid solver inside AABBs; writes volumetric water channels to bricks.
  - API: createRegion(aabb), step(dt, BrickStore&, PlanetParams), destroyRegion(id).
  - Notes: Allocate TSDF/velocity/pressure only in active bricks; gravity uses radial up inside AABB.

- Class Collide
  - Purpose: SDF‑based collisions using TSDF (dynamic bricks) or analytic F for base terrain.
  - API: queryDistance(p), projectToSurface(p), contact(normal, depth).

Tools

- Class Overlays
  - Purpose: Visualize counters, heatmaps (bricks touched, macro masks, queue sizes, streaming cones, gravity vectors).
  - API: draw(renderTargets, debugSSBOs, toggles), addCounter(name,ref).

- Class Capture
  - Purpose: Record a reproducible “golden frame” (camera, RNG seeds, streaming state) and timing stats.
  - API: begin(), end(), save(path), load(path).

Key Data Structures (Device)

- GlobalsUBO
  - camPos (local), curr/prev View/Proj, originDeltaPrevToCurr.
  - voxelSize, brickSize, Rin/Rout/Rsea, exposure, frameIdx, maxBounces, dims.

- Queues (SSBO)
  - QueueHeader { head, tail, capacity } + arrays for Ray, Hit.
  - Enqueue/dequeue via atomics; chunked pops per workgroup to amortize contention.

- Brick Pools (SSBO)
  - Headers (array), occupancy (packed 8×u64 per brick), matIndex bytes (4‑bit packed or 8‑bit), palettes, optional TSDF pool.

- Brick Hash (SSBO)
  - keys[cap] (u64), vals[cap] (u32 index into headers); linear probe ≤ 8 steps at 0.5 load factor.

Primary Control Flow (Per Frame)

- CPU main thread
  - Poll input → update camera.
  - Compute OriginRebase (worldOriginCurr=cameraWorld; originDeltaPrevToCurr = curr−prev).
  - Streaming.update(cameraWorld, rebase, budgets, params, BrickStore, RegionCache, UploadContext).
  - UploadContext.submit() and pass semaphore to FrameGraph.
  - Raytracer.render(params, BrickStore, BrickHash, wait=uploadSem, signal=raytraceSem).
  - Swapchain.present(wait=composeSem).

- GPU (FrameGraph passes)
  - generate_rays: write RayQueueIn with camera jitter; set tmin/tmax.
  - traverse_bricks: clamp to shell (Rin/Rout); brick‑DDA; micro‑DDA; refine F=0; normals from ∇F or TSDF.
  - shade: PBR + env; spawn secondaries (limited); write g‑buffers for denoise.
  - denoise_atrous: temporal reprojection with motion vectors incl. origin rebase; A‑trous filter.
  - composite: tonemap + overlays to swapchain image.

Lifetimes & Ownership

- VulkanContext owns VkDevice and allocators.
- GpuAllocator owns buffers/images; Raytracer/GpuBuffers request resources via it.
- Descriptors/Pipelines own VkDescriptorPools/PipelineCache.
- BrickStore owns CPU copies and GPU SSBOs for bricks; BrickHash references BrickStore lifetimes.
- RegionCache/Streaming own residency sets; UploadContext owns transient staging until submit.

Threading Model

- Main thread: orchestration, render submits, descriptor updates.
- Jobs threads: worldgen, streaming IO, CPU brick packing; avoid direct Vulkan; push results back to main.
- GPU: single compute queue to start; later split async denoise.

Error Handling & Validation

- All Vulkan calls validated in Debug; abort on validation error.
- Brick palette invariants asserted; flip to 8‑bit indices if >16 materials.
- Queue capacity monitored; drop secondaries on overflow and record counters.

Extensibility Notes

- Add far‑field LOD via coarse bricks or implicit crust fallback with a clipmap texture; keep BrickHash interface stable.
- Integrate fluids later by adding dynamic channels to BrickStore and a new sim pass writing liquid material IDs.
- Optional audio rays share the same DDA traversal with a different shading kernel.

Testing Targets (Unit/Integration)

- spherical::IntersectSphereShell edge cases; AABB vs shell overlap.
- Micro‑DDA step sequences on known occupancy patterns.
- BrickHash find/insert; rebuild correctness and probe lengths.
- Motion vectors under origin rebase; reprojection accuracy.

Key Counters (Expose to Overlays)

- Rays launched/hit/missed; average micro‑DDA steps per ray.
- Brick hash probe length distribution; bricks touched per frame.
- Refine iterations (avg/95th); denoise cost; queue occupancy/overflows.
- Streaming loads/evictions; GPU brick residency; macro mask effectiveness.

Initial Class Headers (Sketch)

- math/Spherical.h
  - struct PlanetParams { double R,T,sea,Hmax; };
  - struct NoiseParams { … }; inline constexpr int kNoiseCaveOctaves = 4;
  - struct OriginRebase { glm::dvec3 worldOriginPrev, worldOriginCurr; glm::vec3 originDeltaPrevToCurr; };
  - CrustSample SampleCrust(glm::vec3 p, const PlanetParams&, const NoiseParams&, uint32_t seed);
  - float F_crust(glm::vec3 p, const PlanetParams&, const NoiseParams&, uint32_t seed);
  - glm::vec3 gradF(glm::vec3 p, const PlanetParams&, const NoiseParams&, uint32_t seed, float eps);
  - bool IntersectSphereShell(glm::vec3 o, glm::vec3 d, float Rin, float Rout, float& tEnter, float& tExit);
  - glm::vec3 ApplyDomainWarp(glm::vec3 dir, const NoiseParams&, uint32_t seed);
  - void ENU(const glm::vec3& p, glm::vec3& east, glm::vec3& north, glm::vec3& up);

- world/BrickStore.h
  - struct BrickHeader { int32_t bx,by,bz; uint32_t occOffset, matIdxOffset, paletteOffset, tsdfOffset; uint16_t flags, paletteCount; };
  - class BrickStore { public: BrickHandle getOrCreate(ivec3 bc); void applyEdits(const EditBatch&); void uploadBatch(const std::vector<BrickHandle>&, UploadContext&); const DeviceBuffers& deviceBuffers() const; };

- render/Raytracer.h
  - struct RaytracerParams { int w,h,maxBounces; float voxelSize, brickSize, Rin,Rout,Rsea; uint32_t frameIdx; };
  - class Raytracer { public: void init(VulkanContext&, Descriptors&, Pipelines&, GpuAllocator&); void resize(int w,int h); void render(const RaytracerParams&, const BrickStore&, const BrickHash&, VkSemaphore wait, VkSemaphore signal); };

Glossary

- AABB: Axis‑Aligned Bounding Box (in meters, world space or origin‑relative)
- DDA: Digital Differential Analyzer; grid stepping algorithm
- TSDF: Truncated Signed Distance Field; local micro‑SDF per edited/dynamic brick
- ENU: East‑North‑Up; local tangent frame aligned to radial up
