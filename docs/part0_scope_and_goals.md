# Part 0 — Scope & Stretch Goals

Version: 0.1 (2025-10-15)

This document defines the concrete scope, success criteria, budgets, and risks for the voxel‑planet tech demo and its engine‑ready underpinnings. It translates the end‑to‑end plan into actionable targets that guide design and trade‑offs from day one.

---

## 1) Vision (What we’re building)

- A playable, performance‑bounded tech demo that renders a dense, spherical voxel world using a GPU‑first path tracer with sparse traversal.
- The demo doubles as a thin engine framework: clear module boundaries, Vulkan wrappers, frame graph, streaming, and debug tooling that can evolve into a game engine.

---

## 2) User‑Facing Demo Goals (Must‑haves)

- Navigate a small “planet” (radius 10–20 km) with free‑fly and surface‑walk modes.
- Path‑traced lighting (primary + 1–2 bounces with NEE), temporal accumulation, and a spatiotemporal denoiser.
- Voxel world rendered via sparse brick traversal (8×8×8 bricks) with smooth surfaces (implicit field refinement) — no “corner‑up” artifacts.
- Spherical streaming: keep a working shell of bricks around the camera; hitch‑free flight around the globe.
- Basic editing tools (carve/add using surface‑aligned brushes); dynamic bricks maintain a TSDF for visual + physical smoothness.
- Optional: analytic ocean at sea level with simple refraction into seabed.

---

## 3) Engine‑Architecture Goals (Should‑haves)

- Modular subsystems: platform (window/swapchain), core (frame graph, descriptors, pipelines, uploads), math (spherical), world (bricks/streaming), render (raytracer/denoiser), tools (overlays/capture), sim (optional).
- Clean Vulkan abstraction: descriptor set layout, bindless‑friendly SSBOs, timeline semaphores, VMA for memory, hot‑reloadable compute pipelines.
- Deterministic captures (“golden frames”) with fixed RNG seeds for perf/visual regression.
- Profiling first: Tracy zones/timers, GPU timestamps, and on‑screen counters.

---

## 4) Non‑Goals (Out of scope for the tech demo)

- Full gameplay/AI, networked multiplayer, or content authoring tools beyond simple brushes.
- Complex materials/shaders (keep a compact PBR set; prioritize stability and denoiser compatibility).
- Full atmospheric multiple scattering or volumetric global illumination (single‑scattering sky + simple fog only).
- Production‑grade asset pipeline (handful of blobs/JSON tables suffice).

---

## 5) Target Platforms & Constraints

- OS: Windows 10/11 and Linux (Wayland/X11) — prioritizing one first (developer host).
- GPU: DX12‑class Vulkan 1.2+ hardware (e.g., NVIDIA RTX 20xx/30xx/40xx; AMD RDNA2/3). Ray‑tracing hardware not required (compute‑only path tracer).
- Display targets: 1080p @ 60 fps (1 spp) with temporal denoise; 1440p @ 30–60 fps desirable; fallback 720p for low‑end.
- Input: Keyboard/mouse; optional gamepad.

---

## 6) World Scale & Math Assumptions (Spherical)

- Global space (CPU): geocentric Cartesian in double precision; planet center at (0,0,0).
- Render space (GPU): camera‑relative Cartesian in 32‑bit floats; per‑frame “floating origin” rebase.
- Planet radii:
  - Base radius `R = 10,000–20,000 m` (dev scale).
  - Inner clamp `R_in = R − T` (crust thickness `T ≈ 50–150 m`).
  - Outer clamp `R_out = R + H_max + atmosphereTop` (`H_max ≈ 1000–3000 m`).
  - Sea level `R_sea = R + H_sea`.
- Signed field for crust (continuous): `F(p) = max(|p| − (R + H(û)), −(T − C(p)))`, where `û = normalize(p)`; `H` is domain‑warped noise on the unit sphere; `C` is cave noise in world space.
- Gravity: radial `g(p) = −g0 · normalize(p)` (constant magnitude near surface); optional inverse‑square variant.
- Shell clamp: intersect ray with spheres of radii `R_in` and `R_out`; clamp `tmin/tmax` of traversal to the union interval that lies outside `R_in` and inside `R_out`.
- Local tangent frame (ENU) at point `p`: `up = normalize(p)`, `east = normalize(cross((0,0,1), up))` (fallback to (1,0,0) near poles), `north = cross(up, east)`.

---

## 7) Renderer Scope

- Wavefront compute path tracer with persistent threads and SSBO work queues (primary/secondary rays, hits, misses).
- Brick‑by‑brick DDA in world space; micro‑DDA inside 8×8×8 occupancy with bit‑packed masks.
- On first solid voxel, refine to the `F(p)=0` isosurface via 4–8 steps of bisection; normals from `∇F` (central differences with `ε ≈ 0.5·voxelSize`).
- Materials: ID → compact PBR params (albedo, roughness, metalness, emission, absorption); small table/UBO.
- Denoiser: temporal reprojection (motion vectors account for origin rebases) + A‑trous/SVGF spatial filter (3–5 iters).
- Optional: test analytic ocean (sphere at `R_sea`) on voxel miss; refract into seabed.

---

## 8) World Data Scope (Bricks & Memory)

- Brick size: 8×8×8 voxels; voxel size: 0.25–0.5 m (tunable); brick size in meters = `B·voxelSize`.
- GPU brick payload per brick (typical):
  - Occupancy: 8× `uint64` (512 bits) ≈ 64 B.
  - Material indices: 4‑bit stream (512×0.5 B) ≈ 256 B (fallback 8‑bit = 512 B).
  - Palette (≤16 entries): 32–64 B.
  - Header/flags: ~32 B.
  - Optional TSDF (dynamic bricks only): 512× int16 ≈ 1 KiB.
- Indirection: global linear‑probe hash table `(bx,by,bz) → brickIndex` (cap ≈ 2× resident bricks; short probes).
- Macro masks per macro‑tile (e.g., 8×8×8 bricks) for empty/full/mixed skipping.
- Working set target: ~100–120k resident bricks on GPU (≈ 50–80 MiB base), plus 5–15k TSDF bricks (≈ 10–15 MiB).

---

## 9) Streaming Scope (Spherical Policy)

- Maintain a budgeted working shell: radii `[R_in − marginBelow, R_out + marginAbove]` and an angular window around camera direction.
- Regionization: 64³‑brick regions (AABB vs shell overlap test); prioritize regions by score: `w0/(1+angDist) + w1/(1+|radius−R_sea|) + w2·recent + w3·simActive − w4·age`.
- Two‑tier caches: CPU‑RAM (large) → GPU (small). Promote on use; evict LRU/low score; protect sim‑active bricks.

---

## 10) Editing Scope

- Surface‑aligned brushes using local ENU; disk/sphere shapes; parameters: radius, hardness, material.
- Edits are queued as `(brickCoord, localMaskΔ, materialΔ)` per frame and applied in a batched compute pass.
- A brick becomes “dynamic” on first edit → allocate TSDF tile; recompute TSDF for brick + 1‑voxel halo via jump‑flood or fast sweeping.

---

## 11) Simulation Scope (Stretch)

- Fluids: one active AABB region near camera; MAC grid limited to active bricks; forces + PCG projection; write liquid channel into bricks; renderer blends volumetric water with analytic sea.
- Rigids/Destruction: flood‑fill to clusters; coarse SDF for collisions (on demand in dirty regions); sleep to re‑voxelize.

---

## 12) Tooling, Telemetry & Debug Overlays

- On‑screen overlays: brick residency heatmap, macro mask visualization, ray/hit/miss counters, refine iterations, queue sizes, streaming stats, denoiser timings.
- Golden‑frame capture (camera + RNG seed) for regression.
- Validation: assert brick palette ≤16 for 4‑bit mode; fallback to 8‑bit with a flag; log counts.

---

## 13) Performance Budgets & Quality Bars

- Primary performance target: 1080p, 60 fps with 1 spp (temporal + denoise), 1–2 bounces; average frame time ≤ 16.6 ms (stretch: 1440p/60).
- Traversal: aim for ≤ 2–4 brick lookups per primary ray on average (view‑dependent), micro‑DDA iterations ≤ 16 per hit; refine steps ≤ 8.
- Memory: total GPU footprint ≤ 1.5–2.5 GiB including frame images; world data ≤ 120 MiB typical.
- Hitching: streaming stalls < 2 ms per frame on average; no frame dropouts from uploads.
- Visual stability: denoiser ghosting < mild at fast motion; history clamp with variance; reset when material/normal diverges.

---

## 14) Milestones & Deliverables

- M0 — Platform Boot: Window/swapchain; compute “sky tracer”; Tracy + validation wired.
- M1 — Path Tracer Base: `generate_rays` + `shade` (sky/env); temporal accumulation + tonemap.
- M2 — Bricks & DDA: BrickStore, GPU buffers, hash lookup, shell clamp, binary occupancy hits; macro skipping.
- M3 — Implicit Refinement: `F(p)` + bisection + `∇F` normals; PBR shading; SVGF/A‑trous denoiser.
- M4 — Spherical Streaming: region cache, AABB‑vs‑shell culler, LRU/score eviction; hitch‑free flyaround.
- M5 — Editing + TSDF: brush tools; batched updates; TSDF tiles for dynamic bricks; parity with analytic field visually.
- M6 — Ocean (Optional): analytic sea + refraction; shoreline blending; basic fog.
- M7 — Fluids (Stretch): single active region solver; liquid rendering integration.

Each milestone produces a runnable demo binary and a short test checklist (below).

---

## 15) Risks & Mitigations

- Precision drift/flicker far from origin → double‑precision CPU world + per‑frame rebase; prev/curr origin deltas feed motion vectors.
- Queue overflows under heavy secondary spawning → cap secondaries; drop on overflow with counters; tune bounce count.
- Denoiser ghosting → correct motion vectors (include rebase), history clamp by variance, reset on normal/material divergence.
- Streaming hitching → region prefetch by angular velocity; prioritize regions entering FOV; batch uploads; keep CPU cache warm.
- Palette overflow (>16 materials) → per‑brick flag to 8‑bit indices; track frequency; possibly split content procedurally.
- TSDF update cost spikes → limit dynamic bricks per frame; incremental updates; spread work via job system.

---

## 16) Test Plan & Validation

- Unit tests: sphere‑shell intersection edge cases; AABB‑vs‑shell overlap; key pack/lookup; micro‑DDA traversal of synthetic patterns; `F(p)` gradient finite‑difference sanity.
- Golden captures: 5 camera seeds across biomes; record rays traced, bricks touched, average refine steps, denoise time.
- Manual checklist per milestone: resize, fly to poles/equator, teleport 5,000 km (rebase), edit near brick boundaries, stream across terminator.

---

## 17) Repo/Module Deliverables (Tie‑in)

- `src/platform`: Window + VulkanContext + Swapchain with validation, timeline semaphores.
- `src/core`: FrameGraph, Descriptors (bindless‑friendly), Pipelines (compute), Upload (staging), Jobs (thread pool).
- `src/math`: Spherical math (`F(p)`, shell clamp, ENU, gravity), RNG.
- `src/world`: BrickStore (host/device), RegionCache (streaming), WorldGen (sphere noise), Materials, Streaming policy.
- `src/render`: Raytracer (queues, DDA, refinement), Denoiser (temporal + A‑trous), Tonemap.
- `src/tools`: Overlays, Capture (golden frames), Debug UI.
- `shaders/`: `generate_rays.comp`, `traverse_bricks.comp`, `shade.comp`, `denoise_atrous.comp`, `composite.comp`, `updates_apply.comp`, `sdf_update.comp` (optional).

---

## 18) Acceptance Criteria & Demo Script

- Boot to a flyable scene on a spherical planet with sky.
- Look horizon‑wide: frame time stable; brick counters scale with view, not world size.
- Land and walk: ground appears smooth (no “corner‑up”), normals align with radial up; edits carve/add smoothly.
- Fly to poles/equator; teleport across globe: no precision artifacts; temporal reprojection remains stable.
- Toggle streaming overlays: resident bricks form a spherical shell; region churn limited; no visible hitches.

---

## 19) Appendix: Key Constants & Tunables (Initial)

- `voxelSize = 0.25–0.5 m`, `brickDim = 8`, `brickSize = brickDim·voxelSize`.
- Planet: `R = 10,000–20,000 m`, `T = 100 m`, `H_max = 2000 m`, `R_sea = R + 0 m`.
- DDA: step bias `ε_step = 1e−4·voxelSize`; refine iterations 6–8; gradient `ε = 0.5·voxelSize`.
- Denoiser: temporal alpha ≤ 0.05; A‑trous iterations 3–5; normal σ ≈ 0.1–0.2.
- Streaming: θ_max near LOD 5–10°, far LOD 20–30°; GPU brick budget 120k; dynamic TSDF cap 10–15k.
- Queues: primary capacity ≥ `width·height·raysPerPixel`; secondary ≤ 0.5× primary.

---

## 20) Glossary

- AABB — Axis‑Aligned Bounding Box.
- DDA — Digital Differential Analyzer traversal through a grid.
- ENU — East‑North‑Up local tangent frame on a sphere.
- TSDF — Truncated Signed Distance Field.
- SVGF — Spatiotemporal Variance‑Guided Filter.

