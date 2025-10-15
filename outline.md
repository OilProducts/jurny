Below is a concrete, end‑to‑end plan for a **“V2” voxel engine** that reproduces the “everything looks alive” feel without actually simulating every voxel. It’s organized from fundamentals → renderer → world storage → simulation systems → tooling/perf, with specific data layouts, GPU kernels, and milestones you can build in order.

---

## Design goals (what V2 explicitly optimizes for)

1. **Visual density with bounded work**
   Rays can *see* billions of micro‑voxels, but only a tiny active set is updated each frame.

2. **Single source of spatial truth, multiple internal reps**
   Rendering, fluids, physics, audio share one logical world but keep their *own* fit‑for‑purpose data layouts.

3. **GPU‑first render, mixed CPU/GPU sim**
   Path/ray tracing uses persistent GPU threads and sparse traversal; simulation uses CPU where control flow is messy (rigids, gameplay) and GPU where it’s dense and regular (fluid pressure solves, broad collision, SDF updates).

4. **Streamable, infinite‑feeling world**
   Brick‑paged storage with indirection and background I/O; only touched bricks exist in RAM/VRAM.

---

## Pillar 1 — World representation (brick‑paged “virtual voxels”)

**Unit of storage**: a small 3D brick, e.g. **8×8×8** (“micro‑brick”).
**Why 8³**: cheap to allocate/destroy, very local edits, good cache behavior for DDA traversal.

**In‑brick payload (compact):**

* `64 B` occupancy bitset (512 bits for 8³).
* `palette[<=16]` material IDs (global table of PBR params texturable elsewhere).
* `indices[512]` 4‑bit indices into the palette (if ≤16 materials in the brick), else fall back to 8‑bit per‑voxel material index.
* Optional channels in *dynamic bricks only*: `velocity` (fluids), `temperature`, etc., using half precision.

**Indirection (“page table”)**

* A **hierarchical hash grid** keyed by brick coordinates `(bx,by,bz)` → brick handle (GPU index).
* Level 0 (coarse) buckets world into **regions** (e.g., 64³ bricks). Each region has a compact hash table for its bricks.
* On GPU, a **read‑only perfect hash** (rebuilt occasionally) accelerates lookups; on CPU, a standard Robin Hood hash map for edits.

**Memory budget example** (realistic working set)

* **100k bricks resident** near the player:

    * 100k × (64 B mask + ~512 B indices + ~32 B header) ≈ **~58 MiB** for base occupancy+material.
    * Dynamic channels (fluid velocity/pressure only in a few thousand bricks) add tens of MiB, not hundreds.
* Page tables + region hashes: typically < 16–32 MiB GPU.

> Key rule: **only allocate dynamic channels to bricks in the active set**.

---

## Pillar 2 — Renderer (GPU path tracer over sparse voxels)

**Traversal:**

* Use **3D DDA** to step rays brick‑by‑brick. For each brick hit:

    1. Load the brick header+mask. If empty: skip by advancing DDA to next brick boundary.
    2. If occupied, do a **bit‑grid micro‑DDA** inside the brick: test occupancy mask along the ray step; when a set bit is hit, fetch material from the palette and shade.

**Algorithm structure: wavefront + persistent threads**

* **K0 Generate**: primary rays, jittered for TAA/temporal reuse.
* **K1 Traverse bricks**: DDA steps; output hits to queues (`HitQueue`, `MissQueue`).
* **K2 Shade**: evaluate materials, lights, environment; spawn secondary rays (limited bounces).
* **Denoise**: SVGF/temporal variance‑guided filter or a modern spatiotemporal denoiser (still works great on voxels).

**Empty‑space skipping and LOD**

* Maintain a **macro‑mask** per **32³‑brick tile** (i.e., 256³ voxels): if empty, skip a big span in one step.
* Optional **clipmap LOD**: far distances collapse 2×2×2 bricks into one logical “super‑brick” with aggregated material lobes (prebaked). Rays test coarse levels first, descend as needed.

**Shading**

* Materials are **index→PBR params** (albedo, roughness, metalness, emission) from a small texture.
* Surface normals from **occupancy gradient** (3×3×3 neighborhood) or from a downsampled SDF channel if available for dynamic areas.

**Core kernel sketch (HLSL‑ish pseudocode)**

```cpp
struct Ray { float3 o, d; float tmin, tmax; RNG rng; };
struct Hit { float3 p, n; uint mat; };

[numthreads(64,1,1)]
void TraverseKernel(...) {
  Ray r = RayQueue.pop();
  DDA dda = initDDA(r.o, r.d); // voxel-space with brick-sized steps
  while (dda.t < r.tmax) {
    int3 b = dda.brickCoord();
    BrickHandle h = RegionHashLookup(b);
    if (!h.valid) { dda.stepToNextBrick(); continue; }

    Brick br = BrickPool[h.index];        // coalesced load of header+mask
    if (br.occupancyMask == 0) { dda.stepToNextBrick(); continue; }

    // micro-DDA in local brick space
    LocalDDA ldda = initLocalDDA(r, br);
    int voxelHit = ldda.firstSetBitAlongRay(br.mask);
    if (voxelHit >= 0) {
      Hit hit = shadeSetup(r, br, voxelHit);
      HitQueue.push(hit);
      return;
    } else {
      dda.stepToNextBrick();
    }
  }
  MissQueue.push(r);
}
```

**Why this scales**
The GPU touches *only* bricks intersected by rays, and within them *only* the bits along the ray path. You get “billions of visible voxels” with stable frame times because empty space and uniform bricks are nearly free.

---

## Pillar 3 — Simulation (active sets + local solvers)

### 3.1 Fluids (incompressible, active‑region clipmap)

* Maintain a **bounding AABB** around each fluid volume, padded by a few voxels; grow/shrink as needed.

* Use a **MAC grid** (staggered velocities) *only* inside these AABBs; allocate velocity/pressure channels to those bricks.

* Steps per frame:

    1. **Advection (semi‑Lagrangian or MacCormack)** on `u,v,w` within active bricks.
    2. **Apply forces** (gravity, buoyancy, vorticity confinement).
    3. **Pressure projection**: solve `∇² p = ∇·u` with **PCG**; coefficients from solid walls sampled directly from the voxel occupancy.
    4. **Velocity update**: `u ← u − ∇p`.
    5. **Writeback occupancy** for “liquid voxels” to the world brick masks or keep a dedicated liquid‑channel that the renderer treats as transparent refractive material.

* **Collision**: solid walls from the world occupancy mask; **no need** to voxelize triangles if the world is already voxels.

* **Performance levers**:

    * 16‑bit floats for velocities/pressure.
    * **Red‑black Gauss‑Seidel warm‑start** between frames.
    * **Multigrid** on a per‑AABB basis if needed.

### 3.2 Rigid bodies & destruction

* Static terrain = the world bricks; dynamic debris = **voxel clusters**.

* When something breaks:

    1. Flood‑fill connected solids in the impacted brick neighborhood; each island becomes a **cluster** (limit voxel count per cluster).
    2. Compute cluster **mass & inertia** from voxel count (density per material).
    3. Simulate clusters with **discrete rigid body** integration (CPU okay; a few hundred bodies is fine).
    4. For collision, use a **coarse SDF** updated on demand: in edited regions, run a **jump‑flood** SDF pass at 2× or 4× coarser resolution than display. Cache per region.

* **Sleeping**: clusters that come to rest get **re‑voxelized** back into static world bricks; free their dynamic buffers.

### 3.3 Particles, foliage, and decals

* Particles are **screen‑driven** and spawn rays only when shaded (glints, sparks, volumetric fog).
* Foliage is typically **procedural impostors** that write into a transient voxel layer near the camera (no need to store forever).

### 3.4 Audio occlusion & reverb

* Share the same world occupancy; cast **multi‑band audio rays** (few dozen per source) to compute occlusion and a coarse **late reverb** via stochastic probes. Cheaper than simulating a wave field, and the data are already there.

---

## Pillar 4 — Editing & streaming

**Edits (mining, destruction, fluid paint):**

* Each edit is a **voxel write op** (`(bx,by,bz), localMaskDelta, materialDelta`).
* Accumulate per‑frame into a **brick‑update queue**; on GPU, apply batched updates via a compute kernel.

**Streaming:**

* Keep a **ring of regions** around the camera; background thread loads/synthesizes bricks for entering regions and evicts far ones.
* Materials and small SDF tiles are **immutable blobs** keyed by content hash → dedupe across the world.

---

## Pillar 5 — Tooling & perf regimen

1. **GPU counters**: rays launched/hit/missed, bricks touched, micro‑DDA steps per brick.
2. **Overlay visualizations**: occupancy heat, active simulation bricks, hash table fill %, PCG iteration counts.
3. **Capture “golden frames”** with identical camera & RNG seeds to regression‑test both visuals and timings.
4. **Data validation**: on edits, assert that palette constraints hold (≤16 materials in a brick when using 4‑bit indices); fallback cleanly to 8‑bit layout when exceeded.

---

## Build order (milestones you can hit sequentially)

**M0 – Micro‑renderer PoC**

* Brick store (CPU), trivial scene; GPU compute **primary‑ray DDA** through occupancy masks; hard‑shaded colors.

**M1 – Full path tracer**

* Add secondary rays, emissives, sky/environment, temporal accumulation + simple SVGF.

**M2 – Sparse world**

* Region hash + brick pool on GPU; batched brick uploads; macro‑mask skipping.

**M3 – Editing**

* Brush that toggles occupancy/material; dirty‑brick queue and on‑GPU apply.

**M4 – Fluids**

* Active‑region clipmap; MAC grid in bricks; PCG projection; write liquid into a dedicated channel and render refractive water.

**M5 – Destruction**

* Flood‑fill to clusters; rigid body sim; SDF update; sleep & re‑voxelize.

**M6 – Streaming & LOD**

* Region ring; eviction/prefetch; far‑field clipmap LOD.

**M7 – Audio rays + polish**

* Occlusion & reverb rays; denoiser tuning; content tools.

---

## Concrete data formats

**Brick header (32 bytes)**

```
uint32_t occupancyMaskOffset; // byte offset in brick pool
uint32_t materialIndexOffset; // 4-bit or 8-bit stream
uint32_t paletteOffset;       // <=16 entries, else 0xFFFFFFFF if global
uint16_t flags;               // bits: hasFluid, hasSDF, uses4bit, etc.
uint16_t paletteCount;        // 0..16
uint32_t localAABBMinPacked;  // 10:10:10 fixed-point
uint32_t localAABBMaxPacked;
uint32_t reserved0;
uint32_t reserved1;
```

**Region hash entry (16 bytes)**

```
int32_t bx, by, bz;   // brick coords within region
uint32_t brickIndex;  // index into GPU BrickPool buffers
uint32_t next;        // for linear-probed or bucketed chaining
```

**Material table entry (~32 bytes)**

```
float3 baseColor;
float  emission;
float  roughness;
float  metalness;
float3 absorption; // for participating media (water, fog)
```

---

## Scheduling model (frame graph)

```
CPU: build/edit queues ──► Upload brick deltas ─┐
GPU: K0 Generate Rays                           │
     K1 Traverse (brick DDA) ──► Hit/Miss Qs ───┼─► K2 Shade
     K3 Secondary Traverse ─────────────────────┘
     K4 Denoise/Composite

SIM: (async to render, double-buffered worlds)
CPU/GPU: Fluids (advection, forces, pressure)
CPU: Rigid bodies (broadphase on GPU if needed)
GPU: SDF updates (jump-flood in dirty regions)
Commit sim writes ► next frame's brick deltas
```

---

## Implementation notes that save weeks

* **Persistent GPU threads** keep waves full as some rays die early; use work queues (structured buffers with atomic push/pop).
* **Bit tricks in micro‑DDA**: pack 8×8×8 into 8 64‑bit words; `first set bit` from a parametric line walk is branch‑light and cache‑friendly.
* **Two brick sizes if needed**: 8³ for editable dynamics, 32³ read‑only shading bricks (auto‑generated from 8³) for fewer lookups in far field.
* **SDF only where you need it**: a 4× coarser SDF than display voxels is fine for collisions and smooth normals; compute on demand in dirty AABBs.
* **Materials as IDs**: per‑voxel bitfields explode memory; store small IDs and fetch PBR from a table/atlas.
* **Strict limits**: cap clusters/rigid bodies, cap fluid AABB sizes; when exceeded, split or cull gracefully (and log).

---

## What you’ll “feel” at runtime

* The camera can look across **dense micro‑voxel scenes**; the renderer’s cost scales with rays and *occupied* bricks, not global voxel count.
* Fluids and debris touch only **thousands** of voxels per frame (their AABBs), yet the *renderer* shows waves washing across **millions** of visible voxels.
* Edits are local and cheap; far‑field detail is streamed/LOD’d.

---

If you want, I can follow up with:

* a minimal **C++/HLSL skeleton** (brick store, region hash, DDA kernel, queues), and
* a **fluid active‑region PCG** snippet wired to the same brick layout,
  so you can drop it into a new repo and get M0–M2 running quickly.


Sounds plausible, I'd like to make a spherical world, how might that change the outline of this v2?

Great question—going spherical changes *addressing, streaming, and “what is down”* far more than it changes the core renderer/simulation ideas. Below is a practical way to adapt the V2 plan to a **planet‑scale, fully spherical world** while keeping the “massively dense, sparsely updated” property.

---

## TL;DR of what changes (and what doesn’t)

* **Keep**: voxel bricks (e.g., 8³), GPU ray/path tracing with DDA, active‑set simulation, brick hashing, editing model.
* **Change**:

    1. **Coordinate frames & precision** (floating origin + geocentric doubles).
    2. **Addressing & streaming** (spherical shell of bricks; regionization by cubed‑sphere or world‑space 3D Morton keys).
    3. **Gravity & “down”** (radial).
    4. **Renderer entry/exit** (analytic sphere intersection to clamp ray traversal).
    5. **Water & atmosphere** (planetary sea level and sky model).

Everything else—materials, queues, denoising, destruction—carries over.

---

## 1) Coordinates, precision, and “floating origin”

**Global**: Geocentric Cartesian (planet center at `(0,0,0)`) in **double precision** on CPU and for *accumulated* transforms.
**Local (GPU & gameplay)**: a **camera‑relative frame** in 32‑bit floats to keep precision stable.

**Per‑frame rebasing**

```cpp
// CPU (double precision)
double3 worldCameraPos;          // ECEF
double3 renderOrigin = worldCameraPos; // pick camera as origin

// For each object/brick AABB:
float3 relPos = (double3ToFloat3)(objectWorldPos - renderOrigin);
// Upload relPos and a 3x3 rotation to GPU; the GPU works entirely in this relative space.
```

**Local tangent frame** at a point `p`:

```cpp
float3 up    = normalize(p);                   // radial
float3 east  = normalize(cross(float3(0,0,1), up));
if (length(east) < 1e-3) east = float3(1,0,0); // near poles fallback
float3 north = cross(up, east);
```

Use this ENU basis for UI gizmos, player controls, and to orient tools/brushes so buildings/edits “sit flat” on the surface even though bricks remain axis‑aligned in world space.

---

## 2) Brick layout & addressing on a sphere

You have two solid options. Both end with **axis‑aligned bricks in world space** so the DDA stays simple.

### Option A — World‑space bricks (recommended first)

* Exactly the V2 brick store and hash, keyed by integer **brickCoords = floor(worldPos / BRICK_SIZE)**.
* Memory stays bounded because you only allocate where content exists.
* **Streaming** becomes a **spherical shell** around the camera: load bricks with centers in `[Rmin, Rmax]` (radius from center). Use a prioritization function that mixes angular distance from camera and altitude.

**Regionization**: bucket bricks into fixed 3D “regions” (e.g., 64³ bricks) using world‑space Morton keys. This keeps uploads/evictions chunked.

### Option B — Cubed‑sphere tiling (advanced, nicer LOD seams)

* Index the surface by **(face:0..5, tileX, tileY)** in a quadtree (clipmap), then add **radial layers** (underground and atmosphere): `(face, lod, tileX, tileY, radialLayer)`.
* Each tile holds a *local* 3D brick grid, but you still **instantiate those bricks into world space** (AABBs and transforms) and insert them in the *same* global hash used by the renderer.
* Benefit: more uniform surface area per tile; easier surface LOD. Cost: more plumbing.

> Start with **A**; you can retrofit **B** for prettier far‑field LOD later.

---

## 3) Renderer updates for a planet

### 3.1 Clamp traversal to a spherical shell

Before DDA, intersect rays with an analytic shell that bounds your content:

* Inner radius `R_inner = R_planet - depthBelowSea - cavesMargin`
* Outer radius `R_outer = R_planet + maxMountain + atmosphereTop`

Compute `(tEnter, tExit)` for the ray against `R_inner..R_outer`; if no overlap, skip traversal.
Set `ray.tmin = max(ray.tmin, tEnter)` and `ray.tmax = min(ray.tmax, tExit)` → now your **brick DDA only runs where the planet exists**, avoiding kilometers of empty stepping.

### 3.2 Macro skipping & LOD

* Keep the **macro‑mask** from V2, but organize masks in **concentric regional spheres** (or just reuse your region grid—both work).
* For **far field**, add an optional **implicit SDF fallback** for the *smooth* planetary crust/ocean (no caves). Primary rays test the implicit first, then only dive into voxels where fine detail is required (nearby or when an SDF hit falls in a “refine here” mask produced during baking).

### 3.3 Sky/atmosphere

Replace a flat sky with a planetary **single‑scattering sky model** (Nishita/Bruneton style). In a path tracer this is a lightweight phase‑function integral for primary/secondary rays that miss the planet.

---

## 4) Gravity, physics, and sleeping

**Gravity** is radial with optional falloff:

```cpp
float r = length(p);
float3 g = -normalize(p) * g0 * (R_planet * R_planet) / (r * r); // or clamp to g0 near surface
```

* Use **local ENU** for character controllers and for “upright” constraints.
* **Rigid body sleeping** should consider the *radial* up direction; when a cluster sleeps, voxel‑bake it back into the static world exactly as in V2.

**Collision SDF**: keep the on‑demand SDF in **dirty regions only**. For large structures, a shallow **coarse planetary SDF** (implicit sphere + heightfield) is enough for broadphase; the per‑region SDF handles detail.

---

## 5) Fluids and oceans on a sphere

You don’t want a full sphere of water voxels; treat the ocean as a **thin active band** around sea level.

**Hydrostatic ocean baseline**

* Define **sea level radius** `R_sea`. Voxels at altitude `h = dot(p, normalize(p)) - R_sea` < 0 are “underwater” for *shading* unless overridden by active fluid.
* The **active solver** runs only in AABBs around **shorelines, waterfalls, rivers, or edited regions**, exactly as in V2:

    * Allocate MAC grid channels **only** in those bricks.
    * Apply **open boundary** conditions at the edges of the AABB using the hydrostatic field as a Dirichlet condition (pressure = ρgh).
    * Gravity points radially inward; over tens of kilometers it varies direction, but your **active AABB is small**, so treat `g` as constant inside the AABB and update it if/when the AABB moves.

**Rendering water**

* For the passive ocean (outside active solver), sample **analytic sea surface** at `R_sea` with spectrum‑driven normals near camera (optional), refracting into the voxel world below.
* Where an **active fluid brick** is present, the renderer uses the volumetric channel and blends to the analytic surface at the AABB boundary.

---

## 6) Streaming for a spherical world

Define a **budgeted spherical working set**:

* **Radius window**: `[R_planet - D_below, R_planet + D_above]` (e.g., 300 m below, 2 km above).
* **Angular window** around the camera: load bricks whose **angular distance** ≤ `θ_max(L)` where `L` depends on LOD (e.g., 5° for high LOD, 30° for very coarse far LOD proxies).
* **Priority score** per brick/region:

  ```
  score = w0 * (1 / (1 + angDist)) + w1 * (1 / (1 + abs(alt - R_sea))) + w2 * editProximity + w3 * simActive
  ```
* Keep **two caches**: CPU‑RAM (big) and GPU (small). Promote frequently touched bricks; evict by low score and last‑used time.

For **Option B (cubed‑sphere)**, tie this logic to the quadtree LOD: near camera → deeper LOD tiles, far → coarser tiles. Each tile still ends up as bricks in the global pool.

---

## 7) Worldgen on a sphere (volumetric)

Use **3D world‑space noise** (domain‑warped) so you never get pole seams:

```cpp
double3 x = k * normalize(p) * (R_planet + radial); // 'radial' is signed altitude sample
double h = fbm(warp(noise3D(x)));
```

* Carve the **solid shell** by comparing `radial` against `heightfield(p)`.
* Caves: subtract a second 3D FBM volume (band‑passed) under the crust.
* Materials: map latitude/altitude/slope to palette IDs (ice caps, deserts, etc.).
* Bake **macro masks** and an optional **coarse SDF** per region for renderer/physics skipping.

---

## 8) Renderer kernel deltas (pseudocode)

```cpp
// Before DDA:
bool hit = IntersectSphereShell(ray.o, ray.d, R_inner, R_outer, &tEnter, &tExit);
if (!hit) { MissQueue.push(ray); return; }
ray.tmin = max(ray.tmin, tEnter);
ray.tmax = min(ray.tmax, tExit);

// DDA exactly as V2 (world-space axis-aligned bricks)
TraverseBricksDDA(ray);

// Shading: if we miss voxels but tExit < infinity, check analytic ocean at R_sea for a final surface.
```

---

## 9) Tooling you’ll be glad you added

* **Overlays**: altitude bands, angular distance ring, brick residency heatmap on the sphere, gravity vector debug.
* **Counters**: fraction of rays culled by sphere shell, bricks touched per degree of view, sim AABB surface area.
* **Camera hops**: teleport 5,000 km and validate that precision is still tight after rebasing.

---

## 10) Suggested milestone path (spherical edition)

1. **M0: Planet shell + sky**

    * Analytic sphere in the path tracer + atmosphere. No voxels yet. Floating origin working.

2. **M1: Voxels on the sphere (static)**

    * Instantiate a band of bricks around `R_planet`. DDA clamped by shell. Macro skipping.

3. **M2: Streaming shell**

    * Spherical priority queue; fly around globe without hitching.

4. **M3: Editing in ENU**

    * Brushes oriented to local tangent frame; changes appear correctly on curved ground.

5. **M4: Hydrostatic ocean + shoreline sim**

    * Analytic sea outside active AABBs; volumetric water inside them; render/refraction integration.

6. **M5: Destruction/rigids with radial gravity**

    * Sleep + re‑voxelize, SDF updates in dirty regions.

7. **M6: Far‑field LOD polish**

    * Optional cubed‑sphere quadtree feeding coarse proxy bricks and/or implicit crust SDF fallback.

---

## Practical parameters to start with

* `R_planet` = 10,000 m (small “moon” scale for dev).
* Brick = **8×8×8** voxels, `VOXEL_SIZE` = 0.25 m → brick = 2 m.
* Working shell: **−256 m..+2048 m** around `R_planet`.
* Max resident bricks GPU: ~120k (≈ 70–80 MiB base data).
* Sim AABB caps: 256³ voxels per active fluid region; max 4 concurrent regions.

---

## Key takeaways

* You **don’t** need curved bricks or bespoke spherical DDA: keep axis‑aligned bricks in world space and **clamp traversal** with an analytic shell.
* Radial gravity + ENU frames make gameplay feel natural.
* Treat the ocean as **hydrostatic baseline + local active volumes**; the renderer blends them seamlessly.
* A **spherical streaming policy** (radius + angle) replaces the planar ring from V2; everything else stays pleasantly the same.

If you want, I can turn this into a starter repo outline (folders, core structs, and the handful of shaders/kernels you need for M0–M2), or sketch the brick/region hash and the sphere‑clamped DDA in more concrete C++/HLSL.

I'm curious how the world would actually be conceptualized without curved bricks or some kind of projection of cube to sphere.  How do you mean to do that?

Great question. The trick is to *never* bend your grid. Keep a normal, axis‑aligned 3D brick lattice in Cartesian space, and make the **planet** merely a *region of that space* defined by an implicit function. Everything “spherical” comes from how you **query** and **stream** that region, not from curving bricks or projecting a cube.

Below is the mental model + concrete formulas and snippets you can drop into your V2.

---

## 0) One sentence model

> **World = ℝ³.** Bricks are regular AABBs in ℝ³. The *planet* is the subset of ℝ³ where a signed field `F(p)` says “solid” (or water/air). Rays, edits, and sims interact with this field and with the bricks that intersect it. No projections. No curved bricks.

---

## 1) Define the planet by a signed field

Let `p ∈ ℝ³` be world coordinates with the planet center at the origin.

* Base radius: `R` (meters)
* Unit direction: `û = normalize(p)`
* Procedural terrain height (meters): `H(û)` (e.g., domain‑warped 3D noise sampled on the unit sphere)
* Altitude above terrain:

  ```
  h_terrain(p) = |p| - (R + H(û))
  ```
* Signed “crust” field (negative = solid ground, positive = air above ground). Optionally clamp below to give finite crust thickness `T`:

  ```
  F_crust(p) = max(h_terrain(p), -(T - caves(…)))
  ```

  where `caves(…)` is a 3D noise term that digs tunnels/caverns when > 0.

**Materials** come from simple rules using latitude (from `û`), slope (∥∇F∥), altitude, moisture, etc., mapping to palette IDs.

> Conceptually: the sphere shows up because your *procedural rule* says “solid if `|p|` is near `R + H(û)`,” not because the grid is curved.

---

## 2) Bricks: plain old AABBs + integer addressing

Pick:

* **Voxel size**: `VOXEL` (e.g., 0.25 m)
* **Brick dimension**: `B = 8` (so each brick is 2 m if VOXEL=0.25)
* **Brick size**: `S = B * VOXEL`

**Indexing**

```cpp
// world-space → brick coordinates (int3) and local voxel (0..B-1)
int3 brickCoord(float3 p)  { return floor(p / S); }
int3 voxelCoord(float3 p)  { float3 q = (p / VOXEL) - B * floor(p / S);
                              return clamp((int3)q, 0, B-1); }
// brick → world origin
float3 brickOrigin(int3 bc){ return (float3)bc * S; }
```

Your **hash map** (or region+hash) keys on `brickCoord(p)` exactly as in a flat world.

---

## 3) Make it feel spherical via *queries*, not geometry

### 3.1 Clamp ray traversal to a spherical shell

Before your DDA, intersect the ray with the analytic shell where content can exist:

```cpp
// Radii that bound "stuff" (terrain, caves below, atmosphere/water above)
float R_inner = R - T;                  // e.g., crust thickness below terrain
float R_outer = R + H_max + Atmosphere; // e.g., mts + sky extent

bool IntersectSphereShell(float3 o, float3 d,
                          float R0, float R1,
                          out float tEnter, out float tExit);
```

Set `ray.tmin = max(ray.tmin, tEnter)` and `ray.tmax = min(ray.tmax, tExit)`.
Now your **brick DDA** operates only inside the shell—no curved math required.

### 3.2 Decide occupancy/material at shading time

For a candidate voxel center `x` (or its brick), evaluate the field:

```cpp
float f = F_crust(x);
bool solid = (f < 0.0f);
uint matId = pickMaterial(x, f, gradF(x), /* climate data, etc. */);
```

If you have authored/edited voxels in that brick, they override the procedural value.

---

## 4) Streaming: “sphere of bricks” policy

You only keep bricks that intersect the shell near the camera.

**Fast AABB vs shell test** (origin at 0):

```cpp
struct AABB { float3 c; float3 h; }; // center, half extents in meters

// distance from origin to an AABB: min/max bounds
float minDistToAABB0(AABB b) {
  float3 q = max(abs(b.c) - b.h, 0.0);   // componentwise
  return length(q);
}
float maxDistToAABB0(AABB b) {
  return length(abs(b.c) + b.h);
}

bool overlapsShell(AABB b, float Rin, float Rout) {
  float dmin = minDistToAABB0(b);
  float dmax = maxDistToAABB0(b);
  return (dmin <= Rout) && (dmax >= Rin);
}
```

Build candidate bricks from a **radius band** `[R−T−margin, R+H_max+Atmosphere]` and an **angular window** around the camera (dot product threshold on centers). Prioritize by angular distance, altitude proximity to sea level, “recently visible,” and “sim‑active,” then load/evict.

---

## 5) Gameplay frames and “down”

Everything that needs “down” uses the *radial* up at a point:

```cpp
float3 up    = normalize(p);
float3 east  = normalize(cross(float3(0,0,1), up));
if (dot(east,east) < 1e-6) east = float3(1,0,0); // near poles fallback
float3 north = cross(up, east);
```

Use this local ENU basis for:

* Brush gizmos aligned to the tangent.
* AI/navigation that wants “walkable” relative to the surface.
* Character controllers (upright constraints).
* Gravity `g = -g0 * up` (or inverse‑square if you like, but usually clamp to `g0` near surface).

> The grid stays axis‑aligned in world space; *orientation* for tools and physics is where you express curvature.

---

## 6) Fluids and oceans without a spherical grid

Treat the **ocean** as two layers:

1. **Passive analytic ocean** at radius `R_sea`:
   For shading, a ray that misses voxels but intersects the sphere `|p|=R_sea` hits “water.” Refract rays into the voxel world below for seabed.

2. **Active volumetric water** only in AABBs around shorelines/rivers/waterfalls:

    * Allocate velocity/pressure *only* in those bricks.
    * Use radial `up` for gravity inside the AABB (constant over that small region).
    * When the region is inactive, discard channels; the passive ocean handles visuals.

Again: no curved bricks—just which bricks you allocate.

---

## 7) Worldgen with zero projections (no seams at poles)

Sample **3D noise in world space** for anything volumetric, and use `û` only when you want “on‑sphere” patterns:

```cpp
double3 x = normalize(p);           // û
double  r = length(p);
double  h = fbm( warp( noise3D(x * k1) ) ) * amp; // terrain height
float   f = r - (R + h);                             // altitude over terrain
bool    solid = (f < 0.0) && (caveNoise(p * k2) < caveThreshold);
```

Because you’re not flattening the sphere to 2D, **there are no projection seams**—the only artifacts are the usual voxel stair‑steps, which you control with voxel size and LOD.

---

## 8) Editing “on a sphere” with a flat brick lattice

Edits are still just `(brick, localMaskDelta, materialDelta)` writes. The *shape* of your brush comes from the local ENU frame:

* **Disk brush aligned to surface**:

    1. Build an orthonormal basis `(east,north,up)` at the hit point.
    2. For a voxel center `x` in nearby bricks, compute surface‑plane coords
       `u = dot(x - p0, east)`, `v = dot(x - p0, north)`; inside radius → edit.

* **Flatten**: set `F_crust(x)` toward a target altitude band.

* **Cut sphere/cylinder**: use analytic signed distance against `x` in world space.

No special‑case bricks near poles or the date line—because there is no map projection in play.

---

## 9) Numeric sanity (why this works)

If your brick size is 2 m and the planet radius is 10,000 m, the **curvature over one brick** is tiny:

* Surface bends by ~2 / 10,000 = 0.0002 radians ≈ 0.011° across a brick.
* You can treat the surface as flat inside any small AABB (edits, rigs, solver cells) and recompute “up” per region.

All the spherical behavior appears when you:

* Clamp rays to the shell,
* Choose radial gravity,
* Stream bricks by radius/angle, and
* Define occupancy/materials via the spherical `F(p)`.

---

## 10) Putting it together (pseudocode sketch)

```cpp
// Frame setup (CPU, double precision)
double3 camWorld;                     // geocentric
double3 origin = camWorld;            // floating-origin rebase

// Render (GPU, floats, origin-relative)
for (Ray r : primaryRays) {
  // Clamp to spherical shell
  float t0, t1;
  if (!IntersectSphereShell(r.o, r.d, R_inner, R_outer, t0, t1)) { skyShade(r); continue; }
  r.tmin = max(r.tmin, t0);
  r.tmax = min(r.tmax, t1);

  // Standard brick DDA
  Hit h;
  if (TraverseBricksDDA(r, &h)) {
     ShadeVoxelHit(h);
  } else if (IntersectSphere(r, R_sea, &tSea)) {
     ShadeOcean(r, tSea);
  } else {
     skyShade(r);
  }
}
```

---

## Key takeaway

* The **grid never changes**. You change how you **decide** what’s solid/liquid/air (the signed field), how you **limit traversal** (sphere clamp), how you **stream** (radius+angle), and how you **orient interactions** (local ENU, radial gravity).
* Because you never project the surface to 2D, you avoid all the usual cube‑to‑sphere seam/scale headaches.

If you want, I can turn this into a tiny starter with:

* the `overlapsShell(AABB, Rin, Rout)` culler for streaming,
* a brick DDA that uses a shell clamp, and
* a toy `F_crust(p)` (height + caves) so you can fly around a voxel planet the same day.

Does this method mean that at some places we wont have flat surfaces facing "up" but the corners of the voxels?

Short answer: **only if you render “binary blocks.”**
With the spherical method I proposed, you don’t have to show voxel faces/corners pointing up. You can keep an axis‑aligned grid but render and collide against a **smooth implicit surface** (or a local TSDF) that lives *over* that grid. Here’s how to avoid the “corner up” look:

---

## Two ways to represent the surface (pick 1–2 together)

### 1) **Implicit/analytic field for base terrain (recommended)**

* Define a continuous signed field (F(p)) (planet radius + height + caves).
* Use the grid only for *acceleration/storage*; at hit time you **refine** the intersection to the isosurface (F(p)=0) and compute the normal from (\nabla F).
* Result: the visible ground is smooth and locally aligned with radial “up,” even though your storage is a bitset.

**Ray refinement (inside your DDA)**

1. DDA finds the first “solid” cell along a ray ⇒ you now have a bracket ([t_0,t_1]) where (F) changes sign.
2. Do 4–8 steps of **bisection/secant** on (g(t)=F(o+t,d)) to get sub‑voxel hit (t^*).
3. Normal (n=\text{normalize}(\nabla F)) via 3 central differences around the hit.
   This costs a handful of extra `F()` calls per visible hit and completely removes “corner‑up” shading.

### 2) **Local TSDF/“micro‑SDF” in edited or dynamic bricks**

* For regions the player edits, keep a **16‑bit TSDF** per voxel (truncated to a few voxels’ range).
* Update it incrementally (jump‑flood or fast sweeping) when bricks change.
* Use the TSDF exactly like (F) above for hit refinement + normals, and for **smooth collision**.

> Memory math: an 8³ brick with a half‑precision TSDF is 8×8×8×2 = **1 KiB**. Even if 10k bricks are “dynamic,” that’s ~10 MiB—cheap insurance against blockiness where it matters.

---

## What happens if you don’t do either?

Then you’re rendering **binary occupancy** and intersecting at axis‑aligned cell faces. On diagonals you’ll see stair‑steps and, at certain view angles, “corners” will be the highest point. That’s the classic “Minecraft look.” It’s a *choice*, not a consequence of the spherical approach.

---

## Visuals vs. physics

* **Visuals**: Use (F) or TSDF to refine the hit and compute smooth normals. Your ground looks flat (tangent to “up”) where it should, not faceted to the grid.
* **Physics**: Sample the same field for contacts—e.g., signed distance for penetration depth and (\nabla F) for contact normals—so characters don’t “tick” up stairs. Keep this SDF **only** in dirty/active bricks to stay within memory budgets.

---

## Where the axis‑aligned grid still helps

* **Empty‑space skipping** and **streaming** (what to keep in memory).
* **Edits**: the brush still writes voxel deltas, but the TSDF/field is what you render/collide against.
* **LOD**: far from the camera, render the analytic (F(p)) alone; near the camera, blend in TSDF‑aware bricks so edits show up.

---

## Practical knobs (so you actually don’t see corners)

* **Refine tolerance**: stop when the interval ([t_0,t_1]) < 0.2×voxel size or (|F|<\tau).
* **Normal epsilon**: use (\epsilon\approx 0.5\times) voxel size for (\nabla F) differences—stable and cheap.
* **Fallbacks**: if a brick has no TSDF and no analytic field (rare), compute a **smoothed occupancy normal** (3×3×3 Sobel/Prewitt) as a last resort.

---

### Bottom line

Keeping straight bricks does **not** force “corner‑up” surfaces. Treat the planet as a smooth field, do a tiny ray–surface refinement near the first hit, and (where edited) keep a light TSDF. You’ll get visually and physically smooth, “up‑facing” ground everywhere, while the grid continues to do what it’s best at: culling, caching, and locality.


Ok great.  Say I wanted to build a rendering framework to be the basis (stretch goal here) of a game engine.  But for now really just a playable tech demo.  Can you outline all of the steps necessary, starting from a blank project? I'll use C++ with vulkan.  Please be detailed with the plan, including all of the important bits that we've talked about, especially how the math will work given the goal of having a spherical world.  Repository structure on disk and project structure regarding important classes/abstractions should also be included.


## 0) Scope & stretch goals

**Tech demo deliverable**

* Fly/walk around a small “planet” (e.g., radius 10–20 km).
* Dense voxel visuals (micro‑bricks) with path‑traced lighting, temporal accumulation + denoiser.
* Streaming shell of bricks around the camera; basic edits; optional volumetric water in one active region.
* Smooth surfaces (implicit field/TSDF refinement) so no “corner‑up” artifacts.

**Engine‑ready architecture**

* Modular subsystems (platform, core, math, world, render, sim, tools).
* Clean Vulkan abstractions (device, memory, descriptors, pipelines, frame graph).
* Offline/online shader compilation, hot‑reload.
* Profiling, validation, crash logs, unit tests, CI.

---

## 1) Repository layout

```
voxel-planet/
├─ CMakeLists.txt
├─ extern/                  # third party (as submodules or FetchContent)
│  ├─ volk/                 # Vulkan loader
│  ├─ VMA/                  # Vulkan Memory Allocator
│  ├─ glm/                  # math
│  ├─ spdlog/               # logging
│  ├─ tracy/                # profiler
│  ├─ stb/                  # images
│  └─ xxhash/               # fast hashing
├─ tools/
│  ├─ shaderc_build/        # offline compiler wrapper scripts
│  └─ pack_assets/          # packs materials/env maps to blobs
├─ data/
│  ├─ materials.json
│  ├─ envmaps/
│  └─ noise_tables.bin
├─ shaders/                 # GLSL (compute) → SPIR-V
│  ├─ common.glsl
│  ├─ generate_rays.comp
│  ├─ traverse_bricks.comp
│  ├─ shade.comp
│  ├─ denoise_atrous.comp
│  ├─ composite.comp
│  ├─ updates_apply.comp    # apply voxel edits
│  └─ sdf_update.comp       # optional (jump-flood)
├─ src/
│  ├─ app/                  # entry/UI
│  │  ├─ App.cpp/.h
│  │  └─ Input.cpp/.h
│  ├─ platform/             # window, swapchain, time
│  │  ├─ Window.cpp/.h
│  │  ├─ VulkanContext.cpp/.h
│  │  └─ Swapchain.cpp/.h
│  ├─ core/                 # memory, jobs, frame graph, hot reload
│  │  ├─ FrameGraph.cpp/.h
│  │  ├─ Jobs.cpp/.h
│  │  ├─ Upload.cpp/.h
│  │  ├─ Descriptors.cpp/.h
│  │  ├─ Pipelines.cpp/.h
│  │  └─ Debug.cpp/.h
│  ├─ math/
│  │  ├─ Spherical.cpp/.h   # F(p), shell intersec, ENU, gradients
│  │  └─ RNG.cpp/.h
│  ├─ world/                # voxels, bricks, streaming, edits
│  │  ├─ BrickStore.cpp/.h
│  │  ├─ RegionCache.cpp/.h
│  │  ├─ WorldGen.cpp/.h
│  │  ├─ Materials.cpp/.h
│  │  └─ Streaming.cpp/.h
│  ├─ render/               # ray/path tracer & denoiser
│  │  ├─ Raytracer.cpp/.h
│  │  ├─ Denoiser.cpp/.h
│  │  ├─ GpuBuffers.cpp/.h
│  │  └─ Tonemap.cpp/.h
│  ├─ sim/                  # optional fluids/rigids (scoped)
│  │  ├─ Fluids.cpp/.h
│  │  └─ Collide.cpp/.h
│  └─ tools/                # overlays, captures, inspectors
│     ├─ Overlays.cpp/.h
│     └─ Capture.cpp/.h
└─ tests/
   ├─ MathTests.cpp
   ├─ BrickTests.cpp
   └─ SphericalTests.cpp
```

---

## 2) Build system & dependencies

* **CMake (C++20)**; produce `Debug`, `RelWithDebInfo`, `Release`.
* **Volk** loader + **Vulkan Memory Allocator (VMA)** for GPU memory.
* **GLM** for math (be strict with double vs float).
* **spdlog** for logs; **Tracy** for CPU/GPU profiling.
* Shader toolchain: offline **shaderc/glslc** to SPIR‑V (plus optional runtime compile for hot‑reload).

CMake key points:

* Generate SPIR‑V into `build/shaders/` with a target that re‑builds on changes.
* Add `-fno-exceptions` where comfortable (but keep in tools/tests if you prefer).
* Enable Vulkan validation layers in Debug; crash on validation errors.

---

## 3) Architecture overview (key classes)

### App layer

* `App` – own the main loop, input, camera, UI overlays.
* `Input` – keyboard/mouse/gamepad; camera modes (free fly, surface walk).

### Platform/Vulkan

* `VulkanContext` – instance/device/queues/features; handles descriptor indexing/maintenance.
* `Swapchain` – presentation images & sync.
* `GpuAllocator` – VMA wrapper (buffers/images).
* `UploadContext` – staging queues, persistent staging buffer, timeline semaphores.

### Core

* `FrameGraph` – lightweight pass graph describing compute passes and resource lifetimes.
* `Descriptors` – persistent descriptor pools, set layouts, bindless support.
* `Pipelines` – cache for compute pipelines; hot‑reload (watch SPIR‑V mtime).
* `Jobs` – thread pool for streaming world bricks, worldgen, CPU preprocessing.

### Math

* `Spherical` – all planet math: `IntersectSphereShell`, `F(p)`, `gradF`, ENU basis, gravity, AABB/shell overlap.
* `RNG` – per‑pixel RNG (PCG32) for path tracer.

### World/Voxels

* `BrickStore` – host+device brick pool, formats, handles/IDs; apply edits.
* `RegionCache` – higher‑level grouping (e.g., 64³ bricks) for streaming & eviction.
* `Materials` – palette, PBR parameters, material atlas (read‑only).
* `WorldGen` – deterministic procedural solid/air/material seeds; optional baked macro masks.
* `Streaming` – chooses which bricks to keep around a camera on a spherical shell.

### Render

* `Raytracer` – the compute wavefront: generate → traverse → shade → denoise → composite.
* `GpuBuffers` – queues and framebuffers (accumulation, motion, normal, albedo, output).
* `Denoiser` – SVGF/A‑trous; temporal reprojection.
* `Tonemap` – ACES or simple filmic.

### Simulation (optional early)

* `Fluids` – active AABB solver; writes volumetric water channel for bricks.
* `Collide` – signed distance collisions (TSDF/analytic).

### Tools

* `Overlays` – debug heatmaps/histograms.
* `Capture` – binary capture of a frame’s inputs/resources for repro.

---

## 4) Spherical world math (what makes it “planetary”)

**Global space** (CPU): geocentric Cartesian, **double** precision.
**Render space** (GPU): camera‑relative Cartesian, **float**.

**Per‑frame rebase**

```cpp
struct OriginRebase {
    glm::dvec3 worldOrigin;  // set to camera world pos (double)
    glm::mat3  worldToLocal; // identity (Cartesian), but keep for future
};
inline glm::vec3 ToLocal(const glm::dvec3& p, const OriginRebase& o) {
    return glm::vec3( (p - o.worldOrigin) ); // float truncation here
}
```

**Radial “up” and tangent frame at point `p`**

```cpp
inline glm::vec3 Up(const glm::vec3& p) { return glm::normalize(p); }
inline void ENU(const glm::vec3& p, glm::vec3& east, glm::vec3& north, glm::vec3& up) {
    up = Up(p);
    east = glm::normalize(glm::cross(glm::vec3(0,0,1), up));
    if (glm::length2(east) < 1e-6f) east = glm::vec3(1,0,0);
    north = glm::cross(up, east);
}
```

**Analytic shell clamp for rays**
Solve |o + t d|² = R² for two radii (inner+outer); clamp DDA to overlap of both intervals.

```cpp
struct Interval { float t0, t1; bool hit; };
inline Interval IntersectSphere(float3 o, float3 d, float R) {
    float b = dot(o,d);             // assuming |d|=1
    float c = dot(o,o) - R*R;
    float disc = b*b - c;
    if (disc < 0) return {0,0,false};
    float s = sqrt(disc);
    return { -b - s, -b + s, true };
}
inline bool IntersectSphereShell(float3 o, float3 d, float Rin, float Rout,
                                 float& tEnter, float& tExit) {
    auto in  = IntersectSphere(o,d,Rin);
    auto out = IntersectSphere(o,d,Rout);
    if (!out.hit) return false;
    // Shell = outside inner sphere AND inside outer sphere.
    float t0 = out.t0, t1 = out.t1;
    if (in.hit) { // remove the portion inside inner sphere
        // Two shell intervals: (-inf, in.t0]∩[out] and [in.t1, +inf)∩[out]
        // For clamping DDA, pick the interval overlapping [0, +inf) with smallest positive length.
        float a0 = std::max(t0, 0.f);
        float a1 = std::min(t1, in.t0);
        float b0 = std::max(in.t1, 0.f);
        float b1 = t1;
        // choose whichever has positive length; prefer near one
        if (a1 > a0) { tEnter = a0; tExit = a1; return true; }
        if (b1 > b0) { tEnter = b0; tExit = b1; return true; }
        return false;
    } else { tEnter = std::max(t0, 0.f); tExit = t1; return tExit > tEnter; }
}
```

**Crust signed field** (F(\mathbf{p})) (continuous; negative = solid)

* Base radius `R`, terrain height `H(û)` (noise on unit sphere), optional caves `C(p)`.
* ( F_{\text{crust}}(p) = |p| - (R + H(\hat{p})) )
* With finite thickness and caves:
  ( F(p) = \max\big(F_{\text{crust}}(p), -(T - C(p))\big) )

**Refinement & normals**

* After DDA finds a candidate solid cell, solve for (F(o+t d)=0) by **bisection** (4–8 steps).
* Normal (n = \frac{\nabla F}{|\nabla F|}) by central differences with ( \epsilon \approx 0.5 \times \text{voxel size}).

**Local gravity**
( \mathbf{g}(p) = -g_0 ,\hat{p} ) (constant magnitude near surface), or inverse‑square if desired.

**AABB vs shell overlap** (for streaming culling) uses min/max radius to the AABB (as described earlier).

---

## 5) Voxel data model (host & device)

**Brick size**: 8×8×8 voxels.
**Voxel size**: start with 25–50 cm (tune later).
**Primary payload per brick** (GPU):

* `occupancyMask[8]` – 8× `uint64_t` → 512 bits (8×8×8).
* `matIndex` – 512×4‑bit indices into a **small palette** (≤16 per brick). Fall back to 8‑bit if needed.
* `palette[<=16]` – indices to global material table.
* `flags` – bitfield (hasTSDF, uses4bit, hasFluid, etc.).
* (Optional dynamic) **TSDF** (16‑bit half per voxel) chosen only for “dynamic / edited” bricks.

**Indirection: page tables**

* **Region** = 64×64×64 bricks (CPU concept for streaming + eviction).
* Global **hash map** keyed by brick coords `(bx,by,bz)` → `BrickHandle` (index into GPU pool).
* On GPU: read‑only “perfect-ish” hash (rebuilt when necessary) or flat buckets optimized for coalesced loads.

**Materials**

* Global array with PBR parameters (baseColor, roughness, metalness, emission, absorption).
* Per‑brick palette maps 4‑bit IDs to global indices.

---

## 6) Vulkan resources & descriptors

**Descriptor sets (suggested)**

* **Set 0 – Globals**:
  `GlobalsUBO` (camera, jitter, frame index, render origin, voxel size, radii `R_in/out`, sea level, exposure).
* **Set 1 – World** (all SSBOs, bindless if available):
  region hash, brick headers, occupancy buffer, material index stream, palettes, macro masks; optional TSDF pool; materials table.
* **Set 2 – Ray queues**:
  `RayQueueIn`, `HitQueue`, `MissQueue`, `SecondaryQueue` (SSBO ring buffers with atomics).
* **Set 3 – Frame images**:
  `accumColor`, `moments`, `albedo`, `normal`, `velocity`, `output`.

**Pipelines**

* Compute pipelines: `generate_rays`, `traverse_bricks`, `shade`, `denoise_atrous`, `composite`, `updates_apply`, `sdf_update`.
* Use **specialization constants** for brick size, traversal step epsilon, and feature toggles.

**Synchronization**

* One graphics/compute queue is enough initially; later split async compute.
* Timeline semaphores between upload → raytrace → denoise → composite.
* Double or triple buffering for frame images.

**Memory**

* VMA heaps: device‑local for big pools; host‑visible staging; transient upload ring.

---

## 7) GPU wavefront path‑tracer (compute)

**Pass order**

1. **generate_rays.comp**

    * Create primary rays in camera‑relative space; set `tmin=near`, `tmax=far`.
    * Jitter for TAA; write to `RayQueueIn`.
2. **traverse_bricks.comp**

    * For each ray: clamp to **sphere shell**; run **brick‑DDA** (world‑space axis aligned).
    * If brick found & occupied → micro‑DDA against 8×8×8 mask.
    * On first “solid” voxel: **refine** with bisection to (F=0), compute normal via ( \nabla F ).
    * Emit `Hit{p,n,matId}` to `HitQueue`; otherwise `MissQueue`.
3. **shade.comp**

    * Shade hits (PBR), sample environment for misses.
    * Temporal accumulation: write color/albedo/normal/velocity for denoiser; spawn secondaries (NEE bounce count 1–2).
4. **traverse_bricks (secondary)** → **shade** (again).
5. **denoise_atrous.comp**

    * SVGF or A‑trous with albedo/normal/variance.
6. **composite.comp**

    * Tonemap/GUI → swapchain.

**Brick‑DDA sketch (GLSL‑ish)**

```glsl
// Step through bricks of size S (meters) in ray space
void traverse_bricks(Ray r) {
  if (!intersectSphereShell(r.o, r.d, R_in, R_out, r.tmin, r.tmax)) { enqueueMiss(r); return; }
  DDA dda = initDDA(r, S);
  for (;;) {
    int3 bc = dda.brickCoord;
    BrickHandle h = lookupBrick(bc);
    if (h.valid) {
      Brick br = brickPool[h.idx];
      if (br.occupancyMask != 0ul) {
        if (microDDA_and_refine(r, br, hit)) { enqueueHit(hit); return; }
      }
    }
    if (!dda.advance()) break; // past tmax
  }
  enqueueMiss(r);
}
```

**Micro‑DDA + refinement**

* Convert to **brick‑local** coordinates; parametric walk the 8×8×8 occupancy bits.
* When entering a “solid” bit, take the segment ([t0,t1]) inside that cell and run **bisection** on (F(o+t d)).
* Gradient: central differences on (F) (analytic field), or use TSDF if present.
* This yields smooth shading and normals aligned to the spherical surface.

---

## 8) Denoiser & temporal

* Accumulate with `accumColor = lerp(prev, curr, alpha)` (alpha from sample count).
* **SVGF/A‑trous**: bilateral weights use albedo, normal (from refined hit), and motion vectors from camera delta & rebase.
* Clamp reprojected history using local variance.

---

## 9) World streaming on a spherical shell

**Working set window**

* Radii: `[R_in = R - T - marginBelow, R_out = R + Hmax + atmosphereTop]`.
* Angular window: bricks whose **center direction** is within `θ_max` of camera `Up(camPos)`.

**Priority score**

```
score = w0 / (1 + angDist) + w1 / (1 + |radius - R_sea|)
      + w2 * (recentlyVisible) + w3 * (simActive)
```

**Pipeline**

* `Streaming::update(camera)` collects candidate brick coords in nearby regions; filters with **AABB vs shell**; sorts by score; schedules I/O & worldgen on jobs; uploads in batches via `UploadContext`.
* GPU‐resident pool capped (e.g., 120k bricks). Evict LRU/low‑score.

---

## 10) Editing & dynamic TSDF (optional, but nice)

* Edits are queued as `(brickCoord, localBitMaskΔ, matΔ)`.
* `updates_apply.comp` merges deltas into brick SSBOs.
* If a brick becomes “dynamic,” allocate a **TSDF tile** (16‑bit); `sdf_update.comp` recomputes via jump‑flood or fast sweeping in that brick and its 1‑voxel halo.
* Renderer prefers TSDF field over analytic `F` inside dynamic bricks to match edits exactly.

---

## 11) Minimal gameplay loop

* **Free fly** camera with WASD; toggle “walk” mode (project to surface, keep up=n̂(p)).
* **Brush** aligned to ENU frame (disk/sphere). Left‑click: carve/add; changes go to the update queue.
* **Profiler overlay**: rays traced, bricks touched, refine iterations, denoise time, streaming counts.
* **Options**: voxel size, bounce count, exposure, LOD toggles.

---

## 12) Project skeleton (key headers)

**`math/Spherical.h`**

```cpp
struct PlanetParams { double R; double T; double sea; double Hmax; };
float  F_crust(glm::vec3 p, const PlanetParams& P);
glm::vec3 gradF(glm::vec3 p, const PlanetParams& P, float eps);
bool IntersectSphereShell(glm::vec3 o, glm::vec3 d, float Rin, float Rout,
                          float& tEnter, float& tExit);
void ENU(const glm::vec3& p, glm::vec3& east, glm::vec3& north, glm::vec3& up);
```

**`world/BrickStore.h`**

```cpp
struct BrickHeader {
  uint32_t occMaskOffset;   // byte offset in occupancy buffer
  uint32_t matIndexOffset;  // 4-bit/8-bit stream
  uint32_t paletteOffset;   // or 0xFFFFFFFF if none
  uint16_t flags;           // hasTSDF, uses4bit, hasFluid...
  uint16_t paletteCount;
  int32_t  bx, by, bz;      // brick coords
};

class BrickStore {
public:
  BrickHandle getOrCreate(int3 bc);
  void uploadBatch(const std::vector<BrickHandle>&);
  void applyEdits(const EditBatch&);
  // GPU buffers: headers, occupancy, matIndex, palettes, TSDF
};
```

**`render/Raytracer.h`**

```cpp
struct RaytracerParams {
  int width, height;
  int maxBounces;
  float voxelSize, brickSize;
  float Rin, Rout, Rsea;
  uint32_t frameIndex;
};

class Raytracer {
public:
  void init(VulkanContext&, Descriptors&, Pipelines&);
  void resize(int w, int h);
  void render(const RaytracerParams&, const BrickStore&, VkSemaphore wait, VkSemaphore signal);
  // exposes debug buffers for overlays
};
```

---

## 13) Frame graph (per frame)

```
CPU: Input → Camera → Rebase Origin → Streaming::update → Edits queued

GPU:
  Upload::flush (staging to brick buffers) ─┐
  generate_rays ────────────────────────────┼─► traverse_bricks (primary)
  shade (primary) ──────────────────────────┼─► traverse_bricks (secondary)
  shade (secondary) ────────────────────────┘
  denoise_atrous → composite → present
```

Use **timeline semaphores** to chain, and **barriers** between compute passes for SSBO/UAV hazards.

---

## 14) Worldgen (dev‑friendly)

* Deterministic seeds; `H(û)` via domain‑warped FBM on **unit sphere** (no projection seams).
* Caves `C(p)` via 3D noise band‑pass.
* Materials by altitude/slope/latitude (ice caps, deserts, forest).
* Bake **macro masks** per region (fully empty/solid) to boost DDA skipping.

---

## 15) Testing, profiling, and quality gates

* **Unit tests**: `IntersectSphereShell`, AABB/shell overlap, DDA step alignment, `F(p)`/`gradF` consistency.
* **Tracy** zones around each pass; GPU timers (timestamp queries) per pipeline.
* **Validation**: assert brick palette constraints; log fallback to 8‑bit when palette>16.
* **Golden captures**: fixed camera + RNG seed; compare frame stats over time.
* **CI**: build all configs; run tests; package artifacts.

---

## 16) Milestones you can march through

**M0 – Project boot**

* CMake; Vulkan device; swapchain; triangle; Tracy; logs.

**M1 – Compute path tracer base**

* `generate_rays`, `shade` with sky only; accumulation + tonemap.

**M2 – Bricks & DDA**

* BrickStore on CPU; GPU buffers; brick‑DDA vs occupancy (binary hits); macro mask; shell clamp.

**M3 – Implicit refinement**

* Add `F(p)`, bisection refinement, gradient normals; materials + lights; SVGF/A‑trous denoiser.

**M4 – Streaming shell**

* RegionCache; AABB/shell culler; LRU/score eviction; smooth flyaround.

**M5 – Editing & TSDF**

* Apply edits; dynamic TSDF per edited brick; visual parity with `F(p)`.

**M6 – Ocean pass (optional)**

* Analytic sea at `R_sea` for misses; refract to seabed; shoreline masks.

**M7 – Active fluid AABB (stretch)**

* One solver region near camera; write liquid channel; basic refraction/foamy shading.

---

## 17) Gotchas & guardrails

* **Precision**: always rebase origin each frame; keep *double* in CPU world, *float* on GPU.
* **Traversal robustness**: clamp `tmin` to a small positive ε; add 1e‑4 voxel step bias to escape precision traps.
* **Queue overflows**: bound queue sizes; if overflow, drop secondaries first. Log counters.
* **Denoiser ghosts**: use motion vectors from rebase transform; clamp history with per‑pixel variance.
* **Palettes**: enforce ≤16 materials/brick when using 4‑bit; otherwise flip a flag and use 8‑bit stream.

---

## 18) Example CMake fragments

**Top‑level**

```cmake
cmake_minimum_required(VERSION 3.24)
project(voxel_planet LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 20)

add_subdirectory(extern/volk)
add_subdirectory(extern/VMA)
# ... glm, spdlog, tracy, stb, xxhash ...

add_subdirectory(src)
add_subdirectory(shaders)  # custom target that compiles to SPIR-V
```

**`src/CMakeLists.txt`**

```cmake
file(GLOB_RECURSE SRC CONFIGURE_DEPENDS
    app/*.cpp platform/*.cpp core/*.cpp math/*.cpp
    world/*.cpp render/*.cpp sim/*.cpp tools/*.cpp)
add_executable(voxel_planet ${SRC})
target_include_directories(voxel_planet PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(voxel_planet PRIVATE volk vma glm spdlog tracy)
target_compile_definitions(voxel_planet PRIVATE VK_ENABLE_BETA_EXTENSIONS=1)
```

---

## 19) What to build first this week

1. **M0–M1** (1–2 days): Window + swapchain + compute “sky tracer.”
2. **M2** (2–4 days): BrickStore with hard‑coded test bricks; DDA traversal; visible voxels.
3. **M3** (3–5 days): Implement `F(p)`, refine hits, normals, basic PBR, temporal accumulate + A‑trous.
4. **M4** (2–3 days): Spherical shell streaming; fly around the “planet.”
5. **M5** (3–4 days): Editing + TSDF for dynamic bricks; debug overlays & counters.

You’ll now have a **playable tech demo** that looks “alive,” scales with view angle, and is architected like an engine.

---

If you want, I can drill into any one module next (e.g., **brick hash format & GPU lookup**, or **the exact GLSL for the brick‑DDA + refinement**), or sketch a minimal `GlobalsUBO` + descriptor set layout that matches the shader snippets so you can code straight from this outline.

Let’s nail down the “missing pieces” so you can open a blank repo and start typing with confidence. I’ll drill into:

1. **Descriptor layout & `GlobalsUBO`**
2. **GPU work queues (persistent‑threads pattern)**
3. **Brick hash & GPU lookup** (with memory layout and packing)
4. **Micro‑DDA in an 8×8×8 brick** (bit‑packed occupancy)
5. **Isosurface refinement to (F(p)=0)** + normals
6. **Macro masks & far‑field LOD**
7. **TSDF tiles: data, updates (jump‑flood / fast‑sweep)**
8. **Streaming scheduler details** (radius/angle policy)
9. **Motion vectors & rebase math for temporal**
10. **Denoiser knobs that work on voxels**
11. **Vulkan sync cheat‑sheet (barriers + timeline semaphores)**
12. **Memory budgets & alignment facts**
13. **Worldgen (domain warps on the unit sphere)**
14. **Debug overlays, tests, and telemetry you want from day 1**

---

## 1) Descriptor layout & `GlobalsUBO`

**Bindings (suggested, bindless‑friendly):**

* **Set 0 – Globals (immutable sizes)**

    * `binding 0`: `GlobalsUBO` (UBO, 256B aligned)
    * `binding 1`: blue‑noise / RNG seeds (SSBO or texture)

* **Set 1 – World (SSBOs, potentially bindless)**

    * `b0`: Brick headers (SSBO, struct of arrays)
    * `b1`: Occupancy bitset (SSBO, `uint64_t[8]` per brick, tightly packed)
    * `b2`: Material index stream (SSBO, 4‑bit packed; fallback 8‑bit region)
    * `b3`: Per‑brick palette (SSBO; up to 16 entries/brick)
    * `b4`: Region/global hash table(s) (SSBOs)
    * `b5`: Materials table (SSBO or UBO)
    * `b6`: Macro masks / far‑LOD data
    * `b7`: TSDF pool (optional, 16‑bit SDF per voxel, only for dynamic bricks)

* **Set 2 – Queues**

    * `RayQueueIn`, `HitQueue`, `MissQueue`, `SecondaryQueue` (SSBOs with headers + ring buffers)

* **Set 3 – Frame images**

    * `accumColor`, `albedo`, `normal`, `moment/variance`, `velocity`, `output` (storage images)

**`GlobalsUBO` (CPU→GPU, floats unless noted):**

```cpp
struct GlobalsUBO {
  // Camera (local, origin-rebased)
  glm::vec3 camPos;      float pad0;
  glm::mat4 currView;    // row-major in C++, transpose on upload if needed
  glm::mat4 currProj;    // jittered for TAA
  glm::mat4 prevView;
  glm::mat4 prevProj;

  // Rebase info (world origin delta from prev to curr in *local* units)
  glm::vec3 originDeltaPrevToCurr; float pad1;

  // Planet + render params
  float voxelSize;  float brickSize;  float Rin;      float Rout;
  float Rsea;       float exposure;   uint frameIdx;  uint maxBounces;

  // Dimensions
  uint width; uint height; uint raysPerPixel; uint flags; // flags: useTSDF, useAnalyticSea, etc.
};
static_assert(sizeof(GlobalsUBO) % 256 == 0);
```

> Why keep prev matrices and `originDeltaPrevToCurr`? For stable **motion vectors** and temporal reprojection even when you rebase the world each frame.

---

## 2) GPU work queues (persistent‑threads/wavefront)

Each queue is an SSBO:

```glsl
struct QueueHeader { uint head; uint tail; uint capacity; uint pad; };

// Example payloads
struct Ray { vec3 o; float tmin; vec3 d; float tmax; uint rng; };
struct Hit { vec3 p; float t; vec3 n; uint mat; };

layout(std430, binding=X) buffer RayQueueBuf { QueueHeader H; Ray items[]; } Q;
```

**Chunked dequeue** (avoid 1 atomic per thread):

```glsl
// in a compute kernel with local_size_x = 64
shared uint sBase;  // base index for this workgroup
shared uint sEnd;

void acquireChunk(uint chunkSize) {
  if (gl_LocalInvocationIndex == 0) {
    uint start = atomicAdd(Q.H.head, chunkSize);
    sBase = start;
    sEnd  = min(start + chunkSize, Q.H.tail); // tail may move concurrently
  }
  barrier();
}

bool popRay(out Ray r) {
  for (;;) {
    uint idx = sBase + gl_LocalInvocationIndex;
    if (idx < sEnd) { r = Q.items[idx & (Q.H.capacity - 1u)]; return true; }
    // Fetch next chunk
    acquireChunk(64u); // one warp's worth
    if (sBase >= sEnd) return false; // queue exhausted
  }
}
```

**Enqueue**:

```glsl
uint pushHit(Hit h) {
  uint idx = atomicAdd(HQ.H.tail, 1u);
  HQ.items[idx & (HQ.H.capacity - 1u)] = h;
  return idx;
}
```

> Ring capacities are powers of two; we mask to wrap. Track overflows in a separate counter; if a queue would overflow, **drop secondaries first** (write a stat, skip spawn).

---

## 3) Brick hash & GPU lookup

**Brick addressing:** `bc = floor(worldPos / brickSize)` (3× int32).
**Key pack:** 64‑bit Morton or custom pack:

```cpp
inline uint64_t packKey(int bx,int by,int bz) {
  // 21 bits each, signed bias
  const uint64_t B = 1ull<<20; // bias
  return ((uint64_t)(bx+B) << 42) | ((uint64_t)(by+B) << 21) | (uint64_t)(bz+B);
}
```

**Global linear‑probe hash table** (simple, fast to build):

* **Keys SSBO**: `uint64_t keys[cap];` (EMPTY = 0)
* **Vals SSBO**: `uint brickIndex[cap];` (index into BrickHeaders)
* **Load factor**: ≤ 0.5 (cap ≈ 2× brickCount) → short probes, coherent.

**GPU lookup**:

```glsl
uint encodeKey(ivec3 bc) {
  const uint B = 1u<<20; // same bias as CPU
  return ( (bc.x+B) << 10u ) ^ ( (bc.y+B) << 5u ) ^ (bc.z+B); // small mix as start
}
bool findBrick(ivec3 bc, out uint brickIdx) {
  uint64_t key = packKey(bc.x, bc.y, bc.z);
  uint idx = (xxhash32(key) & (cap-1u)); // any fast 32-bit hash of 64-bit key
  // 4-8 linear probes
  for (uint i=0u;i<8u;++i) {
    uint64_t k = keys[(idx+i) & (cap-1u)];
    if (k == key) { brickIdx = vals[(idx+i) & (cap-1u)]; return true; }
    if (k == 0ul) break;
  }
  return false;
}
```

**Brick header (SoA on GPU; AoS on CPU OK):**

```cpp
struct BrickHeader {
  int32_t bx, by, bz;     // brick coords
  uint32_t occOffset;     // byte offsets into big SSBOs
  uint32_t matIdxOffset;
  uint32_t paletteOffset;
  uint16_t flags;         // bit 0: uses4bit, 1: hasTSDF, 2: hasFluid, ...
  uint16_t paletteCount;
  uint32_t tsdfOffset;    // 0xFFFFFFFF if none
};
```

> **SoA** on GPU is slightly faster for coalesced loads, but a single SSBO of headers works fine initially.

---

## 4) Micro‑DDA (voxel walk in an 8×8×8 brick)

**Bit packing** (Z‑major):

```
idx(x,y,z) = x + y*B + z*B*B   (B=8)
word = idx >> 6;  bit = idx & 63;
occupied = (mask[word] >> bit) & 1u;
```

**Initialize local DDA**:

```glsl
// world -> brick local
vec3 p0 = (ray.o - brickOrigin) / voxelSize; // in voxel units
vec3 rd = ray.d; // (still in world units)
vec3 r  = rd / voxelSize;

ivec3 v = clamp(ivec3(floor(p0)), 0, ivec3(7)); // start cell
ivec3 step = ivec3(sign(r));
vec3 tDelta = abs(vec3(1.0) / r);        // parametric t to cross one voxel on each axis
vec3 nextVoxBoundary = (vec3(v) + stepify(step)) - p0;
vec3 tMax = abs(nextVoxBoundary / r);

for (int iter=0; iter<128; ++iter) {
  // Test occupancy
  uint index = uint(v.x + v.y*8 + v.z*64);
  uint word = index >> 6u; uint bit = index & 63u;
  bool occ = ((occMask[word] >> bit) & 1ul) != 0ul;
  if (occ) {
    // We have a solid cell; [tEnter, tExit] for this cell is bracket for refinement
    float tCellEnter = max(max(tPrevX, tPrevY), tPrevZ); // track previous tMax before increment
    float tCellExit  = min(tMax.x, min(tMax.y, tMax.z));
    // refine here (see §5)
    ...
  }

  // Advance along the smallest tMax
  bvec3 choose = lessThanEqual(tMax, min(tMax.yyy, tMax.zzz));
  if (choose.x) { v.x += step.x; tPrevX = tMax.x; tMax.x += tDelta.x; }
  else if (tMax.y <= tMax.z) { v.y += step.y; tPrevY = tMax.y; tMax.y += tDelta.y; }
  else { v.z += step.z; tPrevZ = tMax.z; tMax.z += tDelta.z; }

  // Exit conditions
  if (any(greaterThan(v, ivec3(7))) || any(lessThan(v, ivec3(0)))) break;
}
```

> Keep `tPrev{X,Y,Z}` to recover the “enter” time for the current cell; the “exit” is the smallest among current `tMax`. That gives you a bracket for (F(o+t d)).

---

## 5) Isosurface refinement to (F(p)=0) + normals

Given `t0` and `t1` that bound the segment inside the solid cell:

**Bisection (robust)**

```glsl
float bisectionToZero(vec3 o, vec3 d, float t0, float t1) {
  float f0 = F(o + t0*d);
  float f1 = F(o + t1*d);
  // If not actually bracketing (can happen at grazing), widen slightly or early-out
  if (f0 * f1 > 0.0) { return t0; } // fallback: use t0
  for (int i=0;i<6;++i) { // 6-8 steps are enough for sub-voxel
    float tm = 0.5*(t0+t1);
    float fm = F(o + tm*d);
    if (fm == 0.0) return tm;
    bool sameSign = (fm * f0) > 0.0;
    t0 = sameSign ? tm : t0;
    f0 = sameSign ? fm : f0;
    t1 = sameSign ? t1 : tm;
  }
  return 0.5*(t0+t1);
}
```

**Normals from (\nabla F)** (central differences):

```glsl
vec3 gradF(vec3 p, float eps) {
  float Fx1 = F(p + vec3(eps,0,0));
  float Fx0 = F(p - vec3(eps,0,0));
  float Fy1 = F(p + vec3(0,eps,0));
  float Fy0 = F(p - vec3(0,eps,0));
  float Fz1 = F(p + vec3(0,0,eps));
  float Fz0 = F(p - vec3(0,0,eps));
  return normalize(vec3(Fx1 - Fx0, Fy1 - Fy0, Fz1 - Fz0));
}
```

Use `eps ≈ 0.5 * voxelSize`. If a brick has a TSDF tile, prefer that TSDF’s gradient (cheaper samples, matches edits exactly).

---

## 6) Macro masks & far‑field LOD

**Macro mask** per **macro‑brick** (e.g., groups of 8×8×8 regular bricks):

* A single byte (or two) that encodes: empty, full, mixed.
* During DDA **before** brick lookup, sample macro mask; if **empty**, skip to next macro boundary in one shot. If **full**, you may still need micro traversal but can skip hash lookup for child bricks if you store a “solid by default” rule for that zone.

**Far LOD** options (one or both):

1. **Implicit crust fallback** for far distances: test the analytic sphere+height field first; only descend to voxels when near camera.
2. **Clipmap of coarser bricks** (16³ or 32³) auto‑generated from fine bricks; use them beyond distance (D). This reduces hash lookups and micro‑DDA iterations.

---

## 7) TSDF tiles: data & updates

**TSDF format per edited/dynamic brick**

* `int16_t sdf[8*8*8]` storing SDF/Δ scaled by `scale` (meters per LSB), truncated to ±`TRUNC` voxels.
* Store `scale` (float) and possibly a `weight` field if you do fusion (not required here).

**Updates**
When a brick is edited (add/remove material), recompute its TSDF **and a 1‑voxel halo** so gradients are correct across brick borders.

Two practical options:

* **Jump‑Flood (2–3 passes):**
  Treat new “surface voxels” as seeds (distance 0), initialize others to `+INF`. In each pass with stride (s), each voxel checks 6–26 neighbors offset by (\pm s) and updates its best distance with the neighbor distance + offset length. Strides (B/2, B/4, …, 1). At the end, assign signs via inside/outside test (e.g., parity count ray or relying on occupancy).

* **Fast sweeping** (6 passes):
  Dynamic programming passes in ±x/±y/±z orders; good for eikonal equations. Simple and accurate for small bricks.

> **When to allocate TSDF**: only for bricks with player edits, destruction, or active collisions. Flag in `BrickHeader.flags`.

---

## 8) Streaming scheduler (radius/angle policy)

Each frame:

1. Compute the **camera direction** `u_cam = normalize(camWorldPos)` (geocentric).
2. Define working shell: `[Rin, Rout]`.
3. Enumerate candidate **region** coords around the camera (e.g., regions are 64³ bricks). For each region:

    * Build the region AABB; test **AABB vs shell** overlap.
    * Compute **angular distance** between region center dir and `u_cam`:
      `ang = acos(dot(u_region, u_cam))` or use 1−dot for a cheaper monotonic score.
4. Accept regions with `ang < θ_max` (θ from 5° near → 30° far LOD).
5. Score each region:

   ```
   score = w0/(1+ang) + w1/(1+abs(radius - Rsea))
         + w2*recentlyVisible + w3*simActive - w4*age
   ```
6. For top‑K regions, generate brick coords, cull by per‑brick AABB vs shell, submit **worldgen** or **IO** jobs if missing, batch **uploads**.

**Eviction**
Maintain an LRU/clock per brick; when over GPU budget, drop the lowest‑score/oldest items first, but **protect** `simActive` bricks.

---

## 9) Motion vectors & rebase math

To reproject history (SVGF/TAA), you need previous NDC of the current hitpoint.

Let:

* `P_world_curr` be world position (double) of a shaded hit.
* `originPrev`, `originCurr` be world origins (doubles) used to rebase prev & curr frames.

**Transform to prev local**:

```
P_local_prev = (P_world_curr - originPrev); // double → float
```

Then:

```
prevClip = prevProj * prevView * vec4(P_local_prev, 1);
prevNDC  = prevClip.xyz / prevClip.w;
currClip = currProj * currView * vec4( P_local_curr, 1);
currNDC  = currClip.xyz / currClip.w;
velocity = 0.5 * (currNDC.xy - prevNDC.xy); // map [-1,1] → [-0.5,0.5]
```

Store `velocity` per pixel; the denoiser will fetch history from the previous frame at `uv - velocity`.

> The **rebase** is already baked into `P_local_prev` by subtracting `originPrev` instead of `originCurr`.

---

## 10) Denoiser knobs that play nice with voxels

* **Temporal**: alpha (= 1 / (1 + sampleCount)) up to a cap (e.g., 0.05).
* **Clamping**: neighborhood min/max clamping of reprojected history by current color ± k·sqrt(variance).
* **Spatial A‑trous**: 3–5 iterations, kernel radius 1,2,4,8,…;
  weights = product of:

    * luminance weight (edge‑aware)
    * normal weight: `exp(- (1 - dot(n_i, n_c)) / σ_n)` with σ_n ≈ 0.1–0.2
    * plane weight: use hit positions and normals to compute plane distance
    * albedo weight: optional if albedo varies a lot
* **History reset**: if material ID changes or geometric divergence (`dot(n_prev, n_curr) < 0.7`), reset accumulation locally.

---

## 11) Vulkan sync cheat‑sheet

Between compute passes that write/read SSBOs or storage images, use a **single global memory barrier**:

```cpp
VkMemoryBarrier2 barrier{
  .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
  .srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
  .srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT | VK_ACCESS_2_SHADER_STORAGE_READ_BIT,
  .dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
  .dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
};
vkCmdPipelineBarrier2(cmd, &(VkDependencyInfo){... .memoryBarrierCount=1, .pMemoryBarriers=&barrier });
```

**Timeline semaphores** for frame graph chaining:

* `upload → raytrace`: signal `uploadSem` value N after staging copies; wait in compute submit.
* `raytrace → denoise → composite`: either single queue with in‑queue barriers or chain via timeline values.

**Images**: use `VK_IMAGE_LAYOUT_GENERAL` for storage images during compute; transition to `PRESENT_SRC_KHR` only at the end.

---

## 12) Memory budgets & alignment

Per brick (typical):

* Occupancy: **64 B** (8× `uint64_t`)
* 4‑bit material indices: **256 B** (512×0.5B)
* Palette (≤16 × 2B index): **32 B** (or 64 B if 4B each)
* Header (GPU): **32 B**
* **Total** ≈ **384–416 B** per brick (no TSDF)

If a brick overflows the 16‑material palette and flips to 8‑bit indices:

* Indices become **512 B** → total ~640–672 B.

**TSDF** (optional): 512 voxels × 2 B = **1 KiB** per brick (plus small header). Allocate only for dynamic bricks.

**Working set example** (GPU):

* 120k standard bricks @ ~416 B → **~50 MB**
* 10k dynamic bricks with TSDF → **~10–12 MB**
* Hash tables, macro masks, queues, frame images → **~80–120 MB** total GPU.

> Pack SSBOs to 16‑byte boundaries for good coalescing. Keep materials and small LUTs in UBOs where convenient.

---

## 13) Worldgen on a unit sphere (no seams)

Use **3D noise** on the **unit direction** `û = normalize(p)` for the height; domain‑warp it to avoid simple FBM look. Caves use 3D noise in **full world space** `p`:

```cpp
double3 u = normalize(p);
float h = fbm( warp( noise3(u * k1) ) ) * amp;   // height in meters
float f_crust = length(p) - (R + h);             // altitude above terrain

float caveBand = bandpass(noise3(p * k2), 0.35, 0.55);
float F = max(f_crust, -(T - caveBand * caveAmp)); // signed field (neg = solid)
```

**Materials** from:

* altitude bands (beach, grass, rock, snow by latitude),
* slope via (|\nabla F|) threshold,
* moisture from another noise or sun exposure proxy (`dot(up, sunDir)`).

Bake **macro masks** per region offline/first‑touch to accelerate empty‑space skipping.

---

## 14) Debug overlays, tests, telemetry

**Overlays (drawn in composite)**:

* Brick residency heat (color by “age” or access count)
* Macro mask (empty/full/mixed)
* Rays per pixel, queue sizes (live counters)
* `tMax` visualization (DDA steps)
* Surface normals and curvature (from (\nabla F))
* Streaming spheres: show `[Rin,Rout]` and angular cone

**Unit tests** (small gtests):

* `IntersectSphereShell` vs analytical cases
* `AABB vs shell` overlap edge cases
* `packKey`/`findBrick` round‑trip
* Micro‑DDA: synthetic brick with diagonal occupancy; compare expected hit cell sequence
* `F(p)` continuity and finite‑difference gradient sanity (directional derivative test)

**Telemetry (Tracy zones & counters):**

* Brick lookups per ray, hit ratio, average probe length
* Micro‑DDA steps per hit/miss
* Refine iterations avg/95th
* Queue overflows / drops
* Streaming: bricks uploaded/evicted this frame; I/O time

---

# Glue code you can copy/paste

### GLSL: sphere‑shell clamp + brick traversal skeleton

```glsl
bool intersectSphere(vec3 o, vec3 d, float R, out float t0, out float t1) {
  float b = dot(o,d);
  float c = dot(o,o) - R*R;
  float disc = b*b - c;
  if (disc < 0.0) return false;
  float s = sqrt(disc);
  t0 = -b - s; t1 = -b + s;
  return t1 > 0.0;
}

bool intersectSphereShell(vec3 o, vec3 d, float Rin, float Rout, out float tEnter, out float tExit) {
  float i0,i1,o0,o1;
  if (!intersectSphere(o,d,Rout,o0,o1)) return false;
  bool hitIn = intersectSphere(o,d,Rin,i0,i1);
  float a0 = max(o0, 0.0), a1 = hitIn ? min(o1, i0) : o1;
  float b0 = hitIn ? max(i1, 0.0) : 1e30, b1 = o1;
  if (a1 > a0) { tEnter = a0; tExit = a1; return true; }
  if (b1 > b0) { tEnter = b0; tExit = b1; return true; }
  return false;
}
```

### C++: AABB vs spherical shell (for streaming)

```cpp
struct AABB { glm::dvec3 c; glm::dvec3 h; }; // center, half extents
inline double minRadiusToAABB0(const AABB& b) {
  glm::dvec3 q = glm::max(glm::abs(b.c) - b.h, glm::dvec3(0));
  return glm::length(q);
}
inline double maxRadiusToAABB0(const AABB& b) {
  return glm::length(glm::abs(b.c) + b.h);
}
inline bool overlapsShell(const AABB& b, double Rin, double Rout) {
  double dmin = minRadiusToAABB0(b);
  double dmax = maxRadiusToAABB0(b);
  return (dmin <= Rout) && (dmax >= Rin);
}
```

### GLSL: 4‑bit material unpack

```glsl
// matIdxOffset points to a byte stream of 512 * 0.5 bytes
uint getMat4bit(uint baseOffset, uint linearIdx) {
  uint byteIdx = baseOffset + (linearIdx >> 1u);
  uint8_t packed = matIndexBytes[byteIdx];         // SSBO byte-addressable via u8 aliasing buffer
  return (linearIdx & 1u) != 0u ? (packed >> 4u) : (packed & 0xFu);
}
```

*(If your backend dislikes byte addressing, keep `uint` buffer and bit‑extract from 32‑bit words.)*

---

## Practical “gotchas” (answers to questions you’ll likely have)

* **Do I need curved bricks?** No. Curvature is encoded in (F(p)), gravity, and the shell clamp. Bricks are plain AABBs.
* **Will surfaces look like corners?** Not if you refine to (F=0) and compute normals from (\nabla F) or a TSDF.
* **Precision far from origin?** Rebase per frame (double on CPU, float on GPU). Never accumulate transforms in float world space.
* **How many rays?** Start with 1 spp, 1 bounce + NEE; rely on temporal + SVGF; scale to 2–4 spp if perf allows.
* **Queue sizing?** `RayQueueIn.capacity >= width*height*raysPerPixel`; `HitQueue ~ same`; `SecondaryQueue` ≤ 0.5× primary; bias to avoid overflow.
* **Palette overflow?** If a brick needs >16 materials, flip a flag and re‑encode its index stream as 8‑bit; it costs more memory but keeps the code simple.

---

## Milestone glue (what to implement next)

1. **Implement Set 0/1 descriptors + `GlobalsUBO`**, build and bind all SSBOs with dummy data.
2. **Write `generate_rays.comp`** (camera in local space, jitter, write queue).
3. **Write `traverse_bricks.comp`** with: shell clamp → hash lookup → micro‑DDA → bisection → normals.
4. **Write `shade.comp`** (Lambert/rough‑metal + sky), temporal accumulation, optional shadows via a single shadow ray.
5. **Add `denoise_atrous`** with motion vectors & reprojection.
6. **Bring up streaming** with AABB‑vs‑shell culler and region scoring.

Once you have those, you’ll have a **playable tech demo** you can iterate on—then grow toward fluids, destruction, and richer materials without changing the core math.

If you want me to produce **one cohesive `.comp` shader** that wires up the ray queues, shell intersection, brick lookup, micro‑DDA, and (F) refinement (ready to paste), say the word and I’ll draft it to match the bindings above.
