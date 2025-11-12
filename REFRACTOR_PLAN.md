# Transition To Analytic SDF Terrain & Edits

> **Goal:** eliminate the TSDF pipeline entirely. Bricks remain as streaming/spatial buckets, but all geometry and edits are described analytically so the ray marcher never reads trilinear TSDF samples.

> **Reminder:** the existing 8×8×8 “brick” chunking stays. We only change what data each brick stores (analytic params + primitive lists) and how traversal evaluates the signed field.

## 1. Simplify Base Terrain SDF (Faster Per-Ray Evaluation)
- [ ] Replace FBM-heavy `math::F_crust` / `shaders/spherical.glsl` with lightweight layers:
  - Radial shell + macro heightfield (few hash-based noises or small LUTs).
  - Mid-frequency detail via small 3D textures or simple analytic bumps.
  - Caves/overhangs from tileable textures or sparse analytic features.
- [ ] Expose parameters (frequencies, amplitudes) to keep ALU budget predictable.
- [ ] Update `WorldGen` and BrickStore sampling to use the new function so CPU/GPU stay in sync.
- [ ] Profile per-hit cost; add optional clipmap cache if needed.

## 2. Drop TSDF Data For Static Bricks (Path To Removal)
- [ ] Static bricks store only occupancy/material; leave `tsdfOffset = kInvalidOffset`.
- [ ] In traversal/shaders, treat `tsdfOffset == Invalid` as “call analytic SDF directly”.
- [ ] Stop allocating/uploading `FieldSamplesBuf`; remove CPU `fieldSamples` for those bricks.
- [ ] Verify visuals/perf; once stable, delete the TSDF interpolation code entirely.

## 3. Represent Player Edits As Analytic Primitives (Per Brick)
- [ ] Define an `EditPrimitive` struct (type, transform, params, operation) shared by CPU/GPU.
- [ ] Modify brush tools to emit primitives instead of rasterizing voxels.
- [ ] Maintain a per-brick primitive list (CPU vectors mirrored in GPU SSBOs) so bricks remain spatial buckets for edits.
- [ ] Update BrickStore/Region uploads to include primitive metadata.

## 4. Evaluate Edits During Ray Marching
- [ ] Bind per-brick primitive lists in `traverse_bricks.comp`.
- [ ] Combine base analytic SDF with primitives using CSG ops (smooth min/max) when rays visit that brick.
- [ ] Keep loops small (limit primitives per brick or bucket by macro tile) to maintain perf.
- [ ] Provide shader helpers for common primitives (sphere, capsule, box, wedge, sweep).

## 5. Physics / Gameplay Integration
- [ ] Replace TSDF-based collision queries with analytic sampling of base field + primitives.
- [ ] Update sculpt/brush previews to visualize primitives directly.
- [ ] Ensure save/load serializes per-brick primitive data efficiently.

## 6. Remove Legacy TSDF Pipeline (Cleanup)
- [ ] Delete TSDF buffers, jump-flood updates, `fieldSamples` arrays, and shader interpolation paths once primitives fully replace them.
- [ ] Simplify BrickStore headers/offsets accordingly (still chunked, just lighter records).
- [ ] Remove any CPU/GPU code that referenced TSDF offsets or field caches.

## 7. Diagnostics & Profiling
- [ ] Extend `dump_brick` to print a brick’s primitive list and analytic sample stats.
- [ ] Instrument traversal shader to log time spent evaluating primitives vs base terrain.
- [ ] Capture perf/visual snapshots before and after each major step.
