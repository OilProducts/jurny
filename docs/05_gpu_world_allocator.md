# Incremental GPU Brick Allocator — Design Notes

Version: 0.1 (2025-10-16)  
Owner: render/world team

---

## 1. Motivation

The current `render::Raytracer` rebuilds the entire GPU brick buffers every time a
streaming region is added or removed. For large resident sets this means:

- O(N) CPU re-packing of occupancy/material streams into contiguous arrays.
- Full SSBO uploads (tens of MiB) on every change.
- Descriptor rewrites and temporal history resets, causing visible hitches.

We need an allocator that lets us ingest or evict *only* the bricks that changed,
while keeping descriptor layouts stable.

---

## 2. Goals & Non-goals

**Goals**
- Append new bricks without touching existing GPU data.
- Evict bricks while keeping buffers compact (no fragmentation growth).
- Update hash/macro tables with ≤ O(dirty) work.
- Keep the shader interface identical (`BrickHeader` byte offsets remain valid).
- Preserve temporal history unless we genuinely invalidate surfaces.

**Non-goals (first pass)**
- Per-brick TSDF allocation (handled later).
- Lock-free uploads or multi-frame staging (single-threaded is fine).
- Avoiding *any* rehash; we accept occasional full rebuilds when load factor
  drifts past a threshold.

---

## 3. High-level Approach

### 3.1 Brick pool as a packed array

Maintain a single packed array of bricks (`std::vector<BrickRecord>`). When adding:

- Append a `BrickRecord` (header + metadata) to the end.
- Append occupancy/material/palette payloads to their respective CPU vectors.
- Stage-copy the *new ranges* only.

When removing:

- Swap the victim brick with the last brick in the array.
- Re-write that last brick’s metadata (offsets) and update its hash entry.
- Pop the tail entries from CPU vectors.
- Stage-copy the overwritten ranges for occupancy/material/palette/headers.

This “swap-with-tail” keeps buffers compact and avoids freelists.

### 3.2 Metadata we must track

For each brick we carry a metadata struct (CPU side only):

```cpp
struct BrickRecord {
    world::BrickHeader header;   // GPU-facing header (byte offsets)
    uint32_t occWordOffset;      // index into occWords_ (8 words per brick)
    uint32_t matWordOffset;      // index into materialIndices_ (64 or 128 words)
    uint16_t matWordCount;       // 64 (4-bit) or 128 (8-bit)
    uint32_t paletteOffset;      // index into palette_ (0 if none)
    uint16_t paletteCount;       // header-friendly copy
    uint16_t flags;              // cached BrickFlags (uses4bit, etc.)
};
```

### 3.3 Region bookkeeping

Maintain `RegionResident` records:

```cpp
struct RegionResident {
    glm::ivec3 coord;
    std::vector<uint64_t> brickKeys; // packed (bx,by,bz) for quick iteration
};
```

When a region is added we push a `RegionResident` with the brick keys we inserted.
On eviction we use those keys to locate bricks and remove them.

### 3.4 Hash & macro tables

- Keep a CPU `std::unordered_map<uint64_t, uint32_t>` (`brickKey -> brickIndex`).
- On insertion we probe and write into a CPU vector of key/value pairs.
- We rebuild the GPU hash arrays when *either*:
  - capacity is too small (load factor > 0.5), or
  - more than e.g. 1/8 of entries changed in a frame.

Rebuild still only uploads `hashKeys` + `hashVals` buffers (lightweight compared
to voxel streams).

Macro occupancy: track `macroCounts[macroKey]` (uint32_t count). When a brick is
added/removed, increment/decrement. Rebuild GPU macro hash when counts changed.

---

## 4. Buffer management

### 4.1 Host-side arrays (owned by `GpuBrickPool`)

```
std::vector<BrickRecord> bricks_;
std::vector<uint64_t>    occWords_;       // 8 words per brick, aligned
std::vector<uint32_t>    matWords_;       // variable: 64 (4-bit) or 128 (8-bit)
std::vector<uint32_t>    palette_;        // tightly packed by paletteCount
```

`BrickRecord`’s offsets are maintained in *words*, then converted to bytes when
filling the GPU-visible `BrickHeader`.

### 4.2 GPU buffers

We reuse `Raytracer::BufferResource` but expose range uploads:

- Extend `core::UploadContext` with `uploadBufferRegion(data, bytes, dst, dstOffset)`.
- When we append bricks, we copy:
  - headers: `bytes = newBrickCount * sizeof(BrickHeader)`
  - occ: `bytes = newBrickCount * 8 * sizeof(uint64_t)`
  - mat: `bytes = Σ newBrick.matWordCount * sizeof(uint32_t)`
  - palette: `bytes = Σ newBrick.paletteCount * sizeof(uint32_t)`
- On swap-removal we copy the moved brick’s payload back down to fill the vacated range.

Buffer growth: if a buffer’s capacity is insufficient, we allocate a *larger*
buffer, copy the entire existing content once, then reuse it for future inserts.
Resizing should be rare (geometric growth policy).

---

## 5. Updated APIs

### 5.1 `GpuBrickPool`

New class (lives inside `render::Raytracer` for now) exposing:

```
struct BuildContext {
    const world::CpuWorld& cpu;
    glm::ivec3 regionCoord;
};

bool init(platform::VulkanContext&, core::UploadContext&);
bool addRegion(const BuildContext&, RegionResident&);
bool removeRegion(const RegionResident&);
void destroy(platform::VulkanContext&);

// Data for descriptors / uploads
VkBuffer headersBuffer() const;
VkDeviceSize headersSize() const;
// ... likewise for occ/mat/palette/hash/macro
```

### 5.2 `RenderGlobals`

No shader-side change, but `GlobalsUBOData` must expose:

- `worldBrickCount` (already there) — keep updated.
- `worldHashCapacity` — updated when hash rebuild occurs.
- `macroHashCapacity`, `macroDimBricks` — updated in lockstep.

### 5.3 `UploadContext`

Add:

```cpp
bool uploadBufferRegion(const void* data,
                        VkDeviceSize bytes,
                        VkBuffer dstBuffer,
                        VkDeviceSize dstOffset);
```

Implementation identical to `uploadBuffer`, but `VkBufferCopy`’s `dstOffset`
is parameterised and we avoid resetting staging memory when copying partial data.

---

## 6. Streaming flow changes

Current flow:

```
Streaming ➜ build CpuWorld ➜ Raytracer::addRegion ➜ rebuildGpuWorld (full)
```

New flow:

```
Streaming ➜ build CpuWorld ➜ Raytracer::addRegion ➜
    GpuBrickPool::addRegion (per-brick append) ➜
    maybeRebuildHashMacro() ➜ refreshDescriptors (if capacity changed)
```

Eviction mirrors the process (remove region ➜ swap-remove bricks ➜ maybe rebuild hash/macro).

Telemetry: increment counters so overlays show bricks appended/evicted each frame.

---

## 7. Hash rebuild policy

- Maintain `dirtyBrickCount_` each frame.
- If `dirtyBrickCount_ > totalBricks / 8` *or* the hash load factor would exceed 0.5,
  rebuild the CPU hash arrays and reupload both `hashKeys` and `hashVals`.
- Else, incremental updates only adjust the handful of entries that changed (write
  directly into the CPU key/value vectors and stage-copy ranges).

Simplest first iteration: **always** rebuild the hash when the brick set changes.
The array is small (~2 * brickCount entries) and reuploading ≈ 8 bytes per entry.
We still eliminate the expensive voxel repack, which is the main win.

---

## 8. Temporal history considerations

Because bricks only swap within the pool, existing indices may change (swap removal).
We *must* update:

- CPU `key -> index` map.
- GPU hash value for the swapped brick.
- Any debug buffers that expose the old index.

Temporal buffers (history color/moments) remain valid; no automatic reset needed.
We only reset accumulation when we actually rebuild the entire brick pool (e.g.,
after buffer reallocation that forces GPU descriptors to be rewritten).

---

## 9. Open questions / follow-ups

- **Thread safety:** streaming upload runs on render thread today; safe. If we move
  to worker thread staging later, we need a mutex around `GpuBrickPool`.
- **TSDF channels:** future work—extend `BrickRecord` to include optional TSDF offsets.
- **Persistent staging:** `UploadContext` is synchronous per call. Consider batching
  multiple range copies into a single command buffer submit.
- **Validation tooling:** add debug overlays to draw brick indices so swap-removal
  issues are visible.

---

## 10. Implementation checklist

1. Add `uploadBufferRegion(...)` to `core::UploadContext`.
2. Introduce `GpuBrickPool` (headers in `src/render`).
3. Refactor `Raytracer::addRegion/removeRegion` to call the pool instead of `rebuildGpuWorld`.
4. Update descriptor refresh to handle buffer reallocation without rebuilding everything.
5. Rework streaming telemetry (counts, overlays) to reflect incremental updates.
6. Extend tests:
   - New unit tests for `GpuBrickPool` append/remove/swap path.
   - Hash rebuild consistency (CPU map vs GPU arrays).

---

By delivering steps 1–4 we get rid of the worst hitch (full voxel repack) and
prepare a foundation for TSDF/edit integrations in subsequent milestones.
