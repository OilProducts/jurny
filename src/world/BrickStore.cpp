#include "BrickStore.h"

#include <glm/glm.hpp>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <atomic>
#include <spdlog/spdlog.h>

namespace world {

uint64_t BrickStore::packKey(int bx, int by, int bz) {
    const uint64_t B = 1ull << 20; // bias for signed coords
    return ((uint64_t)(bx + (int)B) << 42) |
           ((uint64_t)(by + (int)B) << 21) |
            (uint64_t)(bz + (int)B);
}

void BrickStore::buildHash(CpuWorld& world) {
    const size_t n = world.headers.size();
    uint32_t cap = 1u;
    while (cap < n * 2u) cap <<= 1u;
    if (cap < 8u) cap = 8u;
    world.hashKeys.assign(cap, 0ull);
    world.hashVals.assign(cap, 0u);

    auto put = [&](uint64_t key, uint32_t val) {
        const uint32_t mask = cap - 1u;
        uint32_t h = static_cast<uint32_t>(((key ^ (key >> 33)) * 0xff51afd7ed558ccdULL) >> 32) & mask;
        for (uint32_t probe = 0; probe < cap; ++probe) {
            uint32_t idx = (h + probe) & mask;
            if (world.hashKeys[idx] == 0ull) {
                world.hashKeys[idx] = key;
                world.hashVals[idx] = val;
                return;
            }
        }
    };

    for (uint32_t i = 0; i < world.headers.size(); ++i) {
        const auto& h = world.headers[i];
        put(packKey(h.bx, h.by, h.bz), i);
    }
    world.hashCapacity = cap;
}

static int divFloor(int a, int b) {
    int q = a / b;
    int r = a - q * b;
    if (((a ^ b) < 0) && r != 0) --q;
    return q;
}

void BrickStore::buildMacroHash(CpuWorld& world, uint32_t macroDimBricks) {
    world.macroDimBricks = macroDimBricks;
    std::vector<uint64_t> unique;
    unique.reserve(world.headers.size());
    for (const auto& h : world.headers) {
        int mx = divFloor(h.bx, static_cast<int>(macroDimBricks));
        int my = divFloor(h.by, static_cast<int>(macroDimBricks));
        int mz = divFloor(h.bz, static_cast<int>(macroDimBricks));
        unique.push_back(packKey(mx, my, mz));
    }
    std::sort(unique.begin(), unique.end());
    unique.erase(std::unique(unique.begin(), unique.end()), unique.end());
    uint32_t cap = 1u;
    while (cap < unique.size() * 2u) cap <<= 1u;
    if (cap < 8u) cap = 8u;
    world.macroKeys.assign(cap, 0ull);
    world.macroVals.assign(cap, 0u);
    auto put = [&](uint64_t key, uint32_t val) {
        const uint32_t mask = cap - 1u;
        uint64_t kx = key ^ (key >> 33);
        const uint64_t A = 0xff51afd7ed558ccdULL;
        uint32_t h = static_cast<uint32_t>((kx * A) >> 32) & mask;
        for (uint32_t probe = 0; probe < cap; ++probe) {
            uint32_t idx = (h + probe) & mask;
            if (world.macroKeys[idx] == 0ull) {
                world.macroKeys[idx] = key;
                world.macroVals[idx] = val;
                return;
            }
        }
    };
    for (auto key : unique) put(key, 1u);
    world.macroCapacity = cap;
}

void BrickStore::configure(const math::PlanetParams& P, float voxelSize, int brickDim,
                           const WorldGen::NoiseParams& noise, std::uint32_t seed) {
    params_ = P;
    voxelSize_ = voxelSize;
    brickDim_ = brickDim;
    brickSize_ = voxelSize * static_cast<float>(brickDim_);
    initMaterialTable();
    worldGen_.configure(P, noise, seed);
    {
        std::lock_guard<std::mutex> lock(cacheMutex_);
        brickCache_.clear();
    }
}

void BrickStore::initMaterialTable() {
    materialTable_.clear();
    auto makeMaterial = [](glm::vec3 baseColor, float roughness, float metalness, float emission = 0.0f) {
        MaterialGpu m{};
        m.baseColor[0] = baseColor.x;
        m.baseColor[1] = baseColor.y;
        m.baseColor[2] = baseColor.z;
        m.roughness = roughness;
        m.emission = emission;
        m.metalness = metalness;
        m.pad[0] = 0.0f;
        m.pad[1] = 0.0f;
        return m;
    };
    materialTable_.push_back(makeMaterial(glm::vec3(0.36f, 0.34f, 0.33f), 0.85f, 0.0f)); // 0 rock
    materialTable_.push_back(makeMaterial(glm::vec3(0.43f, 0.28f, 0.16f), 0.75f, 0.0f)); // 1 dirt
    materialTable_.push_back(makeMaterial(glm::vec3(0.18f, 0.45f, 0.17f), 0.52f, 0.0f)); // 2 grass
    materialTable_.push_back(makeMaterial(glm::vec3(0.92f, 0.92f, 0.96f), 0.20f, 0.0f)); // 3 snow
    materialTable_.push_back(makeMaterial(glm::vec3(0.76f, 0.68f, 0.45f), 0.55f, 0.0f)); // 4 sand
    materialTable_.push_back(makeMaterial(glm::vec3(0.20f, 0.22f, 0.25f), 0.35f, 0.0f)); // 5 basalt cliff
}

uint32_t BrickStore::classifyMaterial(const glm::vec3& p) const {
    const WorldGen::BiomeSample biome = worldGen_.biomeSample(p);
    const float height = biome.height;
    const float moisture = biome.moisture;
    const float temperature = biome.temperature;
    glm::vec3 up = glm::normalize(p);
    glm::vec3 normal = worldGen_.crustNormal(p, voxelSize_ * 0.75f);
    float slope = 1.0f - glm::clamp(std::abs(glm::dot(normal, up)), 0.0f, 1.0f);

    if (height > 12.0f || temperature < 0.15f) {
        return 3u; // snow / ice
    }

    if (slope > 0.6f) {
        return height > 0.0f ? 5u : 0u; // cliffs vs subterranean rock
    }

    if (height > 4.0f) {
        return (moisture > 0.35f) ? 2u : 1u; // alpine grass vs dry soil
    }

    if (height > 1.5f) {
        if (moisture > 0.55f && temperature > 0.25f) return 2u;
        return 1u;
    }

    if (height > -1.0f) {
        return (moisture > 0.35f) ? 4u : 1u; // sandier beaches near sea level
    }

    return 0u;
}

bool BrickStore::buildBrickOccupancy(const glm::ivec3& bc,
                                     const math::PlanetParams& P,
                                     float voxelSize,
                                     int brickDim,
                                     float brickSize,
                                     std::vector<uint64_t>& outOcc,
                                     std::vector<uint16_t>& outMaterials) const {
    static std::atomic<uint32_t> logged{0};
    uint32_t count = logged.load(std::memory_order_relaxed);
    if (count < 32) {
        if (logged.compare_exchange_strong(count, count + 1, std::memory_order_relaxed)) {
            spdlog::info("BrickStore::buildBrickOccupancy begin bc=({}, {}, {}) voxelSize={} brickSize={}",
                         bc.x, bc.y, bc.z, voxelSize, brickSize);
        }
    }

    const size_t voxelsPerBrick = static_cast<size_t>(brickDim) * static_cast<size_t>(brickDim) * static_cast<size_t>(brickDim);
    const size_t wordsPerBrick = (voxelsPerBrick + 63u) / 64u;

    const uint64_t key = packKey(bc.x, bc.y, bc.z);
    std::lock_guard<std::mutex> lock(cacheMutex_);
    auto it = brickCache_.find(key);
    if (it != brickCache_.end()) {
        const CachedBrick& cached = it->second;
        if (!cached.solid) {
            outOcc.clear();
            outMaterials.clear();
            return false;
        }
        outOcc = cached.occ;
        outMaterials = cached.materials;
        return true;
    }

    // Build brick occupancy while holding the cache lock. This avoids rehash
    // races with other threads; the streaming system currently tolerates the
    // reduced parallelism better than a crash.
    std::vector<uint64_t> occ(wordsPerBrick, 0ull);
    std::vector<uint16_t> mats(voxelsPerBrick, 0u);
    bool any = false;
    const glm::vec3 brickOrigin = glm::vec3(bc) * brickSize;
    for (int vz = 0; vz < brickDim; ++vz) {
        for (int vy = 0; vy < brickDim; ++vy) {
            for (int vx = 0; vx < brickDim; ++vx) {
                glm::vec3 p = brickOrigin + (glm::vec3(vx + 0.5f, vy + 0.5f, vz + 0.5f) * voxelSize);
                float f = worldGen_.crustField(p);
                uint32_t idx = static_cast<uint32_t>(vx + vy * brickDim + vz * brickDim * brickDim);
                if (f < 0.0f) {
                    uint32_t w = idx >> 6u;
                    uint32_t b = idx & 63u;
                    occ[w] |= (1ull << b);
                    any = true;
                }
                mats[idx] = static_cast<uint16_t>(classifyMaterial(p));
            }
        }
    }

    if (!any) {
        uint32_t cnt = logged.load(std::memory_order_relaxed);
        if (cnt < 64) {
            if (logged.compare_exchange_strong(cnt, cnt + 1, std::memory_order_relaxed)) {
                spdlog::info("BrickStore::buildBrickOccupancy empty bc=({}, {}, {})", bc.x, bc.y, bc.z);
            }
        }
        outOcc.clear();
        outMaterials.clear();
        brickCache_[key] = CachedBrick{false, {}, {}};
        return false;
    }

    outOcc = occ;
    outMaterials = mats;

    brickCache_[key] = CachedBrick{true, std::move(occ), std::move(mats)};

    return true;
}

CpuWorld BrickStore::buildCpuWorld(const std::vector<glm::ivec3>& brickCoords,
                                   std::atomic<bool>* cancel) const {
    CpuWorld world{};
    const size_t brickCount = brickCoords.size();
    const size_t voxelsPerBrick = static_cast<size_t>(brickDim_) * static_cast<size_t>(brickDim_) * static_cast<size_t>(brickDim_);
    const size_t wordsPerBrick = (voxelsPerBrick + 63u) / 64u;
    const size_t maxMaterialWords = (voxelsPerBrick + 3u) / 4u;
    world.headers.reserve(brickCount);
    world.occWords.reserve(brickCount * wordsPerBrick);
    world.materialIndices.reserve(brickCount * maxMaterialWords);
    world.palettes.reserve(brickCoords.size() * 16);

    uint32_t occByteOffset = 0;
    bool aborted = false;
    for (const glm::ivec3& bc : brickCoords) {
        if (cancel && cancel->load(std::memory_order_relaxed)) {
            aborted = true;
            break;
        }
        std::vector<uint64_t> occ;
        std::vector<uint16_t> materials;
        if (!buildBrickOccupancy(bc, params_, voxelSize_, brickDim_, brickSize_, occ, materials)) {
            continue;
        }

        const size_t voxelCount = materials.size();
        std::array<uint16_t, 16> palette{};
        std::vector<uint8_t> paletteIndices(voxelCount);
        uint16_t paletteCount = 0;
        bool use4bit = true;
        for (size_t i = 0; i < voxelCount && use4bit; ++i) {
            const uint16_t mat = materials[i];
            int found = -1;
            for (uint16_t j = 0; j < paletteCount; ++j) {
                if (palette[j] == mat) {
                    found = static_cast<int>(j);
                    break;
                }
            }
            if (found < 0) {
                if (paletteCount < 16) {
                    palette[paletteCount] = mat;
                    found = static_cast<int>(paletteCount);
                    ++paletteCount;
                } else {
                    use4bit = false;
                    break;
                }
            }
            paletteIndices[i] = static_cast<uint8_t>(found);
        }

        BrickHeader header{};
        header.bx = bc.x;
        header.by = bc.y;
        header.bz = bc.z;
        header.occOffset = occByteOffset;
        header.flags = 0u;
        header.paletteCount = 0u;
        header.paletteOffset = kInvalidOffset;
        header.tsdfOffset = kInvalidOffset;

        const uint32_t materialOffsetBytes = static_cast<uint32_t>(world.materialIndices.size() * sizeof(uint32_t));
        if (use4bit && paletteCount > 0) {
            header.flags |= kBrickUses4Bit;
            header.paletteCount = paletteCount;
            const uint32_t paletteOffsetBytes = static_cast<uint32_t>(world.palettes.size() * sizeof(uint32_t));
            header.paletteOffset = paletteOffsetBytes;
            for (uint16_t j = 0; j < paletteCount; ++j) {
                world.palettes.push_back(static_cast<uint32_t>(palette[j]));
            }
            const size_t wordCount = (voxelCount + 7u) / 8u;
            world.materialIndices.reserve(world.materialIndices.size() + wordCount);
            for (size_t w = 0; w < wordCount; ++w) {
                uint32_t packed = 0u;
                for (uint32_t nib = 0; nib < 8u; ++nib) {
                    size_t idx = w * 8u + nib;
                    if (idx >= voxelCount) break;
                    packed |= (uint32_t(paletteIndices[idx]) & 0xFu) << (4u * nib);
                }
                world.materialIndices.push_back(packed);
            }
        } else {
            const size_t wordCount = (voxelCount + 3u) / 4u;
            world.materialIndices.reserve(world.materialIndices.size() + wordCount);
            for (size_t w = 0; w < wordCount; ++w) {
                uint32_t packed = 0u;
                for (uint32_t byteIdx = 0; byteIdx < 4u; ++byteIdx) {
                    size_t idx = w * 4u + byteIdx;
                    if (idx >= voxelCount) break;
                    uint16_t mat = materials[idx];
                    uint32_t value = static_cast<uint32_t>(std::min<uint16_t>(mat, 0xFFu));
                    packed |= value << (8u * byteIdx);
                }
                world.materialIndices.push_back(packed);
            }
        }
        header.matIdxOffset = materialOffsetBytes;

        world.headers.push_back(header);
        world.occWords.insert(world.occWords.end(), occ.begin(), occ.end());
        occByteOffset += static_cast<uint32_t>(occ.size() * sizeof(uint64_t));
    }

    if (aborted) {
        return world;
    }

    buildHash(world);
    buildMacroHash(world, /*macroDimBricks*/8);
    world.materialTable = materialTable_;
    return world;
}

} // namespace world
