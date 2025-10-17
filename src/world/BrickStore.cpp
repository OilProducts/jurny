#include "BrickStore.h"
#include <glm/glm.hpp>
#include <glm/gtc/constants.hpp>
#include <algorithm>
#include <cmath>

namespace world {

void BrickStore::clear() {
    cpu_ = CpuWorld{};
}

uint64_t BrickStore::packKey(int bx, int by, int bz) {
    const uint64_t B = 1ull << 20; // bias for signed coords
    return ((uint64_t)(bx + (int)B) << 42) | ((uint64_t)(by + (int)B) << 21) | (uint64_t)(bz + (int)B);
}

void BrickStore::buildHash() {
    // Simple linear-probe hash with 0 as empty key.
    size_t n = cpu_.headers.size();
    uint32_t cap = 1u; while (cap < n * 2u) cap <<= 1u; if (cap < 8u) cap = 8u;
    cpu_.hashKeys.assign(cap, 0ull);
    cpu_.hashVals.assign(cap, 0u);
    auto put = [&](uint64_t key, uint32_t val){
        uint32_t mask = cap - 1u;
        uint32_t h = (uint32_t)((key ^ (key >> 33)) * 0xff51afd7ed558ccdULL >> 32) & mask; // simple mix
        for (uint32_t i=0;i<cap;++i) {
            uint32_t idx = (h + i) & mask;
            if (cpu_.hashKeys[idx] == 0ull) { cpu_.hashKeys[idx] = key; cpu_.hashVals[idx] = val; return; }
        }
    };
    for (uint32_t i=0;i<n;++i) {
        auto& h = cpu_.headers[i];
        uint64_t key = packKey(h.bx, h.by, h.bz);
        put(key, i);
    }
    cpu_.hashCapacity = cap;
}

void BrickStore::buildMacroHash(uint32_t macroDimBricks) {
    cpu_.macroDimBricks = macroDimBricks;
    // Collect unique macro coords that contain at least one brick
    std::vector<uint64_t> unique;
    unique.reserve(cpu_.headers.size());
    auto divFloor = [](int a, int b){ int q = a / (int)b; if ((a ^ b) < 0 && (a % (int)b)) --q; return q; };
    for (const auto& h : cpu_.headers) {
        int mx = divFloor(h.bx, (int)macroDimBricks);
        int my = divFloor(h.by, (int)macroDimBricks);
        int mz = divFloor(h.bz, (int)macroDimBricks);
        unique.push_back(packKey(mx,my,mz));
    }
    std::sort(unique.begin(), unique.end());
    unique.erase(std::unique(unique.begin(), unique.end()), unique.end());
    uint32_t n = (uint32_t)unique.size();
    uint32_t cap = 1u; while (cap < n * 2u) cap <<= 1u; if (cap < 8u) cap = 8u;
    cpu_.macroKeys.assign(cap, 0ull);
    cpu_.macroVals.assign(cap, 0u);
    auto put = [&](uint64_t key, uint32_t val){
        uint32_t mask = cap - 1u;
        uint64_t kx = key ^ (key >> 33);
        const uint64_t A = 0xff51afd7ed558ccdULL;
        uint32_t h = (uint32_t)((kx * A) >> 32) & mask;
        for (uint32_t i=0;i<cap;++i) {
            uint32_t idx = (h + i) & mask;
            if (cpu_.macroKeys[idx] == 0ull) { cpu_.macroKeys[idx] = key; cpu_.macroVals[idx] = val; return; }
        }
    };
    for (auto key : unique) put(key, 1u);
    cpu_.macroCapacity = cap;
}

size_t BrickStore::generateTestSlab(const math::PlanetParams& P,
                                    float voxelSize,
                                    int brickDim,
                                    float yExtentMeters,
                                    float zExtentMeters,
                                    float radialHalfThicknessMeters) {
    clear();
    params_ = P; voxelSize_ = voxelSize; brickDim_ = brickDim; brickSize_ = voxelSize * float(brickDim);

    const float R = static_cast<float>(P.R);
    // Build ranges in brick coordinates around +X axis
    int bxMin = int(std::floor((R - radialHalfThicknessMeters) / brickSize_));
    int bxMax = int(std::floor((R + radialHalfThicknessMeters) / brickSize_));
    int byMin = int(std::floor((-yExtentMeters) / brickSize_));
    int byMax = int(std::floor(( yExtentMeters) / brickSize_));
    int bzMin = int(std::floor((-zExtentMeters) / brickSize_));
    int bzMax = int(std::floor(( zExtentMeters) / brickSize_));

    const int wordPerBrick = 8; // 8x uint64_t = 512 bits
    cpu_.headers.clear(); cpu_.occWords.clear();
    cpu_.headers.reserve((bxMax - bxMin + 1) * (byMax - byMin + 1) * (bzMax - bzMin + 1));
    cpu_.occWords.reserve(cpu_.headers.capacity() * wordPerBrick);

    auto F = [&](const glm::vec3& p) -> float {
        float r = glm::length(p);
        return r - R; // negative = solid
    };

    uint32_t occByteOffset = 0;
    for (int bz=bzMin; bz<=bzMax; ++bz) {
        for (int by=byMin; by<=byMax; ++by) {
            for (int bx=bxMin; bx<=bxMax; ++bx) {
                // Fill occupancy mask for this brick
                uint64_t words[8] = {0,0,0,0,0,0,0,0};
                bool any = false;
                glm::vec3 brickOrigin = glm::vec3(float(bx), float(by), float(bz)) * brickSize_;
                for (int vz=0; vz<brickDim_; ++vz) {
                    for (int vy=0; vy<brickDim_; ++vy) {
                        for (int vx=0; vx<brickDim_; ++vx) {
                            glm::vec3 p = brickOrigin + (glm::vec3(vx + 0.5f, vy + 0.5f, vz + 0.5f) * voxelSize_);
                            float f = F(p);
                            bool solid = (f < 0.0f);
                            if (solid) {
                                uint32_t idx = uint32_t(vx + vy*brickDim_ + vz*brickDim_*brickDim_);
                                uint32_t w = idx >> 6; uint32_t b = idx & 63u;
                                words[w] |= (1ull << b);
                                any = true;
                            }
                        }
                    }
                }
                if (!any) continue; // skip empty bricks
                // Write header
                BrickHeader h{}; h.bx=bx; h.by=by; h.bz=bz;
                h.occOffset = occByteOffset; // bytes
                h.matIdxOffset = 0; h.paletteOffset=0; h.flags=0; h.paletteCount=0; h.tsdfOffset=0xFFFFFFFFu;
                cpu_.headers.push_back(h);
                // Append words
                for (int i=0;i<8;++i) cpu_.occWords.push_back(words[i]);
                occByteOffset += wordPerBrick * sizeof(uint64_t);
            }
        }
    }
    buildHash();
    buildMacroHash(/*macroDimBricks*/8);
    return cpu_.headers.size();
}

size_t BrickStore::generateSphericalCap(const math::PlanetParams& P,
                                        float voxelSize,
                                        int brickDim,
                                        float yExtentMeters,
                                        float zExtentMeters,
                                        float radialHalfThicknessMeters) {
    clear();
    params_ = P; voxelSize_ = voxelSize; brickDim_ = brickDim; brickSize_ = voxelSize * float(brickDim);
    const float R = static_cast<float>(P.R);

    int byMin = int(std::floor((-yExtentMeters) / brickSize_));
    int byMax = int(std::floor(( yExtentMeters) / brickSize_));
    int bzMin = int(std::floor((-zExtentMeters) / brickSize_));
    int bzMax = int(std::floor(( zExtentMeters) / brickSize_));
    int radialHalfBricks = std::max(1, int(std::ceil(radialHalfThicknessMeters / brickSize_)));

    const int wordPerBrick = 8;
    cpu_.headers.clear(); cpu_.occWords.clear();
    // Reserve a conservative estimate: tangential bricks * (2*radialHalfBricks+2)
    size_t tangentialCount = size_t(std::max(0, byMax-byMin+1)) * size_t(std::max(0, bzMax-bzMin+1));
    cpu_.headers.reserve(tangentialCount * size_t(2*radialHalfBricks + 2));
    cpu_.occWords.reserve(cpu_.headers.capacity() * wordPerBrick);

    auto F = [&](const glm::vec3& p) -> float { return glm::length(p) - R; };
    uint32_t occByteOffset = 0;

    for (int bz=bzMin; bz<=bzMax; ++bz) {
        for (int by=byMin; by<=byMax; ++by) {
            // World-space (y,z) at this brick row/column (use brick center in y,z)
            float y = (float(by) + 0.5f) * brickSize_;
            float z = (float(bz) + 0.5f) * brickSize_;
            float yz2 = y*y + z*z;
            if (yz2 >= R*R) continue; // outside sphere projection
            float xSurf = std::sqrt(std::max(R*R - yz2, 0.0f));
            int bxCenter = int(std::floor(xSurf / brickSize_));
            int bxMin = bxCenter - radialHalfBricks;
            int bxMax = bxCenter + radialHalfBricks;

            for (int bx=bxMin; bx<=bxMax; ++bx) {
                uint64_t words[8] = {0,0,0,0,0,0,0,0};
                bool any = false;
                glm::vec3 brickOrigin = glm::vec3(float(bx), float(by), float(bz)) * brickSize_;
                for (int vz=0; vz<brickDim_; ++vz) {
                    for (int vy=0; vy<brickDim_; ++vy) {
                        for (int vx=0; vx<brickDim_; ++vx) {
                            glm::vec3 p = brickOrigin + (glm::vec3(vx + 0.5f, vy + 0.5f, vz + 0.5f) * voxelSize_);
                            bool solid = (F(p) < 0.0f);
                            if (solid) {
                                uint32_t idx = uint32_t(vx + vy*brickDim_ + vz*brickDim_*brickDim_);
                                uint32_t w = idx >> 6; uint32_t b = idx & 63u;
                                words[w] |= (1ull << b);
                                any = true;
                            }
                        }
                    }
                }
                if (!any) continue;
                BrickHeader h{}; h.bx=bx; h.by=by; h.bz=bz; h.occOffset = occByteOffset; h.tsdfOffset=0xFFFFFFFFu;
                cpu_.headers.push_back(h);
                for (int i=0;i<8;++i) cpu_.occWords.push_back(words[i]);
                occByteOffset += wordPerBrick * sizeof(uint64_t);
            }
        }
    }
    buildHash();
    buildMacroHash(/*macroDimBricks*/8);
    return cpu_.headers.size();
}

}
