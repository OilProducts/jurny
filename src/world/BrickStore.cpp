#include "BrickStore.h"

#include <glm/glm.hpp>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <atomic>
#include <limits>
#include <spdlog/spdlog.h>
#include <sstream>
#include <cctype>

#include "core/Assets.h"
#include "world/CoordUtils.h"

namespace world {

uint64_t BrickStore::packKey(int bx, int by, int bz) {
    return coord::packSignedCoord(bx, by, bz);
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

void BrickStore::buildMacroHash(CpuWorld& world, uint32_t macroDimBricks) {
    world.macroDimBricks = macroDimBricks;
    std::vector<uint64_t> unique;
    unique.reserve(world.headers.size());
    for (const auto& h : world.headers) {
        int mx = math::divFloor(h.bx, static_cast<int>(macroDimBricks));
        int my = math::divFloor(h.by, static_cast<int>(macroDimBricks));
        int mz = math::divFloor(h.bz, static_cast<int>(macroDimBricks));
        unique.push_back(coord::packSignedCoord(mx, my, mz));
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

void BrickStore::configure(const ::math::PlanetParams& P, float voxelSize, int brickDim,
                           const WorldGen::NoiseParams& noise, std::uint32_t seed,
                           const core::AssetRegistry* assets) {
    params_ = P;
    voxelSize_ = voxelSize;
    brickDim_ = brickDim;
    brickSize_ = voxelSize * static_cast<float>(brickDim_);
    initMaterialTable(assets);
    worldGen_.configure(P, noise, seed);
    {
        std::lock_guard<std::mutex> lock(cacheMutex_);
        brickCache_.clear();
    }
}

namespace {
MaterialGpu makeMaterial(glm::vec3 baseColor, float roughness, float metalness, float emission) {
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
}

bool parseArray3(std::string_view obj, std::string_view key, glm::vec3& out) {
    size_t pos = obj.find(key);
    if (pos == std::string_view::npos) return false;
    pos = obj.find('[', pos);
    if (pos == std::string_view::npos) return false;
    size_t end = obj.find(']', pos);
    if (end == std::string_view::npos) return false;
    std::string values(obj.substr(pos + 1, end - pos - 1));
    std::stringstream ss(values);
    float v0 = out.x, v1 = out.y, v2 = out.z;
    char comma = 0;
    if (!(ss >> v0)) return false;
    ss >> comma;
    if (!(ss >> v1)) return false;
    ss >> comma;
    if (!(ss >> v2)) return false;
    out = glm::vec3(v0, v1, v2);
    return true;
}

bool parseFloat(std::string_view obj, std::string_view key, float& out) {
    size_t pos = obj.find(key);
    if (pos == std::string_view::npos) return false;
    pos = obj.find(':', pos);
    if (pos == std::string_view::npos) return false;
    pos = obj.find_first_of("-0123456789", pos);
    if (pos == std::string_view::npos) return false;
    size_t end = pos;
    while (end < obj.size()) {
        char c = obj[end];
        if (!(std::isdigit(static_cast<unsigned char>(c)) || c == '.' || c == 'e' || c == 'E' || c == '+' || c == '-')) {
            break;
        }
        ++end;
    }
    try {
        out = std::stof(std::string(obj.substr(pos, end - pos)));
        return true;
    } catch (...) {
        return false;
    }
}

bool loadMaterialsFromJson(std::string_view json, std::vector<MaterialGpu>& out) {
    size_t arrayPos = json.find("\"materials\"");
    if (arrayPos == std::string_view::npos) return false;
    arrayPos = json.find('[', arrayPos);
    if (arrayPos == std::string_view::npos) return false;
    size_t pos = arrayPos;
    std::vector<MaterialGpu> materials;
    while (true) {
        size_t objStart = json.find('{', pos);
        if (objStart == std::string_view::npos) break;
        size_t objEnd = json.find('}', objStart);
        if (objEnd == std::string_view::npos) break;
        std::string_view obj = json.substr(objStart, objEnd - objStart + 1);
        glm::vec3 baseColor(0.8f);
        float emission = 0.0f;
        float roughness = 0.6f;
        float metalness = 0.0f;
        parseArray3(obj, "\"baseColor\"", baseColor);
        parseFloat(obj, "\"emission\"", emission);
        parseFloat(obj, "\"roughness\"", roughness);
        parseFloat(obj, "\"metalness\"", metalness);
        materials.push_back(makeMaterial(baseColor, roughness, metalness, emission));
        pos = objEnd + 1;
    }
    if (!materials.empty()) {
        out = std::move(materials);
        return true;
    }
    return false;
}
} // namespace

void BrickStore::initMaterialTable(const core::AssetRegistry* assets) {
    materialTable_.clear();
    if (assets) {
        std::string json;
        if (assets->loadText("materials.json", json)) {
            if (loadMaterialsFromJson(json, materialTable_)) {
                spdlog::info("Loaded {} materials from assets", materialTable_.size());
                return;
            }
            spdlog::warn("Failed to parse materials.json; falling back to defaults");
        } else {
            spdlog::warn("materials.json not found in asset pack");
        }
    }
    auto addDefault = [&](glm::vec3 baseColor, float roughness, float metalness, float emission = 0.0f) {
        materialTable_.push_back(makeMaterial(baseColor, roughness, metalness, emission));
    };
    addDefault(glm::vec3(0.36f, 0.34f, 0.33f), 0.85f, 0.0f); // rock
    addDefault(glm::vec3(0.43f, 0.28f, 0.16f), 0.75f, 0.0f); // dirt
    addDefault(glm::vec3(0.18f, 0.45f, 0.17f), 0.52f, 0.0f); // grass
    addDefault(glm::vec3(0.92f, 0.92f, 0.96f), 0.20f, 0.0f); // snow
    addDefault(glm::vec3(0.76f, 0.68f, 0.45f), 0.55f, 0.0f); // sand
    addDefault(glm::vec3(0.20f, 0.22f, 0.25f), 0.35f, 0.0f); // basalt cliff
    addDefault(glm::vec3(1.00f, 0.32f, 0.05f), 0.15f, 0.0f, 4.0f); // lava (emissive)
}

uint32_t BrickStore::classifyMaterial(const glm::vec3& p) const {
    const WorldGen::BiomeSample biome = worldGen_.biomeSample(p);
    const float height = biome.height;
    const float moisture = biome.moisture;
    const float temperature = biome.temperature;
    glm::vec3 up = glm::normalize(p);
    glm::vec3 normal = worldGen_.crustNormal(p, voxelSize_ * 0.75f);
    float slope = 1.0f - glm::clamp(std::abs(glm::dot(normal, up)), 0.0f, 1.0f);

    // IDs (match defaults):
    // 0: rock, 1: dirt, 2: grass, 3: snow/ice, 4: sand, 5: basalt cliff, 6: lava (emissive)

    // Deep underground or below sea level: occasional lava pockets
    if (height < -4.0f) {
        float lavaNoise = std::sin(p.x * 0.15f + p.y * 0.07f + p.z * 0.21f);
        if (lavaNoise > 0.85f) {
            return 6u;
        }
        return 0u;
    }

    if (height > 22.0f || temperature < 0.08f) {
        return 3u;
    }

    if (slope > 0.65f) {
        return height > 0.0f ? 5u : 0u;
    }

    if (height > 10.0f) {
        return (moisture > 0.35f) ? 2u : 1u;
    }

    if (height > 3.5f) {
        if (moisture > 0.45f && temperature > 0.25f) return 2u;
        return 1u;
    }

    if (height > -1.0f) {
        if (moisture > 0.45f && temperature > 0.3f) return 2u;
        if (moisture > 0.25f) return 4u;
        return 1u;
    }

    if (height > -3.0f) {
        return 4u;
    }

    return 0u;
}

bool BrickStore::computeBrickData(const glm::ivec3& bc,
                                  const ::math::PlanetParams& P,
                                  float voxelSize,
                                  int brickDim,
                                  float brickSize,
                                  std::vector<uint64_t>& outOcc,
                                  std::vector<uint16_t>& outMaterials) const {
    (void)P;
    static std::atomic<uint32_t> firstLogs{0};
    uint32_t count = firstLogs.load(std::memory_order_relaxed);
    if (count < 32) {
        if (firstLogs.compare_exchange_strong(count, count + 1, std::memory_order_relaxed)) {
            spdlog::info("BrickStore::computeBrickData begin bc=({}, {}, {}) voxelSize={} brickSize={} shell=[{}, {}]",
                         bc.x, bc.y, bc.z, voxelSize, brickSize,
                         static_cast<float>(params_.R) - static_cast<float>(params_.T),
                         static_cast<float>(params_.R) + static_cast<float>(params_.Hmax));
        }
    }

    const size_t voxelsPerBrick = static_cast<size_t>(brickDim) * static_cast<size_t>(brickDim) * static_cast<size_t>(brickDim);
    const size_t wordsPerBrick = (voxelsPerBrick + 63u) / 64u;

    outOcc.assign(wordsPerBrick, 0ull);
    outMaterials.assign(voxelsPerBrick, 0u);

    const glm::vec3 brickOrigin = glm::vec3(bc) * brickSize;
    auto sampleField = [&](const glm::vec3& pos) -> float {
        return worldGen_.crustField(pos);
    };
    float minFieldSample = std::numeric_limits<float>::infinity();
    float maxFieldSample = -std::numeric_limits<float>::infinity();

    // Fast reject: sample brick corners and center. If all comfortably outside, skip voxel walk.
    bool allPositive = true;
    float minField = std::numeric_limits<float>::infinity();
    const float pad = voxelSize * 0.25f;
    const float step = static_cast<float>(brickDim) * 0.5f;
    for (int iz = 0; iz < 3 && allPositive; ++iz) {
        for (int iy = 0; iy < 3 && allPositive; ++iy) {
            for (int ix = 0; ix < 3; ++ix) {
                glm::vec3 offset(
                    (ix == 0) ? 0.0f : (ix == 1 ? step : static_cast<float>(brickDim)),
                    (iy == 0) ? 0.0f : (iy == 1 ? step : static_cast<float>(brickDim)),
                    (iz == 0) ? 0.0f : (iz == 1 ? step : static_cast<float>(brickDim)));
                glm::vec3 samplePos = brickOrigin + offset * voxelSize;
                float field = sampleField(samplePos);
                minField = std::min(minField, field);
                if (field <= 0.0f) {
                    allPositive = false;
                    break;
                }
            }
        }
    }
    if (allPositive && minField > pad) {
        const glm::vec3 brickCenter = brickOrigin + glm::vec3(0.5f * brickDim) * voxelSize;
        const float radius = glm::length(brickCenter);
        const float halfDiag = 0.5f * std::sqrt(3.0f) * brickSize;
        const float minRadius = radius - halfDiag;
        const float maxRadius = radius + halfDiag;

        const float crustInner = static_cast<float>(params_.R) - static_cast<float>(params_.T);
        const float crustOuter = static_cast<float>(params_.R) + static_cast<float>(params_.Hmax);
        const float margin = halfDiag + voxelSize;
        const float shellInner = crustInner - margin;
        const float shellOuter = crustOuter + margin;

        bool intersectsCrust = !(maxRadius < shellInner || minRadius > shellOuter);
        if (!intersectsCrust) {
            outOcc.clear();
            outMaterials.clear();
            return false;
        }
    }

    const int fieldApron = kFieldApron;
    const int fieldResolution = kFieldResolution;
    const int samplesPerAxis =
        brickDim * fieldResolution + 1 +
        2 * fieldApron * fieldResolution;
    const size_t samplesPerBrick = static_cast<size_t>(samplesPerAxis) *
                                   static_cast<size_t>(samplesPerAxis) *
                                   static_cast<size_t>(samplesPerAxis);
    std::vector<float> fieldSamplesLocal(samplesPerBrick);
    const glm::vec3 fieldOrigin = brickOrigin - glm::vec3(fieldApron) * voxelSize;
    const float sampleStep = voxelSize / static_cast<float>(fieldResolution);

    size_t tsdfIndex = 0;
    for (int iz = 0; iz < samplesPerAxis; ++iz) {
        for (int iy = 0; iy < samplesPerAxis; ++iy) {
            for (int ix = 0; ix < samplesPerAxis; ++ix, ++tsdfIndex) {
                glm::vec3 vertexPos = fieldOrigin + glm::vec3(ix, iy, iz) * sampleStep;
                float f = sampleField(vertexPos);
                minFieldSample = std::min(minFieldSample, f);
                maxFieldSample = std::max(maxFieldSample, f);
                fieldSamplesLocal[tsdfIndex] = f;
            }
        }
    }

    auto sampleGrid = [&](int ix, int iy, int iz) -> float {
        ix = std::clamp(ix, 0, samplesPerAxis - 1);
        iy = std::clamp(iy, 0, samplesPerAxis - 1);
        iz = std::clamp(iz, 0, samplesPerAxis - 1);
        size_t idx = (static_cast<size_t>(iz) * samplesPerAxis + static_cast<size_t>(iy)) * samplesPerAxis + static_cast<size_t>(ix);
        return fieldSamplesLocal[idx];
    };

    auto sampleTrilinear = [&](float x, float y, float z) -> float {
        int ix = static_cast<int>(glm::floor(x));
        int iy = static_cast<int>(glm::floor(y));
        int iz = static_cast<int>(glm::floor(z));
        float fx = x - static_cast<float>(ix);
        float fy = y - static_cast<float>(iy);
        float fz = z - static_cast<float>(iz);
        ix = std::clamp(ix, 0, samplesPerAxis - 2);
        iy = std::clamp(iy, 0, samplesPerAxis - 2);
        iz = std::clamp(iz, 0, samplesPerAxis - 2);

        float c000 = sampleGrid(ix,     iy,     iz);
        float c100 = sampleGrid(ix + 1, iy,     iz);
        float c010 = sampleGrid(ix,     iy + 1, iz);
        float c110 = sampleGrid(ix + 1, iy + 1, iz);
        float c001 = sampleGrid(ix,     iy,     iz + 1);
        float c101 = sampleGrid(ix + 1, iy,     iz + 1);
        float c011 = sampleGrid(ix,     iy + 1, iz + 1);
        float c111 = sampleGrid(ix + 1, iy + 1, iz + 1);

        float c00 = glm::mix(c000, c100, fx);
        float c01 = glm::mix(c001, c101, fx);
        float c10 = glm::mix(c010, c110, fx);
        float c11 = glm::mix(c011, c111, fx);
        float c0 = glm::mix(c00, c10, fy);
        float c1 = glm::mix(c01, c11, fy);
        return glm::mix(c0, c1, fz);
    };

    std::vector<float> centerFieldSamples(voxelsPerBrick, 0.0f);
    bool any = false;
    const glm::vec3 voxelSteps = glm::vec3(voxelSize);
    const float samplesPerVoxel = static_cast<float>(fieldResolution);
    const int sampleApronOffset = fieldApron * fieldResolution;

    for (int vz = 0; vz < brickDim; ++vz) {
        const int vzSampleBase = sampleApronOffset + vz * fieldResolution;
        for (int vy = 0; vy < brickDim; ++vy) {
            const int vySampleBase = sampleApronOffset + vy * fieldResolution;
            for (int vx = 0; vx < brickDim; ++vx) {
                const int vxSampleBase = sampleApronOffset + vx * fieldResolution;
                float relX = static_cast<float>(vxSampleBase) + 0.5f * samplesPerVoxel;
                float relY = static_cast<float>(vySampleBase) + 0.5f * samplesPerVoxel;
                float relZ = static_cast<float>(vzSampleBase) + 0.5f * samplesPerVoxel;
                float voxelField = sampleTrilinear(relX, relY, relZ);
                minFieldSample = std::min(minFieldSample, voxelField);
                maxFieldSample = std::max(maxFieldSample, voxelField);

                uint32_t idx = static_cast<uint32_t>(vx + vy * brickDim + vz * brickDim * brickDim);
                centerFieldSamples[idx] = voxelField;

                bool occupied = (voxelField < 0.0f);
                if (!occupied) {
                    for (int corner = 0; corner < 8; ++corner) {
                        int cx = vxSampleBase + ((corner & 1) ? fieldResolution : 0);
                        int cy = vySampleBase + ((corner & 2) ? fieldResolution : 0);
                        int cz = vzSampleBase + ((corner & 4) ? fieldResolution : 0);
                        float cornerField = sampleGrid(cx, cy, cz);
                        minFieldSample = std::min(minFieldSample, cornerField);
                        maxFieldSample = std::max(maxFieldSample, cornerField);
                        if (cornerField < 0.0f) {
                            occupied = true;
                            break;
                        }
                    }
                }

                if (occupied) {
                    uint32_t w = idx >> 6u;
                    uint32_t b = idx & 63u;
                    outOcc[w] |= (1ull << b);
                    any = true;
                }
            }
        }
    }

    if (!any) {
        uint32_t cnt = firstLogs.load(std::memory_order_relaxed);
        if (cnt < 64) {
            if (firstLogs.compare_exchange_strong(cnt, cnt + 1, std::memory_order_relaxed)) {
                spdlog::info("BrickStore::computeBrickData empty bc=({}, {}, {}) minF={} maxF={}",
                             bc.x, bc.y, bc.z, minFieldSample, maxFieldSample);
            }
        }
        outOcc.clear();
        outMaterials.clear();
        return false;
    }

    constexpr float kInteriorBias = 0.75f;
    const float interiorThreshold = -voxelSize * kInteriorBias;
    for (int vz = 0; vz < brickDim; ++vz) {
        for (int vy = 0; vy < brickDim; ++vy) {
            for (int vx = 0; vx < brickDim; ++vx) {
                glm::vec3 voxelMin = brickOrigin + glm::vec3(vx, vy, vz) * voxelSize;
                glm::vec3 centerPos = voxelMin + glm::vec3(0.5f) * voxelSteps;
                uint32_t idx = static_cast<uint32_t>(vx + vy * brickDim + vz * brickDim * brickDim);
                float voxelField = centerFieldSamples[idx];

                uint16_t materialId = 0u; // subterranean rock default
                if (voxelField > interiorThreshold) {
                    materialId = static_cast<uint16_t>(classifyMaterial(centerPos));
                }
                outMaterials[idx] = materialId;
            }
        }
    }

    static std::atomic<uint32_t> loggingBudget{0};
    if (maxFieldSample < -1e-3f) {
        uint32_t idx = loggingBudget.fetch_add(1, std::memory_order_relaxed);
        if (idx < 32) {
            spdlog::debug("BrickStore: fully negative field brick ({}, {}, {}) minF={} maxF={}",
                          bc.x, bc.y, bc.z, minFieldSample, maxFieldSample);
        }
    } else if (minFieldSample > 1e-3f) {
        uint32_t idx = loggingBudget.fetch_add(1, std::memory_order_relaxed);
        if (idx < 32) {
            spdlog::debug("BrickStore: fully positive field brick ({}, {}, {}) minF={} maxF={}",
                          bc.x, bc.y, bc.z, minFieldSample, maxFieldSample);
        }
    } else if (minFieldSample < -1e-3f && maxFieldSample > 1e-3f) {
        uint32_t idx = loggingBudget.fetch_add(1, std::memory_order_relaxed);
        if (idx < 32) {
            spdlog::debug("BrickStore: surface-crossing brick ({}, {}, {}) minF={} maxF={}",
                          bc.x, bc.y, bc.z, minFieldSample, maxFieldSample);
        }
    }

    return true;
}

bool BrickStore::acquireBrick(const glm::ivec3& bc,
                              std::shared_ptr<const BrickPayload>& payload,
                              std::atomic<bool>* cancel) const {
    const uint64_t key = packKey(bc.x, bc.y, bc.z);
    {
        std::lock_guard<std::mutex> lock(cacheMutex_);
        auto it = brickCache_.find(key);
        if (it != brickCache_.end()) {
            payload = it->second;
            return payload != nullptr;
        }
    }

    if (cancel && cancel->load(std::memory_order_relaxed)) {
        return false;
    }

    std::vector<uint64_t> occ;
    std::vector<uint16_t> mats;
    bool solid = computeBrickData(bc, params_, voxelSize_, brickDim_, brickSize_, occ, mats);

    std::shared_ptr<const BrickPayload> newPayload;
    if (solid) {
        auto owned = std::make_shared<BrickPayload>();
        owned->occ = std::move(occ);
        owned->materials = std::move(mats);
        newPayload = std::move(owned);
    }

    {
        std::lock_guard<std::mutex> lock(cacheMutex_);
        auto [it, inserted] = brickCache_.emplace(key, newPayload);
        if (!inserted) {
            payload = it->second;
            return payload != nullptr;
        }
    }

    payload = std::move(newPayload);
    return payload != nullptr;
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

        std::shared_ptr<const BrickPayload> payload;
        if (!acquireBrick(bc, payload, cancel)) {
            continue;
        }

        if (!payload) {
            continue;
        }

        const std::vector<uint64_t>& occ = payload->occ;
        const std::vector<uint16_t>& materials = payload->materials;
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

        header.tsdfOffset = kInvalidOffset;

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
