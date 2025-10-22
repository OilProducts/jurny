#pragma once

#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

#include <glm/vec3.hpp>

#include "BrickFormats.h"
#include "math/Spherical.h"
#include "world/WorldGen.h"

namespace core { class AssetRegistry; }

// BrickStore — CPU brick generation & packing for a test slab near +X.
// GPU upload is handled by the renderer for M2.
namespace world {

struct CpuWorld {
    std::vector<BrickHeader> headers;
    std::vector<uint64_t>    occWords;   // occupancy words per brick (brickDim-dependent)
    std::vector<uint32_t>    materialIndices; // packed material data (4-bit or 8-bit)
    std::vector<uint32_t>    palettes;        // packed per-brick palettes (uint32 per entry)
    std::vector<MaterialGpu> materialTable;
    std::vector<uint64_t>    hashKeys;   // packed (bx,by,bz)
    std::vector<uint32_t>    hashVals;   // index into headers
    uint32_t                 hashCapacity = 0; // power-of-two
    // Macro mask (non-empty macro tiles)
    std::vector<uint64_t>    macroKeys;  // packed macro coords
    std::vector<uint32_t>    macroVals;  // 1 = non-empty tile
    uint32_t                 macroCapacity = 0;
    uint32_t                 macroDimBricks = 8; // default grouping: 8x8x8 bricks per macro tile
};

class BrickStore {
public:
    void configure(const math::PlanetParams& P, float voxelSize, int brickDim,
                   const WorldGen::NoiseParams& noise, std::uint32_t seed,
                   const core::AssetRegistry* assets = nullptr);

    CpuWorld buildCpuWorld(const std::vector<glm::ivec3>& brickCoords,
                           std::atomic<bool>* cancel = nullptr) const;

    float brickSize() const { return brickSize_; }
    float voxelSize() const { return voxelSize_; }
    int brickDim() const { return brickDim_; }
    const math::PlanetParams& params() const { return params_; }
    const WorldGen& worldGen() const { return worldGen_; }

private:
    struct BrickPayload {
        std::vector<uint64_t> occ;
        std::vector<uint16_t> materials;
    };

    static uint64_t packKey(int bx, int by, int bz);
    static void buildHash(CpuWorld& world);
    static void buildMacroHash(CpuWorld& world, uint32_t macroDimBricks);
    bool computeBrickData(const glm::ivec3& bc,
                          const math::PlanetParams& P,
                          float voxelSize,
                          int brickDim,
                          float brickSize,
                          std::vector<uint64_t>& outOcc,
                          std::vector<uint16_t>& outMaterials) const;
    bool acquireBrick(const glm::ivec3& bc,
                      std::shared_ptr<const BrickPayload>& payload,
                      std::atomic<bool>* cancel) const;

private:
    int brickDim_ = 8;
    float brickSize_ = 4.0f;
    float voxelSize_ = 0.5f;
    math::PlanetParams params_{};
    std::vector<MaterialGpu> materialTable_;
    void initMaterialTable(const core::AssetRegistry* assets);
    uint32_t classifyMaterial(const glm::vec3& p) const;
    mutable std::unordered_map<uint64_t, std::shared_ptr<const BrickPayload>> brickCache_;
    mutable std::mutex cacheMutex_;
    WorldGen worldGen_{};
};

}
