#pragma once

#include <cstdint>
#include <unordered_map>
#include <vector>

#include <glm/glm.hpp>

#include "world/BrickStore.h"

// RegionCache — groups bricks into fixed-size regions for streaming/eviction decisions.
namespace world {

class RegionCache {
public:
    struct Region {
        enum class Residency : uint8_t { Dropped, Load, Keep };

        glm::ivec3 coord{};          // region coordinate in region-space
        glm::vec3  center{};         // world-space center (meters)
        glm::vec3  halfExtent{};     // half-size in meters
        float minRadius = 0.0f;      // min distance to planet center
        float maxRadius = 0.0f;      // max distance to planet center
        float distanceToCamera = 0.0f;
        Residency residency = Residency::Dropped;
        bool withinShell = false;
        bool simActive = false;
        uint32_t brickCount = 0;
        std::vector<uint32_t> bricks; // indices into BrickStore::CpuWorld::headers
    };

    void build(const std::vector<glm::ivec3>& bricks, int regionDimBricks, float brickSizeMeters);
    std::vector<Region>& regions() { return regions_; }
    const std::vector<Region>& regions() const { return regions_; }

    int regionDimBricks() const { return regionDimBricks_; }
    float regionSizeMeters() const { return regionSizeMeters_; }

private:
    static int floorDiv(int a, int b);
    static std::int64_t packCoord(const glm::ivec3& coord);
    static void computeBounds(const glm::ivec3& coord, float regionSize, Region& region);

    int regionDimBricks_ = 64;
    float regionSizeMeters_ = 0.0f;
    std::vector<Region> regions_;
    std::unordered_map<std::int64_t, std::size_t> indexLookup_;
};

}
