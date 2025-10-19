#include "RegionCache.h"

#include <algorithm>
#include <limits>

#include <glm/geometric.hpp>

namespace world {

namespace {
constexpr std::int64_t kBias = (1ll << 20); // matches brick hash packing (±2^20 support)
}

int RegionCache::floorDiv(int a, int b) {
    int q = a / b;
    int r = a - q * b;
    if (((a ^ b) < 0) && r) --q;
    return q;
}

std::int64_t RegionCache::packCoord(const glm::ivec3& coord) {
    return ((static_cast<std::int64_t>(coord.x) + kBias) << 42) |
           ((static_cast<std::int64_t>(coord.y) + kBias) << 21) |
            (static_cast<std::int64_t>(coord.z) + kBias);
}

void RegionCache::computeBounds(const glm::ivec3& coord, float regionSize, Region& region) {
    const glm::vec3 regionMin = glm::vec3(coord) * regionSize;
    const glm::vec3 extent(regionSize);
    region.center = regionMin + 0.5f * extent;
    region.halfExtent = 0.5f * extent;

    float minRadius = std::numeric_limits<float>::max();
    float maxRadius = 0.0f;
    for (int sx = 0; sx < 2; ++sx) {
        for (int sy = 0; sy < 2; ++sy) {
            for (int sz = 0; sz < 2; ++sz) {
                glm::vec3 corner = regionMin + glm::vec3(sx ? extent.x : 0.0f,
                                                         sy ? extent.y : 0.0f,
                                                         sz ? extent.z : 0.0f);
                float radius = glm::length(corner);
                minRadius = std::min(minRadius, radius);
                maxRadius = std::max(maxRadius, radius);
            }
        }
    }
    region.minRadius = minRadius;
    region.maxRadius = maxRadius;
}

void RegionCache::build(const std::vector<glm::ivec3>& bricks, int regionDimBricks, float brickSizeMeters) {
    regions_.clear();
    indexLookup_.clear();
    regionDimBricks_ = std::max(regionDimBricks, 1);
    regionSizeMeters_ = static_cast<float>(regionDimBricks_) * brickSizeMeters;
    regions_.reserve(bricks.size());

    for (uint32_t i = 0; i < bricks.size(); ++i) {
        const glm::ivec3& bc = bricks[i];
        glm::ivec3 rc{
            floorDiv(bc.x, regionDimBricks_),
            floorDiv(bc.y, regionDimBricks_),
            floorDiv(bc.z, regionDimBricks_)
        };
        std::int64_t key = packCoord(rc);
        auto it = indexLookup_.find(key);
        Region* region = nullptr;
        if (it == indexLookup_.end()) {
            Region newRegion{};
            newRegion.coord = rc;
            computeBounds(rc, regionSizeMeters_, newRegion);
            newRegion.bricks.clear();
            regions_.push_back(newRegion);
            std::size_t idx = regions_.size() - 1;
            indexLookup_[key] = idx;
            region = &regions_.back();
        } else {
            region = &regions_[it->second];
        }
        region->bricks.push_back(i);
        region->brickCount = static_cast<uint32_t>(region->bricks.size());
    }
}

}
