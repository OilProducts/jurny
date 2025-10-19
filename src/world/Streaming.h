#pragma once

#include <cstdint>
#include <vector>

#include <glm/glm.hpp>

#include <glm/vec3.hpp>

#include "world/BrickStore.h"
#include "world/RegionCache.h"

namespace core { class Jobs; }

// Streaming — maintains resident brick coords around the camera.
namespace world {

class Streaming {
public:
    struct Config {
        float shellInner = 0.0f;     // minimum radius from planet center
        float shellOuter = 0.0f;     // maximum radius from planet center
        float keepRadius = 60.0f;    // always keep within this radius of the camera
        float loadRadius = 90.0f;    // bricks within this distance are requested
        float simRadius  = 55.0f;    // bricks within this distance are flagged for simulation
        int   regionDimBricks = 32;  // bricks per region cube
    };

    struct Stats {
        uint32_t keepCount = 0;
        uint32_t loadCount = 0;
        uint32_t dropCount = 0;
        uint32_t simCount  = 0;
    };

    struct SelectionDiff {
        std::vector<glm::ivec3> target;
        std::vector<glm::ivec3> toUpload;
        std::vector<glm::ivec3> toRemove;
    };

    void initialize(const BrickStore& store, const Config& config, core::Jobs* jobs = nullptr);
    void update(const glm::vec3& cameraPos, uint64_t frameIndex);

    const Stats& stats() const { return stats_; }
    const glm::vec3& lastCameraOrigin() const { return lastCameraWorld_; }

    SelectionDiff collectSelectionDiff(const std::vector<glm::ivec3>& currentSelection) const;

private:
    const BrickStore* store_ = nullptr;
    Config config_{};
    core::Jobs* jobs_ = nullptr;
    Stats stats_{};
    glm::vec3 lastCameraWorld_{0.0f};
    std::vector<glm::ivec3> lastTarget_;
    RegionCache regionCache_;
};

}
