#include "Streaming.h"

#include <algorithm>
#include <cmath>
#include <unordered_set>
#include <vector>

#include <glm/common.hpp>
#include <glm/vec3.hpp>
#include <glm/geometric.hpp>
#include <spdlog/spdlog.h>

namespace {
inline bool glmLess(const glm::ivec3& a, const glm::ivec3& b) {
    if (a.x != b.x) return a.x < b.x;
    if (a.y != b.y) return a.y < b.y;
    return a.z < b.z;
}

inline std::int64_t packKey(const glm::ivec3& c) {
    const std::int64_t B = (1ll << 20);
    return ((static_cast<std::int64_t>(c.x) + B) << 42) |
           ((static_cast<std::int64_t>(c.y) + B) << 21) |
            (static_cast<std::int64_t>(c.z) + B);
}
}

namespace world {

void Streaming::initialize(const BrickStore& store, const Config& config, core::Jobs* jobs) {
    store_ = &store;
    config_ = config;
    jobs_ = jobs;
    stats_ = {};
    lastTarget_.clear();
}

void Streaming::update(const glm::vec3& cameraPos, uint64_t /*frameIndex*/) {
    if (!store_) return;

    const bool prevEmpty = lastTarget_.empty();

    lastCameraWorld_ = cameraPos;
    stats_ = {};

    const float brickSize = store_->brickSize();
    const float keepRadius = config_.keepRadius;
    const float loadRadius = config_.loadRadius;
    const float simRadius  = config_.simRadius;
    const float shellInner = (config_.shellInner > 0.0f) ? config_.shellInner : static_cast<float>(store_->params().R) - 5.0f;
    const float shellOuter = (config_.shellOuter > 0.0f) ? config_.shellOuter : static_cast<float>(store_->params().R) + 5.0f;

    const int regionDim = std::max(config_.regionDimBricks, 1);
    const float regionSize = static_cast<float>(regionDim) * brickSize;
    const float regionHalfDiag = 0.5f * regionSize * std::sqrt(3.0f);
    const glm::vec3 cameraRegionF = cameraPos / regionSize;
    glm::ivec3 cameraRegion = glm::ivec3(glm::floor(cameraRegionF));
    const float loadRadiusExpanded = loadRadius + regionHalfDiag;
    const int regionRadius = int(std::ceil(loadRadiusExpanded / regionSize)) + 1;

    glm::vec3 camDir = glm::vec3(0.0f);
    bool hasCamDir = (glm::dot(cameraPos, cameraPos) > 1e-6f);
    if (hasCamDir) camDir = glm::normalize(cameraPos);

    struct Candidate {
        glm::ivec3 coord;
        float priority;
        float distance;
    };

    std::vector<Candidate> candidates;
    const int dimRegions = 2 * regionRadius + 1;
    candidates.reserve(static_cast<size_t>(dimRegions * dimRegions * dimRegions));
    for (int rz = -regionRadius; rz <= regionRadius; ++rz) {
        for (int ry = -regionRadius; ry <= regionRadius; ++ry) {
            for (int rx = -regionRadius; rx <= regionRadius; ++rx) {
                glm::ivec3 rc = cameraRegion + glm::ivec3(rx, ry, rz);
                glm::vec3 regionMin = glm::vec3(rc) * regionSize;
                glm::vec3 regionCenter = regionMin + glm::vec3(regionSize * 0.5f);
                float regionRadiusCenter = glm::length(regionCenter);
                float minRadius = regionRadiusCenter - regionHalfDiag;
                float maxRadius = regionRadiusCenter + regionHalfDiag;
                if (maxRadius < shellInner || minRadius > shellOuter) {
                    continue;
                }
                float distance = glm::length(regionCenter - cameraPos);
                if (distance > loadRadiusExpanded) {
                    continue;
                }
                float angleScore = 1.0f;
                if (hasCamDir && regionRadiusCenter > 1e-3f) {
                    glm::vec3 regionDir = regionCenter / regionRadiusCenter;
                    float cosAngle = glm::clamp(glm::dot(camDir, regionDir), -1.0f, 1.0f);
                    angleScore = 0.5f * (cosAngle + 1.0f);
                }
                float distanceScore = 1.0f - glm::clamp(distance / loadRadiusExpanded, 0.0f, 1.0f);
                float shellCenter = 0.5f * (shellInner + shellOuter);
                float shellRange = std::max(1.0f, shellOuter - shellInner);
                float shellScore = 1.0f - glm::clamp(std::abs(regionRadiusCenter - shellCenter) / shellRange, 0.0f, 1.0f);
                float priority = angleScore * 0.6f + distanceScore * 0.3f + shellScore * 0.1f;
                candidates.push_back({ rc, priority, distance });
            }
        }
    }

    if (candidates.empty()) {
        spdlog::debug("Streaming.update: no region candidates (cameraRadius={:.2f}, loadRadius={}, shell=[{:.2f},{:.2f}], regionDim={})",
                      glm::length(cameraPos), loadRadius, shellInner, shellOuter, regionDim);
    }

    std::sort(candidates.begin(), candidates.end(), [](const Candidate& a, const Candidate& b) {
        if (a.priority != b.priority) return a.priority > b.priority;
        return a.distance < b.distance;
    });

    std::unordered_set<std::int64_t> uniqueBricks;
    std::vector<glm::ivec3> target;
    const float brickHalfDiag = std::sqrt(3.0f) * brickSize * 0.5f;

    for (const Candidate& cand : candidates) {
        glm::ivec3 baseBrick = cand.coord * regionDim;
        for (int z = 0; z < regionDim; ++z) {
            for (int y = 0; y < regionDim; ++y) {
                for (int x = 0; x < regionDim; ++x) {
                    glm::ivec3 bc = baseBrick + glm::ivec3(x, y, z);
                    glm::vec3 centerWorld = (glm::vec3(bc) + glm::vec3(0.5f)) * brickSize;
                    float radial = glm::length(centerWorld);
                    float minRadius = radial - brickHalfDiag;
                    float maxRadius = radial + brickHalfDiag;
                    if (maxRadius < shellInner || minRadius > shellOuter) continue;
                    float distToCamera = glm::length(centerWorld - cameraPos);
                    if (distToCamera - brickHalfDiag > loadRadius) continue;
                    std::int64_t key = packKey(bc);
                    if (!uniqueBricks.insert(key).second) continue;
                    target.push_back(bc);
                }
            }
        }
    }

    std::sort(target.begin(), target.end(), glmLess);

    stats_.loadCount = static_cast<uint32_t>(target.size());
    stats_.dropCount = 0;
    stats_.keepCount = 0;
    stats_.simCount = 0;
    for (const glm::ivec3& bc : target) {
        glm::vec3 centerWorld = (glm::vec3(bc) + glm::vec3(0.5f)) * brickSize;
        float distToCamera = glm::length(centerWorld - cameraPos);
        if (distToCamera - brickHalfDiag <= keepRadius) ++stats_.keepCount;
        if (distToCamera - brickHalfDiag <= simRadius) ++stats_.simCount;
    }

    regionCache_.build(target, regionDim, brickSize);

    if (target.empty()) {
        if (prevEmpty) {
            spdlog::warn("Streaming.update: target still empty (cameraRadius={:.2f}, loadRadius={}, shell=[{:.2f},{:.2f}], regionDim={})",
                         glm::length(cameraPos), loadRadius, shellInner, shellOuter, regionDim);
        } else {
            spdlog::info("Streaming.update: target became empty (cameraRadius={:.2f}, loadRadius={}, shell=[{:.2f},{:.2f}], regionDim={})",
                         glm::length(cameraPos), loadRadius, shellInner, shellOuter, regionDim);
        }
    } else if (prevEmpty) {
        spdlog::info("Streaming.update: populated {} bricks (cameraRadius={:.2f})",
                     target.size(), glm::length(cameraPos));
    }

    lastTarget_ = std::move(target);
}

Streaming::SelectionDiff Streaming::collectSelectionDiff(const std::vector<glm::ivec3>& currentSelection) const {
    SelectionDiff diff;
    diff.target = lastTarget_;

    std::vector<glm::ivec3> currentSorted = currentSelection;
    if (!std::is_sorted(currentSorted.begin(), currentSorted.end(), glmLess)) {
        std::sort(currentSorted.begin(), currentSorted.end(), glmLess);
    }

    diff.toUpload.reserve(diff.target.size());
    diff.toRemove.reserve(currentSorted.size());

    std::set_difference(diff.target.begin(), diff.target.end(),
                        currentSorted.begin(), currentSorted.end(),
                        std::back_inserter(diff.toUpload), glmLess);

    std::set_difference(currentSorted.begin(), currentSorted.end(),
                        diff.target.begin(), diff.target.end(),
                        std::back_inserter(diff.toRemove), glmLess);

    return diff;
}

} // namespace world
