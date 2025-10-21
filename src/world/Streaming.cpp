#include "Streaming.h"

#include <algorithm>
#include <cmath>
#include <utility>

#include <glm/geometric.hpp>
#include <spdlog/spdlog.h>

#include "core/Jobs.h"
#include "world/BrickStore.h"
#include "math/Spherical.h"

namespace {
inline int divFloor(int a, int b) {
    int q = a / b;
    int r = a - q * b;
    if (((a ^ b) < 0) && r != 0) --q;
    return q;
}
}

namespace world {

void Streaming::initialize(const BrickStore& store, const Config& config, core::Jobs* jobs) {
    store_ = &store;
    config_ = config;
    jobs_ = jobs;
    stats_ = {};
    lastCameraWorld_ = glm::vec3(0.0f);
    clearQueues();
}

void Streaming::shutdown() {
    clearQueues();
    store_ = nullptr;
}

void Streaming::clearQueues() {
    pendingRegions_ = decltype(pendingRegions_)();
    inFlight_.clear();
    {
        std::lock_guard<std::mutex> lock(completedMutex_);
        completedRegions_.clear();
    }
    readyQueue_ = decltype(readyQueue_)();
    regionRecords_.clear();
    evictedRegions_.clear();
}

void Streaming::update(const glm::vec3& cameraPos, uint64_t frameIndex) {
    if (!store_) return;

    lastCameraWorld_ = cameraPos;
    stats_ = {};
    evictedRegions_.clear();

    enqueueCandidateRegions(cameraPos, frameIndex);
    launchRegionBuilds();
    drainCompleted();
    evaluateEvictions(frameIndex);

    stats_.queuedRegions = static_cast<uint32_t>(pendingRegions_.size());
    stats_.buildingRegions = static_cast<uint32_t>(inFlight_.size());
    stats_.readyRegions = static_cast<uint32_t>(readyQueue_.size());
}

bool Streaming::popReadyRegion(ReadyRegion& out) {
    if (readyQueue_.empty()) {
        return false;
    }
    out = std::move(const_cast<ReadyRegion&>(readyQueue_.top()));
    readyQueue_.pop();

    const uint64_t key = packRegionCoord(out.regionCoord);
    auto it = regionRecords_.find(key);
    if (it != regionRecords_.end()) {
        it->second.state = RegionState::Ready;
    }
    return true;
}

void Streaming::markRegionUploaded(const glm::ivec3& regionCoord) {
    const uint64_t key = packRegionCoord(regionCoord);
    auto it = regionRecords_.find(key);
    if (it != regionRecords_.end()) {
        it->second.state = RegionState::Resident;
    }
}

void Streaming::markRegionEvicted(const glm::ivec3& regionCoord) {
    const uint64_t key = packRegionCoord(regionCoord);
    auto it = regionRecords_.find(key);
    if (it != regionRecords_.end()) {
        regionRecords_.erase(it);
    }
}

bool Streaming::popEvictedRegion(glm::ivec3& out) {
    if (evictedRegions_.empty()) {
        return false;
    }
    out = evictedRegions_.back();
    evictedRegions_.pop_back();
    return true;
}

void Streaming::enqueueCandidateRegions(const glm::vec3& cameraPos, uint64_t frameIndex) {
    const auto [shellInner, shellOuter] = shellBounds();
    const float brickSize = store_->brickSize();
    const int regionDim = std::max(config_.regionDimBricks, 1);
    const float regionSize = static_cast<float>(regionDim) * brickSize;
    const float regionHalfDiag = 0.5f * regionSize * std::sqrt(3.0f);

    const glm::vec3 cameraRegionF = cameraPos / regionSize;
    glm::ivec3 cameraRegion = glm::ivec3(glm::floor(cameraRegionF));

    const float loadRadius = config_.loadRadius;
    const float loadRadiusExpanded = loadRadius + regionHalfDiag;
    const int regionRadius = int(std::ceil(loadRadiusExpanded / regionSize)) + 1;

    const float camLen2 = glm::dot(cameraPos, cameraPos);
    const glm::vec3 camDir = camLen2 > 1e-6f ? glm::normalize(cameraPos) : glm::vec3(0.0f);
    const bool hasCamDir = camLen2 > 0.0f;

    uint32_t enqueuedThisFrame = 0;

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

                float distanceToCamera = glm::length(regionCenter - cameraPos);
                if (distanceToCamera > loadRadiusExpanded) {
                    continue;
                }

                float priority = regionPriority(regionCenter, distanceToCamera, shellInner, shellOuter);
                if (hasCamDir && regionRadiusCenter > 1e-3f) {
                    glm::vec3 regionDir = regionCenter / regionRadiusCenter;
                    float cosAngle = glm::clamp(glm::dot(camDir, regionDir), -1.0f, 1.0f);
                    priority += 0.15f * (0.5f * (cosAngle + 1.0f));
                }

                const uint64_t key = packRegionCoord(rc);
                auto [it, inserted] = regionRecords_.emplace(key, RegionRecord{});
                RegionRecord& record = it->second;
                record.lastTouchedFrame = frameIndex;

                if (!inserted && (record.state == RegionState::Pending ||
                                  record.state == RegionState::Building ||
                                  record.state == RegionState::Ready ||
                                  record.state == RegionState::Resident ||
                                  record.state == RegionState::Evicting)) {
                    continue;
                }

                if (enqueuedThisFrame >= static_cast<uint32_t>(config_.maxRegionSelectionsPerFrame)) {
                    continue;
                }

                RegionTask task;
                task.coord = rc;
                task.priority = priority;
                task.key = key;
                task.frameEnqueued = frameIndex;
                pendingRegions_.push(task);

                record.state = RegionState::Pending;
                record.priority = priority;
                ++enqueuedThisFrame;
            }
        }
    }

    stats_.selectedRegions = enqueuedThisFrame;
}

void Streaming::launchRegionBuilds() {
    if (!jobs_) return;

    const auto [shellInner, shellOuter] = shellBounds();

    while (!pendingRegions_.empty() && static_cast<int>(inFlight_.size()) < config_.maxConcurrentGenerations) {
        RegionTask task = pendingRegions_.top();
        pendingRegions_.pop();

        auto it = regionRecords_.find(task.key);
        if (it == regionRecords_.end() || it->second.state != RegionState::Pending) {
            continue;
        }

        auto cancelFlag = std::make_shared<std::atomic<bool>>(false);
        BuildingJob job{task.coord, task.priority, cancelFlag};
        inFlight_.push_back(job);
        it->second.state = RegionState::Building;
        it->second.priority = task.priority;

        std::vector<glm::ivec3> bricks = enumerateRegionBricks(task.coord, shellInner, shellOuter);
        if (bricks.empty()) {
            it->second.state = RegionState::None;
            continue;
        }
        stats_.queuedRegions = static_cast<uint32_t>(pendingRegions_.size());

        jobs_->schedule([this, coord = task.coord, priority = task.priority, bricks = std::move(bricks), cancelFlag]() mutable {
            CpuWorld cpu = store_->buildCpuWorld(bricks, cancelFlag.get());
            bool cancelled = cancelFlag->load(std::memory_order_relaxed);
            CompletedRegion completed;
            completed.coord = coord;
            completed.priority = priority;
            completed.bricks = std::move(cpu);
            completed.cancelled = cancelled;
            {
                std::lock_guard<std::mutex> lock(completedMutex_);
                completedRegions_.push_back(std::move(completed));
            }
        });
    }
}

void Streaming::drainCompleted() {
    std::vector<CompletedRegion> local;
    {
        std::lock_guard<std::mutex> lock(completedMutex_);
        local.swap(completedRegions_);
    }

    if (local.empty()) {
        return;
    }

    for (CompletedRegion& completed : local) {
        promoteToReady(std::move(completed));
    }

    // Remove finished jobs from inFlight_
    inFlight_.erase(std::remove_if(inFlight_.begin(), inFlight_.end(), [&](const BuildingJob& job) {
        const uint64_t key = packRegionCoord(job.coord);
        auto it = regionRecords_.find(key);
        if (it == regionRecords_.end()) return true;
        return it->second.state != RegionState::Building;
    }), inFlight_.end());
}

void Streaming::promoteToReady(CompletedRegion&& completed) {
    const uint64_t key = packRegionCoord(completed.coord);
    auto it = regionRecords_.find(key);
    if (it == regionRecords_.end()) {
        return;
    }

    if (completed.cancelled || completed.bricks.headers.empty()) {
        it->second.state = RegionState::None;
        return;
    }

    ReadyRegion ready;
    ready.regionCoord = completed.coord;
    ready.priority = completed.priority;
    ready.bricks = std::move(completed.bricks);
    readyQueue_.push(std::move(ready));
    it->second.state = RegionState::Ready;
}

void Streaming::evaluateEvictions(uint64_t frameIndex) {
    (void)frameIndex;
    if (!store_) return;

    const auto [shellInner, shellOuter] = shellBounds();
    const int regionDim = std::max(config_.regionDimBricks, 1);
    const float regionSize = static_cast<float>(regionDim) * store_->brickSize();
    const float regionHalfDiag = 0.5f * regionSize * std::sqrt(3.0f);
    const float keepRadius = (config_.keepRadius > 0.0f) ? config_.keepRadius : config_.loadRadius;

    for (auto& entry : regionRecords_) {
        RegionRecord& record = entry.second;
        if (record.state != RegionState::Resident) {
            continue;
        }

        glm::ivec3 coord = unpackRegionCoord(entry.first);
        glm::vec3 regionMin = glm::vec3(coord) * regionSize;
        glm::vec3 center = regionMin + glm::vec3(regionSize * 0.5f);
        float regionRadiusCenter = glm::length(center);
        float minRadius = regionRadiusCenter - regionHalfDiag;
        float maxRadius = regionRadiusCenter + regionHalfDiag;
        bool outsideShell = (maxRadius < shellInner) || (minRadius > shellOuter);

        float distanceToCamera = glm::length(center - lastCameraWorld_);
        bool outsideKeep = distanceToCamera > (keepRadius + regionHalfDiag);

        if (outsideShell || outsideKeep) {
            record.state = RegionState::Evicting;
            evictedRegions_.push_back(coord);
        }
    }
}

std::vector<glm::ivec3> Streaming::enumerateRegionBricks(const glm::ivec3& regionCoord,
                                                         float shellInner,
                                                         float shellOuter) const {
    const int regionDim = std::max(config_.regionDimBricks, 1);
    std::vector<glm::ivec3> bricks;
    bricks.reserve(static_cast<size_t>(regionDim) * regionDim * regionDim);
    const glm::ivec3 base = regionCoord * regionDim;
    const float brickSize = store_->brickSize();
    const float brickHalfDiag = 0.5f * brickSize * std::sqrt(3.0f);

    for (int z = 0; z < regionDim; ++z) {
        for (int y = 0; y < regionDim; ++y) {
            for (int x = 0; x < regionDim; ++x) {
                glm::ivec3 bc = base + glm::ivec3(x, y, z);
                glm::vec3 center = (glm::vec3(bc) + glm::vec3(0.5f)) * brickSize;
                float radius = glm::length(center);
                float minRadius = radius - brickHalfDiag;
                float maxRadius = radius + brickHalfDiag;
                if (maxRadius < shellInner || minRadius > shellOuter) {
                    continue;
                }
                bricks.emplace_back(bc);
            }
        }
    }
    return bricks;
}

float Streaming::regionPriority(const glm::vec3& regionCenter,
                                float distance,
                                float shellInner,
                                float shellOuter) const {
    float shellCenter = 0.5f * (shellInner + shellOuter);
    float shellRange = std::max(1.0f, shellOuter - shellInner);
    float shellScore = 1.0f - glm::clamp(std::abs(glm::length(regionCenter) - shellCenter) / shellRange, 0.0f, 1.0f);
    float distanceScore = 1.0f - glm::clamp(distance / (config_.loadRadius + 1e-3f), 0.0f, 1.0f);
    return 0.6f * distanceScore + 0.4f * shellScore;
}

std::pair<float, float> Streaming::shellBounds() const {
    const auto& params = store_->params();
    float defaultInner = static_cast<float>(params.R) - static_cast<float>(params.T);
    float defaultOuter = static_cast<float>(params.R) + static_cast<float>(params.Hmax) + 10.0f;
    float inner = (config_.shellInner > 0.0f) ? config_.shellInner : defaultInner;
    float outer = (config_.shellOuter > 0.0f) ? config_.shellOuter : defaultOuter;
    if (outer <= inner) {
        outer = inner + 10.0f;
    }
    return {inner, outer};
}

uint64_t Streaming::packRegionCoord(const glm::ivec3& coord) {
    const int64_t B = (1ll << 20);
    return (static_cast<uint64_t>(coord.x + B) << 42) |
           (static_cast<uint64_t>(coord.y + B) << 21) |
            (static_cast<uint64_t>(coord.z + B));
}

glm::ivec3 Streaming::unpackRegionCoord(uint64_t key) {
    const int64_t B = (1ll << 20);
    glm::ivec3 coord;
    coord.x = static_cast<int>((key >> 42) & ((1ull << 21) - 1ull)) - static_cast<int>(B);
    coord.y = static_cast<int>((key >> 21) & ((1ull << 21) - 1ull)) - static_cast<int>(B);
    coord.z = static_cast<int>(key & ((1ull << 21) - 1ull)) - static_cast<int>(B);
    return coord;
}

} // namespace world
