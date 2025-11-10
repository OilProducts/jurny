#include "Streaming.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <utility>

#include <glm/geometric.hpp>
#include <spdlog/spdlog.h>

#include "core/Jobs.h"
#include "world/BrickStore.h"
#include "world/CoordUtils.h"
#include "math/Spherical.h"

namespace world {

void Streaming::initialize(const BrickStore& store, const Config& config, core::Jobs* jobs) {
    store_ = &store;
    config_ = config;
    jobs_ = jobs;
    stats_ = {};
    stats_.buildMsAvg = 0.0;
    lastCameraWorld_ = glm::vec3(0.0f);
    clearQueues();
}

void Streaming::shutdown() {
    clearQueues();
    store_ = nullptr;
}

void Streaming::clearQueues() {
    pendingRegions_ = decltype(pendingRegions_)();
    for (auto& job : inFlight_) {
        if (job.cancel) {
            job.cancel->store(true, std::memory_order_relaxed);
        }
    }
    inFlight_.clear();
    {
        std::lock_guard<std::mutex> lock(completedMutex_);
        completedRegions_.clear();
    }
    readyQueue_ = decltype(readyQueue_)();
    regionRecords_.clear();
    evictedRegions_.clear();
    buildMsAvgValid_ = false;
    buildMsAvg_ = 0.0;
    solidRatioLast_ = 0.0;
}

void Streaming::update(const glm::vec3& cameraPos, uint64_t frameIndex) {
    if (!store_) return;

    currentFrame_ = frameIndex;
    lastCameraWorld_ = cameraPos;
    Stats prevStats = stats_;
    stats_ = {};
    stats_.buildMsLast = prevStats.buildMsLast;
    stats_.bricksGeneratedLast = prevStats.bricksGeneratedLast;
    stats_.bricksRequestedLast = prevStats.bricksRequestedLast;
    stats_.solidRatioLast = prevStats.solidRatioLast;
    stats_.buildMsMax = prevStats.buildMsMax;
    stats_.buildSamples = prevStats.buildSamples;
    stats_.buildMsAvg = buildMsAvgValid_ ? buildMsAvg_ : 0.0;
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
    spdlog::debug("Streaming: update cameraRadius={} shell=[{}, {}] keepRadius={} loadRadius={}",
                  glm::length(cameraPos), shellInner, shellOuter, config_.keepRadius, config_.loadRadius);
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
                                  record.state == RegionState::Evicting ||
                                  record.state == RegionState::Empty)) {
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
    spdlog::debug("Streaming: launch builds shell=[{}, {}] loadRadius={} keepRadius={}",
                  shellInner, shellOuter, config_.loadRadius, config_.keepRadius);

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
            const auto start = std::chrono::steady_clock::now();
            uint32_t bricksRequested = static_cast<uint32_t>(bricks.size());
            CpuWorld cpu = store_->buildCpuWorld(bricks, cancelFlag.get());
            const auto end = std::chrono::steady_clock::now();
            const double ms = std::chrono::duration<double, std::milli>(end - start).count();
            bool cancelled = cancelFlag->load(std::memory_order_relaxed);
            CompletedRegion completed;
            completed.coord = coord;
            completed.priority = priority;
            completed.bricks = std::move(cpu);
            completed.cancelled = cancelled;
            completed.buildMs = ms;
            completed.bricksRequested = bricksRequested;
            completed.bricksGenerated = static_cast<uint32_t>(completed.bricks.headers.size());
            if (!cancelled) {
                double solidRatio = (bricksRequested > 0)
                    ? (static_cast<double>(completed.bricksGenerated) / static_cast<double>(bricksRequested))
                    : 0.0;
                spdlog::info("Streaming built region ({}, {}, {}): {:.2f} ms, bricks {} / {} ({:.1f}% solid)",
                             coord.x, coord.y, coord.z,
                             ms,
                             completed.bricksGenerated,
                             bricksRequested,
                             solidRatio * 100.0);
            }
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

    if (completed.cancelled) {
        it->second.state = RegionState::None;
        return;
    }

    if (completed.bricks.headers.empty()) {
        it->second.state = RegionState::Empty;
        return;
    }

    stats_.buildMsLast = completed.buildMs;
    stats_.buildSamples += 1;
    stats_.buildMsMax = std::max(stats_.buildMsMax, completed.buildMs);
    stats_.bricksGeneratedLast = completed.bricksGenerated;
    stats_.bricksRequestedLast = completed.bricksRequested;
    double solidRatio = (completed.bricksRequested > 0)
        ? (static_cast<double>(completed.bricksGenerated) / static_cast<double>(completed.bricksRequested))
        : 0.0;
    stats_.solidRatioLast = solidRatio;
    solidRatioLast_ = solidRatio;
    if (completed.buildMs > 0.0) {
        if (!buildMsAvgValid_) {
            buildMsAvg_ = completed.buildMs;
            buildMsAvgValid_ = true;
        } else {
            const double alpha = 0.15;
            buildMsAvg_ = (1.0 - alpha) * buildMsAvg_ + alpha * completed.buildMs;
        }
    }
    stats_.buildMsAvg = buildMsAvgValid_ ? buildMsAvg_ : 0.0;

    constexpr double kInteriorThreshold = 0.9;
    const bool mostlySolid = solidRatio >= kInteriorThreshold;
    if (mostlySolid) {
        // Keep the current region so surface voxels bound to its brick coordinates remain available,
        // but continue to push the next outward ring so streaming can expand.
        queueOutwardNeighbor(completed.coord, static_cast<float>(solidRatio));
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
    float keepRadius = (config_.keepRadius > 0.0f) ? config_.keepRadius : config_.loadRadius;
    if (config_.loadRadius > 0.0f) {
        keepRadius = std::max(keepRadius, config_.loadRadius);
    }

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
    if (bricks.empty()) {
        spdlog::debug("Streaming: region ({}, {}, {}) yielded 0 bricks (shell=[{}, {}])",
                      regionCoord.x, regionCoord.y, regionCoord.z, shellInner, shellOuter);
    } else {
        glm::vec3 center = (glm::vec3(bricks.front()) + glm::vec3(0.5f)) * brickSize;
        float radius = glm::length(center);
        spdlog::debug("Streaming: region ({}, {}, {}) kept {} bricks; first radius={} shell=[{}, {}] halfDiag={}",
                      regionCoord.x, regionCoord.y, regionCoord.z,
                      bricks.size(), radius, shellInner, shellOuter, brickHalfDiag);
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
    if (store_) {
        const float brickHalfDiag = 0.5f * store_->brickSize() * std::sqrt(3.0f);
        inner = std::max(0.0f, inner - brickHalfDiag);
        outer += brickHalfDiag;
    }
    if (outer <= inner) {
        outer = inner + 10.0f;
    }
    return {inner, outer};
}

uint64_t Streaming::packRegionCoord(const glm::ivec3& coord) {
    return coord::packSignedCoord(coord.x, coord.y, coord.z);
}

glm::ivec3 Streaming::unpackRegionCoord(uint64_t key) {
    return coord::unpackSignedCoord(key);
}

void Streaming::queueOutwardNeighbor(const glm::ivec3& coord, float solidRatio) {
    if (!store_) {
        return;
    }
    const int regionDim = std::max(config_.regionDimBricks, 1);
    const float regionSize = static_cast<float>(regionDim) * store_->brickSize();
    glm::vec3 center = (glm::vec3(coord) + glm::vec3(0.5f)) * regionSize;
    float radius = glm::length(center);
    if (radius < 1e-3f) {
        return;
    }
    glm::vec3 dir = center / radius;
    glm::vec3 outwardCenter = center + dir * regionSize;
    glm::ivec3 outwardCoord = glm::floor(outwardCenter / regionSize);
    if (outwardCoord == coord) {
        outwardCoord += glm::ivec3((dir.x >= 0.f) ? 1 : -1,
                                   (dir.y >= 0.f) ? 1 : -1,
                                   (dir.z >= 0.f) ? 1 : -1);
    }

    const uint64_t key = packRegionCoord(outwardCoord);
    auto [recIt, inserted] = regionRecords_.emplace(key, RegionRecord{});
    RegionRecord& record = recIt->second;
    if (!inserted && record.state != RegionState::None) {
        return;
    }

    const auto [shellInner, shellOuter] = shellBounds();
    glm::vec3 outwardCenterExact = (glm::vec3(outwardCoord) + glm::vec3(0.5f)) * regionSize;
    float distanceToCamera = glm::length(outwardCenterExact - lastCameraWorld_);
    float priority = regionPriority(outwardCenterExact, distanceToCamera, shellInner, shellOuter);

    record.state = RegionState::Pending;
    record.priority = priority;
    record.lastTouchedFrame = currentFrame_;

    RegionTask task;
    task.coord = outwardCoord;
    task.priority = priority;
    task.key = key;
   task.frameEnqueued = currentFrame_;
   pendingRegions_.push(task);

    spdlog::debug("Streaming: queued outward region ({}, {}, {}) after solid ratio {:.2f}",
                  outwardCoord.x, outwardCoord.y, outwardCoord.z, solidRatio);
}

} // namespace world
