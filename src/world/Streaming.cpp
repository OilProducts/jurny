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

namespace {
constexpr glm::ivec3 kNeighborOffsets[6] = {
    {1, 0, 0},
    {-1, 0, 0},
    {0, 1, 0},
    {0, -1, 0},
    {0, 0, 1},
    {0, 0, -1}
};
}

void Streaming::transitionState(RegionRecord& record, RegionState newState, const glm::ivec3& coord) {
    (void)coord;
    if (record.state == newState) {
        return;
    }
    if (record.state == RegionState::Building && buildingCount_ > 0) {
        --buildingCount_;
    }
    record.state = newState;
    if (newState == RegionState::Building) {
        ++buildingCount_;
    }
}

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
    readyRegions_.clear();
    frontier_.clear();
    hasCameraCell_ = false;
    lastFrontierOrigin_ = glm::ivec3(std::numeric_limits<int>::max(),
                                     std::numeric_limits<int>::max(),
                                     std::numeric_limits<int>::max());
    {
        std::lock_guard<std::mutex> lock(completedMutex_);
        completedRegions_.clear();
    }
    regionRecords_.clear();
    buildingCount_ = 0;
    evictedRegions_.clear();
    buildMsAvgValid_ = false;
    buildMsAvg_ = 0.0;
    solidRatioLast_ = 0.0;
    regionCtx_ = {};
}

void Streaming::updateRegionContext(const glm::vec3& cameraPos) {
    regionCtx_ = {};
    if (!store_) {
        return;
    }
    regionCtx_.regionDim = std::max(config_.regionDimBricks, 1);
    regionCtx_.brickSize = store_->brickSize();
    regionCtx_.regionSize = regionCtx_.brickSize * static_cast<float>(regionCtx_.regionDim);
    regionCtx_.regionHalfDiag = 0.5f * regionCtx_.regionSize * std::sqrt(3.0f);
    auto [shellInner, shellOuter] = shellBounds();
    regionCtx_.shellInner = shellInner;
    regionCtx_.shellOuter = shellOuter;
    regionCtx_.loadRadius = config_.loadRadius;
    regionCtx_.loadRadiusExpanded = config_.loadRadius + regionCtx_.regionHalfDiag;
    float keep = config_.keepRadius > 0.0f ? config_.keepRadius : config_.loadRadius;
    if (config_.loadRadius > 0.0f) {
        keep = std::max(keep, config_.loadRadius);
    }
    regionCtx_.keepRadiusExpanded = keep + regionCtx_.regionHalfDiag;
    if (regionCtx_.regionSize > 0.0f) {
        regionCtx_.regionRadius = static_cast<int>(std::ceil(regionCtx_.loadRadiusExpanded / regionCtx_.regionSize)) + 1;
    } else {
        regionCtx_.regionRadius = 0;
    }
    (void)cameraPos;
}

void Streaming::seedFrontier(const glm::ivec3& cameraCell) {
    if (!hasCameraCell_ || cameraCell != lastFrontierOrigin_) {
        frontier_.clear();
        pushFrontierCell(cameraCell);
        lastFrontierOrigin_ = cameraCell;
    }
}

void Streaming::pushFrontierCell(const glm::ivec3& cell) {
    const uint64_t key = packRegionCoord(cell);
    auto [it, inserted] = regionRecords_.emplace(key, RegionRecord{});
    RegionRecord& record = it->second;
    if (record.frontierVisited) {
        return;
    }
    record.frontierQueued = true;
    record.frontierVisited = true;
    frontier_.push_back(cell);
}

void Streaming::enqueuePendingRegion(RegionRecord& record, const glm::ivec3& coord, float priority, uint64_t frameIndex) {
    PendingEntry entry{coord, priority, frameIndex};
    pendingRegions_.push(entry);
    record.priority = priority;
    record.lastTouchedFrame = frameIndex;
    transitionState(record, RegionState::Pending, coord);
}

void Streaming::expandFrontier(uint64_t frameIndex) {
    const RegionContext& ctx = regionCtx_;
    if (ctx.regionSize <= 0.0f || frontier_.empty()) {
        stats_.selectedRegions = 0;
        return;
    }

    size_t expansions = 0;
    while (!frontier_.empty() && pendingRegions_.size() < static_cast<size_t>(config_.maxQueuedRegions)) {
        glm::ivec3 cell = frontier_.front();
        frontier_.pop_front();
        const uint64_t key = packRegionCoord(cell);
        auto it = regionRecords_.find(key);
        if (it == regionRecords_.end()) {
            continue;
        }
        RegionRecord& record = it->second;
        record.frontierQueued = false;
        updateRecordMetrics(record, cell);
        record.lastTouchedFrame = frameIndex;

        bool withinShell = !(record.maxRadius < ctx.shellInner || record.minRadius > ctx.shellOuter);
        bool withinLoad = record.distanceToCamera <= ctx.loadRadiusExpanded;
        if (withinShell && withinLoad && record.state == RegionState::None) {
            float priority = regionPriority(record.center, record.distanceToCamera, ctx);
            enqueuePendingRegion(record, cell, priority, frameIndex);
            ++expansions;
        }

        if (hasCameraCell_) {
            for (const glm::ivec3& dir : kNeighborOffsets) {
                glm::ivec3 neighbor = cell + dir;
                glm::ivec3 delta = neighbor - currentCameraCell_;
                int maxDist = std::max({std::abs(delta.x), std::abs(delta.y), std::abs(delta.z)});
                if (maxDist > ctx.regionRadius) {
                    continue;
                }
                pushFrontierCell(neighbor);
            }
        }

        if (expansions >= static_cast<size_t>(config_.maxRegionSelectionsPerFrame)) {
            break;
        }
    }

    stats_.selectedRegions = static_cast<uint32_t>(expansions);
}

void Streaming::update(const glm::vec3& cameraPos, uint64_t frameIndex) {
    if (!store_) return;

    currentFrame_ = frameIndex;
    lastCameraWorld_ = cameraPos;
    updateRegionContext(cameraPos);
    if (regionCtx_.regionSize > 0.0f) {
        glm::ivec3 cameraCell = glm::ivec3(glm::floor(cameraPos / regionCtx_.regionSize));
        currentCameraCell_ = cameraCell;
        hasCameraCell_ = true;
        seedFrontier(cameraCell);
    }
    stats_.selectedRegions = 0;
    stats_.buildMsAvg = buildMsAvgValid_ ? buildMsAvg_ : 0.0;
    evictedRegions_.clear();

    if (pendingRegions_.size() < static_cast<size_t>(config_.maxQueuedRegions)) {
        expandFrontier(frameIndex);
    }
    launchRegionBuilds();
    drainCompleted();
    evaluateEvictions(frameIndex);

    stats_.queuedRegions = static_cast<uint32_t>(pendingRegions_.size());
    stats_.buildingRegions = buildingCount_;
    stats_.readyRegions = static_cast<uint32_t>(readyRegions_.size());
}

bool Streaming::popReadyRegion(ReadyRegion& out) {
    if (readyRegions_.empty()) {
        return false;
    }
    out = std::move(readyRegions_.back());
    readyRegions_.pop_back();
    return true;
}

void Streaming::markRegionUploaded(const glm::ivec3& regionCoord) {
    const uint64_t key = packRegionCoord(regionCoord);
    auto it = regionRecords_.find(key);
    if (it != regionRecords_.end()) {
        transitionState(it->second, RegionState::Resident, regionCoord);
    }
}

void Streaming::markRegionEvicted(const glm::ivec3& regionCoord) {
    const uint64_t key = packRegionCoord(regionCoord);
    auto it = regionRecords_.find(key);
    if (it != regionRecords_.end()) {
        transitionState(it->second, RegionState::None, regionCoord);
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

void Streaming::launchRegionBuilds() {
    if (!jobs_) return;

    spdlog::debug("Streaming: launch builds shell=[{}, {}] loadRadius={} keepRadius={}",
                  regionCtx_.shellInner, regionCtx_.shellOuter, config_.loadRadius, config_.keepRadius);

    while (!pendingRegions_.empty() &&
           static_cast<int>(buildingCount_) < config_.maxConcurrentGenerations) {
        PendingEntry entry = pendingRegions_.top();
        pendingRegions_.pop();
        glm::ivec3 coord = entry.coord;
        const uint64_t key = packRegionCoord(coord);

        auto it = regionRecords_.find(key);
        if (it == regionRecords_.end() || it->second.state != RegionState::Pending) {
            continue;
        }
        RegionRecord& record = it->second;

        auto cancelFlag = std::make_shared<std::atomic<bool>>(false);
        transitionState(record, RegionState::Building, coord);

        std::vector<glm::ivec3> bricks = enumerateRegionBricks(coord, regionCtx_);
        if (bricks.empty()) {
            transitionState(record, RegionState::None, coord);
            continue;
        }
        stats_.queuedRegions = static_cast<uint32_t>(pendingRegions_.size());

        float taskPriority = entry.priority;

        jobs_->schedule([this, coord, priority = taskPriority, bricks = std::move(bricks), cancelFlag]() mutable {
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

    // Building state bucket already reflects active jobs; no additional cleanup needed here.
}

void Streaming::promoteToReady(CompletedRegion&& completed) {
    const uint64_t key = packRegionCoord(completed.coord);
    auto it = regionRecords_.find(key);
    if (it == regionRecords_.end()) {
        return;
    }

    if (completed.cancelled) {
        transitionState(it->second, RegionState::None, completed.coord);
        return;
    }

    if (completed.bricks.headers.empty()) {
        transitionState(it->second, RegionState::Empty, completed.coord);
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
    readyRegions_.push_back(std::move(ready));
    transitionState(it->second, RegionState::Ready, completed.coord);
}

void Streaming::evaluateEvictions(uint64_t frameIndex) {
    (void)frameIndex;
    if (!store_) return;

    const RegionContext& ctx = regionCtx_;
    if (ctx.regionSize <= 0.0f) {
        return;
    }

    for (auto& entry : regionRecords_) {
        RegionRecord& record = entry.second;
        if (record.state != RegionState::Resident) {
            continue;
        }

        glm::ivec3 coord = unpackRegionCoord(entry.first);
        updateRecordMetrics(record, coord);
        bool outsideShell = (record.maxRadius < ctx.shellInner) || (record.minRadius > ctx.shellOuter);
        bool outsideKeep = record.distanceToCamera > ctx.keepRadiusExpanded;

        if (outsideShell || outsideKeep) {
            transitionState(record, RegionState::Evicting, coord);
            evictedRegions_.push_back(coord);
        }
    }
}

std::vector<glm::ivec3> Streaming::enumerateRegionBricks(const glm::ivec3& regionCoord,
                                                         const RegionContext& ctx) const {
    const int regionDim = std::max(ctx.regionDim, 1);
    std::vector<glm::ivec3> bricks;
    bricks.reserve(static_cast<size_t>(regionDim) * regionDim * regionDim);
    const glm::ivec3 base = regionCoord * regionDim;
    const float brickSize = store_ ? store_->brickSize() : ctx.brickSize;
    const float brickHalfDiag = 0.5f * brickSize * std::sqrt(3.0f);

    for (int z = 0; z < regionDim; ++z) {
        for (int y = 0; y < regionDim; ++y) {
            for (int x = 0; x < regionDim; ++x) {
                glm::ivec3 bc = base + glm::ivec3(x, y, z);
                glm::vec3 center = (glm::vec3(bc) + glm::vec3(0.5f)) * brickSize;
                float radius = glm::length(center);
                float minRadius = radius - brickHalfDiag;
                float maxRadius = radius + brickHalfDiag;
                if (maxRadius < ctx.shellInner || minRadius > ctx.shellOuter) {
                    continue;
                }
                bricks.emplace_back(bc);
            }
        }
    }
    if (bricks.empty()) {
        spdlog::debug("Streaming: region ({}, {}, {}) yielded 0 bricks (shell=[{}, {}])",
                      regionCoord.x, regionCoord.y, regionCoord.z, ctx.shellInner, ctx.shellOuter);
    } else {
        glm::vec3 center = (glm::vec3(bricks.front()) + glm::vec3(0.5f)) * brickSize;
        float radius = glm::length(center);
        spdlog::debug("Streaming: region ({}, {}, {}) kept {} bricks; first radius={} shell=[{}, {}] halfDiag={}",
                      regionCoord.x, regionCoord.y, regionCoord.z,
                      bricks.size(), radius, ctx.shellInner, ctx.shellOuter, brickHalfDiag);
    }
    return bricks;
}

float Streaming::regionPriority(const glm::vec3& regionCenter,
                                float distance,
                                const RegionContext& ctx) const {
    float shellCenter = 0.5f * (ctx.shellInner + ctx.shellOuter);
    float shellRange = std::max(1.0f, ctx.shellOuter - ctx.shellInner);
    float shellScore = 1.0f - glm::clamp(std::abs(glm::length(regionCenter) - shellCenter) / shellRange, 0.0f, 1.0f);
    float loadRadius = ctx.loadRadius > 0.0f ? ctx.loadRadius : std::max(1.0f, config_.loadRadius);
    float distanceScore = 1.0f - glm::clamp(distance / (loadRadius + 1e-3f), 0.0f, 1.0f);
    return 0.6f * distanceScore + 0.4f * shellScore;
}

void Streaming::updateRecordMetrics(RegionRecord& record, const glm::ivec3& coord) {
    const RegionContext& ctx = regionCtx_;
    if (ctx.regionSize <= 0.0f) {
        record.center = glm::vec3(0.0f);
        record.minRadius = 0.0f;
        record.maxRadius = 0.0f;
        record.distanceToCamera = 0.0f;
        return;
    }
    glm::vec3 regionMin = glm::vec3(coord) * ctx.regionSize;
    record.center = regionMin + glm::vec3(ctx.regionSize * 0.5f);
    float radiusCenter = glm::length(record.center);
    record.minRadius = radiusCenter - ctx.regionHalfDiag;
    record.maxRadius = radiusCenter + ctx.regionHalfDiag;
    record.distanceToCamera = glm::length(record.center - lastCameraWorld_);
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
    const RegionContext& ctx = regionCtx_;
    const int regionDim = std::max(ctx.regionDim, 1);
    const float regionSize = ctx.regionSize > 0.0f ? ctx.regionSize : static_cast<float>(regionDim) * store_->brickSize();
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
    updateRecordMetrics(record, outwardCoord);
    record.lastTouchedFrame = currentFrame_;

    glm::vec3 outwardCenterExact = (glm::vec3(outwardCoord) + glm::vec3(0.5f)) * regionSize;
    float distanceToCamera = glm::length(outwardCenterExact - lastCameraWorld_);
    float priority = regionPriority(outwardCenterExact, distanceToCamera, ctx);

    enqueuePendingRegion(record, outwardCoord, priority, currentFrame_);

    spdlog::debug("Streaming: queued outward region ({}, {}, {}) after solid ratio {:.2f}",
                  outwardCoord.x, outwardCoord.y, outwardCoord.z, solidRatio);
}

} // namespace world
