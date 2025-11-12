#pragma once

#include <atomic>
#include <array>
#include <cstdint>
#include <deque>
#include <limits>
#include <memory>
#include <queue>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <utility>

#include <glm/vec3.hpp>

#include "world/BrickStore.h"
#include "world/RegionCache.h"

namespace core { class Jobs; }

namespace world {

class Streaming {
public:
    struct Config {
        float shellInner = 0.0f;
        float shellOuter = 0.0f;
        float keepRadius = 60.0f;
        float loadRadius = 90.0f;
        float simRadius  = 55.0f;
        int   regionDimBricks = 16;
        int   maxQueuedRegions = 64;
        int   maxRegionSelectionsPerFrame = 4;
        int   maxConcurrentGenerations = 2;
    };

    struct Stats {
        uint32_t selectedRegions = 0;
        uint32_t queuedRegions   = 0;
        uint32_t buildingRegions = 0;
        uint32_t readyRegions    = 0;
        double   buildMsLast     = 0.0;
        double   buildMsAvg      = 0.0;
        double   buildMsMax      = 0.0;
        uint32_t buildSamples    = 0;
        uint32_t bricksGeneratedLast = 0;
        uint32_t bricksRequestedLast = 0;
        double   solidRatioLast  = 0.0;
    };

    struct ReadyRegion {
        glm::ivec3 regionCoord{};
        float priority = 0.0f;
        CpuWorld bricks;
    };

    void initialize(const BrickStore& store, const Config& config, core::Jobs* jobs = nullptr);
    void shutdown();

    void update(const glm::vec3& cameraPos, uint64_t frameIndex);

    bool popReadyRegion(ReadyRegion& out);
    void markRegionUploaded(const glm::ivec3& regionCoord);
    void markRegionEvicted(const glm::ivec3& regionCoord);
    bool popEvictedRegion(glm::ivec3& out);

    const Stats& stats() const { return stats_; }
    const glm::vec3& lastCameraOrigin() const { return lastCameraWorld_; }
    int regionDimBricks() const { return config_.regionDimBricks; }

private:
    void queueOutwardNeighbor(const glm::ivec3& coord, float solidRatio);
    struct PendingEntry {
        glm::ivec3 coord{};
        float priority = 0.0f;
        uint64_t frameEnqueued = 0;
    };
    struct PendingCompare {
        bool operator()(const PendingEntry& a, const PendingEntry& b) const {
            if (a.priority == b.priority) {
                return a.frameEnqueued > b.frameEnqueued;
            }
            return a.priority < b.priority;
        }
    };
    struct RegionContext {
        float brickSize = 0.0f;
        int regionDim = 1;
        float regionSize = 0.0f;
        float regionHalfDiag = 0.0f;
        float shellInner = 0.0f;
        float shellOuter = 0.0f;
        float loadRadius = 0.0f;
        float loadRadiusExpanded = 0.0f;
        float keepRadiusExpanded = 0.0f;
        int regionRadius = 0;
    };
    void updateRegionContext(const glm::vec3& cameraPos);

    struct CompletedRegion {
        glm::ivec3 coord;
        float priority = 0.0f;
        CpuWorld bricks;
        bool cancelled = false;
        double buildMs = 0.0;
        uint32_t bricksRequested = 0;
        uint32_t bricksGenerated = 0;
    };

    enum class RegionState : uint8_t {
        None,
        Pending,
        Building,
        Ready,
        Resident,
        Evicting,
        Empty
    };
    static constexpr int kRegionStateCount = static_cast<int>(RegionState::Empty) + 1;

    struct RegionRecord {
        RegionState state = RegionState::None;
        float priority = 0.0f;
        uint64_t lastTouchedFrame = 0;
        glm::vec3 center{0.0f};
        float minRadius = 0.0f;
        float maxRadius = 0.0f;
        float distanceToCamera = 0.0f;
        bool frontierQueued = false;
        uint32_t frontierVisitSerial = 0;
    };

private:
    void clearQueues();
    void seedFrontier(const glm::ivec3& cameraCell);
    void expandFrontier(uint64_t frameIndex);
    void launchRegionBuilds();
    void drainCompleted();
    void promoteToReady(CompletedRegion&& completed);
    void evaluateEvictions(uint64_t frameIndex);

    std::vector<glm::ivec3> enumerateRegionBricks(const glm::ivec3& regionCoord,
                                                  const RegionContext& ctx) const;
    float regionPriority(const glm::vec3& regionCenter, float distance, const RegionContext& ctx) const;
    std::pair<float, float> shellBounds() const;
    void updateRecordMetrics(RegionRecord& record, const glm::ivec3& coord);
    void enqueuePendingRegion(RegionRecord& record, const glm::ivec3& coord, float priority, uint64_t frameIndex);
    void pushFrontierCell(const glm::ivec3& cell);
    void transitionState(RegionRecord& record, RegionState newState, const glm::ivec3& coord);

    static uint64_t packRegionCoord(const glm::ivec3& coord);
    static glm::ivec3 unpackRegionCoord(uint64_t key);

private:
    const BrickStore* store_ = nullptr;
    Config config_{};
    core::Jobs* jobs_ = nullptr;

    Stats stats_{};
    double buildMsAvg_ = 0.0;
    bool buildMsAvgValid_ = false;
    double solidRatioLast_ = 0.0;
    glm::vec3 lastCameraWorld_{0.0f};
    uint64_t currentFrame_ = 0;
    RegionContext regionCtx_{};

    std::priority_queue<PendingEntry, std::vector<PendingEntry>, PendingCompare> pendingRegions_;
    std::vector<ReadyRegion> readyRegions_;
    std::deque<glm::ivec3> frontier_;
    glm::ivec3 lastFrontierOrigin_{std::numeric_limits<int>::max(), std::numeric_limits<int>::max(), std::numeric_limits<int>::max()};
    glm::ivec3 currentCameraCell_{0};
    bool hasCameraCell_ = false;
    std::vector<glm::ivec3> evictedRegions_;
    uint32_t frontierSerial_ = 1;

    std::mutex completedMutex_;
    std::vector<CompletedRegion> completedRegions_;

    std::unordered_map<uint64_t, RegionRecord> regionRecords_;
    uint32_t buildingCount_ = 0;
};

}
