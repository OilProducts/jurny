#pragma once

#include <atomic>
#include <cstdint>
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
    struct RegionTask {
        glm::ivec3 coord;
        float priority = 0.0f;
        uint64_t key = 0;
        uint64_t frameEnqueued = 0;
    };

    struct BuildingJob {
        glm::ivec3 coord;
        float priority = 0.0f;
        std::shared_ptr<std::atomic<bool>> cancel;
    };

    struct CompletedRegion {
        glm::ivec3 coord;
        float priority = 0.0f;
        CpuWorld bricks;
        bool cancelled = false;
    };

    enum class RegionState : uint8_t {
        None,
        Pending,
        Building,
        Ready,
        Resident,
        Evicting
    };

    struct RegionRecord {
        RegionState state = RegionState::None;
        float priority = 0.0f;
        uint64_t lastTouchedFrame = 0;
    };

    struct RegionTaskCompare {
        bool operator()(const RegionTask& a, const RegionTask& b) const {
            if (a.priority == b.priority) {
                return a.frameEnqueued > b.frameEnqueued; // FIFO for equal priority
            }
            return a.priority < b.priority; // max-heap (higher priority first)
        }
    };

    struct ReadyRegionCompare {
        bool operator()(const ReadyRegion& a, const ReadyRegion& b) const {
            return a.priority < b.priority;
        }
    };

private:
    void clearQueues();
    void enqueueCandidateRegions(const glm::vec3& cameraPos, uint64_t frameIndex);
    void launchRegionBuilds();
    void drainCompleted();
    void promoteToReady(CompletedRegion&& completed);
    void evaluateEvictions(uint64_t frameIndex);

    std::vector<glm::ivec3> enumerateRegionBricks(const glm::ivec3& regionCoord,
                                                  float shellInner,
                                                  float shellOuter) const;
    float regionPriority(const glm::vec3& regionCenter, float distance, float shellInner, float shellOuter) const;
    std::pair<float, float> shellBounds() const;

    static uint64_t packRegionCoord(const glm::ivec3& coord);
    static glm::ivec3 unpackRegionCoord(uint64_t key);

private:
    const BrickStore* store_ = nullptr;
    Config config_{};
    core::Jobs* jobs_ = nullptr;

    Stats stats_{};
    glm::vec3 lastCameraWorld_{0.0f};

    std::priority_queue<RegionTask, std::vector<RegionTask>, RegionTaskCompare> pendingRegions_;
    std::vector<BuildingJob> inFlight_;

    std::mutex completedMutex_;
    std::vector<CompletedRegion> completedRegions_;

    std::priority_queue<ReadyRegion, std::vector<ReadyRegion>, ReadyRegionCompare> readyQueue_;

    std::unordered_map<uint64_t, RegionRecord> regionRecords_;
    std::vector<glm::ivec3> evictedRegions_;
};

}
