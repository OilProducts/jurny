#pragma once

// RegionCache — groups bricks into 64^3 regions for streaming/eviction.
namespace world {
class RegionCache {
public:
    void tick();
};
}

