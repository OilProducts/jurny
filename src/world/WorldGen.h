#pragma once

#include <cstdint>

// WorldGen — procedural height on unit sphere, caves in 3D.
namespace world {
class WorldGen {
public:
    void seed(std::uint64_t s);
};
}
