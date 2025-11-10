#pragma once

#include <cstdint>

#include <glm/vec3.hpp>

namespace world::math {

inline int divFloor(int a, int b) {
    int q = a / b;
    int r = a - q * b;
    if (((a ^ b) < 0) && r != 0) --q;
    return q;
}

} // namespace world::math

namespace world::coord {

constexpr uint64_t kSignedCoordBias = 1ull << 20;

inline uint64_t packSignedCoord(int x, int y, int z) {
    return (static_cast<uint64_t>(x + static_cast<int>(kSignedCoordBias)) << 42) |
           (static_cast<uint64_t>(y + static_cast<int>(kSignedCoordBias)) << 21) |
            static_cast<uint64_t>(z + static_cast<int>(kSignedCoordBias));
}

inline glm::ivec3 unpackSignedCoord(uint64_t key) {
    constexpr int32_t bias = static_cast<int32_t>(kSignedCoordBias);
    glm::ivec3 coord;
    coord.x = static_cast<int32_t>((key >> 42) & ((1ull << 21) - 1ull)) - bias;
    coord.y = static_cast<int32_t>((key >> 21) & ((1ull << 21) - 1ull)) - bias;
    coord.z = static_cast<int32_t>(key & ((1ull << 21) - 1ull)) - bias;
    return coord;
}

} // namespace world::coord
