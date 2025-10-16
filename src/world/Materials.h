#pragma once

#include <cstdint>

// Materials — load JSON and build GPU table.
namespace world {
struct MaterialParams {
    float baseColor[3]{}; float emission{}; float roughness{}; float metalness{}; float absorption[3]{};
};
class Materials {
public:
    bool load(const char* path);
};
}

