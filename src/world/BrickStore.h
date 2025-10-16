#pragma once

#include <vector>
#include "BrickFormats.h"

// BrickStore — CPU+GPU brick pools; edits and uploads.
namespace world {
class BrickStore {
public:
    void clear();
    void uploadBatch();
private:
    std::vector<BrickHeader> headers_;
};
}

