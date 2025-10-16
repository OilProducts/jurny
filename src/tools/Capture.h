#pragma once

// Capture — frame capture/replay for reproducible tests.
namespace tools {
class Capture {
public:
    void save();
    void load();
};
}

