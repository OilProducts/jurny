#pragma once

// Jobs — thread pool for streaming & worldgen tasks.
namespace core {
class Jobs {
public:
    void start(int threads = 0);
    void stop();
};
}

