#pragma once

// Jobs — lightweight thread pool for streaming & worldgen tasks.
//
// The pool owns a fixed set of worker threads that execute submitted tasks in FIFO order.
// Tasks are `std::function<void()>` so callers can capture whatever context they need.
// `start()` launches the worker threads (default = hardware_concurrency-1). `schedule()`
// enqueues work, `waitIdle()` blocks until the queue drains, and `stop()` shuts the pool down.

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <functional>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

namespace core {

class Jobs {
public:
    using Task = std::function<void()>;

    // Launch the worker threads. If `threads <= 0`, the pool will spawn
    // max(1, hardware_concurrency-1) workers to leave one core for the render thread.
    void start(int threads = 0);

    // Enqueue a task for asynchronous execution.
    void schedule(Task task);

    // Block until all queued work has finished (useful for shutdown or barriers).
    void waitIdle();

    // Stop workers and flush remaining tasks. Safe to call even if start() failed.
    void stop();

    std::size_t workerCount() const { return workers_; }

private:
    void workerLoop();

    std::vector<std::thread> threads_;
    std::queue<Task> tasks_;
    mutable std::mutex mutex_;
    std::condition_variable cv_;
    std::atomic<int> active_{0};
    bool stopRequested_ = false;
    std::size_t workers_ = 0;
};

}
