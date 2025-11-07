#include "Jobs.h"

namespace core {

void Jobs::workerLoop() {
    for (;;) {
        Jobs::Task task;
        {
            std::unique_lock<std::mutex> lock(mutex_);
            cv_.wait(lock, [&]{ return stopRequested_ || !tasks_.empty(); });
            if (stopRequested_ && tasks_.empty()) {
                return;
            }
            task = std::move(tasks_.front());
            tasks_.pop();
            active_.fetch_add(1, std::memory_order_relaxed);
        }

        task();

        active_.fetch_sub(1, std::memory_order_relaxed);
        cv_.notify_all();
    }
}

void Jobs::start(int threads) {
    stop();

    stopRequested_ = false;
    const auto hc = static_cast<int>(std::thread::hardware_concurrency());
    int desired = threads;
    if (desired <= 0) {
        if (hc > 0) {
            int half = hc / 2;
            desired = std::max(1, half - 1);
        } else {
            desired = 1;
        }
    }

    threads_.reserve(static_cast<std::size_t>(desired));
    for (int i = 0; i < desired; ++i) {
        threads_.emplace_back([this]{ workerLoop(); });
    }
    workers_ = threads_.size();
}

void Jobs::schedule(Task task) {
    if (!task) return;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        tasks_.emplace(std::move(task));
    }
    cv_.notify_one();
}

void Jobs::waitIdle() {
    std::unique_lock<std::mutex> lock(mutex_);
    cv_.wait(lock, [&]{ return tasks_.empty() && active_.load(std::memory_order_relaxed) == 0; });
}

void Jobs::stop() {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        stopRequested_ = true;
    }
    cv_.notify_all();
    for (auto& t : threads_) {
        if (t.joinable()) t.join();
    }
    threads_.clear();
    workers_ = 0;

    // Drain remaining tasks if stop() is called while work is pending.
    {
        std::lock_guard<std::mutex> lock(mutex_);
        std::queue<Task> empty;
        std::swap(tasks_, empty);
        active_.store(0, std::memory_order_relaxed);
        stopRequested_ = false;
    }
}

}
