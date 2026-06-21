#pragma once

#include "compute_engine/core/macros.h"
#include <thread>
#include <vector>
#include <deque>
#include <functional>
#include <mutex>
#include <condition_variable>

namespace compute_engine {

class ThreadPool {
public:
    explicit ThreadPool(int num_threads = 0);
    ~ThreadPool();

    void ParallelFor(int64_t begin, int64_t end,
                     const std::function<void(int64_t)>& fn);

    void ParallelFor(int64_t begin, int64_t end,
                     const std::function<void(int64_t, int64_t)>& fn,
                     int64_t grain_size = 1);

    template <typename Iter, typename Func>
    void ParallelForEach(Iter begin, Iter end, Func fn) {
        int64_t total = std::distance(begin, end);
        ParallelFor(0, total, [&](int64_t i) {
            fn(*(begin + i));
        });
    }

    void ParallelFor2D(int64_t h_begin, int64_t h_end,
                       int64_t w_begin, int64_t w_end,
                       const std::function<void(int64_t, int64_t)>& fn);

    int NumThreads() const { return static_cast<int>(threads_.size()); }

    static ThreadPool& Global();

private:
    std::vector<std::thread> threads_;
    std::deque<std::function<void()>> tasks_;
    std::mutex mutex_;
    std::condition_variable cv_;
    bool stop_ = false;

    void WorkerLoop();
};

} // namespace compute_engine
