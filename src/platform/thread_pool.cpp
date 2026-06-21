#include "compute_engine/platform/thread_pool.h"
#include <algorithm>
#include <numeric>
#include "compute_engine/platform/logger.h"

namespace compute_engine {

ThreadPool::ThreadPool(int num_threads) {
    if (num_threads <= 0) {
        num_threads = std::max(1, static_cast<int>(std::thread::hardware_concurrency()));
    }
    for (int i = 0; i < num_threads; ++i) {
        threads_.emplace_back([this] { WorkerLoop(); });
    }
    LOGI("ThreadPool created with {} threads", num_threads);
}

ThreadPool::~ThreadPool() {
    LOGI("ThreadPool destroying");
    {
        std::lock_guard<std::mutex> lock(mutex_);
        stop_ = true;
    }
    cv_.notify_all();
    for (auto& t : threads_) {
        if (t.joinable()) t.join();
    }
}

void ThreadPool::WorkerLoop() {
    while (true) {
        std::function<void()> task;
        {
            std::unique_lock<std::mutex> lock(mutex_);
            cv_.wait(lock, [this] { return stop_ || !tasks_.empty(); });
            if (stop_ && tasks_.empty()) return;
            task = std::move(tasks_.front());
            tasks_.pop_front();
        }
        task();
    }
}

void ThreadPool::ParallelFor(int64_t begin, int64_t end,
                              const std::function<void(int64_t)>& fn) {
    int64_t total = end - begin;
    int num_workers = NumThreads();
    if (total <= 0) return;
    if (num_workers <= 1 || total <= 1) {
        for (int64_t i = begin; i < end; ++i) fn(i);
        return;
    }
    int64_t chunk = (total + num_workers - 1) / num_workers;
    std::atomic<int64_t> counter(0);
    std::atomic<int64_t> done(0);
    for (int w = 0; w < num_workers; ++w) {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            tasks_.push_back([&, begin, chunk, num_workers, w]() {
                while (true) {
                    int64_t idx = counter.fetch_add(1);
                    if (idx >= num_workers) break;
                    int64_t s = begin + idx * chunk;
                    int64_t e = std::min(s + chunk, end);
                    for (int64_t i = s; i < e; ++i) fn(i);
                    done.fetch_add(e - s);
                }
            });
        }
    }
    cv_.notify_all();
    while (done.load() < total) std::this_thread::yield();
}

void ThreadPool::ParallelFor(int64_t begin, int64_t end,
                              const std::function<void(int64_t, int64_t)>& fn,
                              int64_t grain_size) {
    int64_t total = end - begin;
    int num_workers = NumThreads();
    if (total <= 0) return;
    if (num_workers <= 1 || total <= grain_size) {
        fn(begin, end);
        return;
    }
    int64_t chunk = std::max(grain_size, (total + num_workers - 1) / num_workers);
    std::atomic<int64_t> counter(0);
    std::atomic<int64_t> done(0);
    for (int w = 0; w < num_workers; ++w) {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            tasks_.push_back([&, begin, chunk, num_workers]() {
                while (true) {
                    int64_t idx = counter.fetch_add(1);
                    if (idx >= num_workers) break;
                    int64_t s = begin + idx * chunk;
                    int64_t e = std::min(s + chunk, end);
                    fn(s, e);
                    done.fetch_add(e - s);
                }
            });
        }
    }
    cv_.notify_all();
    while (done.load() < total) std::this_thread::yield();
}

void ThreadPool::ParallelFor2D(int64_t h_begin, int64_t h_end,
                                int64_t w_begin, int64_t w_end,
                                const std::function<void(int64_t, int64_t)>& fn) {
    ParallelFor(h_begin, h_end, [&](int64_t h) {
        for (int64_t w = w_begin; w < w_end; ++w) {
            fn(h, w);
        }
    });
}

ThreadPool& ThreadPool::Global() {
    static ThreadPool pool;
    LOGI("Global ThreadPool accessed");
    return pool;
}

} // namespace compute_engine
