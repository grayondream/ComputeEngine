#pragma once

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

namespace ie {

// Deterministic work-sharing pool. `parallel_for(total, grain, fn)` splits
// [0, total) into non-overlapping [begin, end) chunks of at most `grain`
// indices and invokes fn for every chunk exactly once. Results are therefore
// reproducible as long as fn only writes to its own range.
//
// Requests with `threads <= 1` run inline on the caller with no worker
// threads, so accumulating into a shared scalar is race-free in that mode.
//
// parallel_for is non-reentrant: a task must not call parallel_for on the same
// pool. Concurrent calls from different threads are serialized.
//
// Task bodies must not throw. If the task running on the calling thread throws,
// parallel_for lets the remaining round finish and then rethrows, leaving the
// pool usable; an exception escaping a worker task calls std::terminate.
class ThreadPool {
public:
    explicit ThreadPool(std::size_t threads);
    ~ThreadPool();

    ThreadPool(const ThreadPool&) = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;

    void parallel_for(std::size_t total, std::size_t grain,
                      const std::function<void(std::size_t, std::size_t)>& fn);

    std::size_t size() const { return thread_count_; }

private:
    void worker_loop();
    void consume_chunks(
        const std::function<void(std::size_t, std::size_t)>& fn);

    std::vector<std::thread> workers_;
    std::size_t thread_count_{1};

    std::mutex call_mutex_;
    std::mutex mutex_;
    std::condition_variable work_cv_;
    std::condition_variable done_cv_;

    bool stop_{false};
    const std::function<void(std::size_t, std::size_t)>* job_{nullptr};
    std::size_t total_{0};
    std::size_t grain_{1};
    std::size_t chunk_count_{0};
    std::atomic<std::size_t> next_chunk_{0};
    std::size_t active_{0};
    std::size_t generation_{0};
};

}  // namespace ie
