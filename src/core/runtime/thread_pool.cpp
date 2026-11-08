#include "core/runtime/thread_pool.h"

#include <algorithm>
#include <exception>

namespace ie {

ThreadPool::ThreadPool(std::size_t threads)
    : thread_count_(threads <= 1 ? 1 : threads) {
    if (thread_count_ <= 1) {
        return;
    }
    workers_.reserve(thread_count_);
    try {
        for (std::size_t i = 0; i < thread_count_; ++i) {
            workers_.emplace_back([this] { worker_loop(); });
        }
    } catch (...) {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stop_ = true;
        }
        work_cv_.notify_all();
        for (std::thread& worker : workers_) {
            if (worker.joinable()) {
                worker.join();
            }
        }
        throw;
    }
}

ThreadPool::~ThreadPool() {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        stop_ = true;
    }
    work_cv_.notify_all();
    for (std::thread& worker : workers_) {
        if (worker.joinable()) {
            worker.join();
        }
    }
}

void ThreadPool::parallel_for(
    std::size_t total, std::size_t grain,
    const std::function<void(std::size_t, std::size_t)>& fn) {
    if (total == 0) {
        return;
    }
    if (thread_count_ <= 1) {
        fn(0, total);
        return;
    }

    std::lock_guard<std::mutex> call_lock(call_mutex_);
    const std::size_t effective_grain =
        grain == 0 ? total : std::min(grain, total);

    {
        std::lock_guard<std::mutex> lock(mutex_);
        job_ = &fn;
        total_ = total;
        grain_ = effective_grain;
        chunk_count_ = total_ / effective_grain +
                       (total_ % effective_grain != 0 ? 1 : 0);
        next_chunk_.store(0, std::memory_order_relaxed);
        active_ = workers_.size();
        ++generation_;
    }
    work_cv_.notify_all();

    std::exception_ptr caller_error;
    try {
        consume_chunks(fn);
    } catch (...) {
        caller_error = std::current_exception();
    }

    {
        std::unique_lock<std::mutex> lock(mutex_);
        done_cv_.wait(lock, [this] { return active_ == 0; });
        job_ = nullptr;
    }

    if (caller_error) {
        std::rethrow_exception(caller_error);
    }
}

void ThreadPool::consume_chunks(
    const std::function<void(std::size_t, std::size_t)>& fn) {
    for (;;) {
        const std::size_t chunk =
            next_chunk_.fetch_add(1, std::memory_order_relaxed);
        if (chunk >= chunk_count_) {
            return;
        }
        const std::size_t begin = chunk * grain_;
        const std::size_t end =
            begin + grain_ < total_ ? begin + grain_ : total_;
        fn(begin, end);
    }
}

void ThreadPool::worker_loop() {
    std::size_t seen_generation = 0;
    for (;;) {
        {
            std::unique_lock<std::mutex> lock(mutex_);
            work_cv_.wait(lock, [this, seen_generation] {
                return stop_ || generation_ != seen_generation;
            });
            if (stop_) {
                return;
            }
            seen_generation = generation_;
        }

        consume_chunks(*job_);

        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (--active_ == 0) {
                done_cv_.notify_all();
            }
        }
    }
}

}  // namespace ie
