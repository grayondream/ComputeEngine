#include <gtest/gtest.h>

#include <atomic>
#include <vector>

#include "core/runtime/thread_pool.h"

using namespace ie;

TEST(ThreadPoolTest, CoversEveryIndexExactlyOnce) {
    ThreadPool pool(4);
    std::vector<int> hits(1000, 0);
    std::atomic<bool> ranges_ok{true};
    pool.parallel_for(1000, 37, [&](std::size_t begin, std::size_t end) {
        if (begin >= end || end > hits.size()) {
            ranges_ok.store(false);
            return;
        }
        for (std::size_t i = begin; i < end; ++i) {
            ++hits[i];
        }
    });
    EXPECT_TRUE(ranges_ok.load());
    for (int hit : hits) {
        EXPECT_EQ(hit, 1);
    }
}

TEST(ThreadPoolTest, EmptyRangeNeverCallsTask) {
    ThreadPool pool(4);
    std::size_t calls = 0;
    pool.parallel_for(0, 16, [&](std::size_t, std::size_t) { ++calls; });
    EXPECT_EQ(calls, 0u);
}

TEST(ThreadPoolTest, ZeroOrOversizedGrainYieldsOneFullChunk) {
    for (std::size_t grain : {std::size_t{0}, std::size_t{1000}}) {
        ThreadPool pool(4);
        std::size_t calls = 0;
        std::size_t chunk_begin = 1;
        std::size_t chunk_end = 0;
        pool.parallel_for(100, grain, [&](std::size_t begin, std::size_t end) {
            ++calls;
            chunk_begin = begin;
            chunk_end = end;
        });
        EXPECT_EQ(calls, 1u) << "grain=" << grain;
        EXPECT_EQ(chunk_begin, 0u) << "grain=" << grain;
        EXPECT_EQ(chunk_end, 100u) << "grain=" << grain;
    }
}

TEST(ThreadPoolTest, SequentialRoundsCoverEveryIndex) {
    ThreadPool pool(4);
    std::vector<int> first(500, 0);
    std::vector<int> second(500, 0);
    pool.parallel_for(500, 31, [&](std::size_t begin, std::size_t end) {
        for (std::size_t i = begin; i < end; ++i) {
            ++first[i];
        }
    });
    for (int hit : first) {
        EXPECT_EQ(hit, 1);
    }
    pool.parallel_for(500, 7, [&](std::size_t begin, std::size_t end) {
        for (std::size_t i = begin; i < end; ++i) {
            ++second[i];
        }
    });
    for (int hit : second) {
        EXPECT_EQ(hit, 1);
    }
}

TEST(ThreadPoolTest, SingleThreadAndZeroThreadFallback) {
    for (std::size_t threads : {std::size_t{0}, std::size_t{1}}) {
        ThreadPool pool(threads);
        EXPECT_EQ(pool.size(), 1u);
        std::size_t sum = 0;
        pool.parallel_for(100, 10, [&](std::size_t begin, std::size_t end) {
            for (std::size_t i = begin; i < end; ++i) {
                sum += i;
            }
        });
        EXPECT_EQ(sum, 4950u);
    }
}
