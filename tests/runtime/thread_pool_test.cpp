#include <gtest/gtest.h>

#include <vector>

#include "core/runtime/thread_pool.h"

using namespace ie;

TEST(ThreadPoolTest, CoversEveryIndexExactlyOnce) {
    ThreadPool pool(4);
    std::vector<int> hits(1000, 0);
    pool.parallel_for(1000, 37, [&](std::size_t begin, std::size_t end) {
        for (std::size_t i = begin; i < end; ++i) {
            ++hits[i];
        }
    });
    for (int hit : hits) {
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
