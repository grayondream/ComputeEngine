# P1 · CPU 后端与硬件能力探测 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把 CPU 后端从直通标量升级为能力位驱动的正确实现：完整探测 CPU 能力、按能力选择标量/SSE2/AVX2/AVX-512 kernel、打通 KernelVariant 选择，并用 PyTorch 参考验证小 MLP。

**Architecture:** 探测层产出统一的 `HardwareProfile`；核心层用 `KernelVariantRegistry` 按能力位选择变体；CPU 后端把选中的函数指针填入 `KernelFns`，由 MatMul/Relu/Softmax 在 `ExecutionContext` 中调用。SIMD 函数用函数级 `target` 属性编译，运行时按能力位分派，不使用全局 `-march=native`。

**Tech Stack:** C++20、CMake + Ninja、GoogleTest 1.17、PyTorch 2.15（仅用于生成一次参考数据）、std::thread。

**Spec:** `docs/superpowers/specs/m2-backend-core/p1-cpu-hardware.md`（设计权威：`docs/readme.md` §2、§5.2、§6.2、§19）

## Global Constraints

- C++20，`CMAKE_CXX_EXTENSIONS OFF`，命名空间 `ie`；`-Wall -Wextra -Wpedantic` 必须无警告。
- 代码在 `src/`，测试在 `tests/`，文档在 `docs/`；不改 P0.5 已冻结接口的既有签名，只做增量扩展。
- 不引入 MLIR、vcpkg、Google Benchmark、JSON 库或新第三方依赖；基准用仓库内 `std::chrono` 工具。
- 编译期不添加全局 `-march=native`/`-mavx2`/`-mavx512*`；SIMD 函数用 `__attribute__((target(...)))`，运行时用能力位选择。
- 仅支持 F32；非 F32、空/动态 shape、空 buffer 必须返回明确错误，不得崩溃。
- MatMul 归约顺序固定；线程只按输出行切分，不改变单元素求和顺序；结果必须确定。
- 每个 Task 先写失败测试、再实现；结束时 `ctest --test-dir build --output-on-failure` 全绿并独立提交。
- 每新增测试文件必须加入 `tests/CMakeLists.txt`；每新增 `.cpp` 必须加入相应 CMake 目标。

## Review Focus

1. **无 AVX2/AVX-512 的机器**：绝不能执行不支持的指令；选择必须由 `HardwareProfile.caps` 与 `__builtin_cpu_supports` 双重保证，合成低能力 profile 的测试必须只落到标量/SSE2。
2. **非 8/16 整除的 shape**：MatMul/Relu/Softmax 的尾部不能越界读写；测试覆盖 1、7、17、33、65 等尺寸，并在 ASan 下运行。
3. **数值正确性**：每个 SIMD 变体必须与标量 oracle 逐元素对齐；小 MLP 必须与 PyTorch 参考在 `atol=1e-5, rtol=1e-3` 内。
4. **线程确定性**：`parallel_for` 必须覆盖每个输出行恰好一次；多线程与单线程的 MatMul 输出逐位一致。
5. **探测真实性**：能力位、`matrix.*` 布尔值与 `caps` 三者必须一致；AVX-512/AMX 不能被误报导致非法指令。
6. **失败路径**：allocator 缺失、kernel 表为空、shape 不匹配、Softmax 维数不支持时返回错误而非空指针解引用。

---

## 文件结构

```text
src/core/hardware/          detector.h/.cpp（扩展 CpuFeatures/CpuTopology/build_cpu_profile）
src/core/runtime/           thread_pool.h/.cpp（新增）；execution_context.h（增量字段）；executor.h/.cpp（增量参数）
src/core/ops/               kernel_fns.h（新增）；common/relu_op.h、softmax_op.h（新增）；common/matmul_op.h（改）
src/core/tuning/            cpu_kernel_variants.h（新增：注册/选择 CPU 变体）
src/core/hardware/          optimization_planner.cpp（改：能力驱动 preferred_kernels）
src/backends/cpu/           cpu_allocator.h（新增）；cpu_backend.h/.cpp（改）
src/backends/cpu/simd/      simd_kernels.h/.cpp（新增）
src/backends/cpu/kernels/   cpu_kernel_catalog.h/.cpp（新增）
src/benchmarks/             ie_bench.cpp（新增）
tests/hardware/             hardware_test.cpp（扩展）、cpu_kernel_catalog_test.cpp（新增）
tests/memory/               cpu_allocator_test.cpp（新增）
tests/runtime/              thread_pool_test.cpp、mlp_e2e_test.cpp（新增）
tests/ops/                  ops_test.cpp（扩展）、activation_test.cpp、simd_kernels_test.cpp（新增）
tests/backend/              backend_test.cpp（扩展）
tests/performance/          matmul_benchmark_test.cpp（新增）
tests/reference/            gen_mlp_reference.py（新增）
tests/data/                 mlp_reference.h（生成并提交）
```

---

### Task 1: 硬件探测平台化

**Files:**
- Modify: `src/core/hardware/detector.h`、`src/core/hardware/detector.cpp`
- Test: `tests/hardware/hardware_test.cpp`

**Interfaces:**
- Consumes: `HardwareProfile`、`CapabilitySet`、`Status`/`Result<T>`
- Produces:
  - `struct CpuFeatures`：`sse2/avx/avx2/fma/avx512f/avx512bw/avx512vl/avx512_vnni/avx512_bf16/avx512_fp16/amx_bf16/amx_int8/amx_fp16/neon/sve/sve2/sme/sme2`，全部 `false`。
  - `struct CpuTopology`：`vendor/device_name/num_cores/threads_per_core/max_threads_per_core/l2_cache_size/l3_cache_size/num_memory_channels/total_memory/available_memory/numa_nodes`。
  - `static HardwareProfile HardwareDetector::build_cpu_profile(const CpuFeatures&, const CpuTopology&);`
  - `detect_cpu()` 仍返回 `Result<HardwareProfile>`，内部探测后调用 `build_cpu_profile`。
  - 能力位规范名：`sse2`、`avx`、`avx2`、`fma`、`avx512f`、`avx512bw`、`avx512vl`、`avx512vnni`、`avx512bf16`、`avx512fp16`、`amx-bf16`、`amx-int8`、`amx-fp16`、`neon`、`sve`、`sve2`、`sme`、`sme2`。

- [ ] **Step 1: 写失败测试**

```cpp
#include <gtest/gtest.h>

#include "core/hardware/detector.h"

using namespace ie;

TEST(CpuProfileBuildTest, MirrorsFeaturesAndTopology) {
    CpuFeatures f;
    f.sse2 = f.avx = f.avx2 = f.fma = true;
    f.avx512f = f.avx512bw = f.avx512vl = f.avx512_vnni = true;
    f.amx_bf16 = true;
    CpuTopology t;
    t.vendor = "amd";
    t.device_name = "test-cpu";
    t.num_cores = 6;
    t.threads_per_core = 2;
    t.max_threads_per_core = 2;
    t.l2_cache_size = 3u << 20;
    t.l3_cache_size = 32u << 20;
    t.num_memory_channels = 2;
    t.total_memory = 16ull << 30;
    t.available_memory = 8ull << 30;
    t.numa_nodes = 1;

    const HardwareProfile p = HardwareDetector::build_cpu_profile(f, t);
    EXPECT_EQ(p.vendor, "amd");
    EXPECT_EQ(p.device_name, "test-cpu");
    EXPECT_EQ(p.cu.num_cores, 6);
    EXPECT_EQ(p.cu.threads_per_core, 2);
    EXPECT_EQ(p.cu.l2_cache_size, 3u << 20);
    EXPECT_EQ(p.cu.l3_cache_size, 32u << 20);
    EXPECT_EQ(p.mem.vram_total, 16ull << 30);
    EXPECT_EQ(p.mem.vram_available, 8ull << 30);
    EXPECT_EQ(p.deploy.numa_nodes, 1);
    EXPECT_TRUE(p.matrix.has_sse2);
    EXPECT_TRUE(p.matrix.has_avx2);
    EXPECT_TRUE(p.matrix.has_avx512_vnni);
    EXPECT_TRUE(p.matrix.has_amx_bf16);
    EXPECT_TRUE(p.caps.has("avx2"));
    EXPECT_TRUE(p.caps.has("avx512vnni"));
    EXPECT_TRUE(p.caps.has("amx-bf16"));
    EXPECT_TRUE(p.matrix.has_vector_unit);
    EXPECT_TRUE(p.matrix.has_matrix_unit);
    EXPECT_FALSE(p.fingerprint.empty());
}

TEST(CpuProfileBuildTest, DisabledFeaturesHaveNoCapsOrFlags) {
    CpuFeatures f;
    CpuTopology t;
    t.num_cores = 1;
    t.threads_per_core = 1;
    t.numa_nodes = 1;
    const HardwareProfile p = HardwareDetector::build_cpu_profile(f, t);
    EXPECT_FALSE(p.matrix.has_sse2);
    EXPECT_FALSE(p.matrix.has_avx2);
    EXPECT_FALSE(p.caps.has("avx2"));
    EXPECT_FALSE(p.matrix.has_vector_unit);
    EXPECT_FALSE(p.matrix.has_matrix_unit);
    EXPECT_EQ(p.cu.num_cores, 1);
}

TEST(HardwareDetectorTest, DetectCpuIsConsistentAndStable) {
    auto first = HardwareDetector::detect_cpu();
    auto second = HardwareDetector::detect_cpu();
    ASSERT_TRUE(first.has_value());
    ASSERT_TRUE(second.has_value());
    EXPECT_EQ(first->fingerprint, second->fingerprint);
    EXPECT_GT(first->cu.num_cores, 0);
    EXPECT_EQ(first->matrix.has_sse2, first->caps.has("sse2"));
    EXPECT_EQ(first->matrix.has_avx2, first->caps.has("avx2"));
    EXPECT_EQ(first->matrix.has_avx512_vnni,
              first->caps.has("avx512vnni"));
}
```

- [ ] **Step 2: 跑测试确认失败**

Run: `cmake --build build`
Expected: 编译失败（`CpuFeatures`/`build_cpu_profile` 不存在）。

- [ ] **Step 3: 实现**

`detector.h` 增加 `CpuFeatures`、`CpuTopology` 与 `build_cpu_profile`。`build_cpu_profile` 逐项把 feature 写入 `matrix.*` 与 `caps.set(name, true)`，并设置 `has_vector_unit = sse2 || neon`、`has_matrix_unit = avx512_vnni || amx_int8 || amx_bf16`、`matrix_unit_generation`（VNNI=1，AMX=2，否则 0）。
`detect_cpu` 的 x86 路径用 `__builtin_cpu_supports` 探测标准指令集（含 `fma`），AMX 用 CPUID leaf 7（`__get_cpuid_count`）补齐；ARM 路径用 `getauxval(AT_HWCAP/AT_HWCAP2)` 探测 NEON/SVE/SVE2/SME/SME2。Linux 下从 `/sys/devices/system/cpu/cpu0/cache/index*/` 读 L2/L3，从 `/proc/meminfo` 读总/可用内存，从 `/sys/devices/system/node/online` 读 NUMA 节点；非 Linux 回退 `hardware_concurrency()` 与 1 个 NUMA 节点。vendor/device_name 用 `/proc/cpuinfo` 的 `vendor_id`/`model name`，失败时 `"generic"`。
`make_fingerprint` 纳入 vendor/device、核心数、L2/L3、全部启用的 caps。

- [ ] **Step 4: 跑测试确认通过**

Run: `cmake --build build && ctest --test-dir build --output-on-failure`
Expected: 全部通过。

- [ ] **Step 5: 提交**

```bash
git add src/core/hardware/detector.h src/core/hardware/detector.cpp tests/hardware/hardware_test.cpp
git commit -m "feat(p1): platformized cpu capability detection"
```

---

### Task 2: CPU 分配器与确定性线程池

**Files:**
- Create: `src/backends/cpu/cpu_allocator.h`
- Create: `src/core/runtime/thread_pool.h`、`src/core/runtime/thread_pool.cpp`
- Modify: `CMakeLists.txt`
- Test: `tests/memory/cpu_allocator_test.cpp`、`tests/runtime/thread_pool_test.cpp`

**Interfaces:**
- Consumes: `Allocator`、`MemoryKind`、`Result<T>`
- Produces:
  - `class CpuAllocator final : public Allocator`：`static constexpr std::size_t kAlignment = 64;`、`alloc/free/available`、`outstanding_blocks()`、`outstanding_bytes()`。
  - `class ThreadPool`：
    ```cpp
    explicit ThreadPool(std::size_t threads);
    ~ThreadPool();
    ThreadPool(const ThreadPool&) = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;
    void parallel_for(std::size_t total, std::size_t grain,
                      const std::function<void(std::size_t, std::size_t)>& fn);
    std::size_t size() const;
    ```
  - `threads == 0` 时按 1 个线程处理；`fn` 处理互不重叠的 `[begin,end)`。

- [ ] **Step 1: 写失败测试**

```cpp
#include <gtest/gtest.h>
#include <cstdint>
#include <vector>

#include "backends/cpu/cpu_allocator.h"
#include "core/runtime/thread_pool.h"

using namespace ie;

TEST(CpuAllocatorTest, AlignsAndTracksBytes) {
    CpuAllocator alloc;
    auto ptr = alloc.alloc(65, MemoryKind::HOST);
    ASSERT_TRUE(ptr.has_value());
    EXPECT_EQ(reinterpret_cast<std::uintptr_t>(*ptr) % CpuAllocator::kAlignment,
              0u);
    EXPECT_EQ(alloc.outstanding_blocks(), 1u);
    EXPECT_GE(alloc.outstanding_bytes(), 65u);
    alloc.free(*ptr);
    EXPECT_EQ(alloc.outstanding_blocks(), 0u);
    EXPECT_EQ(alloc.outstanding_bytes(), 0u);
    alloc.free(*ptr);
    alloc.free(nullptr);
}

TEST(CpuAllocatorTest, RejectsZeroBytes) {
    CpuAllocator alloc;
    auto ptr = alloc.alloc(0, MemoryKind::HOST);
    ASSERT_FALSE(ptr.has_value());
    EXPECT_EQ(ptr.error().code, Error::Code::InvalidArgument);
}

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
```

- [ ] **Step 2: 跑测试确认失败**

Run: `cmake --build build`
Expected: 编译失败（`cpu_allocator.h`/`thread_pool.h` 不存在）。

- [ ] **Step 3: 实现**

`CpuAllocator` 用 `posix_memalign`（Windows 用 `_aligned_malloc`）分配 64 字节对齐块，向上取整到 64 的倍数以便统计；用 mutex + `unordered_map<void*, size_t>` 跟踪，`free(nullptr)`/重复释放安全；`alloc(0)` 返回 `InvalidArgument`。
`ThreadPool` 构造时启动 `max(1, threads)` 个工作线程；`threads <= 1` 时 `parallel_for` 直接在调用线程执行 `fn(0,total)`，不启动 worker（测试里的累加因此无数据竞争）。多线程时用原子 chunk 索引把 `[0,total)` 按 `grain` 切成块，工作线程与调用线程共同消费；用 mutex/condition_variable 等待本轮 `active` 归零。一次只允许一个 `parallel_for`，析构时置停止位、唤醒并 join。结果只依赖调用方对不重叠区间的写入，因此确定。

- [ ] **Step 4: 跑测试确认通过**

Run: `cmake --build build && ctest --test-dir build --output-on-failure`
Expected: 全部通过。

- [ ] **Step 5: 提交**

```bash
git add src/backends/cpu/cpu_allocator.h src/core/runtime/thread_pool.h src/core/runtime/thread_pool.cpp CMakeLists.txt tests/memory/cpu_allocator_test.cpp tests/runtime/thread_pool_test.cpp
git commit -m "feat(p1): add aligned cpu allocator and deterministic thread pool"
```

---

### Task 3: SIMD 内核、KernelFns 与能力目录

**Files:**
- Create: `src/core/ops/kernel_fns.h`
- Create: `src/core/tuning/cpu_kernel_variants.h`
- Create: `src/backends/cpu/simd/simd_kernels.h`、`src/backends/cpu/simd/simd_kernels.cpp`
- Create: `src/backends/cpu/kernels/cpu_kernel_catalog.h`、`src/backends/cpu/kernels/cpu_kernel_catalog.cpp`
- Modify: `CMakeLists.txt`（`simd_kernels.cpp`、`cpu_kernel_catalog.cpp` 加入 `ie_cpu`）
- Test: `tests/ops/simd_kernels_test.cpp`、`tests/hardware/cpu_kernel_catalog_test.cpp`

**Interfaces:**
- Consumes: `HardwareProfile`、`KernelVariantRegistry`、`KernelVariant`、`OpType`、`DType`、`Shape`
- Produces:
  - `struct KernelFns`：
    ```cpp
    using MatMulRowFn = void (*)(const float* a_row, const float* b,
                                 float* out_row, std::int64_t k, std::int64_t n);
    using UnaryFn = void (*)(const float* in, float* out, std::int64_t n);
    using SoftmaxFn = void (*)(const float* in, float* out,
                               std::int64_t rows, std::int64_t cols);
    MatMulRowFn matmul_row{nullptr};
    UnaryFn relu{nullptr};
    SoftmaxFn softmax{nullptr};
    ```
  - `ie::cpu` 函数：`matmul_row_scalar/sse2/avx2/avx512`、`relu_scalar/sse2/avx2/avx512`、`softmax_scalar/avx2/avx512`。
  - `register_cpu_kernel_variants(KernelVariantRegistry&)`；变体名 `matmul_scalar/sse2/avx2/avx512`、`relu_*`、`softmax_*`。
  - `Result<KernelVariant> select_best_cpu_variant(OpType, DType, const Shape&, const HardwareProfile&);`（按 `cost.estimated_tflops` 取最大匹配）
  - `class CpuKernelCatalog`：
    ```cpp
    struct Selection { std::string matmul, relu, softmax; KernelFns fns; };
    static Result<Selection> select(const HardwareProfile&);
    ```
  - 变体前置条件：`sse2` 需 `sse2`；`avx2` 需 `avx2 + fma`；`avx512` 需 `avx512f + avx512bw + avx512vl`；`scalar` 无前置。`cost.estimated_tflops` 依次 1/2/4/8。

- [ ] **Step 1: 写失败测试**

```cpp
#include <gtest/gtest.h>
#include <cstdint>
#include <string_view>
#include <tuple>
#include <vector>

#include "backends/cpu/kernels/cpu_kernel_catalog.h"
#include "backends/cpu/simd/simd_kernels.h"

using namespace ie;

namespace {
HardwareProfile profile_with(std::initializer_list<const char*> caps) {
    HardwareProfile p;
    for (const char* cap : caps) {
        p.caps.set(cap, true);
    }
    return p;
}

bool has(std::string_view feature) {
#if defined(__x86_64__) || defined(__i386__)
    if (feature == "sse2") return __builtin_cpu_supports("sse2");
    if (feature == "avx2") return __builtin_cpu_supports("avx2");
    if (feature == "fma") return __builtin_cpu_supports("fma");
    if (feature == "avx512f") return __builtin_cpu_supports("avx512f");
    if (feature == "avx512bw") return __builtin_cpu_supports("avx512bw");
    if (feature == "avx512vl") return __builtin_cpu_supports("avx512vl");
    return false;
#else
    (void)feature;
    return false;
#endif
}
}  // namespace

TEST(CpuKernelCatalogTest, SelectsBestVariantByCaps) {
    Shape shape{{16, 16}};
    auto scalar = select_best_cpu_variant(OpType::MatMul, DType::F32, shape,
                                          HardwareProfile{});
    ASSERT_TRUE(scalar.has_value());
    EXPECT_EQ(scalar->name, "matmul_scalar");

    auto sse2 = select_best_cpu_variant(
        OpType::MatMul, DType::F32, shape, profile_with({"sse2"}));
    ASSERT_TRUE(sse2.has_value());
    EXPECT_EQ(sse2->name, "matmul_sse2");

    auto avx2 = select_best_cpu_variant(
        OpType::MatMul, DType::F32, shape, profile_with({"avx2", "fma"}));
    ASSERT_TRUE(avx2.has_value());
    EXPECT_EQ(avx2->name, "matmul_avx2");

    auto avx512 = select_best_cpu_variant(
        OpType::MatMul, DType::F32, shape,
        profile_with({"avx2", "fma", "avx512f", "avx512bw", "avx512vl"}));
    ASSERT_TRUE(avx512.has_value());
    EXPECT_EQ(avx512->name, "matmul_avx512");
}

TEST(CpuKernelCatalogTest, Avx2RequiresFmaAndAvx512RequiresBwVl) {
    Shape shape{{16, 16}};
    auto no_fma = select_best_cpu_variant(
        OpType::MatMul, DType::F32, shape, profile_with({"sse2", "avx2"}));
    ASSERT_TRUE(no_fma.has_value());
    EXPECT_EQ(no_fma->name, "matmul_sse2");

    auto partial_512 = select_best_cpu_variant(
        OpType::MatMul, DType::F32, shape, profile_with({"avx2", "fma", "avx512f"}));
    ASSERT_TRUE(partial_512.has_value());
    EXPECT_EQ(partial_512->name, "matmul_avx2");
}

TEST(CpuKernelCatalogTest, SelectionFillsFunctionPointers) {
    auto selection = CpuKernelCatalog::select(profile_with({"sse2"}));
    ASSERT_TRUE(selection.has_value());
    EXPECT_EQ(selection->matmul, "matmul_sse2");
    EXPECT_NE(selection->fns.matmul_row, nullptr);
    EXPECT_NE(selection->fns.relu, nullptr);
    EXPECT_NE(selection->fns.softmax, nullptr);
}

TEST(SimdKernelsTest, MatMulVariantsMatchScalar) {
    const std::vector<std::tuple<int64_t, int64_t, int64_t>> shapes = {
        {1, 1, 1}, {3, 5, 7}, {17, 33, 65}, {64, 64, 64}};
    std::uint32_t state = 12345;
    auto next = [&] {
        state = state * 1664525u + 1013904223u;
        return static_cast<float>((state >> 8) & 0xffff) / 65535.0f - 0.5f;
    };
    for (auto [m, k, n] : shapes) {
        std::vector<float> a(m * k), b(k * n), expected(m * n), actual(m * n);
        for (float& v : a) v = next();
        for (float& v : b) v = next();
        for (int64_t i = 0; i < m; ++i) {
            cpu::matmul_row_scalar(a.data() + i * k, b.data(), expected.data() + i * n, k, n);
        }
        if (has("sse2")) {
            for (int64_t i = 0; i < m; ++i) {
                cpu::matmul_row_sse2(a.data() + i * k, b.data(), actual.data() + i * n, k, n);
            }
            for (int64_t j = 0; j < m * n; ++j) {
                EXPECT_NEAR(actual[j], expected[j], 1e-4f);
            }
        }
        if (has("avx2") && has("fma")) {
            for (int64_t i = 0; i < m; ++i) {
                cpu::matmul_row_avx2(a.data() + i * k, b.data(), actual.data() + i * n, k, n);
            }
            for (int64_t j = 0; j < m * n; ++j) {
                EXPECT_NEAR(actual[j], expected[j], 1e-4f);
            }
        }
        if (has("avx512f") && has("avx512bw") && has("avx512vl")) {
            for (int64_t i = 0; i < m; ++i) {
                cpu::matmul_row_avx512(a.data() + i * k, b.data(), actual.data() + i * n, k, n);
            }
            for (int64_t j = 0; j < m * n; ++j) {
                EXPECT_NEAR(actual[j], expected[j], 1e-4f);
            }
        }
    }
}

TEST(SimdKernelsTest, ReluAndSoftmaxVariantsMatchScalar) {
    std::vector<float> in = {-3.0f, -0.0f, 0.0f, 2.0f, 1000.0f, -1000.0f, 7.0f};
    std::vector<float> relu_expected(in.size()), relu_actual(in.size());
    cpu::relu_scalar(in.data(), relu_expected.data(), in.size());
    if (has("avx2") && has("fma")) {
        cpu::relu_avx2(in.data(), relu_actual.data(), in.size());
        EXPECT_EQ(relu_actual, relu_expected);
    }
    if (has("avx512f") && has("avx512bw") && has("avx512vl")) {
        cpu::relu_avx512(in.data(), relu_actual.data(), in.size());
        EXPECT_EQ(relu_actual, relu_expected);
    }

    std::vector<float> softmax_expected(in.size()), softmax_actual(in.size());
    cpu::softmax_scalar(in.data(), softmax_expected.data(), 1, in.size());
    if (has("avx2") && has("fma")) {
        cpu::softmax_avx2(in.data(), softmax_actual.data(), 1, in.size());
        for (std::size_t i = 0; i < in.size(); ++i) {
            EXPECT_NEAR(softmax_actual[i], softmax_expected[i], 1e-6f);
        }
    }
    if (has("avx512f") && has("avx512bw") && has("avx512vl")) {
        cpu::softmax_avx512(in.data(), softmax_actual.data(), 1, in.size());
        for (std::size_t i = 0; i < in.size(); ++i) {
            EXPECT_NEAR(softmax_actual[i], softmax_expected[i], 1e-6f);
        }
    }
}
```

- [ ] **Step 2: 跑测试确认失败**

Run: `cmake --build build`
Expected: 编译失败（`cpu_kernel_catalog.h`/`simd_kernels.h` 不存在）。

- [ ] **Step 3: 实现**

`cpu_kernel_variants.h` 用 `KernelVariantRegistry` 注册 12 个变体（3 op × 4 档），`target_kind=UNKNOWN`，按上表设置 `req.required_caps` 与 `cost.estimated_tflops`；`select_best_cpu_variant` 调 `select` 后取 `estimated_tflops` 最大者，空匹配返回 `NotFound`。
`simd_kernels.cpp`：
- 标量 MatMul 行：`for j: acc=0; for t: acc += a[t]*b[t*n+j]; out[j]=acc`。
- SSE2：每次处理 4 个 `j`，`_mm_loadu_ps`/`mul_ps`/`add_ps`，尾部标量。
- AVX2：`target("avx2,fma")`，每次 8 个 `j`，`_mm256_fmadd_ps`，尾部标量。
- AVX-512：`target("avx512f,avx512bw,avx512vl")`，每次 16 个 `j`，`_mm512_fmadd_ps`，尾部标量。
- Relu 同理；Softmax 逐行：先求 max（AVX2/AVX512 向量规约），再 `std::exp(float(x-max))` 求和，最后除法；`cols<=0` 直接返回。
`cpu_kernel_catalog.cpp` 的静态 registry 调用 `register_cpu_kernel_variants`；`select` 对 MatMul/Relu/Softmax 分别选变体，按名称映射到函数指针。非 x86 或编译器不支持 target 属性时，AVX 函数定义为转发到标量的安全占位（能力位不会命中），保证头文件与测试跨平台可编译。

- [ ] **Step 4: 跑测试确认通过**

Run: `cmake --build build && ctest --test-dir build --output-on-failure`
Expected: 全部通过；AVX-512 用例在无该指令集的机器上跳过，AVX2 用例在本机执行。

- [ ] **Step 5: 提交**

```bash
git add src/core/ops/kernel_fns.h src/core/tuning/cpu_kernel_variants.h src/backends/cpu/simd src/backends/cpu/kernels CMakeLists.txt tests/ops/simd_kernels_test.cpp tests/hardware/cpu_kernel_catalog_test.cpp
git commit -m "feat(p1): add simd kernels and capability-driven kernel catalog"
```

---

### Task 4: Relu/Softmax 算子、MatMul 分派与优化计划

**Files:**
- Create: `src/core/ops/common/relu_op.h`、`src/core/ops/common/softmax_op.h`
- Modify: `src/core/ops/common/matmul_op.h`、`src/core/ops/op_registry.cpp`
- Modify: `src/core/runtime/execution_context.h`、`src/core/runtime/executor.h`、`src/core/runtime/executor.cpp`
- Modify: `src/core/hardware/optimization_planner.cpp`
- Test: `tests/ops/activation_test.cpp`、`tests/ops/ops_test.cpp`、`tests/hardware/hardware_test.cpp`

**Interfaces:**
- Consumes: `KernelFns`、`ThreadPool`、`HardwareProfile`、`KernelVariant`
- Produces:
  - `ExecutionContext` 增字段：`const KernelFns* kernels{nullptr}; ThreadPool* thread_pool{nullptr};`
  - `Executor::run(..., std::shared_ptr<Allocator> allocator, const KernelFns* kernels = nullptr, ThreadPool* pool = nullptr)`。
  - `class ReluOp final : public Op`：1 输入 1 输出，shape 不变；优先 `context.kernels->relu`，否则标量。
  - `class SoftmaxOp final : public Op`：1 输入 1 输出，支持 1D 与 2D（按最后一维）；优先 `context.kernels->softmax`，否则标量。
  - `MatMulOp` 优先 `context.kernels->matmul_row`；当 `pool != nullptr && m * n * k >= 32768` 时按行 `parallel_for`，否则单线程。
  - `OptimizationPlanner::Plan.preferred_kernels` 为 `{matmul_variant, relu_variant, softmax_variant}`，`reason` 以 `"capability-driven:"` 开头。

- [ ] **Step 1: 写失败测试**

```cpp
#include <gtest/gtest.h>
#include <algorithm>
#include <atomic>
#include <cmath>
#include <memory>

#include "core/hardware/optimization_planner.h"
#include "core/memory/malloc_allocator.h"
#include "core/ops/common/matmul_op.h"
#include "core/ops/common/relu_op.h"
#include "core/ops/common/softmax_op.h"
#include "core/runtime/executor.h"

using namespace ie;

TEST(ReluOpTest, AppliesElementwiseAndValidates) {
    auto alloc = std::make_shared<MallocAllocator>();
    ExecutionContext ctx;
    ctx.allocator = alloc;
    ReluOp op;
    Tensor in = Tensor::from_data({-2.0f, -0.0f, 0.0f, 3.0f});
    Tensor out;
    std::vector<Tensor*> ins{&in}, outs{&out};
    ASSERT_TRUE(op.forward(ins, outs, ctx).has_value());
    EXPECT_FLOAT_EQ(out.data<float>()[0], 0.0f);
    EXPECT_FLOAT_EQ(out.data<float>()[3], 3.0f);

    std::vector<Tensor*> empty;
    EXPECT_EQ(op.forward(empty, outs, ctx).error().code,
              Error::Code::InvalidArgument);
}

TEST(SoftmaxOpTest, NormalizesLastAxisAndIsStable) {
    auto alloc = std::make_shared<MallocAllocator>();
    ExecutionContext ctx;
    ctx.allocator = alloc;
    SoftmaxOp op;
    Tensor in = Tensor::from_data({1000.0f, 1000.0f, 1000.0f, 1000.0f});
    in.shape = Shape{{2, 2}};
    Tensor out;
    std::vector<Tensor*> ins{&in}, outs{&out};
    ASSERT_TRUE(op.forward(ins, outs, ctx).has_value());
    EXPECT_NEAR(out.data<float>()[0], 0.5f, 1e-6f);
    EXPECT_NEAR(out.data<float>()[2], 0.5f, 1e-6f);
    EXPECT_TRUE(std::isfinite(out.data<float>()[0]));
}

namespace {
std::atomic<int> g_matmul_calls{0};
void fake_matmul_row(const float*, const float*, float* out, int64_t, int64_t n) {
    ++g_matmul_calls;
    for (int64_t j = 0; j < n; ++j) out[j] = 1.0f;
}
}  // namespace

TEST(MatMulOpTest, UsesInjectedKernelAndThreadsRows) {
    auto alloc = std::make_shared<MallocAllocator>();
    ThreadPool pool(4);
    KernelFns fns;
    fns.matmul_row = fake_matmul_row;
    ExecutionContext ctx;
    ctx.allocator = alloc;
    ctx.kernels = &fns;
    ctx.thread_pool = &pool;

    auto a_alloc = Tensor::allocate(Shape{{64, 64}}, DType::F32, alloc);
    ASSERT_TRUE(a_alloc.has_value());
    std::fill(a_alloc->data<float>(), a_alloc->data<float>() + 64 * 64, 1.0f);
    Tensor a = std::move(*a_alloc);
    Tensor b = a;
    Tensor out;
    std::vector<Tensor*> ins{&a, &b}, outs{&out};
    g_matmul_calls = 0;
    MatMulOp op;
    ASSERT_TRUE(op.forward(ins, outs, ctx).has_value());
    EXPECT_EQ(g_matmul_calls.load(), 64);
    EXPECT_FLOAT_EQ(out.data<float>()[0], 1.0f);
}

TEST(OptimizationPlannerTest, SelectsKernelsFromCaps) {
    EngineConfig cfg;
    Graph g;
    HardwareProfile scalar;
    auto p0 = OptimizationPlanner::plan(scalar, g, ModelKind::UNKNOWN, cfg);
    EXPECT_EQ(p0.preferred_kernels.size(), 3u);
    EXPECT_EQ(p0.preferred_kernels[0], "matmul_scalar");

    HardwareProfile avx2;
    avx2.caps.set("avx2", true);
    avx2.caps.set("fma", true);
    auto p2 = OptimizationPlanner::plan(avx2, g, ModelKind::UNKNOWN, cfg);
    EXPECT_EQ(p2.preferred_kernels[0], "matmul_avx2");

    HardwareProfile avx512;
    avx512.caps.set("avx2", true);
    avx512.caps.set("fma", true);
    avx512.caps.set("avx512f", true);
    avx512.caps.set("avx512bw", true);
    avx512.caps.set("avx512vl", true);
    auto p5 = OptimizationPlanner::plan(avx512, g, ModelKind::UNKNOWN, cfg);
    EXPECT_EQ(p5.preferred_kernels[0], "matmul_avx512");
}
```

- [ ] **Step 2: 跑测试确认失败**

Run: `cmake --build build`
Expected: 编译失败（`relu_op.h`/`softmax_op.h` 不存在）。

- [ ] **Step 3: 实现**

`execution_context.h` 增加 `kernels`/`thread_pool` 字段（前向声明 `ThreadPool`，包含 `kernel_fns.h`）。`Executor::run` 把参数写入 context。
`MatMulOp` 保留现有校验，分配输出后：若 `context.kernels && context.kernels->matmul_row`，每个输出行调用该函数；否则调用本地标量行函数。线程条件与切分按 Interfaces。
`ReluOp`/`SoftmaxOp` 校验输入/输出非空、F32、allocator；Softmax 仅接受 1D/2D，2D 逐行；`std::exp` 用 `float`，减 max 保证稳定。
`op_registry.cpp` 注册 `OpType::Relu`、`OpType::Softmax`。
`OptimizationPlanner` 用 `select_best_cpu_variant` 为三个 op 取名（依赖 Task 3 的头），失败时回退 `*_scalar`；`reason = "capability-driven:" + matmul name`。

- [ ] **Step 4: 跑测试确认通过**

Run: `cmake --build build && ctest --test-dir build --output-on-failure`
Expected: 全部通过。

- [ ] **Step 5: 提交**

```bash
git add src/core/ops/common/relu_op.h src/core/ops/common/softmax_op.h src/core/ops/common/matmul_op.h src/core/ops/op_registry.cpp src/core/runtime/execution_context.h src/core/runtime/executor.h src/core/runtime/executor.cpp src/core/hardware/optimization_planner.cpp tests/ops/activation_test.cpp tests/ops/ops_test.cpp tests/hardware/hardware_test.cpp
git commit -m "feat(p1): add relu/softmax ops and capability-driven matmul dispatch"
```

---

### Task 5: CpuBackend 集成

**Files:**
- Modify: `src/backends/cpu/cpu_backend.h`、`src/backends/cpu/cpu_backend.cpp`
- Modify: `tests/backend/backend_test.cpp`
- Modify: `CMakeLists.txt`（如需新增源文件）

**Interfaces:**
- Consumes: `HardwareDetector`、`CpuKernelCatalog`、`CpuAllocator`、`ThreadPool`、`KernelFns`
- Produces:
  - `CpuBackend` 成员：`HardwareProfile profile_`、`std::unique_ptr<ThreadPool> pool_`、`std::shared_ptr<CpuAllocator> allocator_`、`KernelFns kernels_`、`bool initialized_`。
  - `init()`：幂等；探测 profile、创建线程池（线程数 = `num_cores * threads_per_core`，至少 1）、调用 `CpuKernelCatalog::select` 填充 kernels_。
  - `shutdown()`：幂等；释放池与图，`initialized_ = false`。
  - `hardware_profile()` 返回 `profile_`；`caps()` 置 `supports_deterministic_kernels=true`、`supports_batch_invariant=true`。
  - `execute_async` 把 `&kernels_`、`pool_.get()` 传给 `Executor::run`。

- [ ] **Step 1: 写失败测试**

```cpp
#include <gtest/gtest.h>
#include <cstdint>

#include "backends/cpu/cpu_backend.h"

using namespace ie;

TEST(CpuBackendTest, InitDetectsProfileAndIsIdempotent) {
    CpuBackend backend;
    ASSERT_TRUE(backend.init().has_value());
    ASSERT_TRUE(backend.init().has_value());
    auto profile = backend.hardware_profile();
    ASSERT_TRUE(profile.has_value());
    EXPECT_GT(profile->cu.num_cores, 0);
    auto caps = backend.caps();
    EXPECT_TRUE(caps.supports_deterministic_kernels);
    EXPECT_TRUE(caps.supports_batch_invariant);
    ASSERT_TRUE(backend.shutdown().has_value());
    ASSERT_TRUE(backend.shutdown().has_value());
}

TEST(CpuBackendTest, AllocIsAlignedAndFreeIsSafe) {
    CpuBackend backend;
    ASSERT_TRUE(backend.init().has_value());
    auto ptr = backend.alloc(65);
    ASSERT_TRUE(ptr.has_value());
    EXPECT_EQ(reinterpret_cast<std::uintptr_t>(*ptr) % 64, 0u);
    backend.free(*ptr);
    backend.free(*ptr);
    backend.free(nullptr);
}
```

- [ ] **Step 2: 跑测试确认失败**

Run: `cmake --build build`
Expected: 编译失败或断言失败（`caps` 仍为默认、`alloc` 未对齐）。

- [ ] **Step 3: 实现**

`CpuBackend` 构造时创建 `CpuAllocator`；`init` 用 `HardwareDetector::detect_cpu()`，失败则返回错误；`CpuKernelCatalog::select` 失败则返回错误；线程数从 profile 计算并钳制到 `[1, 64]`。
`alloc/free` 走 `CpuAllocator`。`execute_async` 继续使用 `context.allocator`（若有），但把 `&kernels_`/`pool_.get()` 传入 Executor；`compile` 不变。
`shutdown` 先 `pool_.reset()`，再清 graph，避免线程引用已销毁对象。

- [ ] **Step 4: 跑测试确认通过**

Run: `cmake --build build && ctest --test-dir build --output-on-failure`
Expected: 全部通过（含 P0/P0.5 回归）。

- [ ] **Step 5: 提交**

```bash
git add src/backends/cpu/cpu_backend.h src/backends/cpu/cpu_backend.cpp tests/backend/backend_test.cpp CMakeLists.txt
git commit -m "feat(p1): wire cpu profile kennels and thread pool into backend"
```

---

### Task 6: PyTorch 参考 MLP、端到端对齐与基准

**Files:**
- Create: `tests/reference/gen_mlp_reference.py`
- Create: `tests/data/mlp_reference.h`（由脚本生成并提交）
- Create: `tests/runtime/mlp_e2e_test.cpp`
- Create: `tests/performance/matmul_benchmark_test.cpp`
- Create: `src/benchmarks/ie_bench.cpp`
- Modify: `CMakeLists.txt`、`tests/CMakeLists.txt`
- Test: 上述两个测试文件 + `ie_bench` 手动运行

**Interfaces:**
- Consumes: `Engine`、`CpuBackend`、`Executor`、`KernelFns`、`cpu::matmul_row_*`
- Produces:
  - `tests/data/mlp_reference.h` 常量：`kMlpBatch=4`、`kMlpIn=8`、`kMlpH1=16`、`kMlpH2=8`、`kMlpOut=4`，数组 `kMlpInput`、`kMlpW1`、`kMlpB1`、`kMlpW2`、`kMlpB2`、`kMlpW3`、`kMlpB3`、`kMlpExpected`（float）。
  - MLP：`x(4,8) → Linear(8,16) → ReLU → Linear(16,8) → ReLU → Linear(8,4) → Softmax`，float32，固定 seed `20261106`。
  - `Graph build_mlp_graph()`（测试内 helper）：6 个节点 `MatMul, Relu, MatMul, Relu, MatMul, Softmax`；权重作为图输入，偏置折叠进增广权重：`x_aug=[x,1]`、`W_aug=[W;b]`。
  - `src/benchmarks/ie_bench.cpp`：打印探测 profile、选中变体、64/128/256 方阵的 scalar/selected GFLOPS。

- [ ] **Step 1: 写失败测试并生成参考数据**

先写脚本（不参与 ctest）：

```python
import torch

torch.manual_seed(20261106)
g = torch.Generator().manual_seed(20261106)
W1 = torch.randn(8, 16, generator=g) * 0.5
b1 = torch.randn(16, generator=g) * 0.1
W2 = torch.randn(16, 8, generator=g) * 0.5
b2 = torch.randn(8, generator=g) * 0.1
W3 = torch.randn(8, 4, generator=g) * 0.5
b3 = torch.randn(4, generator=g) * 0.1
x = torch.randn(4, 8, generator=g)
h = torch.relu(x @ W1 + b1)
h = torch.relu(h @ W2 + b2)
y = torch.softmax(h @ W3 + b3, dim=-1)
# 按 C++ header 格式写出 x/W1/b1/W2/b2/W3/b3/y，保留 9 位小数
```

生成并提交 `tests/data/mlp_reference.h`。测试代码：

```cpp
#include <gtest/gtest.h>
#include <algorithm>
#include <memory>
#include <span>
#include <string>
#include <unordered_map>
#include <vector>

#include "core/ir/graph.h"
#include "core/memory/malloc_allocator.h"
#include "core/runtime/engine.h"
#include "core/runtime/executor.h"
#include "data/mlp_reference.h"

using namespace ie;

namespace {
Tensor make_tensor(const Shape& shape, std::span<const float> values) {
    auto alloc = std::make_shared<MallocAllocator>();
    auto tensor = Tensor::allocate(shape, DType::F32, alloc);
    EXPECT_TRUE(tensor.has_value());
    Tensor result = std::move(*tensor);
    std::copy(values.begin(), values.end(), result.data<float>());
    return result;
}

std::vector<float> augment_input() {
    std::vector<float> out(kMlpBatch * (kMlpIn + 1));
    for (int i = 0; i < kMlpBatch; ++i) {
        for (int j = 0; j < kMlpIn; ++j) {
            out[i * (kMlpIn + 1) + j] = kMlpInput[i * kMlpIn + j];
        }
        out[i * (kMlpIn + 1) + kMlpIn] = 1.0f;
    }
    return out;
}

std::vector<float> augment_weight(const float* w, int in, int out,
                                  const float* bias) {
    std::vector<float> result((in + 1) * out);
    std::copy(w, w + in * out, result.begin());
    for (int j = 0; j < out; ++j) {
        result[in * out + j] = bias[j];
    }
    return result;
}

Graph build_mlp_graph() {
    Graph graph;
    auto value = [&](std::string name, Shape shape, bool input, bool output) {
        Value v;
        v.id = static_cast<ValueId>(graph.values.size());
        v.name = std::move(name);
        v.shape = std::move(shape);
        v.is_input = input;
        v.is_output = output;
        graph.values.push_back(std::move(v));
        return graph.values.back().id;
    };
    auto node = [&](OpType op, std::vector<ValueId> inputs, ValueId output) {
        Node n;
        n.id = static_cast<NodeId>(graph.nodes.size());
        n.op = op;
        n.inputs = std::move(inputs);
        n.outputs = {output};
        graph.nodes.push_back(std::move(n));
    };

    const ValueId x = value("x", Shape{{kMlpBatch, kMlpIn + 1}}, true, false);
    const ValueId w1 = value("w1", Shape{{kMlpIn + 1, kMlpH1}}, true, false);
    const ValueId w2 = value("w2", Shape{{kMlpH1 + 1, kMlpH2}}, true, false);
    const ValueId w3 = value("w3", Shape{{kMlpH2 + 1, kMlpOut}}, true, false);
    const ValueId h1 = value("h1", Shape{{kMlpBatch, kMlpH1}}, false, false);
    const ValueId h1r = value("h1r", Shape{{kMlpBatch, kMlpH1}}, false, false);
    const ValueId h2 = value("h2", Shape{{kMlpBatch, kMlpH2}}, false, false);
    const ValueId h2r = value("h2r", Shape{{kMlpBatch, kMlpH2}}, false, false);
    const ValueId logits = value("logits", Shape{{kMlpBatch, kMlpOut}}, false, false);
    const ValueId probs = value("probs", Shape{{kMlpBatch, kMlpOut}}, false, true);

    node(OpType::MatMul, {x, w1}, h1);
    node(OpType::Relu, {h1}, h1r);
    node(OpType::MatMul, {h1r, w2}, h2);
    node(OpType::Relu, {h2}, h2r);
    node(OpType::MatMul, {h2r, w3}, logits);
    node(OpType::Softmax, {logits}, probs);
    return graph;
}

std::unordered_map<std::string, Tensor> make_inputs() {
    const std::vector<float> x = augment_input();
    const std::vector<float> w1 = augment_weight(kMlpW1, kMlpIn, kMlpH1, kMlpB1);
    const std::vector<float> w2 = augment_weight(kMlpW2, kMlpH1, kMlpH2, kMlpB2);
    const std::vector<float> w3 = augment_weight(kMlpW3, kMlpH2, kMlpOut, kMlpB3);
    return {
        {"x", make_tensor(Shape{{kMlpBatch, kMlpIn + 1}}, x)},
        {"w1", make_tensor(Shape{{kMlpIn + 1, kMlpH1}}, w1)},
        {"w2", make_tensor(Shape{{kMlpH1 + 1, kMlpH2}}, w2)},
        {"w3", make_tensor(Shape{{kMlpH2 + 1, kMlpOut}}, w3)},
    };
}
}  // namespace

TEST(MlpE2ETest, MatchesPyTorchReferenceThroughEngine) {
    Engine engine(EngineConfig{});
    Graph graph = build_mlp_graph();
    ASSERT_TRUE(engine.load(graph).has_value());
    auto outputs = engine.infer(make_inputs());
    ASSERT_TRUE(outputs.has_value());
    ASSERT_EQ(outputs->size(), 1u);
    const Tensor& y = outputs->front();
    ASSERT_EQ(y.shape.dims, (std::vector<int64_t>{kMlpBatch, kMlpOut}));
    for (int i = 0; i < kMlpBatch * kMlpOut; ++i) {
        EXPECT_NEAR(y.data<float>()[i], kMlpExpected[i], 1e-5f);
    }
}

TEST(MlpE2ETest, ScalarPathMatchesGolden) {
    auto alloc = std::make_shared<MallocAllocator>();
    Executor executor;
    auto outputs = executor.run(build_mlp_graph(), make_inputs(), alloc);
    ASSERT_TRUE(outputs.has_value());
    for (int i = 0; i < kMlpBatch * kMlpOut; ++i) {
        EXPECT_NEAR(outputs->front().data<float>()[i], kMlpExpected[i], 1e-5f);
    }
}
```

性能测试（Release 才启用）：

```cpp
#include <gtest/gtest.h>
#include <chrono>
#include <vector>

#include "backends/cpu/simd/simd_kernels.h"

using namespace ie;

namespace {
bool has_avx2_fma() {
#if defined(__x86_64__) || defined(__i386__)
    return __builtin_cpu_supports("avx2") &&
           __builtin_cpu_supports("fma");
#else
    return false;
#endif
}

bool has_avx512() {
#if defined(__x86_64__) || defined(__i386__)
    return __builtin_cpu_supports("avx512f") &&
           __builtin_cpu_supports("avx512bw") &&
           __builtin_cpu_supports("avx512vl");
#else
    return false;
#endif
}
}  // namespace

TEST(MatMulBenchmarkSanity, Avx2BeatsScalarOnLargeMatrix) {
#ifndef NDEBUG
    GTEST_SKIP() << "performance sanity runs in Release";
#endif
    if (!has_avx2_fma()) {
        GTEST_SKIP() << "AVX2+FMA not available";
    }
    constexpr int64_t m = 256, k = 256, n = 256;
    std::vector<float> a(m * k, 1.0f), b(k * n, 1.0f), out(m * n);
    auto run = [&](auto fn) {
        auto start = std::chrono::steady_clock::now();
        for (int rep = 0; rep < 3; ++rep) {
            for (int64_t i = 0; i < m; ++i) {
                fn(a.data() + i * k, b.data(), out.data() + i * n, k, n);
            }
        }
        return std::chrono::duration<double>(
                   std::chrono::steady_clock::now() - start)
            .count();
    };
    const double scalar = run(cpu::matmul_row_scalar);
    const double avx2 = run(cpu::matmul_row_avx2);
    EXPECT_LT(avx2, scalar * 0.85) << "scalar=" << scalar << " avx2=" << avx2;
    if (has_avx512()) {
        const double avx512 = run(cpu::matmul_row_avx512);
        EXPECT_LT(avx512, scalar * 0.85)
            << "scalar=" << scalar << " avx512=" << avx512;
    }
}
```

- [ ] **Step 2: 跑测试确认失败**

Run: `cmake --build build`
Expected: 编译失败（`data/mlp_reference.h`/`ie_bench` 不存在）或 MLP 断言失败。

- [ ] **Step 3: 实现**

生成 `mlp_reference.h`；`build_mlp_graph` 按 Interfaces 建图；`make_inputs` 做偏置折叠。Engine 用 CpuBackend（SIMD+线程），标量测试用空 `KernelFns` 的 Executor。
`ie_bench.cpp`：检测 profile，调用 `CpuKernelCatalog::select`，对每个方阵分别用标量与选中行内核计时，输出 GFLOPS；无 AVX2 时只打印 scalar。CMake 增加 `ie_bench` 目标（链 `ie_cpu`）和两个测试源；`tests/CMakeLists.txt` 增加 `target_include_directories(ie_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})` 以包含 `tests/data`。

- [ ] **Step 4: 跑测试确认通过**

Run:
```bash
cmake --build build && ctest --test-dir build --output-on-failure
cmake -S . -B build-release -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build-release && ctest --test-dir build-release --output-on-failure
./build-release/ie_bench
```
Expected: Debug 全绿（性能用例 skip）；Release 全绿且 AVX2 性能断言通过；`ie_bench` 打印 selected variant 与 GFLOPS。

- [ ] **Step 5: 提交**

```bash
git add tests/reference tests/data/mlp_reference.h tests/runtime/mlp_e2e_test.cpp tests/performance/matmul_benchmark_test.cpp src/benchmarks/ie_bench.cpp CMakeLists.txt tests/CMakeLists.txt
git commit -m "feat(p1): add pytorch-aligned mlp e2e and simd benchmark"
```

---

### Task 7: 决策回写、接口增补清单与最终验证

**Files:**
- Modify: `docs/superpowers/specs/m2-backend-core/p1-cpu-hardware.md`
- Modify: `docs/superpowers/specs/00-overview/decisions-log.md`
- Modify: `docs/superpowers/specs/m1-bootstrap/p0.5-interface-inventory.md`（追加 P1 增量段）
- Modify: `todo.md`、`progress.md`

**Interfaces:**
- Consumes: 全部 Task 1–6 的实现与测试结果
- Produces: P1 实现期裁定、接口增量清单、进度记录

- [ ] **Step 1: 回写裁定**

在 `p1-cpu-hardware.md` 末追加「实现期裁定」：
- 基准：不引入 Google Benchmark；用仓库内 `ie_bench` + Release 性能 sanity。
- SIMD：函数级 target 属性 + 能力位分派，无全局 `-march`；AVX-512 用例在无该指令集机器 skip。
- MLP 偏置：折叠进增广权重，不改 Add 广播语义。
- 参考数据：PyTorch 脚本一次性生成 `tests/data/mlp_reference.h`，测试离线运行。
- 线程：持久 `ThreadPool`，按输出行切分，归约顺序固定。
- 分配器：`CpuAllocator` 64 字节对齐；Engine 现有 `MallocAllocator` 路径保留。
- 能力位名：沿用 `avx512vnni`/`avx512bf16`/`amx-*` 规范名。
- VNNI/AMX：P1 仅登记能力位，不提供 F32 内核；后续 dtype 里程碑实现。
- Cost model：按 `estimated_tflops` 优先级，实测微调留 P7。

`decisions-log.md` P1 三行改 ✅（能力位深度、cost model、基准）。`p0.5-interface-inventory.md` 追加 P1 增量表：`kernel_fns.h`、`thread_pool.h`、`cpu_allocator.h`、`relu_op.h`、`softmax_op.h`、`cpu_kernel_variants.h`、`cpu_kernel_catalog.h`、`simd_kernels.h`。
`todo.md`/`progress.md` 增加 P1 完成段，记录测试数、bench 数字、AVX-512 跳过原因。

- [ ] **Step 2: 全量验证**

Run:
```bash
cmake --build build && ctest --test-dir build --output-on-failure
cmake --build build-release && ctest --test-dir build-release --output-on-failure
cmake -S . -B build-asan -G Ninja -DCMAKE_BUILD_TYPE=Debug \
  -DCMAKE_CXX_FLAGS='-fsanitize=address,undefined -fno-omit-frame-pointer' \
  -DCMAKE_EXE_LINKER_FLAGS='-fsanitize=address,undefined'
cmake --build build-asan && ASAN_OPTIONS=detect_leaks=1 ctest --test-dir build-asan --output-on-failure
./build/ie_demo
./build-release/ie_bench
```
Expected: 三套构建全绿；demo 仍输出 `result: 15, 21, 27`；bench 打印 AVX2/selected 吞吐。

- [ ] **Step 3: 提交**

```bash
git add docs/superpowers/specs/m2-backend-core/p1-cpu-hardware.md docs/superpowers/specs/00-overview/decisions-log.md docs/superpowers/specs/m1-bootstrap/p0.5-interface-inventory.md todo.md progress.md
git commit -m "docs(p1): record cpu hardware decisions and verification"
```

---

## 自检

- Spec DoD 1（MLP 对齐）：Task 6。
- Spec DoD 2（能力位平台化）：Task 1。
- Spec DoD 3（MatMul 标量 + 加速 + 自动选择）：Task 3/4/5。
- Spec DoD 4（AVX2/AVX512 数值与性能）：Task 3/6。
- 模块清单：HardwareDetector（Task 1）、OptimizationPlanner（Task 4）、KernelVariantRegistry（Task 3）、MatMul/Relu/Softmax（Task 3/4）、CpuBackend（Task 5）、`simd/` 与 `kernels/`（Task 3）。
- Review Focus 每条均有测试：低能力分派（Task 3）、尾部 shape（Task 3）、数值/PyTorch（Task 3/6）、线程覆盖（Task 2/4）、探测一致性（Task 1）、失败路径（Task 4/5）。
