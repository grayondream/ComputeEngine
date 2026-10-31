# P0-MVP 静态图 CPU 推理 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 跑通 "GraphBuilder 构建静态图 → CpuBackend 同步执行 → `(a+b)*3` 得到 `{15,21,27}`" 的最小闭环，同时把 P0 规定的核心接口冻结为可编译、可测试的骨架。

**Architecture:** 单仓 C++20 静态库：`ie_core` 承载 base/ir/memory/backend/hardware/scheduler/runtime/plugin/tuning 接口与最小实现，`ie_cpu` 承载 `CpuBackend`，`ie_demo` 是端到端演示。算子按 `Op` 接口 + `OpRegistry` 注册；执行器按图节点顺序同步执行。GPU 在 P0 完全不实现，`DeviceSelector::select()` 恒返回 `GpuBackend::NONE`。

**Tech Stack:** C++20、CMake + Ninja、GoogleTest 1.17（系统 `find_package(GTest)`）、GCC 15.2。

**Spec:** `docs/superpowers/specs/m1-bootstrap/p0-mvp.md`（设计权威：`docs/readme.md` §2.1–§2.3、§3.1–§3.7、§4.1–§4.5、§5.1、§5.5、§15、§17）

## Global Constraints

- C++ 标准 `C++20`，禁止编译器扩展（`CMAKE_CXX_EXTENSIONS OFF`）；命名空间统一 `ie`。
- 代码统一放在 `src/` 下（`src/core/...`、`src/backends/...`、`src/demo/...`），测试在 `tests/`，文档在 `docs/`；头文件与实现同目录；头文件一律 `#pragma once`。
- 公共头用 `#include "core/base/shape.h"` 形式（`target_include_directories(... PUBLIC <repo-root>/src)`）。
- 不使用 vcpkg / 不下载依赖：P0 仅用系统 `libgtest-dev`，`find_package(GTest REQUIRED)`。
- 不引入 MLIR（P0 IR 手写；MLIR 在 P0.5 复核）；不实现 JSON 序列化（P0 无 exporter）；不引入 Google Benchmark（P1 再加）。
- 所有测试先失败后通过；每个 Task 结束必须跑完整 `ctest`。
- 每个 Task 独立提交；提交信息用 `feat(p0): ...` / `test(p0): ...`。
- 每新增一个 `.cpp`，必须加入 `CMakeLists.txt` 中对应目标的源列表；每新增一个测试文件，必须加入 `tests/CMakeLists.txt` 的 `add_executable(ie_tests ...)` 列表。
- `tests/CMakeLists.txt` 从 Task 1 起链接 `ie_core`；Task 8 起改为链接 `ie_cpu`（`ie_cpu` 已 PUBLIC 继承 `ie_core`）。
- `Engine::infer` 的输入表键是图输入 `Value.name`（P0 为 `"a"`、`"b"`）。
- 数值一律 `float`（`DType::F32`）；MatMul 只支持 2D。
- 错误必须返回 `Error`，不得抛异常穿透公共 API。

**实现期裁定（写入 `p0-mvp.md` §8 与 `decisions-log.md`）：**

1. `MLIR`：P0 不引入，手写 IR；P0.5 末复核是否引入。代价：若 P0.5 决定引入，`src/core/ir` 需加一层到 MLIR Dialect 的 lowering。
2. `vcpkg`：推迟到首个非系统依赖出现时；P0 用系统 GTest。代价：后续接入第三方库时补依赖管理。
3. `Backend::device()`：返回 `Device` 值（设计文档写 `Device*`，但 `Device` 同时被定义为 `enum class`，二者同名冲突）。代价：与设计原文签名不同，语义等价。
4. 代码布局：统一 `src/`（代码）、`tests/`（测试）、`docs/`（文档）。代价：与 §15 的 `infer-engine/` 示例路径不同。
5. Google Benchmark / JSON：P0 不落地，P1 / P0.5 起再引入。
6. C++20：不用 C++23 `std::expected`，自研 `Status`/`Result<T>`（基于 `std::optional`，见 Task 1）。代价：与设计原文 `using Status = std::expected<...>` 的字面签名不同，语义等价（`has_value/error` 保留）。
7. `Graph::backward_graph`：用 `std::shared_ptr<Graph>`（原文 `std::optional<Graph>` 是递归完整类型，C++20 不可编译）。代价：与设计字段类型不同；`has_backward()` 语义不变。

## Review Focus

- 输入名缺失或拼错：`Engine::infer` 必须返回明确 `Error`，不得解引用空指针。
- 形状不匹配：Add/Mul/MatMul 必须返回 `InvalidArgument`，不得越界写。
- 空图或没有输出值的图：`infer` 必须返回明确错误，不得返回未定义数据。
- 重复注册后端/算子/插件：必须返回 `AlreadyExists` 且保留原注册，不得覆盖或崩溃。
- Tensor 拷贝后原对象销毁：数据仍可读（共享所有权），不得悬垂。
- 分配 0 字节 / 非法参数：返回 `InvalidArgument`，不得返回可误用的非空指针。

---

## 文件结构（先锁边界）

```text
CMakeLists.txt
src/core/
  base/      error.h types.h shape.h rng.h
  ir/        ids.h model_kind.h attr.h op_type.h graph.h graph_builder.h
  memory/    memory_kind.h policy.h allocator.h malloc_allocator.h tensor.h arena.h transfer_manager.h
  backend/   backend.h backend_caps.h backend_matrix.h backend_registry.h device_selector.h engine_config.h
  hardware/  capability_set.h profile.h policies.h detector.h optimization_planner.h
  plugin/    plugin_meta.h plugin_registry.h kv_cache_types.h kv_cache_plugin.h ring_kv_cache.h
  tuning/    kernel_variant.h kernel_variant_registry.h
  scheduler/ plan.h scheduler.h static_scheduler.h
  runtime/   execution_context.h stream.h workspace.h executor.h engine.h
  ops/       op.h op_registry.h common/add_op.h common/mul_op.h common/matmul_op.h
src/backends/cpu/  cpu_backend.h cpu_backend.cpp
src/demo/         main.cpp
tests/         CMakeLists.txt base/ ir/ memory/ backend/ ops/ hardware/ scheduler/ runtime/engine_e2e_test.cpp plugin/
```

---

### Task 1: 构建系统 + `src/core/base` 基础类型

**Files:**
- Create: `CMakeLists.txt`、`tests/CMakeLists.txt`、`src/core/base/error.h`、`src/core/base/types.h`、`src/core/base/shape.h`、`src/core/base/shape.cpp`
- Test: `tests/base/shape_test.cpp`

**Interfaces:**
- Consumes: 无
- Produces:
  - `struct Error { enum class Code { InvalidArgument, Unsupported, NotFound, AlreadyExists, OutOfMemory, Runtime }; Code code; std::string message; static Error invalid_argument(std::string); ... };`
  - `using Status = ...`：C++20 无 `std::expected`，自研：
    ```cpp
    class Status {
    public:
        Status() = default;                       // 成功
        Status(Error e);                          // 失败（非 explicit，允许 `return Error::...`）
        bool has_value() const;                   // == 成功
        explicit operator bool() const;
        Error& error(); const Error& error() const;
    private:
        std::optional<Error> error_;
    };
    template<class T>
    class Result {
    public:
        Result(T value);                          // 成功（支持 move-only，如 unique_ptr）
        Result(Error e);                          // 失败（非 explicit）
        bool has_value() const; explicit operator bool() const;
        T& operator*(); const T& operator*() const;
        T* operator->(); const T* operator->() const;
        T& value(); const T& value() const;
        Error& error(); const Error& error() const;
    private:
        std::optional<T> value_; Error error_;
    };
    ```
  - `enum class Device { CPU, CUDA, VULKAN };`
  - `enum class DType { F32, F16, BF16, I8, I4, U8, FP8_E4M3, FP8_E5M2, MXFP4, MXFP6, BOOL };`
  - `enum class Layout { ROW_MAJOR, COL_MAJOR, BLOCKED, NHWC, NCHW };`
  - `struct Shape { std::vector<int64_t> dims; int64_t numel() const; int64_t stride(int dim) const; };`
  - 自由函数 `size_t dtype_size(DType); std::string to_string(DType); std::string to_string(Device); std::string to_string(const Shape&); bool operator==(const Shape&, const Shape&);`

- [ ] **Step 1: 写失败测试 `tests/base/shape_test.cpp`**

```cpp
#include <gtest/gtest.h>
#include "core/base/shape.h"
#include "core/base/types.h"
#include "core/base/error.h"

using namespace ie;

TEST(ShapeTest, NumelAndRowMajorStride) {
    Shape s{{2, 3, 4}};
    EXPECT_EQ(s.numel(), 24);
    EXPECT_EQ(s.stride(0), 12);
    EXPECT_EQ(s.stride(1), 4);
    EXPECT_EQ(s.stride(2), 1);
}

TEST(ShapeTest, Equality) {
    EXPECT_TRUE((Shape{{2, 3}} == Shape{{2, 3}}));
    EXPECT_FALSE((Shape{{2, 3}} == Shape{{3, 2}}));
}

TEST(BaseTypesTest, DTypeSizes) {
    EXPECT_EQ(dtype_size(DType::F32), 4u);
    EXPECT_EQ(dtype_size(DType::F16), 2u);
    EXPECT_EQ(dtype_size(DType::I8), 1u);
}

TEST(ErrorTest, FactoriesCarryCodeAndMessage) {
    auto e = Error::invalid_argument("bad shape");
    EXPECT_EQ(e.code, Error::Code::InvalidArgument);
    EXPECT_EQ(e.message, "bad shape");
    Result<int> r = e;
    ASSERT_FALSE(r.has_value());
    EXPECT_EQ(r.error().code, Error::Code::InvalidArgument);
}
```

- [ ] **Step 2: 加构建文件并确认失败**

`CMakeLists.txt`（本 Task 先只含 `ie_core` + 测试）：

```cmake
cmake_minimum_required(VERSION 3.28)
project(compute_engine VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(ie_core STATIC src/core/base/shape.cpp)
target_include_directories(ie_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_compile_options(ie_core PRIVATE -Wall -Wextra -Wpedantic)
find_package(Threads REQUIRED)
target_link_libraries(ie_core PUBLIC Threads::Threads)

enable_testing()
find_package(GTest REQUIRED)
add_subdirectory(tests)
```

`tests/CMakeLists.txt`：

```cmake
add_executable(ie_tests
    base/shape_test.cpp
)
target_link_libraries(ie_tests PRIVATE ie_core GTest::gtest_main)
include(GoogleTest)
gtest_discover_tests(ie_tests)
```

Run: `cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Debug && cmake --build build`
Expected: 编译失败，`fatal error: src/core/base/shape.h: No such file or directory`

- [ ] **Step 3: 实现三个头文件 + `shape.cpp`**

`error.h` 实现 C++20 自研的 `Status`/`Result<T>`（基于 `std::optional`，见上；禁止用 C++23 `std::expected`）；`types.h` 定义三个枚举与 `dtype_size`/`to_string`；`shape.h` 声明 `numel()`/`stride()`/`operator==`/`to_string`；`shape.cpp` 实现它们（空 dims 的 `numel()` 返回 0，行主序 `stride(dim)=dims[dim+1..]` 的后缀乘积，越界 `dim` 返回 0）。

- [ ] **Step 4: 跑测试**

Run: `cmake --build build && ./build/tests/ie_tests --gtest_filter='Shape*:Base*:Error*'`
Expected: PASS，4 个测试全绿。

- [ ] **Step 5: 提交**

```bash
git add CMakeLists.txt tests src/core/base
git commit -m "feat(p0): add build scaffold and base types"
```

---

### Task 2: `src/core/ir` 图模型 + GraphBuilder

**Files:**
- Create: `src/core/ir/ids.h`、`src/core/ir/model_kind.h`、`src/core/ir/attr.h`、`src/core/ir/op_type.h`、`src/core/ir/graph.h`、`src/core/ir/graph_builder.h`
- Test: `tests/ir/graph_test.cpp`

**Interfaces:**
- Consumes: `Shape`、`DType`、`Device`、`Result`、`Error`
- Produces:
  - `using NodeId = uint32_t; using ValueId = uint32_t; inline constexpr ValueId kInvalidValue = 0xFFFFFFFFu;`
  - `enum class ModelKind { TRADITIONAL, TRANSFORMER, HYBRID, RULE, UNKNOWN };`
  - `using AttrValue = std::variant<int64_t, double, bool, std::string>; using Attrs = std::unordered_map<std::string, AttrValue>;`
  - `enum class OpType { ... }`（完整复制 `docs/readme.md` §4.2 枚举成员）
  - `struct Value { ValueId id; std::string name; Shape shape; DType dtype = DType::F32; bool is_input = false; bool is_output = false; };`
  - `struct Node { NodeId id; OpType op; std::vector<ValueId> inputs, outputs; Attrs attrs; std::optional<Device> placed_device; ModelKind kind = ModelKind::UNKNOWN; };`
  - `struct Edge { ValueId value; std::optional<NodeId> from; std::optional<NodeId> to; };`
  - `using ShapeMap = std::unordered_map<std::string, Shape>; struct BoundGraph { std::unordered_map<ValueId, Shape> shapes; };`
  - `struct LoopInfo { NodeId loop_body{kInvalidValue}; int64_t max_iterations{0}; ValueId condition{kInvalidValue}; std::vector<ValueId> carried_values; }; struct Subgraph { std::vector<NodeId> nodes; }; struct ControlFlow { std::vector<Subgraph> subgraphs; std::vector<LoopInfo> loops; };`
  - `class Graph`：字段 `nodes/values/edges/dynamic_dims/control_flow/is_training/backward_graph`；`struct DynamicDim { ValueId value; int dim_index; std::string name; };`；方法 `bind(const ShapeMap&)`、`build_backward(const std::vector<ValueId>&)`、`is_dynamic()`、`has_control_flow()`、`has_backward()`、`find_value(ValueId)`、`find_value(std::string_view)`、`input(std::string_view)`。`backward_graph` 用 `std::shared_ptr<Graph>`（设计原文 `std::optional<Graph>` 因递归完整类型在 C++20 下不可编译）。
  - `class GraphBuilder { GraphBuilder(); GraphBuilder& add(int num_inputs); GraphBuilder& mul(double scalar); Graph build(); };`（`build()` 不限定 `&&`，以便 `std::move(b).add(2).mul(3.0).build()` 链式调用）

- [ ] **Step 1: 写失败测试 `tests/ir/graph_test.cpp`**

```cpp
#include <gtest/gtest.h>
#include "core/ir/graph_builder.h"

using namespace ie;

TEST(GraphBuilderTest, BuildsAddThenScalarMul) {
    GraphBuilder b;
    Graph g = std::move(b).add(2).mul(3.0).build();

    ASSERT_EQ(g.nodes.size(), 2u);
    EXPECT_EQ(g.nodes[0].op, OpType::Add);
    EXPECT_EQ(g.nodes[0].inputs.size(), 2u);
    EXPECT_EQ(g.nodes[1].op, OpType::Mul);
    ASSERT_TRUE(std::holds_alternative<double>(g.nodes[1].attrs.at("scalar")));
    EXPECT_DOUBLE_EQ(std::get<double>(g.nodes[1].attrs.at("scalar")), 3.0);

    ASSERT_NE(g.input("a"), nullptr);
    ASSERT_NE(g.input("b"), nullptr);
    EXPECT_TRUE(g.input("a")->is_input);
    EXPECT_EQ(g.nodes[1].outputs.size(), 1u);
    const Value* out = g.find_value(g.nodes[1].outputs[0]);
    ASSERT_NE(out, nullptr);
    EXPECT_TRUE(out->is_output);
}

TEST(GraphTest, BindRequiresEveryInput) {
    GraphBuilder b;
    Graph g = std::move(b).add(2).mul(3.0).build();

    auto missing = g.bind(ShapeMap{{"a", Shape{{3}}}});
    ASSERT_FALSE(missing.has_value());
    EXPECT_EQ(missing.error().code, Error::Code::InvalidArgument);

    auto ok = g.bind(ShapeMap{{"a", Shape{{3}}}, {"b", Shape{{3}}}});
    ASSERT_TRUE(ok.has_value());
    EXPECT_EQ(ok->shapes.size(), 2u);
}

TEST(GraphTest, EmptyGraphIsDetectable) {
    Graph g;
    EXPECT_FALSE(g.is_dynamic());
    EXPECT_FALSE(g.has_control_flow());
    EXPECT_FALSE(g.has_backward());
}
```

- [ ] **Step 2: 跑测试确认失败**

Run: `cmake --build build && ./build/tests/ie_tests --gtest_filter='Graph*'`
Expected: 编译失败（`graph_builder.h` 不存在）。

- [ ] **Step 3: 实现 IR 头文件**

要点：
- `GraphBuilder()` 预置两个输入 `Value{name="a"}`、`Value{name="b"}`，`shape` 为动态占位 `{{-1}}`，`dtype=F32`，`is_input=true`。
- `add(num_inputs)` 从输入值中取前 `num_inputs` 个作为 Add 输入，创建输出值 `t0`，记录 Node。
- `mul(scalar)` 取上一节点输出为输入，创建输出值 `t1` 并标记 `is_output=true`，`attrs["scalar"]=scalar`。
- `bind`：遍历 `values` 中 `is_input` 的值，按 `name` 查 `ShapeMap`，缺任何一个返回 `Error::invalid_argument("missing shape for input: <name>")`，成功则返回全部输入 value→shape 映射。
- `build_backward` 返回 `Error::unsupported("backward not implemented in P0")`。
- `Edge` 不强制构建（P0 执行按 `nodes` 顺序）；`GraphBuilder` 填 `edges` 可选，测试不依赖。

- [ ] **Step 4: 跑测试**

Run: `cmake --build build && ./build/tests/ie_tests --gtest_filter='Graph*'`
Expected: PASS。

- [ ] **Step 5: 提交**

```bash
git add src/core/ir tests/ir
git commit -m "feat(p0): add static graph model and graph builder"
```

---

### Task 3: `src/core/memory` 分配器 + Tensor

**Files:**
- Create: `src/core/memory/memory_kind.h`、`src/core/memory/policy.h`、`src/core/memory/allocator.h`、`src/core/memory/malloc_allocator.h`、`src/core/memory/tensor.h`、`src/core/memory/arena.h`、`src/core/memory/transfer_manager.h`
- Test: `tests/memory/memory_test.cpp`

**Interfaces:**
- Consumes: `Shape/DType/Device/Layout`、`Result/Status/Error`
- Produces:
  - `using BufferHandle = void*;`
  - `enum class MemoryKind { HOST, DEVICE, PINNED, UNIFIED, MANAGED, GRADIENT, OPTIMIZER_STATE, ACTIVATION, WEIGHT_CONST, KV_CACHE };`
  - `struct MemoryPolicy { size_t tensor_pool_size{0}; size_t activation_pool_size{0}; size_t weight_cache_size{0}; bool enable_weight_prefetch{false}; size_t kv_cache_block_size{0}; bool enable_prefix_caching{false}; size_t max_concurrent_requests{0}; };`
  - `class Allocator { virtual Result<BufferHandle> alloc(size_t, MemoryKind) = 0; virtual void free(BufferHandle) = 0; virtual size_t available(MemoryKind) const = 0; virtual Status free_gradients(); virtual Status free_activations(); };`
  - `class MallocAllocator final : public Allocator` + `size_t outstanding_bytes() const; size_t outstanding_blocks() const;`
  - `class Tensor`（字段同设计 §3.5 + `data<T>()`、`numel()`、`bytes()`、`static Tensor from_data(std::initializer_list<float>)`、`static Result<Tensor> allocate(const Shape&, DType, Allocator&, MemoryKind = MemoryKind::ACTIVATION)`、深/浅拷贝语义：拷贝共享底层 buffer）
  - `class Arena {}; class MemoryPool {};`
  - `class TransferManager { Status copy(Device src, Device dst, BufferHandle, size_t); };`（同设备返回成功；跨设备返回 `Unsupported`）

- [ ] **Step 1: 写失败测试 `tests/memory/memory_test.cpp`**

```cpp
#include <gtest/gtest.h>
#include "core/memory/malloc_allocator.h"
#include "core/memory/tensor.h"

using namespace ie;

TEST(MallocAllocatorTest, AllocFreeTracksOutstanding) {
    MallocAllocator a;
    auto h = a.alloc(64, MemoryKind::HOST);
    ASSERT_TRUE(h.has_value());
    EXPECT_NE(*h, nullptr);
    EXPECT_EQ(a.outstanding_blocks(), 1u);
    a.free(*h);
    EXPECT_EQ(a.outstanding_blocks(), 0u);
}

TEST(MallocAllocatorTest, ZeroSizeIsInvalid) {
    MallocAllocator a;
    auto h = a.alloc(0, MemoryKind::HOST);
    ASSERT_FALSE(h.has_value());
    EXPECT_EQ(h.error().code, Error::Code::InvalidArgument);
}

TEST(TensorTest, FromDataHasShapeAndValues) {
    Tensor t = Tensor::from_data({1.0f, 2.0f, 3.0f});
    EXPECT_EQ(t.shape.numel(), 3);
    ASSERT_NE(t.data<float>(), nullptr);
    EXPECT_FLOAT_EQ(t.data<float>()[2], 3.0f);
}

TEST(TensorTest, CopyOutlivesOriginal) {
    Tensor original = Tensor::from_data({4.0f, 5.0f});
    Tensor copy = original;
    { Tensor tmp = Tensor::from_data({9.0f}); }
    EXPECT_FLOAT_EQ(copy.data<float>()[1], 5.0f);
}

TEST(TransferManagerTest, SameDeviceCopySucceeds) {
    TransferManager tm;
    EXPECT_TRUE(tm.copy(Device::CPU, Device::CPU, nullptr, 0).has_value());
    EXPECT_EQ(tm.copy(Device::CPU, Device::CUDA, nullptr, 0).error().code, Error::Code::Unsupported);
}
```

- [ ] **Step 2: 跑测试确认失败**

Run: `cmake --build build`
Expected: 编译失败（头文件不存在）。

- [ ] **Step 3: 实现**

要点：
- `MallocAllocator::alloc`：`size==0` → `invalid_argument("alloc size is zero")`；`malloc` 失败 → `out_of_memory`；记录块数与字节数（`std::mutex` 保护）。
- `Tensor`：`std::shared_ptr<void> owner_` 持有 `malloc` 指针（deleter 调 `free`）；`buffer` 指向同块内存；拷贝共享 `owner_`；`data<T>()` 返回 `static_cast<T*>(buffer)`。
- `from_data(initializer_list<float>)`：分配 `n*sizeof(float)`，拷贝数据，`shape={{n}}`，`dtype=F32`，`device=CPU`，`layout=ROW_MAJOR`。
- `allocate(shape, dtype, allocator, kind)`：`allocator.alloc(bytes, kind)`，失败时返回其 `Error`；成功后返回填充 `shape/dtype/device/layout/buffer` 的 `Tensor`。`bytes = numel*dtype_size(dtype)`；`numel==0` 也返回 `InvalidArgument`（复用 alloc 的 0 字节规则）。
- `bytes()` = `numel()*dtype_size(dtype)`；空 shape 返回 0。
- `TransferManager::copy`：`src==dst` 返回 `{}`，否则 `Error::unsupported("cross-device copy not implemented in P0")`。加注释说明 P0 的空实现边界。

- [ ] **Step 4: 跑测试**

Run: `cmake --build build && ./build/tests/ie_tests --gtest_filter='Malloc*:Tensor*:Transfer*'`
Expected: PASS。

- [ ] **Step 5: 提交**

```bash
git add src/core/memory tests/memory
git commit -m "feat(p0): add malloc allocator and tensor"
```

---

### Task 4: `src/core/backend` 接口 + 注册表 + EngineConfig

**Files:**
- Create: `src/core/backend/backend_caps.h`、`src/core/backend/backend.h`、`src/core/backend/backend_matrix.h`、`src/core/backend/backend_registry.h`、`src/core/backend/device_selector.h`、`src/core/backend/engine_config.h`、`src/core/backend/engine_config.cpp`、`src/core/plugin/plugin_meta.h`
- Test: `tests/backend/backend_test.cpp`

**Interfaces:**
- Consumes: `Graph`、`HardwareProfile`、`MemoryPolicy`、`PluginMeta`、`ExecutionConfig`
- Produces:
  - `struct BackendCaps`（§3.7 全字段，默认值 false/0）
  - `using Timeout = std::chrono::milliseconds;`
  - `class Backend`（§3.3；`virtual Device device() const = 0;`）
  - `enum class BackendKind { CPU, CUDA, VULKAN };`
  - `struct PluginMeta { std::string name, version, api_version; std::vector<std::string> tags; std::string description; };`（本 Task 创建，供 `Backend::meta()` 使用）
  - `class BackendRegistry { static BackendRegistry& instance(); Status register_backend(BackendKind, std::function<std::unique_ptr<Backend>()>); Result<std::unique_ptr<Backend>> create(BackendKind); bool contains(BackendKind) const; std::vector<BackendKind> list() const; };`
  - `enum class GpuBackend { NONE, CUDA, VULKAN };`
  - `class DeviceSelector { static GpuBackend select(); };`（P0 恒 `NONE`）
  - `struct HybridConfig { bool enable_hybrid{false}; int gpu_layers{0}; }; struct ValidationConfig { bool enable{false}; std::string trace_dir; };`
  - `struct EngineConfig`（§5.1 全字段）+ `Status validate(const EngineConfig&)`
  - `struct BackendMatrix { std::unordered_map<OpType, std::array<bool,3>> op_support; std::unordered_map<DType, std::array<bool,3>> dtype_support; bool supports(BackendKind, OpType) const; };`

  > `Backend` 引用 `BatchPlan`/`ExecutionContext` 时必须用前置声明 + 在 `.cpp` 中 include，避免 Task 7/8 之前循环依赖；本 Task 只保证头文件可编译（`execute_async` 参数用引用）。

- [ ] **Step 1: 写失败测试 `tests/backend/backend_test.cpp`**

```cpp
#include <gtest/gtest.h>
#include "core/backend/backend_registry.h"
#include "core/backend/device_selector.h"
#include "core/backend/engine_config.h"
#include "core/backend/backend_matrix.h"

using namespace ie;

TEST(BackendRegistryTest, RegistersAndCreates) {
    auto& reg = BackendRegistry::instance();
    auto st = reg.register_backend(BackendKind::CUDA, [] { return std::unique_ptr<Backend>{}; });
    EXPECT_TRUE(st.has_value() || st.error().code == Error::Code::AlreadyExists);
    EXPECT_TRUE(reg.contains(BackendKind::CUDA));
    auto created = reg.create(BackendKind::CUDA);
    ASSERT_TRUE(created.has_value());
    EXPECT_EQ(created->get(), nullptr);
}

TEST(BackendRegistryTest, DuplicateRegistrationIsRejected) {
    auto& reg = BackendRegistry::instance();
    ASSERT_TRUE(reg.register_backend(BackendKind::VULKAN,
        [] { return std::unique_ptr<Backend>{}; }).has_value());
    auto st = reg.register_backend(BackendKind::VULKAN,
        [] { return std::unique_ptr<Backend>{}; });
    EXPECT_EQ(st.error().code, Error::Code::AlreadyExists);
}

TEST(DeviceSelectorTest, ReturnsNoneForP0) {
    EXPECT_EQ(DeviceSelector::select(), GpuBackend::NONE);
}

TEST(EngineConfigTest, RejectsEmptyDeviceSet) {
    EngineConfig cfg;
    cfg.enable_cpu = false;
    cfg.gpu_backend = GpuBackend::NONE;
    auto st = validate(cfg);
    ASSERT_FALSE(st.has_value());
    EXPECT_EQ(st.error().code, Error::Code::InvalidArgument);
}

TEST(BackendMatrixTest, DefaultsToUnsupported) {
    BackendMatrix m;
    EXPECT_FALSE(m.supports(BackendKind::CPU, OpType::Add));
}
```

- [ ] **Step 2: 跑测试确认失败**

Run: `cmake --build build`
Expected: 编译失败（头文件不存在）。

- [ ] **Step 3: 实现**

要点：
- `BackendRegistry` 用 `std::unordered_map<BackendKind, Factory>`；`register_backend` 已存在返回 `Error::already_exists("backend already registered")`；`create` 未注册返回 `not_found`；`list` 按枚举顺序返回。
- `validate`：(a) `!enable_cpu && gpu_backend==NONE` → invalid；(b) `hardware_adapt.autotune_budget_ms <= 0` → invalid；其余返回 `{}`（`tuning_cache_dir` 为空在 P0 允许，P7 再收紧）。
- `BackendMatrix::supports`：查 `op_support`，缺失返回 false。
- `Backend` 纯虚接口按 §3.3 全部落地；`device()` 返回 `Device`。

- [ ] **Step 4: 跑测试**

Run: `cmake --build build && ./build/tests/ie_tests --gtest_filter='Backend*:DeviceSelector*:EngineConfig*'`
Expected: PASS。

- [ ] **Step 5: 提交**

```bash
git add src/core/backend tests/backend
git commit -m "feat(p0): add backend interfaces, registry and engine config"
```

---

### Task 5: 算子接口 + OpRegistry + Add/Mul/MatMul

**Files:**
- Create: `src/core/ops/op.h`、`src/core/ops/op_registry.h`、`src/core/ops/op_registry.cpp`、`src/core/ops/common/add_op.h`、`src/core/ops/common/mul_op.h`、`src/core/ops/common/matmul_op.h`、`src/core/runtime/execution_context.h`
- Test: `tests/ops/ops_test.cpp`

**Interfaces:**
- Consumes: `Tensor`、`OpType`、`ExecutionContext`
- Produces:
  - `struct ExecutionContext { Device device{Device::CPU}; Allocator* allocator{nullptr}; std::unordered_map<std::string, Tensor> inputs; std::vector<Tensor> outputs; void* stream{nullptr}; };`
  - `class Op`（§3.6 五方法 + `virtual Status configure(const Attrs&)`，`backward` 默认 `Unsupported`）
  - `class OpRegistry { using Creator = std::function<std::unique_ptr<Op>()>; static OpRegistry& instance(); Status register_op(OpType, Creator); Result<std::unique_ptr<Op>> create(OpType); std::vector<OpType> list_by_kind(ModelKind); };`（重复注册返回 `AlreadyExists` 且保留原注册）
  - `void register_builtin_ops(OpRegistry&);`（幂等：忽略 `AlreadyExists`）
  - `class AddOp : public Op`（2 输入逐元素，形状必须相同）
  - `class MulOp : public Op`（2 输入逐元素；1 输入时使用 `configure(attrs["scalar"])` 得到的标量逐元素相乘）
  - `class MatMulOp : public Op`（2D，`[M,K] x [K,N]`）

  > 为了让 `MulOp` 拿到 scalar，`Op` 增加可选的 `virtual Status configure(const Attrs&)`，默认返回 `{}`；执行器在调用 `forward` 前调用 `configure(node.attrs)`。

- [ ] **Step 1: 写失败测试 `tests/ops/ops_test.cpp`**

```cpp
#include <gtest/gtest.h>
#include "core/ops/op_registry.h"
#include "core/ops/common/add_op.h"
#include "core/ops/common/mul_op.h"
#include "core/ops/common/matmul_op.h"
#include "core/memory/malloc_allocator.h"

using namespace ie;

TEST(OpRegistryTest, RegisterCreateAndLookup) {
    auto& reg = OpRegistry::instance();
    register_builtin_ops(reg);
    auto op = reg.create(OpType::Add);
    ASSERT_TRUE(op.has_value());
    EXPECT_NE(dynamic_cast<AddOp*>(op->get()), nullptr);
    EXPECT_FALSE(reg.list_by_kind(ModelKind::TRADITIONAL).empty());
}

TEST(OpRegistryTest, UnknownOpReturnsNotFound) {
    auto& reg = OpRegistry::instance();
    auto op = reg.create(OpType::Conv2D);
    ASSERT_FALSE(op.has_value());
    EXPECT_EQ(op.error().code, Error::Code::NotFound);
}

TEST(OpRegistryTest, DuplicateRegistrationIsRejected) {
    auto& reg = OpRegistry::instance();
    auto first = reg.register_op(OpType::Sub, [] { return std::make_unique<AddOp>(); });
    EXPECT_TRUE(first.has_value() || first.error().code == Error::Code::AlreadyExists);
    auto second = reg.register_op(OpType::Sub, [] { return std::make_unique<AddOp>(); });
    ASSERT_FALSE(second.has_value());
    EXPECT_EQ(second.error().code, Error::Code::AlreadyExists);
}

TEST(AddOpTest, AddsElementwiseAndRejectsShapeMismatch) {
    MallocAllocator alloc;
    ExecutionContext ctx; ctx.allocator = &alloc;
    AddOp op;
    Tensor a = Tensor::from_data({1, 2, 3});
    Tensor b = Tensor::from_data({4, 5, 6});
    Tensor out;
    std::vector<Tensor*> in{&a, &b};
    std::vector<Tensor*> outs{&out};
    ASSERT_TRUE(op.forward(in, outs, ctx).has_value());
    EXPECT_FLOAT_EQ(out.data<float>()[0], 5.0f);

    Tensor c = Tensor::from_data({7, 8});
    std::vector<Tensor*> bad{&a, &c};
    EXPECT_EQ(op.forward(bad, outs, ctx).error().code, Error::Code::InvalidArgument);
}

TEST(MulOpTest, ScalarMultiply) {
    MallocAllocator alloc;
    ExecutionContext ctx; ctx.allocator = &alloc;
    MulOp op;
    Attrs attrs{{"scalar", 3.0}};
    ASSERT_TRUE(op.configure(attrs).has_value());
    Tensor a = Tensor::from_data({1, 2, 3});
    Tensor out;
    std::vector<Tensor*> in{&a};
    std::vector<Tensor*> outs{&out};
    ASSERT_TRUE(op.forward(in, outs, ctx).has_value());
    EXPECT_FLOAT_EQ(out.data<float>()[2], 9.0f);
}

TEST(MatMulOpTest, TwoDimensionalProduct) {
    MallocAllocator alloc;
    ExecutionContext ctx; ctx.allocator = &alloc;
    MatMulOp op;
    Tensor a = Tensor::from_data({1, 2, 3, 4}); a.shape = Shape{{2, 2}};
    Tensor b = Tensor::from_data({5, 6, 7, 8}); b.shape = Shape{{2, 2}};
    Tensor out;
    std::vector<Tensor*> in{&a, &b};
    std::vector<Tensor*> outs{&out};
    ASSERT_TRUE(op.forward(in, outs, ctx).has_value());
    ASSERT_EQ(out.shape.dims, (std::vector<int64_t>{2, 2}));
    EXPECT_FLOAT_EQ(out.data<float>()[0], 19.0f);
    EXPECT_FLOAT_EQ(out.data<float>()[3], 50.0f);
}
```

- [ ] **Step 2: 跑测试确认失败**

Run: `cmake --build build`
Expected: 编译失败（算子头文件不存在）。

- [ ] **Step 3: 实现**

要点：
- `Op::configure(const Attrs&)` 默认为 `{}`；`MulOp` 读取 `attrs.at("scalar")`（`double` 或 `int64_t`），缺失返回 `invalid_argument`。
- `AddOp::forward` 校验 `inputs.size()==2`、输出缓冲区可写；形状不同返回 invalid；结果写入 `outputs[0]`（执行器保证已分配）。
- `MulOp`：2 输入逐元素；1 输入 + 已 configure 的 scalar 逐元素乘。
- `MatMulOp`：校验 2D 与 `K` 相等；`outputs[0]` 分配 `{M,N}`；用三层循环（P0 不优化）。
- `register_builtin_ops` 注册 Add/Mul/MatMul；`list_by_kind` 对 `TRADITIONAL`/`TRANSFORMER`/`UNKNOWN` 返回已注册算子（P0 用同一列表）。
- `OpRegistry::instance()` 用函数内 static。

- [ ] **Step 4: 跑测试**

Run: `cmake --build build && ./build/tests/ie_tests --gtest_filter='OpRegistry*:AddOp*:MulOp*:MatMul*'`
Expected: PASS。

- [ ] **Step 5: 提交**

```bash
git add src/core/ops src/core/runtime/execution_context.h tests/ops
git commit -m "feat(p0): add op interface, registry and cpu reference kernels"
```

---

### Task 6: `src/core/hardware` 能力画像 + CPU 探测 + OptimizationPlanner

**Files:**
- Create: `src/core/hardware/capability_set.h`、`src/core/hardware/profile.h`、`src/core/hardware/policies.h`、`src/core/hardware/detector.h`、`src/core/hardware/detector.cpp`、`src/core/hardware/optimization_planner.h`、`src/core/hardware/optimization_planner.cpp`
- Test: `tests/hardware/hardware_test.cpp`

**Interfaces:**
- Consumes: `Result`、`Graph`、`EngineConfig`、`ModelKind`
- Produces:
  - `class CapabilitySet`（`has/set/list`）
  - `struct HardwareProfile`（§2.1 全字段，含 `fingerprint`；所有标量字段带 `{}` 默认初始化，保证 `HardwareProfile{}` 可用）
  - `struct QuantPolicy { enum class Mode { NONE, INT8, FP8 }; Mode mode{Mode::NONE}; };`
  - `class HardwareDetector { static Result<HardwareProfile> detect_cpu(); static Result<HardwareProfile> detect_cuda(int); static Result<HardwareProfile> detect_vulkan(int); private: static std::string make_fingerprint(const HardwareProfile&); };`
  - `class OptimizationPlanner { struct Plan { std::vector<std::string> enabled_passes; std::vector<std::string> preferred_kernels; MemoryPolicy memory_policy; QuantPolicy quant_policy; std::string reason; }; static Plan plan(const HardwareProfile&, const Graph&, ModelKind, const EngineConfig&); };`

- [ ] **Step 1: 写失败测试 `tests/hardware/hardware_test.cpp`**

```cpp
#include <gtest/gtest.h>
#include "core/hardware/detector.h"
#include "core/hardware/optimization_planner.h"

using namespace ie;

TEST(CapabilitySetTest, SetHasList) {
    CapabilitySet c;
    EXPECT_FALSE(c.has("avx2"));
    c.set("avx2", true);
    EXPECT_TRUE(c.has("avx2"));
    c.set("avx2", false);
    EXPECT_FALSE(c.has("avx2"));
}

TEST(HardwareDetectorTest, DetectCpuReturnsProfileWithFingerprint) {
    auto p = HardwareDetector::detect_cpu();
    ASSERT_TRUE(p.has_value());
    EXPECT_FALSE(p->fingerprint.empty());
    EXPECT_FALSE(p->vendor.empty());
    EXPECT_GE(p->cu.num_cores, 0);
}

TEST(HardwareDetectorTest, DetectCudaIsUnsupportedInP0) {
    auto p = HardwareDetector::detect_cuda(0);
    ASSERT_FALSE(p.has_value());
    EXPECT_EQ(p.error().code, Error::Code::Unsupported);
}

TEST(OptimizationPlannerTest, ReturnsConservativePlan) {
    EngineConfig cfg;
    Graph g;
    auto plan = OptimizationPlanner::plan(HardwareProfile{}, g, ModelKind::UNKNOWN, cfg);
    EXPECT_FALSE(plan.reason.empty());
}
```

- [ ] **Step 2: 跑测试确认失败**

Run: `cmake --build build`
Expected: 编译失败（头文件不存在）。

- [ ] **Step 3: 实现**

要点：
- `detect_cpu`：填充 `vendor="generic"`（或读 `__builtin_cpu_is`）、`device_name`、`driver_version="n/a"`、`cu.num_cores=std::thread::hardware_concurrency()`；仅在 `#if defined(__x86_64__) || defined(__i386__)` 下用 `__builtin_cpu_supports` 设置 `has_sse2/avx/avx2/avx512f/avx512bw/avx512vl/avx512vnni/avx512bf16/avx512fp16/amx-bf16/amx-int8/amx-fp16`；非 x86 平台保持全 false（测试不硬断言某能力位为 true）。
- 同步把关键能力写入 `profile.caps`（`set("avx2", ...)` 等），`list()` 排序返回。
- `fingerprint`：用 FNV-1a 对 vendor、device_name、核心数、caps 列表做哈希，格式 `"cpu-%016llx"`。
- `detect_cuda/vulkan` 返回 `Error::unsupported("gpu detection not implemented in P0")`。
- `OptimizationPlanner::plan` 返回 `enabled_passes={}`、`preferred_kernels={"cpu_scalar"}`、`memory_policy{}`、`quant_policy{NONE}`、`reason="p0-conservative"`。

- [ ] **Step 4: 跑测试**

Run: `cmake --build build && ./build/tests/ie_tests --gtest_filter='Capability*:HardwareDetector*:OptimizationPlanner*'`
Expected: PASS（x86 与非 x86 均通过；不硬断言 AVX2 为 true）。

- [ ] **Step 5: 提交**

```bash
git add src/core/hardware tests/hardware
git commit -m "feat(p0): add hardware profile and cpu detection"
```

---

### Task 7: `src/core/scheduler` 静态调度器

**Files:**
- Create: `src/core/scheduler/plan.h`、`src/core/scheduler/scheduler.h`、`src/core/scheduler/static_scheduler.h`、`src/core/scheduler/static_scheduler.cpp`
- Test: `tests/scheduler/scheduler_test.cpp`

**Interfaces:**
- Consumes: `Tensor`、`BufferHandle`、`Status`
- Produces:
  - `using RequestId = uint64_t;`
  - `struct Request { RequestId id; std::string model; std::vector<Tensor> inputs; };`
  - `struct PrefillItem { RequestId request; std::vector<Tensor> inputs; }; struct DecodeItem { RequestId request; size_t num_tokens{1}; }; struct TransferItem { BufferHandle src{nullptr}; BufferHandle dst{nullptr}; size_t bytes{0}; };`
  - `struct BatchPlan { std::vector<PrefillItem> prefills; std::vector<DecodeItem> decodes; std::vector<TransferItem> transfers; size_t token_budget{0}; };`
  - `class Scheduler`（§3.2 五方法）
  - `class StaticScheduler final : public Scheduler { using ExecutorFn = std::function<Status(const BatchPlan&)>; explicit StaticScheduler(ExecutorFn fn = {}); ... };`

- [ ] **Step 1: 写失败测试 `tests/scheduler/scheduler_test.cpp`**

```cpp
#include <gtest/gtest.h>
#include "core/scheduler/static_scheduler.h"

using namespace ie;

TEST(StaticSchedulerTest, PlanContainsOnePrefillPerRequest) {
    StaticScheduler s;
    s.submit(Request{1, "m", {Tensor::from_data({1, 2})}});
    s.submit(Request{2, "m", {Tensor::from_data({3, 4})}});
    auto plan = s.plan_next_iteration();
    ASSERT_EQ(plan.prefills.size(), 2u);
    EXPECT_TRUE(plan.decodes.empty());
}

TEST(StaticSchedulerTest, ExecuteIterationInvokesExecutor) {
    int calls = 0;
    StaticScheduler s([&](const BatchPlan&) { ++calls; return Status{}; });
    s.submit(Request{7, "m", {}});
    auto plan = s.plan_next_iteration();
    ASSERT_TRUE(s.execute_iteration(plan).has_value());
    EXPECT_EQ(calls, 1);
    s.on_request_complete(7);
    EXPECT_TRUE(s.plan_next_iteration().prefills.empty());
}

TEST(StaticSchedulerTest, PreemptRemovesPendingRequest) {
    StaticScheduler s;
    s.submit(Request{9, "m", {}});
    s.preempt(9);
    EXPECT_TRUE(s.plan_next_iteration().prefills.empty());
}
```

- [ ] **Step 2: 跑测试确认失败**

Run: `cmake --build build`
Expected: 编译失败。

- [ ] **Step 3: 实现**

要点：`pending_` 保存请求；`plan_next_iteration` 为每个 pending 生成 `PrefillItem` 且 `token_budget = 各请求输入 numel 之和`；`execute_iteration` 在 `exec_` 存在时调用，否则返回 `{}`；`on_request_complete`/`preempt` 从 `pending_` 删除；`Scheduler` 的其余方法按接口实现。`ContinuousBatchScheduler`/`HybridScheduler` P0 不实现（只在 Spec 中保留）。

- [ ] **Step 4: 跑测试**

Run: `cmake --build build && ./build/tests/ie_tests --gtest_filter='StaticScheduler*'`
Expected: PASS。

- [ ] **Step 5: 提交**

```bash
git add src/core/scheduler tests/scheduler
git commit -m "feat(p0): add static scheduler"
```

---

### Task 8: `src/core/runtime` 执行器 + Engine + CpuBackend（端到端）

**Files:**
- Create: `src/core/runtime/stream.h`、`src/core/runtime/workspace.h`、`src/core/runtime/executor.h`、`src/core/runtime/executor.cpp`、`src/core/runtime/engine.h`、`src/core/runtime/engine.cpp`、`src/backends/cpu/cpu_backend.h`、`src/backends/cpu/cpu_backend.cpp`
- Modify: `CMakeLists.txt`（新增 `ie_cpu`、`ie_demo` 目标）
- Test: `tests/runtime/engine_e2e_test.cpp`

**Interfaces:**
- Consumes: 前 7 个 Task 的全部公共接口
- Produces:
  - `struct Stream { Device device{Device::CPU}; void* handle{nullptr}; };`
  - `class Workspace { std::unordered_map<ValueId, Tensor>& tensors(); };`
  - `class Executor { explicit Executor(OpRegistry& reg = OpRegistry::instance()); Result<std::vector<Tensor>> run(const Graph&, const std::unordered_map<std::string, Tensor>& inputs, Allocator&); };`
  - `class Engine { explicit Engine(EngineConfig cfg); ~Engine(); Status load(const Graph&); Result<std::vector<Tensor>> infer(const std::unordered_map<std::string, Tensor>& inputs); };`
  - `class CpuBackend final : public Backend`（全部接口落地；`compile` 存图；`execute_async` 同步跑 `Executor`）

- [ ] **Step 1: 写失败测试 `tests/runtime/engine_e2e_test.cpp`**

```cpp
#include <gtest/gtest.h>
#include "core/runtime/engine.h"
#include "core/ir/graph_builder.h"

using namespace ie;

static Graph BuildMulAddGraph() {
    GraphBuilder b;
    return std::move(b).add(2).mul(3.0).build();
}

TEST(EngineE2ETest, ComputesAddThenMul) {
    EngineConfig cfg;
    Engine engine(cfg);
    ASSERT_TRUE(engine.load(BuildMulAddGraph()).has_value());

    std::unordered_map<std::string, Tensor> inputs;
    inputs.emplace("a", Tensor::from_data({1.0f, 2.0f, 3.0f}));
    inputs.emplace("b", Tensor::from_data({4.0f, 5.0f, 6.0f}));

    auto out = engine.infer(inputs);
    ASSERT_TRUE(out.has_value());
    ASSERT_EQ(out->size(), 1u);
    ASSERT_EQ(out->at(0).shape.numel(), 3);
    EXPECT_FLOAT_EQ(out->at(0).data<float>()[0], 15.0f);
    EXPECT_FLOAT_EQ(out->at(0).data<float>()[1], 21.0f);
    EXPECT_FLOAT_EQ(out->at(0).data<float>()[2], 27.0f);
}

TEST(EngineE2ETest, MissingInputIsReported) {
    EngineConfig cfg;
    Engine engine(cfg);
    ASSERT_TRUE(engine.load(BuildMulAddGraph()).has_value());
    std::unordered_map<std::string, Tensor> inputs;
    inputs.emplace("a", Tensor::from_data({1.0f, 2.0f, 3.0f}));
    auto out = engine.infer(inputs);
    ASSERT_FALSE(out.has_value());
    EXPECT_EQ(out.error().code, Error::Code::InvalidArgument);
}

TEST(EngineE2ETest, InferBeforeLoadIsRejected) {
    EngineConfig cfg;
    Engine engine(cfg);
    auto out = engine.infer({});
    ASSERT_FALSE(out.has_value());
    EXPECT_EQ(out.error().code, Error::Code::InvalidArgument);
}

TEST(EngineE2ETest, GraphWithoutOutputsIsRejected) {
    EngineConfig cfg;
    Engine engine(cfg);
    ASSERT_TRUE(engine.load(Graph{}).has_value());
    auto out = engine.infer({});
    ASSERT_FALSE(out.has_value());
    EXPECT_EQ(out.error().code, Error::Code::InvalidArgument);
}
```

- [ ] **Step 2: 跑测试确认失败**

Run: `cmake --build build`
Expected: 编译失败（`engine.h` 不存在）。

- [ ] **Step 3: 实现执行器**

先更新 `CMakeLists.txt`：把本 Task 的 `.cpp` 加入 `ie_core`，新增 `ie_cpu` 目标（源 `src/backends/cpu/cpu_backend.cpp`，`target_link_libraries(ie_cpu PUBLIC ie_core)`）；把 `tests/CMakeLists.txt` 的链接目标从 `ie_core` 改为 `ie_cpu` 并加入 `runtime/engine_e2e_test.cpp`。

`Executor::run` 步骤：
1. `graph.find_value` 校验每个 `is_input` 的 value 在 `inputs` 中存在（按 name），缺失 → `invalid_argument("missing input: <name>")`；同时校验 dtype 为 F32。
2. 用 `value_id -> Tensor` 的 `unordered_map` 保存已计算结果；输入 value 直接放进去（浅拷贝）。
3. 按 `graph.nodes` 顺序遍历：`reg_.create(node.op)`，失败 → `not_found("no kernel for op ...")`；调用 `configure(node.attrs)`。
4. 为每个 `node.outputs` 创建输出 `Tensor`：`shape` 从节点推导（P0 只支持 Add/Mul 同形状、MatMul `[M,N]`；由算子 `forward` 负责设置/分配）。实现细节：先放默认 `Tensor` 到表里，传 `&outputs[i]` 给 `forward`，算子内部用 Task 3 提供的 `Tensor::allocate(shape, dtype, allocator, MemoryKind::ACTIVATION)` 分配并 `*outputs[i] = std::move(*tmp)`。**不要**让算子直接 `Tensor::from_data` 赋给 `outputs[i]`。
5. 执行完成后，收集 `is_output==true` 的 value（按 `values` 顺序）对应 Tensor，返回 `std::vector<Tensor>`。
6. 若图没有输出值 → `invalid_argument("graph has no outputs")`。

- [ ] **Step 4: 实现 Engine + CpuBackend**

- `Engine` 构造：`validate(cfg)` 失败 → 保存错误状态，`load/infer` 返回该错误；确保 `BackendRegistry` 有 CPU 工厂（没有则注册 `CpuBackend`），`create(BackendKind::CPU)` 并 `init()`。
- `Engine::load(const Graph&)`：`backend_->compile(g)`，保存图副本，`loaded_=true`。
- `Engine::infer`：未 load → invalid；构造 `ExecutionContext`（`allocator = &allocator_`）；填 `ctx.inputs`；构造 `BatchPlan` 单 prefill；`backend_->execute_async(plan, ctx)`；`backend_->wait(Timeout{1000})`；返回 `ctx.outputs`。
- `CpuBackend`：持有 `MallocAllocator`、`Executor`、`std::optional<Graph> graph_`；`alloc/free` 转发；`copy_*` 用 `memcpy`（仅 HOST）；`compile(Graph&)` 复制；`execute_async` 用 `plan.prefills[0].inputs` 或 `ctx.inputs` 调 `Executor::run`，写入 `ctx.outputs`；`wait` 返回 `{}`；`is_idle` 返回 true；`caps` 全 false；`meta` name=`"cpu"`、version=`"0.1.0"`、api_version=`"0.1"`；`hardware_profile` 返回 `HardwareDetector::detect_cpu()`；`memory_policy` 返回默认。

- [ ] **Step 5: 跑测试**

Run: `cmake --build build && ./build/tests/ie_tests --gtest_filter='EngineE2E*'`
Expected: PASS，`{15,21,27}` 断言全绿。

- [ ] **Step 6: 提交**

```bash
git add src/core/runtime backends/cpu tests/runtime CMakeLists.txt src/core/memory/tensor.h
git commit -m "feat(p0): add executor, engine and cpu backend"
```

---

### Task 9: 预留接口骨架（KVCache/Plugin/RNG/Tuning）

**Files:**
- Create: `src/core/plugin/plugin_registry.h`、`src/core/plugin/kv_cache_types.h`、`src/core/plugin/kv_cache_plugin.h`、`src/core/plugin/ring_kv_cache.h`、`src/core/base/rng.h`、`src/core/tuning/kernel_variant.h`、`src/core/tuning/kernel_variant_registry.h`
- Modify: `CMakeLists.txt`、`tests/CMakeLists.txt`
- Test: `tests/plugin/plugin_test.cpp`

**Interfaces:**
- Consumes: `Tensor`、`Status/Result`
- Produces:
  - `struct PluginMeta`（Task 4 已创建于 `src/core/plugin/plugin_meta.h`；本 Task 直接复用）
  - `template<class PluginT> class PluginRegistry`（§4.4；`load_shared_library` 返回 `Unsupported`）
  - `using BlockId = uint32_t; using BlockHash = uint64_t; using TensorHash = uint64_t; struct BlockTable { std::vector<BlockId> blocks; }; enum class QuantFormat { NONE, INT8, FP8 };`
  - `class KVCachePlugin`（§3.1 全部纯虚 + 默认方法）
  - `class RingKVCache final : public KVCachePlugin`（append/load 返回 `Unsupported`，`memory_usage` 返回 0）
  - `class RNG`（§4.5）+ `class SamplingRNG`（xorshift/PCG，确定性 seed）+ `class RNGRegistry`
  - `struct KernelVariant { std::string name; OpType op; DType dtype; CapabilitySet required; };`
  - `class KernelVariantRegistry { void register_variant(KernelVariant); std::vector<KernelVariant> list(OpType) const; };`

- [ ] **Step 1: 写失败测试 `tests/plugin/plugin_test.cpp`**

```cpp
#include <gtest/gtest.h>
#include "core/plugin/plugin_registry.h"
#include "core/plugin/ring_kv_cache.h"
#include "core/base/rng.h"

using namespace ie;

struct FakePlugin {
    PluginMeta m{"fake", "1.0", "0.1", {"test"}, "fake plugin"};
    PluginMeta meta() const { return m; }
};

TEST(PluginRegistryTest, RegisterGetList) {
    auto& reg = PluginRegistry<FakePlugin>::instance();
    ASSERT_TRUE(reg.register_plugin(std::make_unique<FakePlugin>()).has_value());
    auto p = reg.get("fake");
    ASSERT_TRUE(p.has_value());
    EXPECT_EQ((*p)->meta().version, "1.0");
    EXPECT_EQ(reg.list_by_tag("test").size(), 1u);
}

TEST(PluginRegistryTest, DuplicateRejected) {
    auto& reg = PluginRegistry<FakePlugin>::instance();
    reg.register_plugin(std::make_unique<FakePlugin>());
    auto st = reg.register_plugin(std::make_unique<FakePlugin>());
    EXPECT_EQ(st.error().code, Error::Code::AlreadyExists);
}

TEST(RingKVCacheTest, AppendUnsupportedInP0) {
    RingKVCache kv;
    EXPECT_EQ(kv.append(0, Tensor{}, Tensor{}).error().code, Error::Code::Unsupported);
    EXPECT_EQ(kv.memory_usage(), 0u);
}

TEST(SamplingRNGTest, SameSeedSameSequence) {
    SamplingRNG a, b;
    a.set_seed(42); b.set_seed(42);
    EXPECT_EQ(a.next_u64(), b.next_u64());
    a.set_seed(7); b.set_seed(7);
    EXPECT_FLOAT_EQ(a.next_float(), b.next_float());
}
```

- [ ] **Step 2: 跑测试确认失败**

Run: `cmake --build build`
Expected: 编译失败。

- [ ] **Step 3: 实现**

要点：`PluginRegistry::register_plugin` 以 `meta().name` 为键，重复返回 `AlreadyExists`；`load_shared_library` 返回 `Unsupported("plugin loading not implemented in P0")`；`RNGRegistry` 以名字登记；`SamplingRNG` 用 `std::mt19937_64` + `std::uniform_real_distribution<float>(0,1)`，`next_u64` 直接返回引擎输出。

- [ ] **Step 4: 跑测试**

Run: `cmake --build build && ./build/tests/ie_tests --gtest_filter='PluginRegistry*:RingKVCache*:SamplingRNG*'`
Expected: PASS。

- [ ] **Step 5: 提交**

```bash
git add src/core/plugin src/core/base/rng.h src/core/tuning tests/plugin
git commit -m "feat(p0): add plugin, kv-cache, rng and tuning skeletons"
```

---

### Task 10: 演示程序 + 全量验证 + 文档回写

**Files:**
- Create: `src/demo/main.cpp`
- Modify: `CMakeLists.txt`（`ie_demo` + `add_test`）、`tests/CMakeLists.txt`（加入全部测试源）、`todo.md`、`progress.md`、`docs/superpowers/specs/m1-bootstrap/p0-mvp.md`、`docs/superpowers/specs/00-overview/decisions-log.md`
- Test: 全量 `ctest`

**Interfaces:**
- Consumes: 所有前序 Task
- Produces: `ie_demo` 可执行文件 + 全绿测试

- [ ] **Step 1: 写 `src/demo/main.cpp`（对照 §17 骨架）**

```cpp
#include <cassert>
#include <iostream>
#include "core/hardware/detector.h"
#include "core/backend/backend_registry.h"
#include "core/ir/graph_builder.h"
#include "core/runtime/engine.h"
#include "backends/cpu/cpu_backend.h"

using namespace ie;

int main() {
    auto profile = HardwareDetector::detect_cpu();
    if (!profile) { std::cerr << "cpu detect failed\n"; return 1; }
    std::cout << "CPU fingerprint: " << profile->fingerprint << "\n";
    std::cout << "AVX2: " << profile->matrix.has_avx2 << "\n";
    std::cout << "AVX-512 VNNI: " << profile->matrix.has_avx512_vnni << "\n";

    BackendRegistry::instance().register_backend(
        BackendKind::CPU, [] { return std::make_unique<CpuBackend>(); });

    GraphBuilder builder;
    auto graph = std::move(builder).add(2).mul(3.0).build();

    EngineConfig config;
    config.gpu_backend = GpuBackend::NONE;
    config.default_model_kind = ModelKind::UNKNOWN;
    Engine engine(config);
    if (!engine.load(graph)) { std::cerr << "load failed\n"; return 1; }

    std::unordered_map<std::string, Tensor> inputs;
    inputs.emplace("a", Tensor::from_data({1.0f, 2.0f, 3.0f}));
    inputs.emplace("b", Tensor::from_data({4.0f, 5.0f, 6.0f}));

    auto result = engine.infer(inputs);
    if (!result) { std::cerr << "infer failed: " << result.error().message << "\n"; return 1; }
    const float* out = result->at(0).data<float>();
    assert(out[0] == 15.0f);
    std::cout << "result: " << out[0] << ", " << out[1] << ", " << out[2] << "\n";
    return out[0] == 15.0f && out[1] == 21.0f && out[2] == 27.0f ? 0 : 2;
}
```

- [ ] **Step 2: 更新 CMake 并跑全量**

`CMakeLists.txt` 末尾追加：

```cmake
add_library(ie_cpu STATIC src/backends/cpu/cpu_backend.cpp)
target_link_libraries(ie_cpu PUBLIC ie_core)
target_compile_options(ie_cpu PRIVATE -Wall -Wextra -Wpedantic)

add_executable(ie_demo src/demo/main.cpp)
target_link_libraries(ie_demo PRIVATE ie_cpu)
add_test(NAME demo_e2e COMMAND ie_demo)
```

`tests/CMakeLists.txt` 的 `add_executable` 汇总所有 `*_test.cpp`（base/ir/memory/backend/ops/hardware/scheduler/runtime/plugin）。

Run: `cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Debug && cmake --build build && ctest --test-dir build --output-on-failure`
Expected: 构建成功；`ctest` 全绿（含 `demo_e2e`）；demo 输出 `result: 15, 21, 27`。

- [ ] **Step 3: 跑一次 Release 验证**

Run: `cmake -S . -B build-release -G Ninja -DCMAKE_BUILD_TYPE=Release && cmake --build build-release && ctest --test-dir build-release --output-on-failure`
Expected: 全绿（`assert` 在 Release 下被 `NDEBUG` 关闭，但 demo 的返回值检查仍生效）。

- [ ] **Step 4: 回写文档（实现期裁定）**

在 `p0-mvp.md` §8 追加“实现期裁定”小节，写入本计划 Global Constraints 下的 7 条裁定及日期；在 `00-overview/decisions-log.md` 将对应行状态改为“已按实现期裁定”，并记录：C++20 + 自研 Result/Status、代码布局 `src/`、`Graph::backward_graph` 用 `shared_ptr`、MLIR 延后 P0.5、vcpkg 延后、Backend::device() 返回 `Device` 值、Benchmark/JSON 延后。

- [ ] **Step 5: 更新 `todo.md` / `progress.md`**

记录 P0-MVP 完成、测试命令与结果、后续进入 P0.5。

- [ ] **Step 6: 提交**

```bash
git add src CMakeLists.txt tests todo.md progress.md docs/superpowers/specs
git commit -m "feat(p0): complete mvp demo, tests and decision writeback"
```
