# P2a · ONNX 传统模型推理 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 打通 ONNX 传统 CNN 模型到自研 IR/CPU 后端的推理闭环，Conv/Pool/BN/Gemm 等算子正确，`TraditionalEngine` 可加载、单样本推理与批量推理，并用 ONNX Runtime 黄金数据验证 ResNet50 兼容路径。

**Architecture:** 在 P0.5 IR 上做增量扩展：`Graph` 增加常量权重表，`AttrValue` 增加整数/浮点数组；新增无第三方依赖的最小 ONNX protobuf wire 解析器，将 ONNX 子集转换为 `Graph` + 常量权重；新增 `src/ops/traditional/` 算子与 `TraditionalEngine`，后者复用现有 `Engine`/`CpuBackend` 的 SIMD/线程池路径。ONNX Runtime/PyTorch 只用于一次性生成并提交黄金数据，测试离线运行。

**Tech Stack:** C++20, CMake + Ninja, GoogleTest, 现有 `Status`/`Result<T>`, 现有 CPU 后端；ONNX protobuf wire format 手写解析；PyTorch/onnx/onnxruntime 仅用于 fixture 生成。

**Spec:** `docs/superpowers/specs/m2-backend-core/p2-traditional-models.md`

## Scope

本计划是 P2 的垂直切片（P2a），覆盖 P2 DoD #1–#4：

- ONNX loader 覆盖 CNN 推理子集；
- Conv2D/DepthwiseConv2D/BatchNorm/MaxPool2D/AvgPool2D/GlobalAvgPool；
- Gemm/Flatten/Reshape/Transpose/Concat/Slice/Pad/Upsample/Gather/Reduce/Clip/Sigmoid/Tanh/Softmax；
- `TraditionalEngine::load_model/infer/infer_batch`；
- `WEIGHT_CONST` 权重常驻；
- 小 CNN 逐层/端到端 ONNX Runtime 对齐，以及可选的 ResNet50 全模型本地对齐。

明确延后到 P2b：`ModelConverter::convert_onnx_to_ie` 的 JSON 序列化、`ConversionValidator` 五阶段、TFLite/Caffe loader、Conv+BN+ReLU 融合 Pass、Winograd/FFT、`ConvTranspose2D`/`GroupNorm`/LSTM/GRU 真实执行。`OnnxLoader`、`OnnxMapping`、`ModelConverter::load_model` 在 P2a 实现；其余转换/校验入口保持 `Unsupported`，并在文档中写明。

P2a 额外限制：`Add`/`Mul` 只支持同形状（`Mul` 仍支持 P1 标量）；通用 broadcasting 延后。`Softmax` 沿用 P1 的 1D/2D 语义，ONNX 3D+ / 任意 axis 的 Softmax 延后。ResNet50 不使用这些未覆盖形态；遇到时 loader/op 返回明确 `Unsupported`。

## Global Constraints

- C++20；`CMAKE_CXX_EXTENSIONS OFF`；namespace `ie`；`-Wall -Wextra -Wpedantic` 无警告。
- 不引入运行时第三方依赖；ONNX 解析器必须仓库内实现。PyTorch/onnx/onnxruntime 只用于离线生成 fixture，测试运行期不依赖 Python。
- 不破坏 P0.5 冻结接口；只做增量字段/新头文件；`Status` 仍为 void 结果，禁止 `std::expected`。
- 算子只支持 F32 输入/权重；非 F32、动态形状、未知 ONNX op 必须返回明确 `Unsupported`/`InvalidArgument`，不得崩溃或越界。
- 权重使用 `MemoryKind::WEIGHT_CONST`；权重必须随 `Graph` 拷贝存活，不能依赖调用方临时 buffer。
- ONNX 形状常量（Reshape shape、Slice starts/ends、Pad pads 等）内部可转为 F32 张量读取，但算子必须按整数语义截断/校验。
- 测试数据（`.onnx` 与黄金 header）提交到 `tests/data/`；生成脚本提交到 `tests/reference/`，但测试不执行 Python。
- 不提交 ResNet50 权重（约 100MB）；ResNet50 测试用环境变量指向本地生成的 ONNX/reference，缺省 `GTEST_SKIP`。
- 代码任务在 `master` 上增量提交；每个 Task 先 RED 后 GREEN，结束时全量测试并提交。
- 文档中文、面向实现者；实现期裁定回写 Spec/`decisions-log.md`/`todo.md`/`progress.md`。

## Review Focus

- 截断/畸形/未知字段 ONNX 输入：必须返回 `InvalidArgument`，不能越界读取或无限递归。
- NCHW 卷积/池化形状：pads/strides/dilations/ceil_mode/group 的边界，1x1、stride 2、奇数尺寸、全 padding 窗口。
- 权重生命周期：`Graph` 拷贝、`Engine::load` 再拷贝、`TraditionalEngine` 多次 `infer` 后权重仍有效。
- 不支持/可选输入的语义：缺省 bias、Clip min/max、Pad value、空字符串 optional input、非 F32 常量。
- 批量不变性：`infer_batch` 每个样本与单独 `infer` 结果在容差内一致；线程化 MatMul 结果与串行一致。

---

### Task 1: IR 常量权重与属性数组

**Files:**
- Modify: `src/core/ir/graph.h`
- Modify: `src/core/ir/attr.h`
- Modify: `src/core/ir/op_type.h`
- Modify: `src/core/runtime/executor.cpp`
- Test: `tests/ir/graph_test.cpp`
- Test: `tests/ir/attr_test.cpp`（新建；若并入 `graph_test.cpp` 也可，但需在 CMake 注册）

**Interfaces:**
- Consumes: 现有 `Graph`/`Tensor`/`Executor`。
- Produces:
  - `Graph::weights`：`std::unordered_map<std::string, Tensor>`，键为 ONNX initializer/Constant 名称。
  - `AttrValue` 追加 `std::vector<int64_t>`、`std::vector<double>` 两个 variant 分支。
  - `OpType::Flatten` 追加在 enum 末尾（保持既有枚举值不变）。
  - `Executor::run` 在读取输入前，把 `graph.weights` 中非 input 的值注入 `values`。

- [ ] **Step 1: 写失败测试**

在 `tests/ir/graph_test.cpp` 增加：

```cpp
TEST(GraphWeightsTest, ExecutorSeedsConstantWeights) {
    auto alloc = std::make_shared<MallocAllocator>();

    Graph graph;
    Value x; x.id = 0; x.name = "x"; x.shape = Shape{{2}}; x.is_input = true;
    Value w; w.id = 1; w.name = "w"; w.shape = Shape{{2}};
    Value y; y.id = 2; y.name = "y"; y.shape = Shape{{2}}; y.is_output = true;
    graph.values = {x, w, y};

    Node node; node.id = 0; node.op = OpType::Add; node.inputs = {0, 1};
    node.outputs = {2}; graph.nodes.push_back(node);

    auto weight = Tensor::allocate(Shape{{2}}, DType::F32, alloc,
                                   MemoryKind::WEIGHT_CONST);
    ASSERT_TRUE(weight.has_value());
    weight->data<float>()[0] = 10.0f;
    weight->data<float>()[1] = 20.0f;
    graph.weights.emplace("w", std::move(*weight));

    auto input = Tensor::allocate(Shape{{2}}, DType::F32, alloc);
    ASSERT_TRUE(input.has_value());
    input->data<float>()[0] = 1.0f;
    input->data<float>()[1] = 2.0f;

    Executor executor;
    auto out = executor.run(graph, {{"x", std::move(*input)}}, alloc);
    ASSERT_TRUE(out.has_value());
    EXPECT_FLOAT_EQ(out->at(0).data<float>()[0], 11.0f);
    EXPECT_FLOAT_EQ(out->at(0).data<float>()[1], 22.0f);
}

TEST(GraphWeightsTest, MissingWeightIsAnError) {
    auto alloc = std::make_shared<MallocAllocator>();
    Graph graph;
    Value x; x.id = 0; x.name = "x"; x.shape = Shape{{1}}; x.is_input = true;
    Value w; w.id = 1; w.name = "w"; w.shape = Shape{{1}};
    Value y; y.id = 2; y.name = "y"; y.shape = Shape{{1}}; y.is_output = true;
    graph.values = {x, w, y};
    Node node; node.id = 0; node.op = OpType::Add; node.inputs = {0, 1};
    node.outputs = {2}; graph.nodes.push_back(node);

    auto input = Tensor::allocate(Shape{{1}}, DType::F32, alloc);
    ASSERT_TRUE(input.has_value());
    input->data<float>()[0] = 1.0f;

    Executor executor;
    auto out = executor.run(graph, {{"x", std::move(*input)}}, alloc);
    ASSERT_FALSE(out.has_value());
    EXPECT_EQ(out.error().code, Error::Code::InvalidArgument);
}
```

在 `tests/ir/attr_test.cpp` 增加：

```cpp
TEST(AttrValueTest, HoldsIntAndFloatLists) {
    Attrs attrs;
    attrs["pads"] = std::vector<int64_t>{1, 1, 1, 1};
    attrs["scales"] = std::vector<double>{1.5, 1.5};
    ASSERT_NE(std::get_if<std::vector<int64_t>>(&attrs["pads"]), nullptr);
    ASSERT_NE(std::get_if<std::vector<double>>(&attrs["scales"]), nullptr);
    EXPECT_EQ(std::get<std::vector<int64_t>>(attrs["pads"])[2], 1);
}
```

- [ ] **Step 2: 运行确认 RED**

```bash
cmake --build build --target ie_tests -j
./build/tests/ie_tests --gtest_filter='GraphWeightsTest.*:AttrValueTest.*'
```

Expected: 编译失败（`Graph::weights`、`Tensor` include、vector variant 分支不存在）。

- [ ] **Step 3: 实现**

- `graph.h` 增加 `#include "core/memory/tensor.h"` 与 `std::unordered_map<std::string, Tensor> weights;`。
- `attr.h` 增加 `<vector>`，`using AttrValue = std::variant<int64_t, double, bool, std::string, std::vector<int64_t>, std::vector<double>>;`。
- `op_type.h` 在 `SoftmaxCrossEntropy` 后追加 `Flatten,`，不改既有枚举顺序。
- `executor.cpp` 在 input 循环前：

```cpp
for (const auto& [name, tensor] : graph.weights) {
    const Value* value = graph.find_value(name);
    if (value == nullptr) {
        return Error::invalid_argument("weight has no graph value: " + name);
    }
    if (value->is_input) {
        continue;  // caller-provided input wins
    }
    values.emplace(value->id, tensor);
}
```

- [ ] **Step 4: 运行确认 GREEN**

```bash
cmake --build build --target ie_tests -j
./build/tests/ie_tests --gtest_filter='GraphWeightsTest.*:AttrValueTest.*'
```

- [ ] **Step 5: 全量测试并提交**

```bash
ctest --test-dir build --output-on-failure
git add src/core/ir/graph.h src/core/ir/attr.h src/core/ir/op_type.h src/core/runtime/executor.cpp tests/ir/graph_test.cpp tests/ir/attr_test.cpp tests/CMakeLists.txt
git commit -m "feat(p2): add graph weights, attr lists, and flatten op type"
```

---

### Task 2: 最小 ONNX wire parser 与 OnnxLoader

**Files:**
- Create: `src/models/traditional/onnx/onnx_wire.h`
- Create: `src/models/traditional/onnx/onnx_wire.cpp`
- Create: `src/models/traditional/onnx/onnx_loader.h`
- Create: `src/models/traditional/onnx/onnx_loader.cpp`
- Modify: `src/converter/format_loader/format_loader.h`（`OnnxLoader::load` 改声明）
- Create: `src/converter/format_loader/format_loader.cpp`
- Modify: `src/converter/op_mapping/op_mapping.h`（`OnnxMapping::map` 改声明）
- Create: `src/converter/op_mapping/onnx_mapping.cpp`
- Modify: `CMakeLists.txt`（`ie_core` 增加 4 个 `.cpp`）
- Modify: `tests/interface/converter_contract_test.cpp`（OnnxLoader/OnnxMapping 不再是 Unsupported）
- Test: `tests/models/onnx_loader_test.cpp`
- Fixture/create: `tests/reference/gen_onnx_fixtures.py`
- Fixture: `tests/data/tiny_add.onnx`（由脚本生成并提交）

**Interfaces:**
- Consumes: Task 1 的 `Graph::weights`/`AttrValue`。
- Produces:
  - `Result<OnnxModel> parse_onnx_model(std::span<const uint8_t> bytes)`（内部类型在 `onnx_wire.h`）。
  - `Result<Graph> load_onnx_model(const std::string& path)`。
  - `OnnxMapping::map(std::string_view)` 覆盖白名单；未知返回 `Unsupported`。
  - `OnnxLoader::load` 委托 `load_onnx_model`。

**ONNX 子集：** 只解析 `ModelProto.graph`、`GraphProto.node/initializer/input/output/value_info`、`NodeProto`、`AttributeProto` 的 `f/i/s/ints/floats/t`、`TensorProto` 的 `dims/data_type/float_data/int32_data/int64_data/double_data/raw_data/name`、`ValueInfoProto` 的 tensor 类型/维度。忽略未知字段；遇到 wire type 3/4、外部数据、未知 dtype、截断长度返回 `InvalidArgument`。

- [ ] **Step 1: 生成并提交 fixture**

`tests/reference/gen_onnx_fixtures.py` 用 onnx helper 生成一个 2x2 Add：

- 输入 `x` `[2,2]` F32；
- initializer `w` `[2,2]` F32，值 `[1,2,3,4]`；
- 节点 `Add(x,w) -> y`；
- 输出 `y` `[2,2]`；
- 写入 `tests/data/tiny_add.onnx`。

脚本运行方式（仅生成时）：

```bash
PYTHONPATH=/tmp/opencode/onnxvenv/lib/python3.14/site-packages python3 tests/reference/gen_onnx_fixtures.py
```

测试运行期不调用 Python。

- [ ] **Step 2: 写失败测试**

`tests/models/onnx_loader_test.cpp`：

```cpp
TEST(OnnxLoaderTest, ParsesTinyAddAndPreservesWeight) {
    auto graph = load_onnx_model(std::string(IE_TEST_DATA_DIR) + "/tiny_add.onnx");
    ASSERT_TRUE(graph.has_value()) << graph.error().message;
    ASSERT_EQ(graph->nodes.size(), 1u);
    EXPECT_EQ(graph->nodes[0].op, OpType::Add);
    ASSERT_EQ(graph->nodes[0].inputs.size(), 2u);
    ASSERT_EQ(graph->nodes[0].outputs.size(), 1u);
    EXPECT_NE(graph->find_value("x"), nullptr);
    EXPECT_NE(graph->find_value("w"), nullptr);
    EXPECT_NE(graph->find_value("y"), nullptr);
    ASSERT_EQ(graph->weights.count("w"), 1u);
    const Tensor& w = graph->weights.at("w");
    EXPECT_TRUE(w.shape == (Shape{{2, 2}}));
    EXPECT_EQ(w.dtype, DType::F32);
    EXPECT_FLOAT_EQ(w.data<float>()[0], 1.0f);
    EXPECT_FLOAT_EQ(w.data<float>()[3], 4.0f);
    EXPECT_TRUE(graph->find_value("y")->is_output);
}

TEST(OnnxLoaderTest, RejectsTruncatedFileAndMissingFile) {
    const std::string path = std::string(IE_TEST_DATA_DIR) + "/tiny_add.onnx";
    std::ifstream input(path, std::ios::binary);
    ASSERT_TRUE(input.good());
    std::vector<uint8_t> bytes((std::istreambuf_iterator<char>(input)),
                               std::istreambuf_iterator<char>());
    ASSERT_GT(bytes.size(), 8u);
    bytes.resize(8);
    auto parsed = parse_onnx_model(bytes);
    ASSERT_FALSE(parsed.has_value());
    EXPECT_EQ(parsed.error().code, Error::Code::InvalidArgument);

    EXPECT_EQ(load_onnx_model(std::string(IE_TEST_DATA_DIR) +
                              "/does_not_exist.onnx").error().code,
              Error::Code::InvalidArgument);
}

TEST(OnnxMappingTest, MapsTraditionalWhitelistAndRejectsUnknown) {
    OnnxMapping mapping;
    EXPECT_EQ(*mapping.map("Conv"), OpType::Conv2D);
    EXPECT_EQ(*mapping.map("GlobalAveragePool"), OpType::GlobalAvgPool);
    EXPECT_EQ(*mapping.map("Flatten"), OpType::Flatten);
    EXPECT_EQ(mapping.map("DefinitelyNotAnOp").error().code,
              Error::Code::Unsupported);
}
```

在 `tests/CMakeLists.txt` 增加 `models/onnx_loader_test.cpp`，并给 `ie_tests` 增加：

```cmake
target_compile_definitions(ie_tests PRIVATE
    IE_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data")
```

同时更新 `tests/interface/converter_contract_test.cpp`：从 `AllFormatLoadersUnsupported` 移除 `OnnxLoader`，从 `AllOpMappingsUnsupported` 移除 `OnnxMapping`；新增 `OnnxLoader` 加载 `tiny_add.onnx` 成功的断言，以及 `OnnxMapping` 的 `Conv/Flatten` 映射断言。`ModelConverter` 的契约暂不改（P2a Task 7 才实现 `load_model`）。

- [ ] **Step 3: 运行确认 RED**

```bash
cmake --build build --target ie_tests -j
./build/tests/ie_tests --gtest_filter='OnnxLoaderTest.*:OnnxMappingTest.*'
```

Expected: 编译失败（`onnx_loader.h`/`onnx_mapping.cpp`/fixture 不存在）。

- [ ] **Step 4: 实现 wire parser**

`onnx_wire.h` 公开最小结构：

```cpp
struct OnnxTensor {
    std::string name;
    std::vector<int64_t> dims;
    int32_t data_type{0};
    std::vector<float> float_data;
    std::vector<int32_t> int32_data;
    std::vector<int64_t> int64_data;
    std::vector<double> double_data;
    std::vector<uint8_t> raw_data;
};

struct OnnxAttribute {
    std::string name;
    std::variant<int64_t, double, std::string, OnnxTensor,
                 std::vector<int64_t>, std::vector<double>> value;
};

struct OnnxNode { std::string name, op_type; std::vector<std::string> inputs, outputs; std::vector<OnnxAttribute> attributes; };
struct OnnxValueInfo { std::string name; int32_t elem_type{0}; std::vector<int64_t> dims; };
struct OnnxGraph { std::vector<OnnxNode> nodes; std::vector<OnnxTensor> initializers; std::vector<OnnxValueInfo> inputs, outputs, value_info; };
struct OnnxModel { OnnxGraph graph; };
```

实现 `ProtoReader`（`std::span<const uint8_t>`、`pos`、`read_varint`、`read_tag`、`read_length_delimited`、`skip`、`enter_message`），按字段号解析；未知字段 `skip`。`TensorProto` 转 F32 时：

- `data_type == 1`（FLOAT）：优先 `raw_data`，否则 `float_data`；
- `data_type == 7`（INT64）：`raw_data` 或 `int64_data`，逐元素转 float；
- `data_type == 6`（INT32）：`int32_data` 转 float；
- 其他 dtype：`Unsupported("ONNX tensor dtype ...")`。
- 标量（`dims` 为空）内部存为 `Shape{{1}}`；`Tensor` 不能分配空 shape。

`onnx_loader.cpp`：

- `load_onnx_model` 读文件到 `std::vector<uint8_t>`（`std::ifstream` binary），空文件/读失败返回 `InvalidArgument`。
- 为每个 input/output/value_info/initializer/node 输入输出建立 `Value`；initializer 建 `Tensor` 并放入 `graph.weights`，`is_input=false`。
- Node `op_type` 经 `OnnxMapping`；`Constant` 节点折叠为 weight，不建 node。
- `Attribute` 映射到 `Attrs`：`f/ints/floats/i/s`；`Tensor` 属性只允许 Constant。
- Node input 为空字符串时跳过；output 为空时跳过。
- 初始 `Value.shape`：graph input/output/value_info/initializer 有则填；否则 `{}`（执行期由算子计算）。
- `edges` 按输入建；Node ID 顺序。
- 不支持的外部数据/未知 op 返回 `Unsupported`，错误信息包含 op/字段名。

`OnnxMapping` 白名单：

```
Add, Mul, MatMul, Gemm, Relu, Softmax, Conv, ConvTranspose, MaxPool,
AveragePool, GlobalAveragePool, BatchNormalization, Flatten, Reshape,
Transpose, Concat, Slice, Gather, ReduceMean, ReduceSum, Pad, Upsample,
Clip, Sigmoid, Tanh
```

映射：`Conv->Conv2D`, `ConvTranspose->ConvTranspose2D`, `MaxPool->MaxPool2D`, `AveragePool->AvgPool2D`, `GlobalAveragePool->GlobalAvgPool`, `BatchNormalization->BatchNorm`, `Flatten->Flatten`，其余同名；`ReduceMean/ReduceSum` 都映射 `Reduce`，用 `reduce_mode` attr 区分。

- [ ] **Step 5: 运行确认 GREEN**

```bash
cmake --build build --target ie_tests -j
./build/tests/ie_tests --gtest_filter='OnnxLoaderTest.*:OnnxMappingTest.*'
```

- [ ] **Step 6: 全量测试并提交**

```bash
ctest --test-dir build --output-on-failure
git add src/models src/converter/format_loader src/converter/op_mapping CMakeLists.txt tests/models tests/interface/converter_contract_test.cpp tests/reference/gen_onnx_fixtures.py tests/data/tiny_add.onnx tests/CMakeLists.txt
git commit -m "feat(p2): add minimal onnx parser and loader"
```

---

### Task 3: Conv2D / DepthwiseConv2D / BatchNorm / Pool / GlobalAvgPool

**Files:**
- Create: `src/ops/traditional/conv2d_op.h`
- Create: `src/ops/traditional/conv2d_op.cpp`
- Create: `src/ops/traditional/pool_ops.h`
- Create: `src/ops/traditional/pool_ops.cpp`
- Create: `src/ops/traditional/batch_norm_op.h`
- Create: `src/ops/traditional/batch_norm_op.cpp`
- Modify: `src/ops/traditional/traditional_ops.h`（改为 include 新头文件；保留仍在 P2b 的 stub 类）
- Modify: `src/core/ops/op_registry.cpp`（注册已实现传统算子）
- Modify: `CMakeLists.txt`（`ie_core` 增加 3 个 `.cpp`）
- Test: `tests/ops/traditional_ops_test.cpp`
- Fixture/create: `tests/reference/gen_traditional_fixtures.py`
- Fixture: `tests/data/traditional_reference.h`

**Interfaces:**
- Consumes: Task 1 的 `AttrValue` 数组；Task 2 的 loader 输出的 `Attrs`。
- Produces: 同名类 `Conv2DOp`、`DepthwiseConv2DOp`、`BatchNormOp`、`MaxPool2DOp`、`AvgPool2DOp`、`GlobalAvgPoolOp`，都实现 `Op`。
- `Conv2DOp` 优先使用 `context.kernels->matmul_row` 做 GEMM 行内核；没有注入时用本地标量行循环，保证 `TraditionalEngine` 与直接 `Executor` 两条路径一致。

**属性约定：**

| 算子 | attrs |
|---|---|
| Conv2D | `strides` int list {sH,sW} 默认 {1,1}; `pads` {top,left,bottom,right} 默认 0; `dilations` 默认 {1,1}; `kernel_shape` 可选；`group` 默认 1; `auto_pad` string |
| BatchNorm | `epsilon` double 默认 1e-5；输入顺序 `[x, scale, bias, mean, var]` |
| MaxPool2D | `kernel_shape` {kH,kW} 必填; `strides` 默认 {1,1}; `pads` 默认 0; `ceil_mode` int 默认 0; `auto_pad` |
| AvgPool2D | 同上，另 `count_include_pad` int 默认 0 |
| GlobalAvgPool | 无属性；输入 `[N,C,H,W]` 输出 `[N,C,1,1]` |

- [ ] **Step 1: 写失败测试**

在 `tests/ops/traditional_ops_test.cpp` 增加直接算子测试，黄金数据来自 `tests/data/traditional_reference.h`（脚本用 PyTorch/ONNX Runtime 生成，测试只 include）：

```cpp
TEST(TraditionalOpsTest, Conv2DMatchesReference) {
    Conv2DOp op;
    Attrs attrs;
    attrs["strides"] = std::vector<int64_t>{1, 1};
    attrs["pads"] = std::vector<int64_t>{1, 1, 1, 1};
    ASSERT_TRUE(op.configure(attrs).has_value());

    auto x = make_tensor({1, 2, 5, 5}, kConvInput);
    auto w = make_tensor({3, 2, 3, 3}, kConvWeight);
    auto b = make_tensor({3}, kConvBias);
    std::vector<Tensor*> inputs{&x, &w, &b};
    std::vector<Tensor> out(1);
    std::vector<Tensor*> outputs{&out[0]};
    ExecutionContext ctx; ctx.allocator = std::make_shared<MallocAllocator>();
    ASSERT_TRUE(op.forward(inputs, outputs, ctx).has_value());
    ASSERT_EQ(out[0].shape, (Shape{{1, 3, 5, 5}}));
    for (std::size_t i = 0; i < kConvExpected.size(); ++i) {
        EXPECT_NEAR(out[0].data<float>()[i], kConvExpected[i], 1e-5f);
    }
}

TEST(TraditionalOpsTest, BatchNormMatchesReference) { /* 5 输入，epsilon 1e-5 */ }
TEST(TraditionalOpsTest, MaxAvgPoolAndGlobalAvgMatchReference) { /* 分别断言 */ }
TEST(TraditionalOpsTest, DepthwiseConv2DMatchesGroupedConv) { /* group=2 小例子 */ }
```

`make_tensor` helper 用 `Tensor::allocate` + `std::copy`；黄金数组在 `tests/data/traditional_reference.h`。

- [ ] **Step 2: 运行确认 RED**

```bash
cmake --build build --target ie_tests -j
./build/tests/ie_tests --gtest_filter='TraditionalOpsTest.*'
```

Expected: 编译失败（类仍是 `UnimplementedOp`，`configure`/`forward` 不存在）。

- [ ] **Step 3: 实现**

- `Conv2DOp`：
  - 校验输入 2 或 3、输出 1、F32、allocator；权重 `[M, C/group, kH, kW]`；bias `[M]`。
  - `auto_pad`: `VALID` 全 0；`SAME_UPPER`/`SAME_LOWER` 按输出 `ceil(H/sH)` 计算总 pad。
  - 输出 `[N, M, Hout, Wout]`，`Hout = (H + pt + pb - dH*(kH-1) - 1)/sH + 1`。
  - im2col 到 `std::vector<float>`（`C*kH*kW x Hout*Wout`），按 group 分块；权重行 `M x C/group*kH*kW`；用 `matmul_row` 或标量行循环累加，bias 加在输出上。
  - 越界 padding 填 0；group 不能整除返回 `InvalidArgument`。
- `DepthwiseConv2DOp`：校验 `group == C`，委托同一个 `detail::conv2d_forward` 函数；不改 `Conv2DOp` 的 `final`。
- `BatchNormOp`：输入 5；按 `N,C` 计算；`inv = 1/sqrt(var+epsilon)`。
- `MaxPool2DOp`/`AvgPool2DOp`：NCHW；padding 窗口按 `ceil_mode` 裁剪；Max 初始 `-inf`；Avg 分母按 `count_include_pad`。
- `GlobalAvgPoolOp`：每 `(n,c)` 对 H*W 求平均。
- `register_builtin_ops` 注册 `Conv2D/DepthwiseConv2D/MaxPool2D/AvgPool2D/GlobalAvgPool/BatchNorm`。
- `traditional_ops.h` 移除已实现类的 stub 定义并 include 新头；`ConvTranspose2D/LayerNorm/GroupNorm/LSTM/GRU/RNN/GBDT*` 继续 `UnimplementedOp`。
- 更新 `tests/interface/ops_families_contract_test.cpp`：从 stub 列表移除已实现的 6 个类，保留其余 27 个；`AllStubCases().size()` 改为 27。

- [ ] **Step 4: 运行确认 GREEN**

```bash
cmake --build build --target ie_tests -j
./build/tests/ie_tests --gtest_filter='TraditionalOpsTest.*:OpsFamiliesTest.*'
```

- [ ] **Step 5: 全量测试并提交**

```bash
ctest --test-dir build --output-on-failure
git add src/ops/traditional src/core/ops/op_registry.cpp CMakeLists.txt tests/ops tests/interface/ops_families_contract_test.cpp tests/reference/gen_traditional_fixtures.py tests/data/traditional_reference.h tests/CMakeLists.txt
git commit -m "feat(p2): implement conv, batchnorm, and pooling ops"
```

---

### Task 4: Gemm / Flatten / Reshape / Transpose

**Files:**
- Create: `src/ops/traditional/gemm_op.h`
- Create: `src/ops/traditional/gemm_op.cpp`
- Create: `src/ops/traditional/layout_ops.h`
- Create: `src/ops/traditional/layout_ops.cpp`
- Modify: `src/ops/traditional/traditional_ops.h`
- Modify: `src/core/ops/op_registry.cpp`
- Modify: `CMakeLists.txt`（`ie_core` 增加 2 个 `.cpp`）
- Test: `tests/ops/traditional_ops_test.cpp`

**Interfaces:**
- Produces: `GemmOp`、`FlattenOp`、`ReshapeOp`、`TransposeOp`，均为 `Op`；注册 `OpType::Gemm/Flatten/Reshape/Transpose`。

**行为：**

- `Gemm`: `Y = alpha * A' * B' + beta * C`；`transA`/`transB` 默认 0；`alpha`/`beta` double 默认 1；C 可缺省，支持标量/`[N]`/`[M]`/`[M,N]` 广播；F32。
- `Flatten`: `axis` int 默认 1；支持负 axis；输出二维 `[prod(before), prod(after)]`。
- `Reshape`: 优先 inputs[1] 作为 shape（F32 读取转 int64）；否则 attr `shape` int list；支持 `0` 复制对应输入维度、`-1` 推断；元素数不匹配返回 `InvalidArgument`。
- `Transpose`: `perm` int list；缺省反转所有维度；校验排列合法。

- [ ] **Step 1: 写失败测试**

```cpp
TEST(TraditionalOpsTest, GemmMatchesReferenceWithTransposeAndBias) {
    // A [2,3], B [4,3] with transB=1, C [4], alpha=1, beta=1
    // 期望 Y[0][0] = sum_k A[0][k]*B[0][k] + C[0]
}

TEST(TraditionalOpsTest, FlattenReshapeTransposePreserveValues) {
    // Flatten axis=1 on [2,3,2] -> [2,6]
    // Reshape [2,3,2] with shape {3,-1} -> [3,4]
    // Transpose perm {1,0,2} on [2,3,2] -> [3,2,2]
}
```

- [ ] **Step 2: 运行确认 RED；Step 3: 实现；Step 4: GREEN**

构建/过滤命令同 Task 3，过滤 `TraditionalOpsTest.*`。

- [ ] **Step 5: 全量测试并提交**

```bash
ctest --test-dir build --output-on-failure
git add src/ops/traditional src/core/ops/op_registry.cpp CMakeLists.txt tests/ops/traditional_ops_test.cpp
git commit -m "feat(p2): implement gemm and layout ops"
```

---

### Task 5: Concat / Slice / Pad / Upsample / Gather / Reduce / Clip / Sigmoid / Tanh

**Files:**
- Create: `src/ops/traditional/data_ops.h`
- Create: `src/ops/traditional/data_ops.cpp`
- Create: `src/ops/traditional/activation_ops.h`
- Create: `src/ops/traditional/activation_ops.cpp`
- Modify: `src/ops/traditional/traditional_ops.h`
- Modify: `src/core/ops/op_registry.cpp`
- Modify: `CMakeLists.txt`（`ie_core` 增加 2 个 `.cpp`）
- Modify: `tests/interface/ops_families_contract_test.cpp`（移除已实现的 5 个 stub）
- Test: `tests/ops/traditional_ops_test.cpp`

**Interfaces:**
- `ConcatOp`, `SliceOp`, `PadOp`, `UpsampleOp`, `GatherOp`, `ReduceOp`, `ClipOp`, `SigmoidOp`, `TanhOp`.
- `ReduceOp` 用 attr `reduce_mode` = `"mean"`/`"sum"`，`axes` int list，`keepdims` int。
- `Slice` 支持 attr `starts/ends/axes/steps` 与 opset 10+ 的 inputs[1..4]（F32 转 int64）。
- `Pad` 支持 attr `pads/value` 与 inputs[1..3]；只实现 `mode=constant`。
- `Upsample` 支持 scales attr 或 input；只实现 nearest；支持 NCHW。
- `Clip` 支持 attr min/max 与 inputs[1..2]。
- `Concat` 校验除 axis 外形状一致；`Gather` 支持 axis 和 indices；`Reduce` 输出 keepdims。

- [ ] **Step 1: 写失败测试**

为每个算子写一个直接张量断言；至少覆盖：

```cpp
TEST(TraditionalOpsTest, ConcatSlicePadGatherReduce) { /* 精确值 */ }
TEST(TraditionalOpsTest, UpsampleNearestAndClipSigmoidTanh) { /* 精确值 + 范围 */ }
TEST(TraditionalOpsTest, SliceHandlesNegativeEndsAndSteps) { /* 反向/步长 */ }
```

- [ ] **Step 2: RED；Step 3: 实现；Step 4: GREEN**

实现后更新 `tests/interface/ops_families_contract_test.cpp`：从 stub 列表移除 `Upsample/Pad/Clip/Sigmoid/Tanh`，`AllStubCases().size()` 从 27 改为 22；保留 `ConvTranspose2D/LayerNorm/GroupNorm/LSTM/GRU/RNN/GBDT*`。

- [ ] **Step 5: 全量测试并提交**

```bash
ctest --test-dir build --output-on-failure
git add src/ops/traditional src/core/ops/op_registry.cpp CMakeLists.txt tests/ops/traditional_ops_test.cpp tests/interface/ops_families_contract_test.cpp
git commit -m "feat(p2): implement data movement and activation ops"
```

---

### Task 6: TraditionalEngine + tiny CNN 端到端

**Files:**
- Modify: `src/engine/traditional/traditional_engine.h`
- Create: `src/engine/traditional/traditional_engine.cpp`
- Modify: `CMakeLists.txt`（`src/engine/traditional/traditional_engine.cpp` 加入 `ie_cpu`）
- Test: `tests/runtime/traditional_engine_test.cpp`
- Fixture/create: `tests/reference/gen_traditional_fixtures.py`（扩展）
- Fixture: `tests/data/tiny_cnn.onnx`
- Fixture: `tests/data/tiny_cnn_reference.h`

**Interfaces:**
- Consumes: Task 2 `load_onnx_model`；Task 3–5 算子；现有 `Engine`。
- Produces:
  - `TraditionalEngine::load_model(const std::string&, ModelFormat)`：仅 `ONNX`，其余 `Unsupported`。
  - `TraditionalEngine::infer(const std::vector<Tensor>&)`：按图中 `is_input` 顺序绑定，返回第一个输出；输入数量不匹配 `InvalidArgument`。
  - `TraditionalEngine::infer_batch(...)`：逐样本调用 `infer`，返回同序结果。

**Tiny CNN fixture：** 1x3x8x8 输入 → Conv(3→4,3x3,pad1) → BatchNorm(4) → ReLU → MaxPool(2x2) → Conv(4→6,3x3,pad1) → ReLU → GlobalAvgPool → Flatten → Gemm(6→3)。导出 ONNX 时 `do_constant_folding=False`，opset 17；用 ONNX Runtime 生成输入/最终 logits/逐层输出，写入 `tiny_cnn.onnx` 与 `tiny_cnn_reference.h`。

- [ ] **Step 1: 写失败测试**

```cpp
TEST(TraditionalEngineTest, LoadsAndRunsTinyCnn) {
    TraditionalEngine engine;
    ASSERT_TRUE(engine.load_model(
        std::string(IE_TEST_DATA_DIR) + "/tiny_cnn.onnx",
        ModelFormat::ONNX).has_value());
    auto input = make_tensor({1, 3, 8, 8}, kTinyCnnInput);
    auto out = engine.infer({input});
    ASSERT_TRUE(out.has_value()) << out.error().message;
    ASSERT_EQ(out->shape, (Shape{{1, 3}}));
    for (std::size_t i = 0; i < kTinyCnnExpected.size(); ++i) {
        EXPECT_NEAR(out->data<float>()[i], kTinyCnnExpected[i], 1e-4f);
    }
}

TEST(TraditionalEngineTest, BatchMatchesSingleAndRejectsWrongFormat) {
    // infer_batch({input, input2}) 与分别 infer 的结果一致
    // load_model("x", ModelFormat::TFLITE) -> Unsupported
}
```

- [ ] **Step 2: RED；Step 3: 实现；Step 4: GREEN**

实现要点：
- `TraditionalEngine` 头文件改用 `std::unique_ptr<Engine>`，析构函数在 `.cpp` 定义；保留现有三个公开方法签名。
- `load_model`：`load_onnx_model` → 收集 `graph.values` 中 `is_input` 的名称 → `engine_ = std::make_unique<Engine>(EngineConfig{})` → `engine_->load(graph)`。
- `infer`：构造 `unordered_map<string, Tensor>`；`engine_->infer`；输出空则 `InvalidArgument`，返回 `outputs[0]`。
- `infer_batch`：逐个调用 `infer`；空 batch 返回空 vector。
- `Engine` 的 `Graph` 拷贝会带上 `weights`；测试 `infer` 两次确认权重仍有效。

- [ ] **Step 5: 全量测试并提交**

```bash
ctest --test-dir build --output-on-failure
git add src/engine/traditional CMakeLists.txt tests/runtime/traditional_engine_test.cpp tests/reference/gen_traditional_fixtures.py tests/data/tiny_cnn.onnx tests/data/tiny_cnn_reference.h tests/CMakeLists.txt
git commit -m "feat(p2): add traditional engine and tiny cnn e2e"
```

---

### Task 7: ResNet50 本地对齐 + 转换入口

**Files:**
- Create: `tests/reference/gen_resnet50_onnx.py`
- Create: `tests/runtime/resnet50_parity_test.cpp`
- Modify: `src/converter/model_converter.h`（`load_model` 改声明）
- Create: `src/converter/model_converter.cpp`
- Modify: `tests/interface/converter_contract_test.cpp`
- Modify: `CMakeLists.txt`（`ie_core` 增加 `model_converter.cpp`）
- Modify: `tests/CMakeLists.txt`（注册 ResNet50 测试）

**Interfaces:**
- `ModelConverter::load_model(path, config)`：`ONNX` → `load_onnx_model`；其他格式 `Unsupported`。`convert_onnx_to_ie` 仍 `Unsupported`（P2b）。
- ResNet50 测试从 `IE_RESNET50_ONNX`、`IE_RESNET50_REFERENCE` 读取；未设置则 `GTEST_SKIP`。

**ResNet50 生成脚本：** `torchvision.models.resnet50(weights=None).eval()`，固定 seed，1x3x224x224 全 1 输入，`torch.onnx.export(..., opset_version=17, do_constant_folding=False)`；用 onnxruntime 生成 1000 维 logits 二进制；脚本输出 ONNX/`.bin` 路径与运行命令。脚本和生成物都不提交。

- [ ] **Step 1: 写失败测试**

```cpp
TEST(ResNet50ParityTest, MatchesOnnxRuntimeReference) {
    const char* onnx = std::getenv("IE_RESNET50_ONNX");
    const char* ref = std::getenv("IE_RESNET50_REFERENCE");
    if (onnx == nullptr || ref == nullptr) {
        GTEST_SKIP() << "set IE_RESNET50_ONNX and IE_RESNET50_REFERENCE";
    }
    TraditionalEngine engine;
    ASSERT_TRUE(engine.load_model(onnx, ModelFormat::ONNX).has_value());
    auto input = make_tensor({1, 3, 224, 224}, std::vector<float>(1 * 3 * 224 * 224, 1.0f));
    auto out = engine.infer({input});
    ASSERT_TRUE(out.has_value());
    auto expected = read_float_binary(ref);
    ASSERT_EQ(out->numel(), 1000);
    float max_err = 0.0f;
    for (std::size_t i = 0; i < 1000; ++i) {
        max_err = std::max(max_err, std::abs(out->data<float>()[i] - expected[i]));
    }
    EXPECT_LT(max_err, 1e-3f);
    EXPECT_EQ(argmax(out->data<float>(), 1000), argmax(expected.data(), 1000));
}

TEST(ConverterTest, LoadsOnnxModelAndRejectsOtherFormats) {
    ModelLoaderConfig cfg;
    cfg.format = ModelFormat::ONNX;
    auto graph = ModelConverter::load_model(
        std::string(IE_TEST_DATA_DIR) + "/tiny_add.onnx", cfg);
    ASSERT_TRUE(graph.has_value());
    EXPECT_FALSE(graph->nodes.empty());
    cfg.format = ModelFormat::TFLITE;
    EXPECT_EQ(ModelConverter::load_model("x", cfg).error().code,
              Error::Code::Unsupported);
}
```

- [ ] **Step 2: RED；Step 3: 实现；Step 4: GREEN**

`model_converter.cpp` 实现 `load_model`；保留其他转换方法 inline `Unsupported`。更新 `converter_contract_test.cpp`：`load_model` 不再无条件断言 Unsupported；改为 CUSTOM 格式 Unsupported + ONNX fixture 成功；`OnnxLoader` 用 tiny_add 成功；其余 loader/mapping 保持 Unsupported/已实现映射。

- [ ] **Step 5: 生成并运行 ResNet50（本地验证）**

```bash
PYTHONPATH=/tmp/opencode/onnxvenv/lib/python3.14/site-packages \
  python3 tests/reference/gen_resnet50_onnx.py --out-dir /tmp/opencode/resnet50
IE_RESNET50_ONNX=/tmp/opencode/resnet50/resnet50.onnx \
IE_RESNET50_REFERENCE=/tmp/opencode/resnet50/resnet50_logits.bin \
  ./build/tests/ie_tests --gtest_filter='ResNet50ParityTest.*'
```

若本机生成/执行超时，记录 `DONE_WITH_CONCERNS`；至少保留 tiny CNN 覆盖，ResNet50 测试可 skip。

- [ ] **Step 6: 全量测试并提交**

```bash
ctest --test-dir build --output-on-failure
git add src/converter tests/runtime/resnet50_parity_test.cpp tests/reference/gen_resnet50_onnx.py tests/interface/converter_contract_test.cpp tests/CMakeLists.txt CMakeLists.txt
git commit -m "test(p2): add resnet50 parity path and converter loader"
```

---

### Task 8: 决策回写、接口清单与最终验证

**Files:**
- Modify: `docs/superpowers/specs/m2-backend-core/p2-traditional-models.md`
- Modify: `docs/superpowers/specs/00-overview/decisions-log.md`
- Modify: `docs/superpowers/specs/m1-bootstrap/p0.5-interface-inventory.md`
- Modify: `todo.md`
- Modify: `progress.md`
- Test: full Debug/Release/ASan/UBSan

**Interfaces:** 仅文档/验证。

- [ ] **Step 1: 回写 P2 决策**

`p2-traditional-models.md` 追加「P2a 实现期裁定」：

- 无第三方 ONNX 依赖，手写 protobuf wire 解析；只覆盖 CNN 子集；未知字段跳过，畸形/外部数据/未知 dtype 报错。
- ONNX 权重存 `Graph::weights`，`WEIGHT_CONST`；`Executor` 注入；`Graph` 拷贝共享权重所有权。
- `AttrValue` 增加 int/float 数组；`OpType::Flatten` 追加在末尾。
- `Flatten` 用独立 Op（不改 `Reshape` 语义）；`Reshape` shape 输入兼容 F32 整数常量。
- Conv 用 im2col+GEMM；`Conv+BN+ReLU` 融合、Winograd/FFT 延后 P2b/P7。
- ResNet50 不提交权重；用本地脚本+环境变量测试；CI 跑 tiny CNN/层对齐。
- `ModelConverter::convert_onnx_to_ie`、`ConversionValidator`、TFLite/Caffe 延后 P2b。
- 算子范围与 ResNet50 所需子集一致；ConvTranspose/GroupNorm/LSTM/GRU 仍 Unsupported。

`decisions-log.md` P2 三行更新为：ONNX 白名单 ✅（列实际覆盖）、Conv 算法 ✅ im2col+GEMM + 融合延后、权重常驻 ✅（`WEIGHT_CONST` + 无预取）；新增 ResNet50 测试策略行。

`p0.5-interface-inventory.md` 追加 P2a 增量头文件：

`core/ir/attr.h`（扩展）、`models/traditional/onnx/onnx_wire.h`、`models/traditional/onnx/onnx_loader.h`、`ops/traditional/conv2d_op.h`、`pool_ops.h`、`batch_norm_op.h`、`gemm_op.h`、`layout_ops.h`、`data_ops.h`、`activation_ops.h`、`engine/traditional/traditional_engine.h`（实现）；说明纯增量/ABI 0.1 不变。

- [ ] **Step 2: 更新 todo/progress**

记录 Task 1–7 提交、测试数、tiny CNN 误差、ResNet50 是否本地运行、跳过原因、延后项。

- [ ] **Step 3: 最终验证**

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Debug
cmake --build build -j
ctest --test-dir build --output-on-failure

cmake -S . -B build-release -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_FLAGS="-Wall -Wextra -Wpedantic"
cmake --build build-release -j
ctest --test-dir build-release --output-on-failure

cmake -S . -B build-asan -G Ninja -DCMAKE_BUILD_TYPE=Debug \
  -DCMAKE_CXX_FLAGS="-fsanitize=address,undefined -fno-omit-frame-pointer"
cmake --build build-asan -j
ASAN_OPTIONS=detect_leaks=1 UBSAN_OPTIONS=halt_on_error=1 \
  ctest --test-dir build-asan --output-on-failure

./build/ie_demo
```

如果 ResNet50 本地资产可用，再运行 Task 7 的 ResNet50 命令；否则在报告中记录 `GTEST_SKIP` 原因。

- [ ] **Step 4: 提交**

```bash
git add docs todo.md progress.md
git commit -m "docs(p2): record traditional inference decisions and verification"
```

---

## Self-Review

- **Spec coverage:** P2 DoD #1 ResNet50 → Task 7（本地）；#2 ONNX loader/IR → Task 2；#3 TraditionalEngine → Task 6；#4 Conv/Pool/BN/GAP 层对齐 → Task 3。P2 范围中的 Concat/Slice/Reshape/Gemm 等 → Task 4/5；`WEIGHT_CONST`/批量 → Task 1/6。converter/validator/其他格式明确延后 P2b。
- **Type consistency:** `Graph::weights` 在 Task 1 定义，Task 2/6 使用；`OnnxModel` 只在 Task 2 内部；`load_onnx_model` Task 2 定义，Task 6/7 使用；`IE_TEST_DATA_DIR` 在 Task 2 定义并在后续测试使用。
- **Review Focus:** 畸形 ONNX → Task 2 测试；NCHW 形状 → Task 3 边界测试；权重生命周期 → Task 1/6 两次 infer；可选输入 → Task 5；批量不变性 → Task 6。
- **比例:** 计划只固定接口、属性和关键断言；算子循环体留给实现者，避免变成代码转录。
