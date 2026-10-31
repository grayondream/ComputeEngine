# P0-MVP · 跑通静态图 CPU 推理

> 设计来源:`docs/readme.md` §16/§17(MVP 骨架与模块清单)、§3(核心接口)、§4(核心抽象)。本 Spec 是编码依据,冲突时以本 Spec 的实现次序与验收为准,语义以设计文档为准。

## 1. 目标与验收(DoD)

- **目标**:跑通"构建静态图 → CPU 执行 → 得结果"的最小闭环;核心接口落地后能在 Release 编译下通过单元测试运行。
- **验收标准**:
  1. 演示程序(仿 §17 骨架)对 `a={1,2,3}, b={4,5,6}`、图 `(a+b)*3` 输出 `{15,21,27}`,首个元素断言 `==15.0f`。
  2. `HardwareDetector::detect_cpu()` 返回 CPU 硬件画像(含 `fingerprint`、`matrix.has_avx2`、`matrix.has_avx512_vnni` 等能力位)。
  3. `BackendRegistry` 静态注册 CPU Backend 并可取用。
  4. 单元测试覆盖 P0 涉及的关键模块(见 §6),在 CI 上通过。
- **代码量参考**:~7000 行(接口 100% 定义,实现 ~15%)。

## 2. 范围与边界

- **范围内**:核心接口骨架、CPU 检测、静态图、同步 CPU 执行、最小算子(Add/Mul/MatMul)、最小内存分配。
- **范围外(本阶段只留接口,不做实现)**:KVCache/RuleEngine/TraditionalEngine/LLMEngine/TraceExporter/AutogradEngine/Trainer/ModelConverter/DeterminismGuard 等仅定义接口。
- **GPU**:本阶段不实现。`DeviceSelector::select()` 返回 `GpuBackend::NONE`。

## 3. 需创建/落地的模块(对应 §15 目录)

```
infer-engine/
  core/base/            Status, Result, Error, DType, Shape, Device, Layout
  core/ir/              Node, Graph, OpType, OpRegistry, Pass(空), ControlFlow(字段)
  core/runtime/         执行器(最小:单次执行), Session(最小), Workspace, Stream
  core/scheduler/       Scheduler(接口), StaticScheduler(MVP 简版)
  core/memory/          Allocator(接口+malloc 包装), Arena/Pool(骨架), TransferManager(空)
  core/backend/         Backend, Device, BackendRegistry, DeviceSelector
  core/plugin/          PluginRegistry/PluginMeta/PluginLoader(空框架)
  core/hardware/        HardwareProfile, CapabilitySet, HardwareDetector(仅 CPU), OptimizationPlanner(骨架)
  core/tuning/          KernelVariant(接口), KernelVariantRegistry(骨架)
  ops/common/           Add, Mul(开发者先落地 MatMul)
  backends/cpu/         CpuBackend(同步), kernels/(最简)
```

## 4. 关键接口冻结清单(本阶段必须可编译可用)

参考 `docs/readme.md` §3.1–§3.7、§4.1、§4.4、§4.5,落地这些头文件:

- `CapabilitySet`(has/set/list)+ `HardwareProfile`(全字段结构,§2.1)
- `HardwareDetector::detect_cpu()`(§2.2)
- `OptimizationPlanner::plan()`(§2.3,可先返回保守默认)
- `KVCachePlugin`(纯虚 + 默认实现,§3.1)
- `Scheduler`(submit/plan_next_iteration/execute_iteration/on_request_complete/preempt,§3.2)+ `BatchPlan`
- `Backend`(init/shutdown/device/alloc/free/copy*/compile/execute_async/wait/is_idle/caps/meta/hardware_profile/memory_policy,§3.3)
- `Graph`(nodes/values/edges/dynamic_dims/control_flow/is_training/backward_graph/bind/build_backward,§3.4)
- `Tensor`(shape/dtype/device/layout/buffer/requires_grad/grad,§3.5)
- `Op`(forward/backward/supports_training/is_control_flow/execute_subgraph,§3.6)
- `BackendCaps`(§3.7 全字段)
- `OpRegistry`(register_op/create/list_by_kind,§4.2)
- `Allocator`(alloc/free/available/free_gradients/free_activations,§4.3)
- `PluginRegistry`(register/get/list/list_by_tag/load_shared_library,§4.4)
- `RNG` + `SamplingRNG`(§4.5)
- `EngineConfig` + `validate(cfg)`(§5.1)
- `BackendMatrix`(§5.5)

## 5. 实现要点

- **正确性**:CPU 后端即 oracle;算子实现必须可复现、顺序无关(结果稳定)。
- **内存**:Allocator 用 `malloc` 包装即可;`MemoryKind` 枚举落地,`HOST/DEVICE/.../WEIGHT_CONST/KV_CACHE` 齐全。
- **静态调度**:MVP 的 `StaticScheduler` 可退化为"一次 prefill,无 decode"的 BatchPlan 直通执行。
- **执行器**:先在 CPU 上以同步 `execute()` 打通,`execute_async/wait` 保留接口。
- **Graph 构建**:落地 `GraphBuilder`(`add/mul/build`,§17 骨架)或等价 DSL,仅作 MVP 快捷入口。

## 6. 测试与验证

- 单测:Status/Result、Shape、Tensor、Graph 绑定、CpuBackend 执行、OpRegistry 注册/创建、HardwareDetector(CPU 能力位)。
- 契约测试:`docs/readme.md` §16 定义"接口契约测试"在 P0.5 验收,但 P0 应为已冻结的接口编译可达。
- 端到端:`(a+b)*3` 演示程序断言。
- 遵循 TDD:先写失败测试再实现;所有测试经 `ctest` 或等价框架跑通。

## 7. 依赖

- P0 为起点,MVP 后交付给 P0.5(接口冻结)。
- 依赖决策点与默认建议见 §8(C++23、CMake+Ninja、vcpkg、GoogleTest、Google Benchmark、MLIR 候选、IR=JSON);实现 P0 时在实现内裁定。

## 8. 决策点(默认建议 · 待实现时裁定)

> 以下为 P0-MVP 期的工程选型**决策点**与**默认建议**(2026-10-18 记录)。按项目约定(见 `00-overview/decisions-log.md`),这些**不在 Spec 期硬锁定**,实现本里程碑时在其实现内裁定并回写本 Spec。

| 决策点 | 默认建议 | 副作用/影响 |
|---|---|---|
| C++ 标准 | **C++20(实现期裁定)** | 用户要求 C++20;无 `std::expected`,改用自研 `Status`/`Result<T>` |
| 构建系统 | **CMake + Ninja** | 后端独立静态库、增量构建(§19) |
| 依赖管理 | **系统 GoogleTest;vcpkg 延后** | P0 仅需 GTest,`find_package(GTest REQUIRED)` |
| 测试框架 | **GoogleTest** | 断言/参数化/CI |
| 基准框架 | **Google Benchmark(延后 P1)** | P0 不做基准 |
| 是否引入 MLIR | **延后到 P0.5 复核;P0 手写 IR** | 多层 IR 仍按设计分层,暂不接 MLIR 依赖 |
| IR 序列化 | **JSON(延后)** | P0 无 exporter/加载器,序列化后续里程碑再落地 |

### MLIR 引入的影响面(如实现时采纳"引入 MLIR"则需一并落地)

- `core/ir/` 的多层 IR 应围绕 MLIR 建:定**自定义 Dialect**(高层语义 Op / 控制流 / 动态维度),用 MLIR Pass infra 做常量折叠/融合/降级。
- CMake 需引入 LLVM/MLIR 依赖(vcpkg 或手工;锁定稳定版本)并接入 `TableGen` 生成 Op 定义。
- 与"模型执行只依赖 IR 语义、不依赖模型类别"不冲突——Dialect 保持模型类别无关。
- `Graph::bind`(§3.4 动态维度绑定)与 MLIR 的动态 shape 机制需在实现时对齐设计。
- 代价:构建时间与体积上升(§19 风险需同时考虑 MLIR 作为核心依赖的地位)。

### 实现期裁定(P0-MVP 已落地,2026-10-31)

实际实现采用以下裁定(覆盖上表的默认建议):

1. **C++20**:用户裁定 C++20。因无 `std::expected`,自研 `Status`/`Result<T>`(基于 `std::optional`,保留 `has_value/error`)。
2. **目录布局**:代码在 `src/`(`src/core/...`、`src/backends/...`、`src/demo/...`),测试在 `tests/`,文档在 `docs/`。
3. **`Backend::device()` 返回 `Device` 值**:设计原文 `Device*` 与 `enum class Device` 同名冲突。
4. **`Graph::backward_graph` 用 `std::shared_ptr<Graph>`**:`std::optional<Graph>` 是递归完整类型,C++20 不可编译。
5. **依赖管理**:vcpkg 延后;P0 用系统 GoogleTest 1.17 + `find_package(GTest)`。
6. **MLIR**:延后到 P0.5 末复核;P0 的 IR 手写并保持分层,便于后续接 Dialect。
7. **Google Benchmark / JSON 序列化**:延后(P1 / 需要 exporter 的里程碑)。

**验收结果**:Debug 与 Release 均 39/39 `ctest` 通过;`ie_demo` 输出 `result: 15, 21, 27`;`HardwareDetector::detect_cpu()` 返回含 `fingerprint`、AVX2 等能力位的画像;`BackendRegistry` 可注册并创建 CPU 后端。