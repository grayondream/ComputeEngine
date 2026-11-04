# 进度

## 背景
仓库 `ComputeEngine` 是一个自研 C++ 推理引擎(CPU + 单一 GPU: CUDA 或 Vulkan),统一 IR 同时支持传统模型(CNN/RNN/MLP/GBDT)与 Transformer,涵盖规则/张量/LLM 推理,预留训练/校验/转换接口,内置调试与可视化工具链。设计文档位于 `docs/readme.md`(2263 行,§0-§21 全栈设计)。

## 目标
把设计文档转译为一整套按里程碑分文件夹、面向实现者的中文 Spec,置于 `docs/superpowers/specs/`,作为各阶段编码依据;并建立决策管理机制。

## 进展与状态

### 2026-10-18 · Spec 编写完成 ✅
- 阅读并理解 `docs/readme.md` 全 21 节。
- 澄清产出初始不确定点:范围=全引擎按里程碑分文件夹;位置=`docs/superpowers/specs/`;语言=中文面向实现者;粒度=里程碑合并。
- 产出 **17 份中文实现者 Spec**(`m1-bootstrap`～`m8-converter` + `00-overview`)。
- 统一结构:目标与验收(DoD)/范围边界/模块清单/接口要点/实现要点/测试验证/依赖/决策点。

### 2026-10-18 · 决策集中盘点 → 用户改为"待实现时定" ◆
- 曾逐项向用户确认自 P0 到 P15-P18 约 30 个决策点(T/V/converter 未逐一确认),并一度写回各 Spec。
- 用户中途提出「回退所有决议,待里程碑实现时再决定」,并确认今后决策方式为 **延迟到实现时定**。
- 处理:
  - 已把全部「已裁定」改写为「决策点(默认建议 · 待实现时定)」,消除硬约束语义。
  - 建立 `00-overview/decisions-log.md` 候选建议台账(决策点+默认建议+副作用)。
  - 总览补「决策约定」段落;T/V/converter 补决策点清单。
  - `todo.md`、`progress.md` 同步此决策模式。
- 结论:本仓库现按"实现时决定、回写 Spec"推进,不再在 Spec 期锁定决策。

### 2026-10-31 · P0-MVP 实现完成 ✅
- 按 `docs/superpowers/plans/2026-10-31-p0-mvp-skeleton.md` 以 TDD 逐 Task 实现(10 个 Task,均先让测试失败再实现)。
- 工程裁定(实现期):
  - C++20(用户要求);因无 `std::expected`,自研 `Status`/`Result<T>`。
  - 目录:代码 `src/`、测试 `tests/`、文档 `docs/`;CMake + Ninja;系统 GoogleTest 1.17。
  - MLIR 延后到 P0.5 复核;vcpkg、Google Benchmark、JSON 序列化延后。
  - `Backend::device()` 返回 `Device` 值;`Graph::backward_graph` 用 `std::shared_ptr<Graph>`。
- 落地模块:base、ir、memory、backend、ops(Add/Mul/MatMul)、hardware(CPU 探测 + 指纹)、scheduler、runtime(Executor/Engine)、CpuBackend、plugin/kv/rng/tuning 骨架、demo。
- 验证:Debug 49/49、Release 49/49 `ctest` 通过;`ie_demo` 输出 `result: 15, 21, 27`。
- 整分支复审后的修复(均 RED→GREEN):
  - Tensor 分配改为通过分配器释放(共享持有 allocator,输出可活过 Engine),并修正 allocator 字节统计。
  - `Shape::is_concrete()` + `Tensor::allocate` 拒绝动态/零/空 shape。
  - CPU kernel 拒绝非 F32、空 buffer;Executor 拒绝空输入 buffer。
  - 注册表拒绝空工厂;create 遇到 null 工厂返回 `Runtime` 错误。
  - `MulOp::configure` 允许无 scalar,恢复双输入 Mul 的 Executor 路径。
- ASan/UBSan 构建 49/49 通过。
- 已回写 `p0-mvp.md` §8 与 `00-overview/decisions-log.md`。

### 2026-11-04 · P0.5 接口冻结(进行中)
- Task 1 配置/特性门控/API 版本已完成(提交 `21dfa9a`)。
- Task 2 编译层与调优接口完成(提交 `a036089`):
  - 新增 `core/ir/pass.h`、`pass_manager.h`、`pass_registry.h`、`pass_plugin.h`。
  - 新增 `core/tuning/kernel.h`、`runtime_autotuner.h`、`hardware_aware_cache.h`、`runtime_hardware_profiler.h`。
  - `KernelVariant` 升级为设计 §6.2(`name/target_kind/req/cost`),`KernelVariantRegistry::select` 按能力位/阈值/`ModelKind` 过滤。
  - 未实现调优接口统一返回 `Error::unsupported(...)`;profiler `replan` 返回 `reason="runtime-profiler-stub"`。
  - 新增 `tests/interface/compiler_tuning_contract_test.cpp`;`ctest` 62/62 通过。
  - 复审修复(提交 `b31d1fa`):补齐 `types.h` 使头文件可独立编译,`replan` 去 `const` 对齐 §6.4,新增 PassRegistry 错误路径、select 全过滤项、PassManager 早退测试;`ctest` 67/67 通过。

## 待办(后续)
- 按里程碑顺序实现,每份 Spec 的 DoD 作验收;实现时裁决各「决策点」并回写。
- P0.5 剩余 Task 3–8 继续按计划推进。