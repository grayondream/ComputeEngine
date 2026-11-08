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

### 2026-11-04 · P0.5 接口冻结完成 ✅
- Task 1 配置/特性门控/API 版本完成(提交 `21dfa9a`)。
- Task 2 编译层与调优接口完成(提交 `a036089`):
  - 新增 `core/ir/pass.h`、`pass_manager.h`、`pass_registry.h`、`pass_plugin.h`。
  - 新增 `core/tuning/kernel.h`、`runtime_autotuner.h`、`hardware_aware_cache.h`、`runtime_hardware_profiler.h`。
  - `KernelVariant` 升级为设计 §6.2(`name/target_kind/req/cost`),`KernelVariantRegistry::select` 按能力位/阈值/`ModelKind` 过滤。
  - 未实现调优接口统一返回 `Error::unsupported(...)`;profiler `replan` 返回 `reason="runtime-profiler-stub"`。
  - 新增 `tests/interface/compiler_tuning_contract_test.cpp`;复审修复(提交 `b31d1fa`):补齐头文件独立编译、`replan` 去 `const`、PassRegistry 错误路径与全过滤项测试。
- Task 3 运行时执行器、传输与混合层完成(提交 `155e8ab`,复审修复 `7a171e6`):
  - 新增 `core/runtime/iterative_executor.h`、`core/backend/backend_plugin.h`、`core/scheduler/scheduler_plugin.h`、`core/hardware/hardware_probe_plugin.h`。
  - 新增 `hybrid/hybrid_config.h`、`planner/hybrid_planner.h`(含其后的 `ExecutionPlan`)、`scheduler/hybrid_scheduler.h`、`memory/hybrid_memory.h`、`orchestrator/hybrid_orchestrator.h`。
  - `HybridConfig` 移入 `hybrid/hybrid_config.h` 并按 §7.2 扩展;`engine_config.h` 改为 include,`EngineConfig` 字段不变。
  - `TransferManager` 增加路由表与 `transfer/prefetch_async/wait`;`copy` 同设备为 no-op(空指针/零字节按契约校验)、跨设备返回 `Unsupported`;未实现能力统一返回 `Unsupported`。
- Task 4 规则推理层完成(提交 `4129dae`):
  - 新增 `rules/fact/fact.h`、`rules/rule/rule.h`、`rules/agenda/agenda.h`、`rules/rule_engine.h`。
  - `RuleEngine` 保存 facts/rules/trace;`fire()` 返回 `Unsupported`;新增 `ops/rule/rule_ops.h` 三个规则算子。
- Task 5 算子族与实验算子完成(提交 `a43ff96`,测试补强 `877fe86`):
  - 新增 `core/ops/unimplemented_op.h`、`core/ir/op_plugin.h`。
  - 新增 19 个传统算子、8 个 Transformer 算子、4 个控制流算子,以及 `FlashAttentionOp`/`MoEDispatchOp`。
  - 所有算子 `forward` 返回 `Unsupported`;控制流算子 `is_control_flow()==true`。
- Task 6 LLM 层完成(提交 `73634d3`):
  - 新增 `llm/tokenizer`、`llm/model_loader`、`llm/transformer`、`llm/sampler`、`llm/streaming` 接口。
  - 新增 `engine/traditional/traditional_engine.h`、`experimental/kv_cache/paged_kv_cache.h`、`experimental/speculative/speculative_decoder.h`。
  - 插件接口保持纯声明;未实现方法返回 `Unsupported`;`PagedKVCache` 的 `clear` 返回 `{}`、`memory_usage` 为 0、`api_version="0.1"`。
- Task 7 训练层完成(提交 `7e3afd2`,补强 `7ca506b`):
  - 新增 autograd/optimizer/lr_scheduler/loss/dataloader/trainer/amp/checkpoint/distributed 九个头文件。
  - 训练插件为纯接口;`AutogradEngine` 可记录与切换训练态,`backward/grad` 返回 `Unsupported`;`Trainer`/`CheckpointIO`/`AmpManager`/`DistributedContext` 未实现方法返回 `Unsupported`。
- Task 8 校验与调试层完成(提交 `815dfd3`):
  - 新增 `validation/reference_tracer/trace_ir.h`、`reference_tracer.h`、`tensor_hash/tensor_hash.h`、`determinism/determinism.h`、`comparator/comparator.h`、`cross_framework/cross_framework.h`、`report/report.h`、`config.h`。
  - `ValidationConfig` 按 §10.7 全字段落地并通过 `engine_config.h` include;所有功能方法返回 `Unsupported`。
- Task 9 模型转换层与量化插件完成(提交 `63b18ca`,复审修复 `b28daf5`):
  - 新增 `converter/model_converter.h`、`format_loader/format_loader.h`、`op_mapping/op_mapping.h`、`validation/conversion_validator.h`、`core/plugin/quant_plugin.h`、`quantization/quant_plugins.h`。
  - 未实现转换/量化方法返回 `Unsupported`;四个量化插件 `api_version` 均为 `0.1`。
  - 转换输出统一使用 `Status`(规范 void 结果),未新增 `Result<void>` 特化。
- Task 10 C ABI、支撑层与文档回写完成(提交 `fa46124`):
  - 新增 `api/c_api/ie_api.h`(`IE_API_VERSION_MAJOR=0`/`MINOR=1`、`IE_API`、三个 `extern "C"` 函数)与 `ie_c_api.cpp`;`ie_create_backend` 仅识别 `"cpu"`,`ie_destroy_backend(nullptr)` 安全,`ie_api_version()` 恒为 `"0.1"`;`.cpp` 已加入 `ie_cpu`。
  - `plugin_api.h` 增加 `kApiVersionString`;`ie_api.h` 用 `static_assert` 锁定宏与 C++ 常量一致。
  - 新增 `api/openai_compat`、`api/grpc`、`api/cli`、`security`、`fault_tolerance`、`observability`、`tools`、`sdk`、`packaging`、`benchmarks` 支撑层接口;`AuditLog::record/entries` 内存留存,安全/容错/可观测等生命周期方法返回 `{}`/空（`Sandbox` 生命周期除外，统一返回 `Unsupported`）,功能方法返回 `Unsupported`;`OpenAIServer`/`GrpcService`/`CliApp` 的 `start/run/stop` 均返回 `Unsupported`。
  - 新增 `docs/superpowers/specs/m1-bootstrap/p0.5-interface-inventory.md`(79 个头文件 / 10 个契约组);回写 `p0.5-interface-freeze.md` §8 与 `decisions-log.md`。
  - 新增 `tests/interface/api_support_contract_test.cpp`(8 项)与 `inventory_test.cpp`(1 项)。
- 最终验证:Debug **116/116**、Release **116/116**、ASan/UBSan **116/116** `ctest` 通过;`./build/ie_demo` 仍输出 `result: 15, 21, 27`;P0-MVP 既有 e2e/49 项基线回归通过。
- 实现期裁定:不引入 MLIR(自定义 `Graph::bind`)、插件 ABI 提前锁定 `0.1`、JSON 延后、`KernelVariant` 对齐 §6.2、`FeatureFlags` 位于 `core/config`、`HybridConfig`/`ValidationConfig` 独立头文件、`Status` 为规范 void 结果。
- 整分支复审最终修复:
  - `FeatureFlags` 明确为唯一权威全局门控,legacy bools 与 `hardware_adapt`/`validation` 开关仅作镜像/收窄;新增默认镜像一致性测试。
  - `TransferManager::transfer` 非零字节先拒绝空指针,再处理同设备别名/路由;`copy` 同设备改为 no-op、跨设备 `Unsupported`。
  - `Sandbox` 生命周期统一返回 `Unsupported`;C ABI 使用 `new (std::nothrow)`;插件注册表补充所有权/生命周期注释。
  - 补齐转换/校验 9 组表驱动 `Unsupported` 默认行为测试;inventory 测试断言 10 组计数合计 79;文档方法数/Sandbox 描述与代码同步。

### 2026-11-08 · P1 Task 1 硬件探测平台化 ✅
- 提交 `7ce453c`;`detector.h` 新增 `CpuFeatures`/`CpuTopology` 与 `build_cpu_profile`,`detect_cpu` 改为「探测 → 构建 profile」两段式,保持 `Result<HardwareProfile>` 签名与 P0.5 冻结接口不变。
- x86:标准指令集用 `__builtin_cpu_supports`(含 `fma`),AMX 用 CPUID leaf 7 补齐并以 AMX-TILE 位做前置校验;ARM:用 `getauxval(AT_HWCAP/AT_HWCAP2)` 探测 NEON/SVE/SVE2/SME/SME2(编译期隔离,未在 x86 主机运行)。
- Linux 拓扑:从 `/proc/cpuinfo` 读 vendor/model 与物理核、`/sys/.../cache/index*` 读 L2/L3、`/proc/meminfo` 读总/可用内存、`/sys/devices/system/node/online` 读 NUMA;非 Linux 回退到 `hardware_concurrency()` 与 1 个 NUMA 节点。
- 指纹纳入 vendor/device、核心数、L2/L3 与全部启用 caps。
- 验证:新增 3 项测试(Debug)后全量 **119/119** `ctest` 通过,编译无警告。
- 初审修复:修正 aarch64 NEON 为 `HWCAP_ASIMD`(bit 1)并优先使用 `<asm/hwcap.h>` 常量;ARM 探测分支限定 `__linux__`,非 Linux ARM 回退全 false;测试补充 FMA cap 与 `matrix_unit_generation` 断言。

### 2026-11-08 · P1 Task 2 CPU 分配器与确定性线程池 ✅
- 提交 `ace983f`;新增 `src/backends/cpu/cpu_allocator.h`(header-only)与 `src/core/runtime/thread_pool.h/.cpp`。
- `CpuAllocator`:`posix_memalign`(Windows `_aligned_malloc`)提供 64B 对齐;按 64 倍数向上取整统计 `outstanding_bytes`;mutex + `unordered_map` 跟踪;`alloc(0)` 返回 `InvalidArgument`;`free(nullptr)` 与重复释放安全;`available` 返回 `SIZE_MAX`。
- `ThreadPool`:构造 `max(1, threads)` 逻辑线程,`threads<=1` 不启动 worker 且 `parallel_for` 在调用线程执行 `fn(0,total)`;多线程用原子 chunk 索引按 `grain` 切分,worker 与调用线程共同消费,互不重叠;`active_` + condition_variable 等待本轮完成;析构置停止位并 join;构造期线程创建失败时先 join 已创建线程再重抛。
- 明确裁定:`parallel_for` 非重入(任务内不得再次调用同一 pool),并发调用由内部 mutex 串行化。
- 验证:新增 4 项测试(Debug),全量 **123/123** `ctest` 通过,`-Wall -Wextra -Wpedantic` 无警告;额外用 TSan(200 轮多线程压力)与 ASan/UBSan(分配器/线程池)本地验证通过。
- 复审修复 round 1(`9cba3f2`):调用线程任务抛异常时先等待本轮 `active_==0`、清空 `job_` 再重抛,避免 worker 引用已销毁任务;头文件明确任务体不得抛出(worker 内异常调用 `std::terminate`)。补齐 `total==0`、`grain==0`/超大 grain、连续两轮 `parallel_for`、区间合法性测试;全量 **126/126** 通过。

## 待办(后续)
- 按里程碑顺序实现,每份 Spec 的 DoD 作验收;实现时裁决各「决策点」并回写。
- 在 P0.5 冻结接口上增量实现 P1 CPU/硬件探测等能力。
