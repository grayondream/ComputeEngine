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

### 2026-11-08 · P1 Task 3 SIMD 内核与能力目录 ✅
- 提交 `3165aab`;新增 `src/core/ops/kernel_fns.h`、`src/core/tuning/cpu_kernel_variants.h`(header-only)、`src/backends/cpu/simd/simd_kernels.h/.cpp`、`src/backends/cpu/kernels/cpu_kernel_catalog.h/.cpp`;两个 `.cpp` 加入 `ie_cpu`。
- `KernelFns` 三分量签名与 brief 一致;`register_cpu_kernel_variants` 注册 12 个变体(MatMul/Relu/Softmax × scalar/sse2/avx2/avx512),前置能力分别为无/`sse2`/`avx2+fma`/`avx512f+avx512bw+avx512vl`,`estimated_tflops` 1/2/4/8;`select_best_cpu_variant` 取最大 TFLOPS,空匹配返回 `NotFound`。
- 内核使用函数级 `__attribute__((target(...)))`,无全局 `-march`;MatMul/ReLU 按 4/8/16 lane 向量化并标量处理尾部,AVX2/AVX-512 使用 FMA;Softmax 先行内减 max、`std::exp` 求和再除法,`cols<=0` 安全返回;非 x86/非 GNU-Clang 全部转发标量占位。
- `CpuKernelCatalog::select` 使用初始化一次的静态 registry,按选中名称映射函数指针;`softmax_sse2` 变体映射到标量 softmax(brief 未要求独立 SSE2 softmax 内核)。
- 验证:TDD 先写测试确认 RED(头文件缺失编译失败),实现后聚焦 5 项通过;Debug/Release 全量均 **131/131** `ctest` 通过、无警告;额外 ASan/UBSan 脚本覆盖尺寸 1/2/3/4/5/7/8/15/16/17/31/32/33/63/64/65、多行 softmax 与退化输入,`matmul/relu/softmax` 失败数均为 0。本机 CPU 无 AVX-512,相应测试按能力位跳过(代码仅编译与审查)。
- 复审修复 round 1(`11ebcd6`):原 ReLU 长度 7 与 `rows=1, cols=7` softmax 未进入 AVX2/AVX-512 向量循环且未调用 `relu_sse2`。新增 ReLU 长度 1/4/7/8/15/16/17/33/65/66 对比(覆盖三种 SIMD 与尾部),新增 `rows=2, cols∈{17,33,65}` 的 softmax 对比与逐行和≈1/有限性断言,行内最大值放大到 1000/700 以确保减 max 的数值稳定性被覆盖。通过临时变异(ReLU 向量 store 去 max、softmax 向量 max→min、向量除法 mul→add)确认 RED 后全部还原;聚焦 7 项、全量 **133/133** 通过,内核实现未改动。

## 待办(后续)
- 按里程碑顺序实现,每份 Spec 的 DoD 作验收;实现时裁决各「决策点」并回写。
- 在 P0.5 冻结接口上增量实现 P1 CPU/硬件探测等能力。

### 2026-11-08 · P1 Task 4 ReLU/Softmax 算子、MatMul 分派与优化计划 ✅
- 提交 `7cd5a6f`;新增 `src/core/ops/common/relu_op.h`、`src/core/ops/common/softmax_op.h`;修改 `matmul_op.h`、`op_registry.cpp`、`execution_context.h`、`executor.h/.cpp`、`optimization_planner.cpp`。
- `ExecutionContext` 增加 `kernels`/`thread_pool` 指针;`Executor::run` 追加默认参数 `kernels=nullptr`、`pool=nullptr`,既有 3 参调用保持兼容。
- `MatMulOp` 保留原校验并优先注入 `matmul_row`;当 `pool != nullptr && m*n*k >= 32768` 时按行 `parallel_for(m, 1, ...)`,否则串行。每行独立、累加顺序不变,线程与串行结果逐字节一致(测试覆盖 64³)。
- `ReluOp` 校验 1 输入/1 输出、F32、allocator,输出 shape 不变;优先注入 `relu`,否则标量 `x>0?x:0`。`SoftmaxOp` 支持 1D/2D(最后一维),减行 max 后 `exp` 求和归一化,其他 rank 返回 `InvalidArgument`,优先注入 `softmax`。
- `register_builtin_ops` 幂等注册 `Relu`/`Softmax`。`OptimizationPlanner::plan` 用 `select_best_cpu_variant` 为 MatMul/Relu/Softmax 选名,失败回退 `*_scalar`,`reason="capability-driven:"+matmul name`。
- 验证:TDD 先确认 RED(缺少 `relu_op.h`/`softmax_op.h` 导致编译失败);实现后聚焦 19 项通过,全量 **145/145** `ctest` 通过;`--clean-first` 重建 `-Wall -Wextra -Wpedantic` 无警告。
- 边界说明:ReLU/Softmax 暂不并行(仅 MatMul 按 brief 要求);`CpuBackend` 尚未注入 kernels/pool,留待 Task 5。

### 2026-11-08 · P1 Task 5 CpuBackend 集成 ✅
- 提交 `9a413fc`;修改 `src/backends/cpu/cpu_backend.h/.cpp`、`tests/backend/backend_test.cpp`。
- `CpuBackend` 成员改为 `HardwareProfile profile_`、`std::unique_ptr<ThreadPool> pool_`、`std::shared_ptr<CpuAllocator> allocator_`、`KernelFns kernels_`、`bool initialized_`(保留 `graph_`);构造时创建 `CpuAllocator`,析构时按 `shutdown` 顺序释放。
- `init`:幂等;`HardwareDetector::detect_cpu()` 失败原样返回,`CpuKernelCatalog::select` 失败原样返回;线程数按 `num_cores * threads_per_core` 计算并钳制到 `[1,64]`;线程池构造异常转为 `Error::runtime`;全部成功后才提交 profile/kernels/pool 并置 `initialized_`。
- `shutdown`:幂等,先 `pool_.reset()` 再 `graph_.reset()`,清空 kernels 并置 `initialized_=false`。
- `hardware_profile()` 返回缓存的 `profile_`;`caps()` 仅置 `supports_deterministic_kernels`/`supports_batch_invariant`,其余保持默认;`alloc/free` 走 `CpuAllocator`(64B 对齐、重复释放安全)。
- `execute_async`:未初始化返回 `InvalidArgument`;保留 `context.allocator` 优先逻辑,并把 `&kernels_`、`pool_.get()` 传入 `Executor::run`;`compile/copy/wait/is_idle/meta/memory_policy` 行为不变。
- 验证:TDD 先写 2 项测试确认 RED(caps 为 false、65B 分配仅 16B 对齐),实现后聚焦 2 项通过,全量 **147/147** `ctest` 通过,`-Wall -Wextra -Wpedantic` 无警告。

### 2026-11-08 · P1 Task 6 PyTorch 参考 MLP、端到端对齐与基准 ✅
- 提交 `1fc4402`;新增 `tests/reference/gen_mlp_reference.py`、`tests/data/mlp_reference.h`、`tests/runtime/mlp_e2e_test.cpp`、`tests/performance/matmul_benchmark_test.cpp`、`src/benchmarks/ie_bench.cpp`;修改顶层与 `tests/CMakeLists.txt`。
- 黄金数据:PyTorch 2.15 float32,seed `20261106`,4x8→16→8→4(ReLU/ReLU/Softmax),9 位小数;脚本重跑两次哈希一致,测试运行期不依赖 Python。
- 6 节点图(`MatMul/Relu/MatMul/Relu/MatMul/Softmax`),权重为图输入;偏置折叠采用齐次常量列(`W1_aug=[9,17]`、`W2_aug=[17,9]`、`W3_aug=[9,4]`),ReLU 精确保留常量 1;修正了实现计划中 `h1r=[4,16]` 与 `w2=[17,8]` 的内维不一致。
- 验证:TDD RED(缺失 `data/mlp_reference.h`;生成后暴露 `MatMul inner dimensions mismatch`)→ GREEN;Engine 路径最大误差 `8.94e-08`,标量路径 `5.96e-08`(容差 `1e-5`);Debug **150/150**(性能用例 skip)、Release **150/150**(AVX2 性能断言通过)。
- `ie_bench`(Ryzen 5 5600X,选中 `matmul_avx2`):64³ / 128³ / 256³ 标量 3.58/1.75/1.62 GFLOPS,选中 23.27/13.39/11.97 GFLOPS;本机无 AVX-512,相关分支仅编译未执行。
- 复审修复(`21ca813`):`CpuBackend::graph_` 由 `std::optional<Graph>` 改为 `std::unique_ptr<Graph>`,消除 GCC 15 在 `std::optional<Graph>` 析构 `shared_ptr<Graph>` 时的 `-Wmaybe-uninitialized`;补齐性能测试 `<cstdint>` 与 `api_support_contract_test` 的聚合初始化,强制 `-Wall -Wextra -Wpedantic` 的全目标 Release 构建零告警;聚焦 5 项、Debug/Release 全量 **150/150** 通过。

### 2026-11-08 · P1 Task 7 决策回写、接口增补与最终验证 ✅
- 提交范围:`7ce453c^..56ac645`(含 Task 1 首提交,共 19 个提交);Task 7 为本次文档提交。
- 文档回写:
  - `docs/superpowers/specs/m2-backend-core/p1-cpu-hardware.md` 追加「§9 实现期裁定」:不引入 Google Benchmark、函数级 target + 能力位分派、AVX-512 主机 skip、偏置折叠齐次坐标、PyTorch 黄金数据离线、持久线程池/固定归约、`CpuAllocator` 64B 对齐且保留 `MallocAllocator`、规范能力位名、VNNI/AMX 仅登记、cost model 按 `estimated_tflops`;并专节记录 17/9 维度修正与 Release `-Wmaybe-uninitialized`(`graph_` 改 `unique_ptr`)。
  - `docs/superpowers/specs/00-overview/decisions-log.md`:P1 能力位深度、cost model 标记 ✅,新增「基准框架:不引入 Google Benchmark,仓库内 `ie_bench`」✅。
  - `docs/superpowers/specs/m1-bootstrap/p0.5-interface-inventory.md`:追加 8 个 P1 增量公开头文件(`kernel_fns.h`、`thread_pool.h`、`cpu_allocator.h`、`relu_op.h`、`softmax_op.h`、`cpu_kernel_variants.h`、`cpu_kernel_catalog.h`、`simd_kernels.h`);纯增量,不改变冻结的 79 头文件/10 契约组与插件 ABI `0.1`。
- 最终验证(全部重新配置并 `--clean-first`/全新构建目录):
  - Debug:`cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Debug` + 构建 + `ctest` → **150/150** 通过;`MatMulBenchmarkSanity` 按设计跳过;构建 0 告警。
  - Release:`cmake -S . -B build-release -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS="-Wall -Wextra -Wpedantic"` + 构建 + `ctest` → **150/150** 通过(含性能 sanity);全目标 **0 warning**。
  - ASan/UBSan:全新 `build-asan`(`-fsanitize=address,undefined -fno-omit-frame-pointer`),`ASAN_OPTIONS=detect_leaks=1 UBSAN_OPTIONS=halt_on_error=1 ctest` → **150/150** 通过(性能用例 skip);构建 0 告警。
  - `./build/ie_demo` 输出仍为 `result: 15, 21, 27`(前有指纹与 `AVX2: 1` / `AVX-512 VNNI: 0`)。
  - `./build-release/ie_bench`:profile `AMD Ryzen 5 5600X 6-Core Processor`,选中 `matmul_avx2`/`relu_avx2`/`softmax_avx2`;64³ **3.66→23.28**、128³ **1.73→13.58**、256³ **1.59→11.71** GFLOPS。
  - MLP 对 PyTorch(沿用 Task 6 黄金数据):Engine 最大误差 **8.94e-08**、标量 Executor **5.96e-08**(容差 `1e-5`)。
- 已知限制/延后:
  - 本机(Zen3)无 AVX-512,AVX-512 数值与吞吐未真机执行;仅编译并通过合成 profile 的分派/目录测试,待 AVX-512 机器补跑。
  - `softmax_sse2` 变体映射到标量 softmax(P1 未单独实现 SSE2 softmax)。
  - ReLU/Softmax 未并行化;仅 MatMul 在 `m*n*k >= 32768` 时按行并行(线程数与串行结果逐字节一致)。
  - VNNI/AMX/BF16/FP16 仅登记能力位,无 F32 内核;INT8/BF16 内核留待 dtype 里程碑;cost model 固定 1/2/4/8,实测微调留 P7。
  - `ThreadPool::parallel_for` 非重入;ARM 探测分支在 x86 主机仅编译未执行。

### 2026-11-14 · P2a Task 1 IR 常量权重与属性数组 ✅
- 提交 `74344fd`;`Graph` 增加 `std::unordered_map<std::string, Tensor> weights`(按值持有,Tensor 拷贝共享 owner,权重生命周期独立于 Executor 分配器);`graph.h` 引入 `core/memory/tensor.h`(无 include 环)。
- `AttrValue` 追加 `std::vector<int64_t>`、`std::vector<double>` 两个 variant 分支(末尾追加,既有索引不变);`OpType::Flatten` 追加在 enum 末尾,既有枚举值不位移。
- `Executor::run` 在读取输入前把 `graph.weights` 注入 `values`:未知权重名返回 `InvalidArgument`,与 input 同名的权重跳过(调用方输入优先);非 input 值缺权重仍由既有 "missing node input value" 返回 `InvalidArgument`。
- 验证:TDD RED(缺 `Graph::weights`、vector variant 分支导致编译失败)→ GREEN;新增 3 项测试(`GraphWeightsTest` 2 项、`AttrValueTest` 1 项);全量 `ctest` **153/153** 通过;`-Wall -Wextra -Wpedantic` 全新构建零告警。

### 2026-11-14 · P2a Task 2 最小 ONNX wire parser 与 OnnxLoader ✅
- 提交 `bc3bbb2`;新增 `src/models/traditional/onnx` 下的 `onnx_wire.h/.cpp` 与 `onnx_loader.h/.cpp`,仓库内手写 protobuf wire 解析(wire type 0/1/2/5,未知字段跳过,截断/组类型/未知 dtype 返回 `InvalidArgument`,外部数据返回 `Unsupported`)。
- `load_onnx_model`:读文件、解析、转 `Graph`;graph input/output/value_info/node 输入输出建 `Value`,initializer 与 `Constant` 节点折叠为 `Graph::weights`(F32、`MallocAllocator`、`MemoryKind::WEIGHT_CONST`,`is_input=false`),weight 的 Tensor owner 共享保持生命周期。
- tensor 转换:FLOAT 用 `float_data`/`raw_data`,INT32/INT64/DOUBLE 逐元素转 F32;标量 `dims` 空存为 `Shape{{1}}`;其他 dtype/外部数据 `Unsupported`。`AttrValue` 映射 `f/i/s/ints/floats`;tensor 属性仅允许 Constant。
- `OnnxMapping` 实现 25 个白名单 op(含 `Conv->Conv2D`、`Flatten`、`ReduceMean/ReduceSum->Reduce`);`OnnxLoader::load` 委托 `load_onnx_model`。为白名单补 `OpType::Gemm`(enum 末尾追加,既有值不变)。
- fixture:`tests/reference/gen_onnx_fixtures.py` 生成并提交 `tests/data/tiny_add.onnx`;测试包含 tiny_add 成功、截断/缺失文件、畸形 wire、未知 dtype、packed 属性、Constant 折叠 + INT64 转 F32、`OnnxMapping` 白名单。
- 验证:TDD RED(CMake 找不到新增 `.cpp`)→ GREEN;聚焦 13 项通过;全量 `ctest` **163/163** 通过(性能用例 skip);`--clean-first` 重建 `-Wall -Wextra -Wpedantic` 零告警。
- 复审修复:`TensorProto` 字段号修正(`raw_data=9`、`double_data=10`、跳过 `uint64_data=11`)、`ints/floats` 多次 wire 出现追加合并、`data_type > UINT32_MAX` 拒绝;新增 raw/typed payload、packed/unpacked 属性、Reduce `reduce_mode`、空 optional name、initializer-as-input 回归;聚焦 17 项、全量 `ctest` **167/167** 通过,零告警。
